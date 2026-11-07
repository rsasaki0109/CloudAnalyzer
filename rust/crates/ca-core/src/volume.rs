//! 2.5D cut/fill volume between two surfaces (like CloudCompare's
//! "Compute 2.5D volume").
//!
//! Both surfaces are rasterised onto the same XY grid; each cell's height
//! difference times its area gives the volume added (after above before) or
//! removed (after below before).

use crate::mesh::TriangleMesh;
use std::cmp::{Ordering, Reverse};
use std::collections::BinaryHeap;

pub const HEIGHT_REVIEW_CELL_LIMIT: usize = 50;

/// Original f64 grid center and heights, not a source-point correspondence.
#[derive(Debug, Clone, PartialEq)]
pub struct HeightReviewCell {
    pub index: usize,
    pub x: f64,
    pub y: f64,
    pub before: f64,
    pub after: f64,
    pub difference: f64,
}

#[derive(Debug)]
struct RankedHeightCell(HeightReviewCell);
impl PartialEq for RankedHeightCell {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}
impl Eq for RankedHeightCell {}
impl PartialOrd for RankedHeightCell {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}
impl Ord for RankedHeightCell {
    fn cmp(&self, other: &Self) -> Ordering {
        self.0
            .difference
            .abs()
            .total_cmp(&other.0.difference.abs())
            .then_with(|| other.0.index.cmp(&self.0.index))
    }
}

/// One side of a volume comparison.
#[derive(Debug, Clone, Copy)]
pub enum Surface<'a> {
    Points(&'a [[f64; 3]]),
    Mesh(&'a TriangleMesh),
    /// A horizontal plane at this height.
    Constant(f64),
}

/// How a cell's height is derived from the points (or triangles) in it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CellHeight {
    Mean,
    Min,
    Max,
}

#[derive(Debug, Clone, Copy)]
pub struct VolumeParams {
    /// Grid cell edge length.
    pub cell: f64,
    pub height: CellHeight,
    /// Fill cells a surface does not cover from their neighbours.
    pub fill_empty: bool,
}

/// The raster both surfaces share.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Grid {
    pub min: [f64; 2],
    pub cell: f64,
    pub nx: usize,
    pub ny: usize,
}

impl Grid {
    pub fn valid(&self) -> bool {
        self.cell.is_finite()
            && self.cell > 0.0
            && self.min.iter().all(|v| v.is_finite())
            && self.nx > 0
            && self.ny > 0
            && self
                .nx
                .checked_mul(self.ny)
                .is_some_and(|n| n <= 200_000_000)
            && (self.cell * self.cell).is_finite()
            && self.cell * self.cell > 0.0
            && self
                .center(self.nx - 1, self.ny - 1)
                .iter()
                .all(|v| v.is_finite())
    }

    pub fn center(&self, i: usize, j: usize) -> [f64; 2] {
        [
            self.min[0] + (i as f64 + 0.5) * self.cell,
            self.min[1] + (j as f64 + 0.5) * self.cell,
        ]
    }

    pub(crate) fn index(&self, x: f64, y: f64) -> Option<usize> {
        let i = ((x - self.min[0]) / self.cell).floor();
        let j = ((y - self.min[1]) / self.cell).floor();
        if i < 0.0 || j < 0.0 {
            return None;
        }
        let (i, j) = (i as usize, j as usize);
        // Points exactly on the far edge belong to the last cell.
        let i = if i == self.nx { i - 1 } else { i };
        let j = if j == self.ny { j - 1 } else { j };
        (i < self.nx && j < self.ny).then_some(j * self.nx + i)
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct VolumeResult {
    pub grid: Grid,
    /// Per-cell heights (row-major, `j * nx + i`); NaN where undefined.
    pub before: Vec<f64>,
    pub after: Vec<f64>,
    /// Volume where `after` is above `before` (fill) and below it (cut),
    /// both positive.
    pub added: f64,
    pub removed: f64,
    pub added_area: f64,
    pub removed_area: f64,
    /// Cells where both surfaces are defined, and cells in the review region
    /// (the full grid when no region is supplied).
    pub matched_cells: usize,
    pub total_cells: usize,
}

/// User-defined height screening, independent of measurement uncertainty.
#[derive(Debug, Clone, PartialEq)]
pub struct HeightReview {
    /// Largest absolute exceedances; ties ordered by ascending row-major index.
    pub cells: Vec<HeightReviewCell>,
    pub threshold: f64,
    pub rise_cells: usize,
    pub fall_cells: usize,
    pub within_cells: usize,
    pub rise_area: f64,
    pub fall_area: f64,
    pub rise_volume: f64,
    pub fall_volume: f64,
}

impl VolumeResult {
    /// Strict `d > threshold` / `d < -threshold` on the original f64 heights.
    /// Equality is within the band. Volumes use the full difference, not the
    /// excess over the threshold. Undefined/outside/masked cells stay excluded.
    pub fn height_review(&self, threshold: f64) -> Option<HeightReview> {
        if !threshold.is_finite()
            || threshold < 0.0
            || !self.grid.valid()
            || self.before.len() != self.grid.nx * self.grid.ny
            || self.after.len() != self.before.len()
        {
            return None;
        }
        let mut result = HeightReview {
            cells: Vec::new(),
            threshold,
            rise_cells: 0,
            fall_cells: 0,
            within_cells: 0,
            rise_area: 0.0,
            fall_area: 0.0,
            rise_volume: 0.0,
            fall_volume: 0.0,
        };
        let area = self.grid.cell * self.grid.cell;
        let mut top =
            BinaryHeap::<Reverse<RankedHeightCell>>::with_capacity(HEIGHT_REVIEW_CELL_LIMIT);
        for (index, (&b, &a)) in self.before.iter().zip(&self.after).enumerate() {
            if b.is_nan() || a.is_nan() {
                continue;
            }
            let d = a - b;
            if !b.is_finite() || !a.is_finite() || !d.is_finite() {
                return None;
            }
            if d > threshold {
                result.rise_cells += 1;
                result.rise_volume += d * area;
            } else if d < -threshold {
                result.fall_cells += 1;
                result.fall_volume -= d * area;
            } else {
                result.within_cells += 1;
            }
            if d.abs() > threshold {
                let [x, y] = self.grid.center(index % self.grid.nx, index / self.grid.nx);
                let candidate = RankedHeightCell(HeightReviewCell {
                    index,
                    x,
                    y,
                    before: b,
                    after: a,
                    difference: d,
                });
                if top.len() < HEIGHT_REVIEW_CELL_LIMIT {
                    top.push(Reverse(candidate));
                } else if top.peek().is_some_and(|smallest| candidate > smallest.0) {
                    top.pop();
                    top.push(Reverse(candidate));
                }
            }
        }
        let mut listed: Vec<_> = top.into_iter().map(|v| v.0).collect();
        listed.sort_unstable_by(|a, b| b.cmp(a));
        result.cells = listed.into_iter().map(|v| v.0).collect();
        result.rise_area = result.rise_cells as f64 * area;
        result.fall_area = result.fall_cells as f64 * area;
        (result.rise_cells + result.fall_cells + result.within_cells == self.matched_cells
            && [
                result.rise_area,
                result.fall_area,
                result.rise_volume,
                result.fall_volume,
            ]
            .iter()
            .all(|v| v.is_finite()))
        .then_some(result)
    }

    pub fn net(&self) -> f64 {
        self.added - self.removed
    }

    /// `after - before` per cell (NaN where either is undefined).
    pub fn difference(&self) -> Vec<f64> {
        self.after
            .iter()
            .zip(&self.before)
            .map(|(a, b)| a - b)
            .collect()
    }
}

/// Compute the volume between `before` and `after`. The grid covers the XY
/// extent of the non-constant surfaces. Returns `None` if both surfaces are
/// constant or empty, or the cell size is not positive.
pub fn volume(before: Surface, after: Surface, params: VolumeParams) -> Option<VolumeResult> {
    volume_in_region(before, after, params, None)
}

/// Keep cells whose centers are inside a closed world-XY rectangle.
/// Rasterization, interpolation and grid origin still use the full surfaces.
/// Included cells contribute their full area; boundary squares are not clipped.
/// Outside cells are undefined in the returned rasters and excluded from counts.
pub fn volume_in_region(
    before: Surface,
    after: Surface,
    params: VolumeParams,
    region: Option<[f64; 4]>,
) -> Option<VolumeResult> {
    let grid = grid_for(&[before, after], params.cell)?;
    volume_on_grid(before, after, params, region, grid)
}

/// Compute on an explicit shared grid, e.g. the union of several survey visits.
/// Each comparison still excludes cells lacking either height; a shared grid
/// does not imply a common observed footprint across visits.
pub fn volume_on_grid(
    before: Surface,
    after: Surface,
    params: VolumeParams,
    region: Option<[f64; 4]>,
    grid: Grid,
) -> Option<VolumeResult> {
    volume_on_grid_with_support(before, after, params, region, grid, None)
}

/// A row-major 0/1 mask intersects raw observed support across loaded surveys.
/// Excluded cells remain uncompared, including cells observed by only this pair.
pub fn volume_on_grid_with_support(
    before: Surface,
    after: Surface,
    params: VolumeParams,
    region: Option<[f64; 4]>,
    grid: Grid,
    support: Option<&[u8]>,
) -> Option<VolumeResult> {
    if let Some(b) = region
        && (!b.iter().all(|v| v.is_finite()) || b[0] >= b[2] || b[1] >= b[3])
    {
        return None;
    }
    if !grid.valid() || grid.cell != params.cell {
        return None;
    }
    if let Some(mask) = support
        && (params.fill_empty
            || !matches!(before, Surface::Points(_))
            || !matches!(after, Surface::Points(_))
            || mask.len() != grid.nx * grid.ny
            || mask.iter().any(|&v| v > 1))
    {
        return None;
    }
    let raster = |s: Surface| {
        let mut h = match (s, support) {
            (Surface::Points(points), Some(_)) => {
                observed_point_heights(points, &grid, params.height)
            }
            _ => rasterize(s, &grid, params.height),
        };
        if params.fill_empty {
            fill_empty(&mut h, &grid);
        }
        h
    };
    let (mut b, mut a) = (raster(before), raster(after));
    let area = grid.cell * grid.cell;
    let (mut added, mut removed, mut added_area, mut removed_area, mut matched) =
        (0.0, 0.0, 0.0, 0.0, 0);
    let mut total_cells = 0;
    for (k, (hb, ha)) in b.iter_mut().zip(&mut a).enumerate() {
        if let Some(bounds) = region {
            let [x, y] = grid.center(k % grid.nx, k / grid.nx);
            if x < bounds[0] || x > bounds[2] || y < bounds[1] || y > bounds[3] {
                *hb = f64::NAN;
                *ha = f64::NAN;
                continue;
            }
        }
        total_cells += 1;
        if support.is_some_and(|mask| mask[k] == 0) {
            *hb = f64::NAN;
            *ha = f64::NAN;
            continue;
        }
        let d = *ha - *hb;
        if d.is_nan() {
            continue;
        }
        matched += 1;
        if d > 0.0 {
            added += d * area;
            added_area += area;
        } else if d < 0.0 {
            removed -= d * area;
            removed_area += area;
        }
    }
    Some(VolumeResult {
        grid,
        total_cells,
        before: b,
        after: a,
        added,
        removed,
        added_area,
        removed_area,
        matched_cells: matched,
    })
}

/// Intersect a mask with cells having a finite raw (unfilled) point height.
/// Constant planes and triangle interpolation are not point observations.
pub fn intersect_observed_support(
    surface: Surface,
    grid: &Grid,
    height: CellHeight,
    mask: &mut [u8],
) -> Option<usize> {
    let Surface::Points(points) = surface else {
        return None;
    };
    if !grid.valid() || mask.len() != grid.nx * grid.ny || mask.iter().any(|&v| v > 1) {
        return None;
    }
    let heights = observed_point_heights(points, grid, height);
    let mut count = 0;
    for (bit, h) in mask.iter_mut().zip(heights) {
        *bit &= u8::from(h.is_finite());
        count += *bit as usize;
    }
    Some(count)
}

fn observed_point_heights(points: &[[f64; 3]], grid: &Grid, height: CellHeight) -> Vec<f64> {
    let mut acc = Accumulator::new(grid.nx * grid.ny, height);
    for p in points {
        if p.iter().all(|v| v.is_finite())
            && let Some(k) = grid.index(p[0], p[1])
        {
            acc.add(k, p[2]);
        }
    }
    acc.finish()
}

pub(crate) fn grid_for(surfaces: &[Surface], cell: f64) -> Option<Grid> {
    if !cell.is_finite() || cell <= 0.0 {
        return None;
    }
    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    let mut extend = |p: &[f64; 3]| {
        for a in 0..2 {
            lo[a] = lo[a].min(p[a]);
            hi[a] = hi[a].max(p[a]);
        }
    };
    for s in surfaces {
        match s {
            Surface::Points(points) => points.iter().for_each(&mut extend),
            Surface::Mesh(mesh) => mesh.vertices.iter().for_each(&mut extend),
            Surface::Constant(_) => {}
        }
    }
    if !(lo[0].is_finite() && lo[1].is_finite()) {
        return None;
    }
    // As CloudCompare: the lowest corner of the data is the centre of the
    // first cell, so a point goes to the cell whose centre is nearest.
    let count = |a: usize| ((hi[a] - lo[a]) / cell + 0.5).floor() as usize + 1;
    let (nx, ny) = (count(0), count(1));
    // Refuse absurd grids rather than exhausting memory.
    if nx.checked_mul(ny)? > 200_000_000 {
        return None;
    }
    Some(Grid {
        min: [lo[0] - 0.5 * cell, lo[1] - 0.5 * cell],
        cell,
        nx,
        ny,
    })
}

/// Height of `surface` per grid cell (NaN where it has no data).
pub(crate) fn rasterize(surface: Surface, grid: &Grid, mode: CellHeight) -> Vec<f64> {
    let n = grid.nx * grid.ny;
    match surface {
        Surface::Constant(z) => vec![z; n],
        Surface::Points(points) => {
            let mut acc = Accumulator::new(n, mode);
            for p in points {
                if let Some(k) = grid.index(p[0], p[1]) {
                    acc.add(k, p[2]);
                }
            }
            acc.finish()
        }
        Surface::Mesh(mesh) => {
            let mut acc = Accumulator::new(n, mode);
            for t in 0..mesh.triangles.len() {
                let [a, b, c] = mesh.corners(t);
                rasterize_triangle(grid, a, b, c, |k, z| acc.add(k, z));
            }
            acc.finish()
        }
    }
}

/// Call `emit(cell, height)` for every cell whose center lies inside the
/// triangle's XY projection, with the height interpolated at that center.
fn rasterize_triangle(
    grid: &Grid,
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
    mut emit: impl FnMut(usize, f64),
) {
    let det = (b[1] - c[1]) * (a[0] - c[0]) + (c[0] - b[0]) * (a[1] - c[1]);
    if det.abs() < 1e-300 {
        return; // vertical or degenerate: no footprint
    }
    let min_x = a[0].min(b[0]).min(c[0]);
    let max_x = a[0].max(b[0]).max(c[0]);
    let min_y = a[1].min(b[1]).min(c[1]);
    let max_y = a[1].max(b[1]).max(c[1]);
    let to_cell =
        |v: f64, lo: f64, n: usize| (((v - lo) / grid.cell - 0.5).ceil().max(0.0) as usize).min(n);
    let (i0, i1) = (
        to_cell(min_x, grid.min[0], grid.nx),
        to_cell(max_x, grid.min[0], grid.nx),
    );
    let (j0, j1) = (
        to_cell(min_y, grid.min[1], grid.ny),
        to_cell(max_y, grid.min[1], grid.ny),
    );
    const EPS: f64 = -1e-12;
    for j in j0..=j1.min(grid.ny - 1) {
        for i in i0..=i1.min(grid.nx - 1) {
            let [x, y] = grid.center(i, j);
            let l1 = ((b[1] - c[1]) * (x - c[0]) + (c[0] - b[0]) * (y - c[1])) / det;
            let l2 = ((c[1] - a[1]) * (x - c[0]) + (a[0] - c[0]) * (y - c[1])) / det;
            let l3 = 1.0 - l1 - l2;
            if l1 >= EPS && l2 >= EPS && l3 >= EPS {
                emit(j * grid.nx + i, l1 * a[2] + l2 * b[2] + l3 * c[2]);
            }
        }
    }
}

struct Accumulator {
    mode: CellHeight,
    value: Vec<f64>,
    count: Vec<u32>,
}

impl Accumulator {
    fn new(n: usize, mode: CellHeight) -> Self {
        let init = match mode {
            CellHeight::Mean => 0.0,
            CellHeight::Min => f64::INFINITY,
            CellHeight::Max => f64::NEG_INFINITY,
        };
        Self {
            mode,
            value: vec![init; n],
            count: vec![0; n],
        }
    }

    fn add(&mut self, k: usize, z: f64) {
        self.count[k] += 1;
        let v = &mut self.value[k];
        match self.mode {
            CellHeight::Mean => *v += z,
            CellHeight::Min => *v = v.min(z),
            CellHeight::Max => *v = v.max(z),
        }
    }

    fn finish(self) -> Vec<f64> {
        self.value
            .iter()
            .zip(&self.count)
            .map(|(&v, &n)| match (n, self.mode) {
                (0, _) => f64::NAN,
                (n, CellHeight::Mean) => v / n as f64,
                _ => v,
            })
            .collect()
    }
}

/// Fill NaN cells with the mean of their defined 8-neighbours, growing
/// inward from the data until nothing changes. Cells that cannot be reached
/// (no data at all) stay NaN.
pub(crate) fn fill_empty(h: &mut [f64], grid: &Grid) {
    let (nx, ny) = (grid.nx as isize, grid.ny as isize);
    loop {
        let mut updates = Vec::new();
        for j in 0..ny {
            for i in 0..nx {
                let k = (j * nx + i) as usize;
                if !h[k].is_nan() {
                    continue;
                }
                let (mut sum, mut n) = (0.0, 0);
                for dj in -1..=1 {
                    for di in -1..=1 {
                        let (x, y) = (i + di, j + dj);
                        if (di, dj) == (0, 0) || x < 0 || y < 0 || x >= nx || y >= ny {
                            continue;
                        }
                        let v = h[(y * nx + x) as usize];
                        if !v.is_nan() {
                            sum += v;
                            n += 1;
                        }
                    }
                }
                if n > 0 {
                    updates.push((k, sum / n as f64));
                }
            }
        }
        if updates.is_empty() {
            return;
        }
        for (k, v) in updates {
            h[k] = v;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn height_screening_lists_largest_f64_cells_with_stable_ties_and_bounded_storage() {
        let points: Vec<_> = (0..70)
            .map(|i| {
                [
                    1000.0 + i as f64,
                    2000.0,
                    if i % 2 == 0 { 2.0 } else { -2.0 },
                ]
            })
            .collect();
        let r = volume(
            Surface::Constant(0.0),
            Surface::Points(&points),
            params(1.0),
        )
        .unwrap();
        let h = r.height_review(1.0).unwrap();
        assert_eq!((h.rise_cells, h.fall_cells), (35, 35));
        assert_eq!(h.cells.len(), HEIGHT_REVIEW_CELL_LIMIT);
        assert_eq!(
            h.cells.iter().map(|c| c.index).collect::<Vec<_>>(),
            (0..50).collect::<Vec<_>>()
        );
        assert_eq!(
            (
                h.cells[0].x,
                h.cells[0].y,
                h.cells[0].before,
                h.cells[0].after
            ),
            (1000.0, 2000.0, 0.0, 2.0)
        );
        assert!(r.height_review(2.0).unwrap().cells.is_empty());
        let mut points = points;
        points[69][2] = 10.0;
        let h = volume(
            Surface::Constant(0.0),
            Surface::Points(&points),
            params(1.0),
        )
        .unwrap()
        .height_review(1.0)
        .unwrap();
        assert_eq!(h.cells[0].index, 69);
        assert_eq!(h.cells[1].index, 0);
        assert_eq!(h.cells.last().unwrap().index, 48);
    }

    #[test]
    fn height_screening_strict_edges_full_volumes_and_reversal() {
        let before: Vec<_> = (0..6).map(|i| [i as f64 * 2.0, 0.0, 0.0]).collect();
        let after: Vec<_> = [2.0, -3.0, 0.5, -0.5, 0.0]
            .into_iter()
            .enumerate()
            .map(|(i, z)| [i as f64 * 2.0, 0.0, z])
            .collect();
        for height in [CellHeight::Mean, CellHeight::Min, CellHeight::Max] {
            let p = VolumeParams {
                cell: 2.0,
                height,
                fill_empty: false,
            };
            let r = volume(Surface::Points(&before), Surface::Points(&after), p).unwrap();
            let original = (
                r.added,
                r.removed,
                r.difference()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
            );
            let h = r.height_review(0.5).unwrap();
            assert_eq!((h.rise_cells, h.fall_cells, h.within_cells), (1, 1, 3));
            assert_eq!(
                (h.rise_area, h.fall_area, h.rise_volume, h.fall_volume),
                (4.0, 4.0, 8.0, 12.0)
            );
            let zero = r.height_review(0.0).unwrap();
            assert_eq!(
                (zero.rise_cells, zero.fall_cells, zero.within_cells),
                (2, 2, 1)
            );
            assert_eq!((zero.rise_volume, zero.fall_volume), (10.0, 14.0));
            assert_eq!(
                (
                    r.added,
                    r.removed,
                    r.difference()
                        .iter()
                        .map(|v| v.to_bits())
                        .collect::<Vec<_>>()
                ),
                original
            );
            let reverse = volume(Surface::Points(&after), Surface::Points(&before), p)
                .unwrap()
                .height_review(0.5)
                .unwrap();
            assert_eq!((reverse.rise_volume, reverse.fall_volume), (12.0, 8.0));
        }
    }

    #[test]
    fn height_screening_uses_f64_not_rounded_display_and_rejects_unrepresentable_values() {
        let before = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let above = 1.0 + f64::EPSILON;
        let after = [[0.0, 0.0, above], [1.0, 0.0, -above], [2.0, 0.0, 1.0]];
        assert_eq!(above as f32, 1.0);
        let r = volume(
            Surface::Points(&before),
            Surface::Points(&after),
            params(1.0),
        )
        .unwrap();
        let h = r.height_review(1.0).unwrap();
        assert_eq!((h.rise_cells, h.fall_cells, h.within_cells), (1, 1, 1));
        assert_eq!(
            h.cells.iter().map(|c| c.index).collect::<Vec<_>>(),
            vec![0, 1]
        );
        assert_eq!(h.cells[0].difference, above);
        for invalid in [-1.0, f64::NAN, f64::INFINITY] {
            assert!(r.height_review(invalid).is_none());
        }
        let huge = [[0.0, 0.0, f64::MAX], [1.0, 0.0, f64::MAX]];
        let overflow = volume(Surface::Constant(0.0), Surface::Points(&huge), params(1.0)).unwrap();
        assert!(overflow.height_review(1.0).is_none());
        let overflow = volume(
            Surface::Constant(-f64::MAX),
            Surface::Points(&huge),
            params(1.0),
        )
        .unwrap();
        assert!(overflow.height_review(1.0).is_none());
    }

    #[test]
    fn height_screening_respects_region_missing_and_common_mask_and_reports_empty_support() {
        let before = [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]];
        let after = [[0.0, 0.0, 2.0], [1.0, 0.0, -100.0], [2.0, 0.0, 10.0]];
        let grid = grid_for(&[Surface::Points(&before), Surface::Points(&after)], 1.0).unwrap();
        let r = volume_on_grid_with_support(
            Surface::Points(&before),
            Surface::Points(&after),
            params(1.0),
            Some([-0.1, -0.1, 1.1, 0.1]),
            grid,
            Some(&[1, 0, 1]),
        )
        .unwrap();
        let h = r.height_review(0.5).unwrap();
        assert_eq!(
            (
                r.total_cells,
                r.matched_cells,
                h.rise_cells,
                h.fall_cells,
                h.within_cells
            ),
            (2, 1, 1, 0, 0)
        );
        assert_eq!(h.rise_volume, 2.0);
        assert_eq!(h.cells.iter().map(|c| c.index).collect::<Vec<_>>(), vec![0]);
        let empty = volume_in_region(
            Surface::Points(&before),
            Surface::Points(&after),
            params(1.0),
            Some([10.0, 10.0, 11.0, 11.0]),
        )
        .unwrap()
        .height_review(0.0)
        .unwrap();
        assert_eq!(
            (empty.rise_cells, empty.fall_cells, empty.within_cells),
            (0, 0, 0)
        );
        assert_eq!((empty.rise_volume, empty.fall_volume), (0.0, 0.0));
        let missing = volume(
            Surface::Points(&before[..1]),
            Surface::Points(&after[1..]),
            params(1.0),
        )
        .unwrap();
        assert_eq!(missing.matched_cells, 0);
        assert_eq!(missing.height_review(0.0).unwrap().within_cells, 0);
    }

    /// Points on a regular grid with spacing `step` over `[0, size)^2`,
    /// lifted by `height(x, y)`. With `step` equal to the cell size there is
    /// one point at the centre of each cell (as CloudCompare, the grid puts
    /// the lowest point at a cell centre), so volumes come out exact.
    fn surface(size: f64, step: f64, height: impl Fn(f64, f64) -> f64) -> Vec<[f64; 3]> {
        let n = (size / step).round() as usize;
        let mut out = Vec::new();
        for j in 0..n {
            for i in 0..n {
                let (x, y) = ((i as f64 + 0.5) * step, (j as f64 + 0.5) * step);
                out.push([x, y, height(x, y)]);
            }
        }
        out
    }

    fn params(cell: f64) -> VolumeParams {
        VolumeParams {
            cell,
            height: CellHeight::Mean,
            fill_empty: false,
        }
    }

    #[test]
    fn common_heights_ignore_nonfinite_xyz_in_both_support_and_volume() {
        let baseline = [[0., 0., 10.], [f64::NAN, 0., 999.]];
        let visit = [[0., 0., 12.], [0., f64::INFINITY, -999.]];
        let grid = Grid {
            min: [-0.5, -0.5],
            cell: 1.,
            nx: 1,
            ny: 1,
        };
        let mut mask = vec![1];
        for points in [&baseline[..], &visit[..]] {
            assert_eq!(
                intersect_observed_support(
                    Surface::Points(points),
                    &grid,
                    CellHeight::Mean,
                    &mut mask
                ),
                Some(1)
            );
        }
        let result = volume_on_grid_with_support(
            Surface::Points(&baseline),
            Surface::Points(&visit),
            params(1.),
            None,
            grid,
            Some(&mask),
        )
        .unwrap();
        assert_eq!(
            (result.added, result.removed, result.matched_cells),
            (2., 0., 1)
        );
    }

    #[test]
    fn observed_intersection_corrects_equal_counts_with_different_footprints() {
        let baseline = [[0., 0., 10.], [1., 0., 10.], [2., 0., 10.]];
        let first = [[0., 0., 12.], [1., 0., 9.]];
        let second = [[1., 0., 14.], [2., 0., 13.]];
        let grid = Grid {
            min: [-0.5, -0.5],
            cell: 1.,
            nx: 3,
            ny: 1,
        };
        for height in [CellHeight::Mean, CellHeight::Min, CellHeight::Max] {
            let params = VolumeParams {
                cell: 1.,
                height,
                fill_empty: false,
            };
            let mut mask = vec![1; 3];
            for points in [&baseline[..], &first[..], &second[..]] {
                intersect_observed_support(Surface::Points(points), &grid, height, &mut mask)
                    .unwrap();
            }
            assert_eq!(mask, vec![0, 1, 0]);
            let run = |points, support| {
                volume_on_grid_with_support(
                    Surface::Points(&baseline),
                    Surface::Points(points),
                    params,
                    None,
                    grid,
                    support,
                )
                .unwrap()
            };
            let a = run(&first, None);
            let b = run(&second, None);
            assert_eq!((a.matched_cells, b.matched_cells), (2, 2));
            assert_eq!(b.net() - a.net(), 6.);
            let a = run(&first, Some(mask.as_slice()));
            let b = run(&second, Some(mask.as_slice()));
            assert_eq!(
                (
                    a.added,
                    a.removed,
                    b.added,
                    b.removed,
                    a.matched_cells,
                    b.matched_cells
                ),
                (0., 1., 4., 0., 1, 1)
            );
            assert_eq!(b.net() - a.net(), 5.);
            let outside = volume_on_grid_with_support(
                Surface::Points(&baseline),
                Surface::Points(&first),
                params,
                Some([-0.1, -0.1, 0.1, 0.1]),
                grid,
                Some(&mask),
            )
            .unwrap();
            assert_eq!((outside.matched_cells, outside.total_cells), (0, 1));
        }
    }

    #[test]
    fn observed_support_rejects_unsafe_masks_and_interpolation_and_preserves_empty_support() {
        let grid = Grid {
            min: [-0.5, -0.5],
            cell: 1.,
            nx: 3,
            ny: 1,
        };
        let points = [[0., 0., f64::NAN], [1., 0., f64::INFINITY], [2., 0., 10.]];
        let mut mask = vec![1; 3];
        assert_eq!(
            intersect_observed_support(
                Surface::Points(&points),
                &grid,
                CellHeight::Mean,
                &mut mask
            ),
            Some(1)
        );
        assert_eq!(mask, vec![0, 0, 1]);
        assert!(
            intersect_observed_support(Surface::Constant(0.), &grid, CellHeight::Mean, &mut mask)
                .is_none()
        );
        assert_eq!(mask, vec![0, 0, 1]);
        for invalid in [vec![1, 1], vec![1, 2, 1]] {
            let mut copy = invalid.clone();
            assert!(
                intersect_observed_support(
                    Surface::Points(&points),
                    &grid,
                    CellHeight::Mean,
                    &mut copy
                )
                .is_none()
            );
            assert_eq!(copy, invalid);
            assert!(
                volume_on_grid_with_support(
                    Surface::Points(&points),
                    Surface::Points(&points),
                    params(1.),
                    None,
                    grid,
                    Some(&invalid)
                )
                .is_none()
            );
        }
        let mut filled = params(1.);
        filled.fill_empty = true;
        assert!(
            volume_on_grid_with_support(
                Surface::Points(&points),
                Surface::Points(&points),
                filled,
                None,
                grid,
                Some(&mask)
            )
            .is_none()
        );
        let empty = [0, 0, 0];
        let r = volume_on_grid_with_support(
            Surface::Points(&points),
            Surface::Points(&points),
            params(1.),
            None,
            grid,
            Some(&empty),
        )
        .unwrap();
        assert_eq!(
            (r.matched_cells, r.total_cells, r.added, r.removed),
            (0, 3, 0., 0.)
        );
    }

    #[test]
    fn shared_grid_keeps_visit_cells_aligned_and_preserves_missing_support() {
        let baseline = [[0., 0., 10.], [1., 0., 10.], [2., 0., 10.]];
        let visit = [[0., 0., 12.], [1., 0., 9.]];
        let params = VolumeParams {
            cell: 1.,
            height: CellHeight::Mean,
            fill_empty: false,
        };
        let grid = Grid {
            min: [-1.1, -0.5],
            cell: 1.,
            nx: 4,
            ny: 1,
        };
        let result = volume_on_grid(
            Surface::Points(&baseline),
            Surface::Points(&visit),
            params,
            None,
            grid,
        )
        .unwrap();
        assert_eq!(
            (
                result.added,
                result.removed,
                result.matched_cells,
                result.total_cells
            ),
            (2., 1., 2, 4)
        );
        let selected = volume_on_grid(
            Surface::Points(&baseline),
            Surface::Points(&visit),
            params,
            Some([0.39, -0.1, 0.41, 0.1]),
            grid,
        )
        .unwrap();
        assert_eq!(
            (
                selected.added,
                selected.removed,
                selected.matched_cells,
                selected.total_cells
            ),
            (2., 0., 1, 1)
        );
        let native = volume_in_region(
            Surface::Points(&baseline),
            Surface::Points(&visit),
            params,
            Some([0.39, -0.1, 0.41, 0.1]),
        )
        .unwrap();
        assert_eq!(native.total_cells, 0);
    }

    #[test]
    fn shared_grid_rejects_invalid_dimensions_origin_cell_and_region_before_allocation() {
        let points = [[0., 0., 0.]];
        let params = VolumeParams {
            cell: 1.,
            height: CellHeight::Mean,
            fill_empty: false,
        };
        let good = Grid {
            min: [-0.5, -0.5],
            cell: 1.,
            nx: 1,
            ny: 1,
        };
        for grid in [
            Grid { nx: 0, ..good },
            Grid { ny: 0, ..good },
            Grid {
                nx: usize::MAX,
                ny: 2,
                ..good
            },
            Grid {
                nx: 200_000_001,
                ..good
            },
            Grid { cell: 2., ..good },
            Grid {
                min: [f64::NAN, 0.],
                ..good
            },
        ] {
            assert!(
                volume_on_grid(
                    Surface::Points(&points),
                    Surface::Constant(1.),
                    params,
                    None,
                    grid
                )
                .is_none()
            );
        }
        assert!(
            volume_on_grid(
                Surface::Points(&points),
                Surface::Constant(1.),
                params,
                Some([1., 0., 0., 1.]),
                good
            )
            .is_none()
        );
    }

    #[test]
    fn review_region_keeps_grid_origin_closed_edges_and_missing_cell_accounting() {
        let before: Vec<_> = (0..6)
            .map(|i| [1000.0 + (i % 3) as f64, 2000.0 + (i / 3) as f64, 10.0])
            .collect();
        let after: Vec<_> = before[..5]
            .iter()
            .enumerate()
            .map(|(i, p)| [p[0], p[1], p[2] + [2.0, -1.0, 0.0, 0.0, 0.0][i]])
            .collect();
        let full = volume(
            Surface::Points(&before),
            Surface::Points(&after),
            params(1.0),
        )
        .unwrap();
        let selected = volume_in_region(
            Surface::Points(&before),
            Surface::Points(&after),
            params(1.0),
            Some([1001.0, 1999.5, 1002.0, 2001.5]),
        )
        .unwrap();
        assert_eq!(selected.grid, full.grid);
        assert_eq!((selected.total_cells, selected.matched_cells), (4, 3));
        assert_eq!(
            (selected.added, selected.removed, selected.removed_area),
            (0.0, 1.0, 1.0)
        );
        assert!(selected.before[0].is_nan() && selected.after[0].is_nan());
        assert!(selected.before[3].is_nan() && selected.after[3].is_nan());
        assert!(selected.after[5].is_nan()); // Missing inside, distinct from outside.
        assert_eq!((full.added, full.removed, full.total_cells), (2.0, 1.0, 6));
        // A narrow region keeps a full square cell, not its intersected fraction.
        let narrow = volume_in_region(
            Surface::Points(&before),
            Surface::Points(&after),
            params(1.0),
            Some([1000.9, 1999.9, 1001.1, 2000.1]),
        )
        .unwrap();
        assert_eq!(
            (narrow.total_cells, narrow.removed_area, narrow.removed),
            (1, 1.0, 1.0)
        );
    }

    #[test]
    fn review_region_handles_empty_selection_invalid_bounds_and_full_grid_interpolation() {
        let points = [[0.0, 0.0, 2.0], [2.0, 0.0, 2.0]];
        let area = Some([0.9, -0.1, 1.1, 0.1]);
        let raw = volume_in_region(
            Surface::Constant(0.0),
            Surface::Points(&points),
            params(1.0),
            area,
        )
        .unwrap();
        assert_eq!((raw.total_cells, raw.matched_cells, raw.added), (1, 0, 0.0));
        let mut interpolated = params(1.0);
        interpolated.fill_empty = true;
        let filled = volume_in_region(
            Surface::Constant(0.0),
            Surface::Points(&points),
            interpolated,
            area,
        )
        .unwrap();
        assert_eq!(
            (filled.total_cells, filled.matched_cells, filled.added),
            (1, 1, 2.0)
        );
        let empty = volume_in_region(
            Surface::Constant(0.0),
            Surface::Points(&points),
            params(1.0),
            Some([10.0, 10.0, 11.0, 11.0]),
        )
        .unwrap();
        assert_eq!(
            (
                empty.total_cells,
                empty.matched_cells,
                empty.added,
                empty.removed
            ),
            (0, 0, 0.0, 0.0)
        );
        assert!(empty.before.iter().all(|v| v.is_nan()));
        for bounds in [
            [0.0, 0.0, 0.0, 1.0],
            [2.0, 0.0, 1.0, 1.0],
            [0.0, f64::NAN, 1.0, 1.0],
        ] {
            assert!(
                volume_in_region(
                    Surface::Constant(0.0),
                    Surface::Points(&points),
                    params(1.0),
                    Some(bounds)
                )
                .is_none()
            );
        }
    }

    #[test]
    fn mound_over_a_flat_constant() {
        // A 10 x 10 m block 2 m high on a 40 x 40 m site: 200 m3 of fill.
        let after = surface(40.0, 1.0, |x, y| {
            if (10.0..20.0).contains(&x) && (10.0..20.0).contains(&y) {
                2.0
            } else {
                0.0
            }
        });
        let r = volume(Surface::Constant(0.0), Surface::Points(&after), params(1.0)).unwrap();
        assert!((r.added - 200.0).abs() < 1e-9, "{}", r.added);
        assert_eq!(r.removed, 0.0);
        assert!((r.added_area - 100.0).abs() < 1e-9);
        assert_eq!(r.matched_cells, r.total_cells);
    }

    #[test]
    fn pit_and_mound_between_two_clouds() {
        let before = surface(30.0, 1.0, |_, _| 5.0);
        let after = surface(30.0, 1.0, |x, _| {
            if x < 10.0 {
                4.0
            } else if x >= 20.0 {
                6.5
            } else {
                5.0
            }
        });
        let r = volume(
            Surface::Points(&before),
            Surface::Points(&after),
            params(1.0),
        )
        .unwrap();
        assert!(
            (r.removed - 10.0 * 30.0 * 1.0).abs() < 1e-9,
            "{}",
            r.removed
        );
        assert!((r.added - 10.0 * 30.0 * 1.5).abs() < 1e-9, "{}", r.added);
        assert!((r.net() - 150.0).abs() < 1e-9);
    }

    #[test]
    fn mesh_surface_equals_the_plane_it_describes() {
        // Tilted plane z = 0.1 x as a two-triangle mesh vs the same plane
        // sampled by points. The mesh reaches half a cell past the points so
        // the points sit at cell centres.
        let (lo, hi) = (-0.25, 20.25);
        let mesh = TriangleMesh {
            vertices: vec![
                [lo, lo, 0.1 * lo],
                [hi, lo, 0.1 * hi],
                [hi, hi, 0.1 * hi],
                [lo, hi, 0.1 * lo],
            ],
            triangles: vec![[0, 1, 2], [0, 2, 3]],
        };
        let points = surface(20.0, 0.5, |x, _| 0.1 * x + 1.0);
        let r = volume(Surface::Mesh(&mesh), Surface::Points(&points), params(0.5)).unwrap();
        // 1 m above the plane everywhere: 20 x 20 m -> 400 m3.
        assert!((r.added - 400.0).abs() < 1e-6, "{}", r.added);
        assert!(r.removed < 1e-9);
        let scoped = volume_in_region(
            Surface::Mesh(&mesh),
            Surface::Points(&points),
            params(0.5),
            Some([0.25, 0.25, 9.75, 9.75]),
        )
        .unwrap();
        assert!((scoped.added - 100.0).abs() < 1e-6);
        assert_eq!((scoped.total_cells, scoped.matched_cells), (400, 400));
    }

    #[test]
    fn empty_cells_are_skipped_or_filled() {
        // A 10 x 10 site with a 4 x 4 hole in the after-survey.
        let after: Vec<[f64; 3]> = surface(10.0, 1.0, |_, _| 1.0)
            .into_iter()
            .filter(|p| !((3.0..7.0).contains(&p[0]) && (3.0..7.0).contains(&p[1])))
            .collect();
        let skip = volume(Surface::Constant(0.0), Surface::Points(&after), params(1.0)).unwrap();
        assert!((skip.added - 84.0).abs() < 1e-9, "{}", skip.added);
        assert_eq!(skip.total_cells - skip.matched_cells, 16);
        let filled = volume(
            Surface::Constant(0.0),
            Surface::Points(&after),
            VolumeParams {
                fill_empty: true,
                ..params(1.0)
            },
        )
        .unwrap();
        assert!((filled.added - 100.0).abs() < 1e-9, "{}", filled.added);
    }

    #[test]
    fn min_and_max_cell_heights() {
        // Both within half a cell of the first point, i.e. in one cell.
        let pts = [[0.2, 0.2, 1.0], [0.6, 0.6, 3.0]];
        let run = |height| {
            volume(
                Surface::Constant(0.0),
                Surface::Points(&pts),
                VolumeParams {
                    cell: 1.0,
                    height,
                    fill_empty: false,
                },
            )
            .unwrap()
            .added
        };
        assert_eq!(run(CellHeight::Min), 1.0);
        assert_eq!(run(CellHeight::Max), 3.0);
        assert_eq!(run(CellHeight::Mean), 2.0);
    }

    #[test]
    fn rejects_degenerate_input() {
        assert!(volume(Surface::Constant(0.0), Surface::Constant(1.0), params(1.0)).is_none());
        assert!(
            volume(
                Surface::Constant(0.0),
                Surface::Points(&[[0.0; 3]]),
                params(0.0)
            )
            .is_none()
        );
    }
}
