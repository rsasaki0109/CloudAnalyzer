//! 2.5D cut/fill volume between two surfaces (like CloudCompare's
//! "Compute 2.5D volume").
//!
//! Both surfaces are rasterised onto the same XY grid; each cell's height
//! difference times its area gives the volume added (after above before) or
//! removed (after below before).

use crate::mesh::TriangleMesh;

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
    pub fn center(&self, i: usize, j: usize) -> [f64; 2] {
        [
            self.min[0] + (i as f64 + 0.5) * self.cell,
            self.min[1] + (j as f64 + 0.5) * self.cell,
        ]
    }

    fn index(&self, x: f64, y: f64) -> Option<usize> {
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
    /// Cells where both surfaces are defined, and all cells.
    pub matched_cells: usize,
    pub total_cells: usize,
}

impl VolumeResult {
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
    if params.cell.is_nan() || params.cell <= 0.0 {
        return None;
    }
    let grid = grid_for(&[before, after], params.cell)?;
    let raster = |s: Surface| {
        let mut h = rasterize(s, &grid, params.height);
        if params.fill_empty {
            fill_empty(&mut h, &grid);
        }
        h
    };
    let (b, a) = (raster(before), raster(after));
    let area = grid.cell * grid.cell;
    let (mut added, mut removed, mut added_area, mut removed_area, mut matched) =
        (0.0, 0.0, 0.0, 0.0, 0);
    for (hb, ha) in b.iter().zip(&a) {
        let d = ha - hb;
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
        total_cells: b.len(),
        before: b,
        after: a,
        added,
        removed,
        added_area,
        removed_area,
        matched_cells: matched,
    })
}

fn grid_for(surfaces: &[Surface], cell: f64) -> Option<Grid> {
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
fn rasterize(surface: Surface, grid: &Grid, mode: CellHeight) -> Vec<f64> {
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
fn fill_empty(h: &mut [f64], grid: &Grid) {
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
