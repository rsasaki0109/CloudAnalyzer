//! Rasterize a cloud into a height grid, a DEM / DSM (like CloudCompare's
//! "Rasterize").
//!
//! Uses the XY grid of the 2.5D volume, so a raster and a volume computed
//! with the same cell size line up cell for cell.

use crate::volume::{self, CellHeight, Grid, Surface};
use crate::{AttributeValues, CLASSIFICATION, PointCloud};

/// How a cell's height is derived from its points.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum RasterHeight {
    Mean,
    Min,
    Max,
    /// Percentile in `[0, 100]`, interpolated linearly between ranks
    /// (50 is the median).
    Percentile(f64),
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RasterParams {
    /// Grid cell edge length.
    pub cell: f64,
    pub height: RasterHeight,
    /// Interpolate empty cells from their neighbours.
    pub fill_empty: bool,
    /// Only use points of this class (e.g. 2 = ground, for a DTM).
    pub class: Option<u8>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Raster {
    pub grid: Grid,
    /// Per-cell heights (row-major from the lowest y, `j * nx + i`); NaN
    /// where there is no data.
    pub heights: Vec<f64>,
    /// Cells that had points (before any filling).
    pub populated_cells: usize,
}

/// Rasterize `cloud`. The grid covers the whole cloud even with a class
/// filter, so e.g. a ground-only DTM and a full DSM share their grid.
/// Returns `None` if the cell size or percentile is invalid, or no point is
/// selected (empty cloud, no classes, or none of the requested class).
pub fn rasterize(cloud: &PointCloud, params: RasterParams) -> Option<Raster> {
    if params.cell.is_nan() || params.cell <= 0.0 {
        return None;
    }
    if let RasterHeight::Percentile(p) = params.height
        && !(0.0..=100.0).contains(&p)
    {
        return None;
    }
    let grid = volume::grid_for(&[Surface::Points(&cloud.positions)], params.cell)?;
    let selected: Vec<[f64; 3]>;
    let points = match params.class {
        None => &cloud.positions,
        Some(class) => {
            let AttributeValues::U8(classes) = &cloud.attribute(CLASSIFICATION)?.values else {
                return None;
            };
            selected = cloud
                .positions
                .iter()
                .zip(classes)
                .filter(|&(_, &c)| c == class)
                .map(|(p, _)| *p)
                .collect();
            &selected
        }
    };
    if points.is_empty() {
        return None;
    }
    let mut heights = match params.height {
        RasterHeight::Mean => volume::rasterize(Surface::Points(points), &grid, CellHeight::Mean),
        RasterHeight::Min => volume::rasterize(Surface::Points(points), &grid, CellHeight::Min),
        RasterHeight::Max => volume::rasterize(Surface::Points(points), &grid, CellHeight::Max),
        RasterHeight::Percentile(p) => percentiles(points, &grid, p),
    };
    let populated_cells = heights.iter().filter(|h| !h.is_nan()).count();
    if params.fill_empty {
        volume::fill_empty(&mut heights, &grid);
    }
    Some(Raster {
        grid,
        heights,
        populated_cells,
    })
}

/// The `p`-th percentile of each cell's heights. Points are bucketed by
/// cell (a counting sort) so each cell's heights can be sorted in place.
fn percentiles(points: &[[f64; 3]], grid: &Grid, p: f64) -> Vec<f64> {
    let n = grid.nx * grid.ny;
    let mut start = vec![0usize; n + 1];
    for q in points {
        if let Some(k) = grid.index(q[0], q[1]) {
            start[k + 1] += 1;
        }
    }
    for k in 0..n {
        start[k + 1] += start[k];
    }
    let mut z = vec![0.0; start[n]];
    let mut next = start.clone();
    for q in points {
        if let Some(k) = grid.index(q[0], q[1]) {
            z[next[k]] = q[2];
            next[k] += 1;
        }
    }
    (0..n)
        .map(|k| percentile(&mut z[start[k]..start[k + 1]], p))
        .collect()
}

fn percentile(values: &mut [f64], p: f64) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.sort_unstable_by(f64::total_cmp);
    let rank = p / 100.0 * (values.len() - 1) as f64;
    let lo = rank.floor() as usize;
    let hi = (lo + 1).min(values.len() - 1);
    values[lo] + (values[hi] - values[lo]) * rank.fract()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Attribute;

    fn params(cell: f64, height: RasterHeight) -> RasterParams {
        RasterParams {
            cell,
            height,
            fill_empty: false,
            class: None,
        }
    }

    fn cloud(positions: Vec<[f64; 3]>) -> PointCloud {
        PointCloud {
            positions,
            ..PointCloud::default()
        }
    }

    #[test]
    fn sloped_plane_gives_the_plane_height_at_each_cell() {
        // z = 0.5 x + 0.25 y sampled every 0.5 m over [0, 10] x [0, 6]. The
        // grid puts the lowest point at a cell centre, so with 1 m cells
        // centred on whole metres every cell but the first row / column
        // holds the four points at (c - 0.5 | c, c - 0.5 | c).
        let plane = |x: f64, y: f64| 0.5 * x + 0.25 * y;
        let mut pts = Vec::new();
        for j in 0..=12 {
            for i in 0..=20 {
                let (x, y) = (i as f64 * 0.5, j as f64 * 0.5);
                pts.push([x, y, plane(x, y)]);
            }
        }
        let run = |height| rasterize(&cloud(pts.clone()), params(1.0, height)).unwrap();
        let (mean, min, max) = (
            run(RasterHeight::Mean),
            run(RasterHeight::Min),
            run(RasterHeight::Max),
        );
        let g = mean.grid;
        assert_eq!((g.nx, g.ny), (11, 7));
        assert_eq!(mean.populated_cells, 77);
        for j in 1..g.ny {
            for i in 1..g.nx {
                let [x, y] = g.center(i, j);
                let k = j * g.nx + i;
                let near = |a: f64, b: f64| (a - b).abs() < 1e-12;
                assert!(near(mean.heights[k], plane(x - 0.25, y - 0.25)), "{i},{j}");
                assert!(near(min.heights[k], plane(x - 0.5, y - 0.5)), "{i},{j}");
                assert!(near(max.heights[k], plane(x, y)), "{i},{j}");
            }
        }
        // The first cell holds just the lowest point.
        assert_eq!(mean.heights[0], 0.0);
    }

    #[test]
    fn percentile_interpolates_between_ranks() {
        let pts: Vec<[f64; 3]> = [1.0, 5.0, 2.0, 4.0, 3.0]
            .iter()
            .map(|&z| [0.1, 0.1, z])
            .collect();
        let run = |p| {
            rasterize(
                &cloud(pts.clone()),
                params(1.0, RasterHeight::Percentile(p)),
            )
            .unwrap()
            .heights[0]
        };
        assert_eq!(run(0.0), 1.0);
        assert_eq!(run(50.0), 3.0);
        assert_eq!(run(100.0), 5.0);
        assert_eq!(run(90.0), 4.6);
        assert!(rasterize(&cloud(pts), params(1.0, RasterHeight::Percentile(101.0))).is_none());
    }

    #[test]
    fn class_filter_keeps_the_grid_and_fill_covers_the_holes() {
        // Ground (class 2) at z = 0 everywhere except under a "building"
        // (class 6) roof at z = 5 over the middle cells.
        let mut pts = Vec::new();
        let mut classes = Vec::new();
        for j in 0..5 {
            for i in 0..5 {
                let roof = (1..4).contains(&i) && (1..4).contains(&j);
                pts.push([i as f64, j as f64, if roof { 5.0 } else { 0.0 }]);
                classes.push(if roof { 6 } else { 2 });
            }
        }
        let cloud = PointCloud {
            positions: pts,
            colors: None,
            attributes: vec![Attribute {
                name: CLASSIFICATION.into(),
                values: AttributeValues::U8(classes),
            }],
        };
        let dsm = rasterize(&cloud, params(1.0, RasterHeight::Max)).unwrap();
        assert_eq!(dsm.heights[2 * 5 + 2], 5.0);
        let ground = RasterParams {
            class: Some(2),
            ..params(1.0, RasterHeight::Max)
        };
        let dtm = rasterize(&cloud, ground).unwrap();
        assert_eq!(dtm.grid, dsm.grid);
        assert_eq!(dtm.populated_cells, 16);
        assert!(dtm.heights[2 * 5 + 2].is_nan());
        let filled = rasterize(
            &cloud,
            RasterParams {
                fill_empty: true,
                ..ground
            },
        )
        .unwrap();
        assert!(filled.heights.iter().all(|&h| h == 0.0));
        assert!(
            rasterize(
                &cloud,
                RasterParams {
                    class: Some(9),
                    ..ground
                }
            )
            .is_none()
        );
    }
}
