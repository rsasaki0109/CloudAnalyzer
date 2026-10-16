//! f64 normalized GICP surface shapes from explicit exact k-neighborhoods.
//! Unsupported geometry is reported, never replaced with an identity shape.

use crate::gicp::Matrix3;
use crate::kdtree::KdTree;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ShapeError {
    pub index: Option<usize>,
    pub reason: &'static str,
}

impl std::fmt::Display for ShapeError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(index) = self.index {
            write!(f, "GICP shape point {index}: {}", self.reason)
        } else {
            write!(f, "GICP shapes: {}", self.reason)
        }
    }
}
impl std::error::Error for ShapeError {}

#[derive(Debug, Clone, PartialEq)]
pub struct SurfaceShapes {
    /// Zero matrix for unsupported rows. These rows must be filtered explicitly.
    pub shapes: Vec<Matrix3>,
    pub supported: Vec<bool>,
    /// Signed ascending population scatter eigenvalues; zero for <5 neighbors.
    pub scatter_eigenvalues: Vec<[f64; 3]>,
    pub neighbor_counts: Vec<usize>,
    /// Empty row vectors unless capture_neighbors was requested.
    pub neighbor_indices: Vec<Vec<usize>>,
    pub neighbor_squared_distances: Vec<Vec<f64>>,
}

struct Row {
    shape: Matrix3,
    supported: bool,
    eigenvalues: [f64; 3],
    count: usize,
    indices: Vec<usize>,
    distances: Vec<f64>,
}

// Fixed-size triangular scatter dependencies retain a declared operation order.
#[allow(clippy::needless_range_loop)]
fn row(
    points: &[[f64; 3]],
    tree: &KdTree,
    index: usize,
    neighbors: usize,
    hits: &mut Vec<(usize, f64)>,
    capture: bool,
) -> Result<Row, ShapeError> {
    let invalid = |reason| ShapeError {
        index: Some(index),
        reason,
    };
    tree.nearest_k_into(&points[index], neighbors, hits);
    if hits.iter().any(|&(_, distance)| !distance.is_finite()) {
        return Err(invalid("nonfinite nearest-neighbor squared distance"));
    }
    let mut result = Row {
        shape: [[0.0; 3]; 3],
        supported: false,
        eigenvalues: [0.0; 3],
        count: hits.len(),
        indices: if capture {
            hits.iter().map(|&(i, _)| i).collect()
        } else {
            Vec::new()
        },
        distances: if capture {
            hits.iter().map(|&(_, d)| d).collect()
        } else {
            Vec::new()
        },
    };
    if hits.len() < 5 {
        return Ok(result);
    }
    let query = points[index];
    let mut mean = [0.0; 3];
    for &(i, _) in hits.iter() {
        for j in 0..3 {
            mean[j] += points[i][j] - query[j];
        }
    }
    let count = hits.len() as f64;
    mean = mean.map(|value| value / count);
    if !mean.iter().all(|v| v.is_finite()) {
        return Err(invalid("nonfinite neighborhood mean"));
    }
    let mut scatter = [[0.0; 3]; 3];
    for &(i, _) in hits.iter() {
        let d: [f64; 3] = std::array::from_fn(|j| (points[i][j] - query[j]) - mean[j]);
        for a in 0..3 {
            for b in 0..=a {
                scatter[a][b] += d[a] * d[b];
            }
        }
    }
    for a in 0..3 {
        for b in 0..=a {
            scatter[a][b] /= count;
            scatter[b][a] = scatter[a][b];
        }
    }
    if !scatter.iter().flatten().all(|v| v.is_finite()) {
        return Err(invalid("nonfinite neighborhood scatter"));
    }
    let scale = scatter
        .iter()
        .flatten()
        .map(|v| v.abs())
        .fold(0.0, f64::max);
    if scale == 0.0 {
        return Ok(result);
    }
    let (values, vectors) = crate::icp::symmetric_eigen(scatter.map(|r| r.map(|v| v / scale)));
    let mut order = [0, 1, 2];
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]).then(a.cmp(&b)));
    result.eigenvalues = order.map(|i| values[i] * scale);
    if !result
        .eigenvalues
        .iter()
        .chain(vectors.iter().flatten())
        .all(|v| v.is_finite())
    {
        return Err(invalid("nonfinite neighborhood eigensystem"));
    }
    let [_, middle, largest] = result.eigenvalues;
    if largest <= 0.0 || middle <= 1e-12 * largest {
        return Ok(result);
    }
    let normal: [f64; 3] = std::array::from_fn(|i| vectors[i][order[0]]);
    let norm = normal.iter().fold(0.0_f64, |a, &b| a.hypot(b));
    if !norm.is_finite() || norm <= 0.0 {
        return Err(invalid("invalid neighborhood normal"));
    }
    let normal = normal.map(|v| v / norm);
    result.shape = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            f64::from(u8::from(i == j)) + (0.001 - 1.0) * (normal[i] * normal[j])
        })
    });
    if crate::gicp::canonical_shape(&result.shape).is_none() {
        return Err(invalid("normalized surface shape is not numerically SPD"));
    }
    result.supported = true;
    Ok(result)
}

/// One owned row per input point, with exact tree tie order and f64 PCA.
/// `neighbors >= 5`; each neighborhood has min(neighbors, points.len()) rows.
pub fn estimate(
    points: &[[f64; 3]],
    neighbors: usize,
    parallel: bool,
    capture_neighbors: bool,
) -> Result<SurfaceShapes, ShapeError> {
    if neighbors < 5 {
        return Err(ShapeError {
            index: None,
            reason: "neighbors must be at least five",
        });
    }
    if points.len() > u32::MAX as usize {
        return Err(ShapeError {
            index: None,
            reason: "point count exceeds tree index capacity",
        });
    }
    for (index, point) in points.iter().enumerate() {
        if !point.iter().all(|v| v.is_finite()) {
            return Err(ShapeError {
                index: Some(index),
                reason: "point must be finite",
            });
        }
    }
    let mut result = SurfaceShapes {
        shapes: Vec::with_capacity(points.len()),
        supported: Vec::with_capacity(points.len()),
        scatter_eigenvalues: Vec::with_capacity(points.len()),
        neighbor_counts: Vec::with_capacity(points.len()),
        neighbor_indices: Vec::with_capacity(points.len()),
        neighbor_squared_distances: Vec::with_capacity(points.len()),
    };
    let Some(tree) = KdTree::new(points) else {
        return Ok(result);
    };
    let k = neighbors.min(points.len());
    let compute = |start: usize, count: usize| {
        let mut hits = Vec::with_capacity(k);
        (start..start + count)
            .map(|i| row(points, &tree, i, k, &mut hits, capture_neighbors))
            .collect::<Vec<_>>()
    };
    #[cfg(feature = "parallel")]
    let rows: Vec<_> = if parallel {
        use rayon::prelude::*;
        points
            .par_chunks(64)
            .enumerate()
            .map(|(i, chunk)| compute(i * 64, chunk.len()))
            .collect::<Vec<_>>()
            .into_iter()
            .flatten()
            .collect()
    } else {
        compute(0, points.len())
    };
    #[cfg(not(feature = "parallel"))]
    let rows = {
        let _ = parallel;
        compute(0, points.len())
    };
    for item in rows {
        let item = item?;
        result.shapes.push(item.shape);
        result.supported.push(item.supported);
        result.scatter_eigenvalues.push(item.eigenvalues);
        result.neighbor_counts.push(item.count);
        result.neighbor_indices.push(item.indices);
        result.neighbor_squared_distances.push(item.distances);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;
    fn plane() -> Vec<[f64; 3]> {
        (0..36)
            .map(|i| [(i % 6) as f64, (i / 6) as f64, 2.])
            .collect()
    }

    #[test]
    fn plane_shape_matches_closed_form_and_has_signed_scatter() {
        let result = estimate(&plane(), 20, false, true).unwrap();
        assert!(result.supported.iter().all(|&v| v));
        assert!(result.neighbor_counts.iter().all(|&n| n == 20));
        for (shape, eig) in result.shapes.iter().zip(result.scatter_eigenvalues) {
            assert!(eig[0].abs() < 1e-14 && eig[1] > 0. && eig[2] >= eig[1]);
            for (i, row) in shape.iter().enumerate() {
                for (j, &value) in row.iter().enumerate() {
                    let expected = if i == j {
                        if i == 2 { 0.001 } else { 1. }
                    } else {
                        0.
                    };
                    assert!((value - expected).abs() < 1e-14);
                }
            }
        }
    }

    #[test]
    fn missing_and_line_geometry_are_explicitly_unsupported() {
        for points in [
            vec![],
            vec![[0.; 3]; 4],
            vec![[0.; 3]; 8],
            (0..20).map(|i| [i as f64, 0., 0.]).collect(),
        ] {
            let result = estimate(&points, usize::MAX, false, true).unwrap();
            assert!(result.supported.iter().all(|&v| !v));
            assert!(result.shapes.iter().all(|&c| c == [[0.; 3]; 3]));
            assert!(result.neighbor_counts.iter().all(|&n| n == points.len()));
        }
    }

    #[test]
    fn indexed_parallel_chunks_preserve_all_rows_and_capture_is_optional() {
        let points: Vec<_> = (0..1025)
            .map(|i| {
                [
                    (i % 31) as f64 * 0.3,
                    (i % 17) as f64 * 0.2,
                    (i % 7) as f64 * 0.4,
                ]
            })
            .collect();
        let serial = estimate(&points, 20, false, true).unwrap();
        assert_eq!(serial, estimate(&points, 20, true, true).unwrap());
        let plain = estimate(&points, 20, false, false).unwrap();
        assert_eq!(serial.shapes, plain.shapes);
        assert_eq!(serial.scatter_eigenvalues, plain.scatter_eigenvalues);
        assert!(plain.neighbor_indices.iter().all(Vec::is_empty));
    }

    #[test]
    fn invalid_inputs_and_numeric_overflow_keep_first_index() {
        assert!(estimate(&[], 4, false, false).is_err());
        let mut points = plane();
        points[7][0] = f64::NAN;
        points[17][1] = f64::INFINITY;
        for parallel in [false, true] {
            assert_eq!(
                estimate(&points, 20, parallel, true).unwrap_err().index,
                Some(7)
            );
            let huge = [[0.; 3], [f64::MAX; 3], [-f64::MAX; 3], [1.; 3], [2.; 3]];
            assert_eq!(
                estimate(&huge, 20, parallel, true).unwrap_err().index,
                Some(0)
            );
        }
    }

    #[test]
    fn repeated_eigenvalues_remain_supported_without_identity_fallback() {
        let points = vec![
            [1., 0., 0.],
            [-1., 0., 0.],
            [0., 1., 0.],
            [0., -1., 0.],
            [0., 0., 1.],
            [0., 0., -1.],
        ];
        let result = estimate(&points, 20, false, false).unwrap();
        assert!(result.supported.iter().all(|&v| v));
        for shape in result.shapes {
            assert!(crate::gicp::canonical_shape(&shape).is_some());
            assert_ne!(shape, [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]]);
        }
    }
}
