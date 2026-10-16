//! Fixed-correspondence GICP residuals with explicit shape frames and center.
//!
//! Shapes may encode normalized surface geometry; this API does not turn them
//! into calibrated measurement uncertainty. Matches, external weights and the
//! transported metric are frozen for this linearization. No solver or fallback
//! is applied, and no existing registration path calls this module implicitly.

use crate::icp::Rigid;

pub type Matrix3 = [[f64; 3]; 3];

/// One declared pair. Source point/shape use sensor axes; target uses world axes.
#[derive(Debug, Clone, Copy)]
pub struct Correspondence {
    pub source: [f64; 3],
    pub target: [f64; 3],
    pub source_shape: Matrix3,
    pub target_shape: Matrix3,
    pub weight: f64,
}

/// Parameters: translation xyz, then world rotation xyz about the fixed center.
#[derive(Debug, Clone, PartialEq)]
pub struct GicpLinearization {
    pub normal_matrix: [[f64; 6]; 6],
    pub right_hand_side: [f64; 6],
    /// Half the weighted squared Mahalanobis residual; metric held fixed.
    pub cost: f64,
    pub weight_sum: f64,
    pub positive_weight_points: usize,
    pub source_world: Vec<[f64; 3]>,
    pub combined_shapes: Vec<Matrix3>,
    /// Unweighted squared Mahalanobis residual, one per input pair.
    pub mahalanobis_squared: Vec<f64>,
}

/// Common-input errors have no index; row errors identify the first input pair.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct GicpError {
    pub index: Option<usize>,
    pub reason: &'static str,
}

impl std::fmt::Display for GicpError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(index) = self.index {
            write!(f, "GICP pair {index}: {}", self.reason)
        } else {
            write!(f, "GICP: {}", self.reason)
        }
    }
}

impl std::error::Error for GicpError {}

fn proper_pose(pose: &Rigid) -> bool {
    let r = &pose.rotation;
    if !r
        .iter()
        .flatten()
        .chain(&pose.translation)
        .all(|v| v.is_finite())
    {
        return false;
    }
    let orthogonal = (0..3).all(|i| {
        (0..3).all(|j| {
            let dot: f64 = r[i].iter().zip(r[j]).map(|(a, b)| a * b).sum();
            (dot - f64::from(u8::from(i == j))).abs() <= 1e-10
        })
    });
    let det = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
    orthogonal && (det - 1.0).abs() <= 1e-10
}

// Fixed-size triangular dependencies are clearer as indexed loops.
#[allow(clippy::needless_range_loop)]
fn cholesky(matrix: &Matrix3) -> Option<Matrix3> {
    let mut lower = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..=i {
            let mut value = matrix[i][j];
            for k in 0..j {
                value -= lower[i][k] * lower[j][k];
            }
            if i == j {
                if !value.is_finite() || value <= 0.0 {
                    return None;
                }
                lower[i][j] = value.sqrt();
            } else {
                lower[i][j] = value / lower[j][j];
                if !lower[i][j].is_finite() {
                    return None;
                }
            }
        }
    }
    Some(lower)
}

fn canonical_shape(matrix: &Matrix3) -> Option<Matrix3> {
    if !matrix.iter().flatten().all(|v| v.is_finite()) {
        return None;
    }
    let scale = matrix.iter().flatten().map(|v| v.abs()).fold(0.0, f64::max);
    if !(0..3).all(|i| (0..3).all(|j| (matrix[i][j] - matrix[j][i]).abs() <= 1e-12 * scale)) {
        return None;
    }
    let symmetric = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            let (a, b) = (matrix[i][j], matrix[j][i]);
            if a == b { a } else { a / 2.0 + b / 2.0 }
        })
    });
    cholesky(&symmetric)?;
    Some(symmetric)
}

struct Row {
    normal: [[f64; 6]; 6],
    rhs: [f64; 6],
    cost: f64,
    world: [f64; 3],
    combined: Matrix3,
    distance: f64,
}

#[allow(clippy::needless_range_loop)]
fn linearize_row(
    pair: &Correspondence,
    pose: &Rigid,
    center: [f64; 3],
) -> Result<Row, &'static str> {
    if !pair
        .source
        .iter()
        .chain(&pair.target)
        .all(|v| v.is_finite())
    {
        return Err("source and target must be finite");
    }
    if !pair.weight.is_finite() || pair.weight < 0.0 {
        return Err("weight must be finite and nonnegative");
    }
    let source = canonical_shape(&pair.source_shape)
        .ok_or("source shape must be finite, symmetric and strictly positive definite")?;
    let target = canonical_shape(&pair.target_shape)
        .ok_or("target shape must be finite, symmetric and strictly positive definite")?;
    let rotated: Matrix3 = std::array::from_fn(|i| {
        std::array::from_fn(|j| (0..3).map(|k| pose.rotation[i][k] * source[k][j]).sum())
    });
    let mut combined = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..=i {
            let value = target[i][j]
                + (0..3)
                    .map(|k| rotated[i][k] * pose.rotation[j][k])
                    .sum::<f64>();
            combined[i][j] = value;
            combined[j][i] = value;
        }
    }
    if !combined.iter().flatten().all(|v| v.is_finite()) {
        return Err("nonfinite transported shape");
    }
    let lower =
        cholesky(&combined).ok_or("transported shape is not numerically positive definite")?;
    let world = pose.apply(&pair.source);
    let arm: [f64; 3] = std::array::from_fn(|i| world[i] - center[i]);
    let residual: [f64; 3] = std::array::from_fn(|i| world[i] - pair.target[i]);
    let jacobian = [
        [1.0, 0.0, 0.0, 0.0, arm[2], -arm[1]],
        [0.0, 1.0, 0.0, -arm[2], 0.0, arm[0]],
        [0.0, 0.0, 1.0, arm[1], -arm[0], 0.0],
    ];
    // Whiten [residual | J] by forward substitution. No explicit matrix inverse.
    let mut whitened = [[0.0; 7]; 3];
    for i in 0..3 {
        for j in 0..7 {
            let mut value = if j == 0 {
                residual[i]
            } else {
                jacobian[i][j - 1]
            };
            for k in 0..i {
                value -= lower[i][k] * whitened[k][j];
            }
            whitened[i][j] = value / lower[i][i];
        }
    }
    if !world
        .iter()
        .chain(&arm)
        .chain(whitened.iter().flatten())
        .all(|v| v.is_finite())
    {
        return Err("nonfinite transformed point or whitened system");
    }
    let mut normal = [[0.0; 6]; 6];
    let mut rhs = [0.0; 6];
    let mut distance = 0.0;
    for row in whitened {
        distance += row[0] * row[0];
        for i in 0..6 {
            rhs[i] -= pair.weight * row[i + 1] * row[0];
            for j in 0..6 {
                normal[i][j] += pair.weight * row[i + 1] * row[j + 1];
            }
        }
    }
    let cost = 0.5 * pair.weight * distance;
    if !normal
        .iter()
        .flatten()
        .chain(&rhs)
        .chain([&cost, &distance])
        .all(|v| v.is_finite())
    {
        return Err("nonfinite row contribution");
    }
    Ok(Row {
        normal,
        rhs,
        cost,
        world,
        combined,
        distance,
    })
}

/// Linearize declared pairs with shapes transported at `pose` and held fixed.
///
/// Input-row work may run in parallel, followed by a serial input-order sum.
/// All inputs, including zero-weight pairs, are validated. No eigenvalue floor,
/// matrix damping, state update, matching or calibrated uncertainty is inferred.
pub fn linearize(
    pairs: &[Correspondence],
    pose: &Rigid,
    center_world: [f64; 3],
    parallel: bool,
) -> Result<GicpLinearization, GicpError> {
    if !proper_pose(pose) {
        return Err(GicpError {
            index: None,
            reason: "pose must be finite and proper rigid",
        });
    }
    if !center_world.iter().all(|v| v.is_finite()) {
        return Err(GicpError {
            index: None,
            reason: "center_world must be finite",
        });
    }
    let compute = |pair: &Correspondence| linearize_row(pair, pose, center_world);
    #[cfg(feature = "parallel")]
    let rows: Vec<_> = if parallel {
        use rayon::prelude::*;
        pairs.par_iter().map(compute).collect()
    } else {
        pairs.iter().map(compute).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let rows: Vec<_> = {
        let _ = parallel;
        pairs.iter().map(compute).collect()
    };
    let mut result = GicpLinearization {
        normal_matrix: [[0.0; 6]; 6],
        right_hand_side: [0.0; 6],
        cost: 0.0,
        weight_sum: 0.0,
        positive_weight_points: 0,
        source_world: Vec::with_capacity(pairs.len()),
        combined_shapes: Vec::with_capacity(pairs.len()),
        mahalanobis_squared: Vec::with_capacity(pairs.len()),
    };
    for (index, row) in rows.into_iter().enumerate() {
        let row = row.map_err(|reason| GicpError {
            index: Some(index),
            reason,
        })?;
        for (i, target) in result.normal_matrix.iter_mut().enumerate() {
            result.right_hand_side[i] += row.rhs[i];
            for (j, value) in target.iter_mut().enumerate() {
                *value += row.normal[i][j];
            }
        }
        result.cost += row.cost;
        result.weight_sum += pairs[index].weight;
        result.positive_weight_points += usize::from(pairs[index].weight > 0.0);
        if !result
            .normal_matrix
            .iter()
            .flatten()
            .chain(&result.right_hand_side)
            .chain([&result.cost, &result.weight_sum])
            .all(|v| v.is_finite())
        {
            return Err(GicpError {
                index: Some(index),
                reason: "nonfinite accumulated system",
            });
        }
        result.source_world.push(row.world);
        result.combined_shapes.push(row.combined);
        result.mahalanobis_squared.push(row.distance);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pair() -> Correspondence {
        Correspondence {
            source: [2.0, 3.0, 4.0],
            target: [1.0, 1.0, 1.0],
            source_shape: [[0.25, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]],
            target_shape: [[0.25, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]],
            weight: 2.0,
        }
    }

    fn bits(result: &GicpLinearization) -> Vec<u64> {
        result
            .normal_matrix
            .iter()
            .flatten()
            .chain(&result.right_hand_side)
            .chain([&result.cost, &result.weight_sum])
            .chain(result.source_world.iter().flatten())
            .chain(result.combined_shapes.iter().flatten().flatten())
            .chain(&result.mahalanobis_squared)
            .map(|v| v.to_bits())
            .collect()
    }

    #[test]
    fn anisotropic_one_pair_agrees_with_closed_form() {
        let result = linearize(&[pair()], &Rigid::IDENTITY, pair().source, false).unwrap();
        assert!((result.cost - 10.5).abs() < 1e-14);
        assert!((result.mahalanobis_squared[0] - 10.5).abs() < 1e-14);
        let expected = [4.0, 2.0, 1.0, 0.0, 0.0, 0.0];
        for (i, row) in result.normal_matrix.iter().enumerate() {
            for (j, value) in row.iter().enumerate() {
                assert!((value - if i == j { expected[i] } else { 0.0 }).abs() < 1e-14);
            }
        }
        for (value, expected) in result
            .right_hand_side
            .iter()
            .zip([-4.0, -4.0, -3.0, 0.0, 0.0, 0.0])
        {
            assert!((value - expected).abs() < 1e-14);
        }
        assert_eq!(result.weight_sum, 2.0);
        assert_eq!(result.positive_weight_points, 1);
    }

    #[test]
    fn empty_system_and_zero_weight_rows_have_explicit_zero_support() {
        let empty = linearize(&[], &Rigid::IDENTITY, [0.0; 3], true).unwrap();
        assert!(empty.source_world.is_empty());
        assert_eq!(empty.cost, 0.0);
        assert_eq!(empty.normal_matrix, [[0.0; 6]; 6]);
        let result = linearize(
            &[Correspondence {
                weight: 0.0,
                ..pair()
            }],
            &Rigid::IDENTITY,
            [0.0; 3],
            false,
        )
        .unwrap();
        assert_eq!(result.weight_sum, 0.0);
        assert_eq!(result.positive_weight_points, 0);
        assert_eq!(result.normal_matrix, empty.normal_matrix);
        assert_eq!(result.right_hand_side, empty.right_hand_side);
        assert!(result.mahalanobis_squared[0] > 0.0);
    }

    #[test]
    fn indexed_parallel_rows_preserve_all_result_bits() {
        let pairs: Vec<_> = (0..1025)
            .map(|i| Correspondence {
                source: [i as f64 * 0.013, (i % 31) as f64 * -0.17, (i % 7) as f64],
                weight: (i % 13) as f64 / 13.0,
                ..pair()
            })
            .collect();
        let pose = Rigid {
            rotation: crate::pose_graph::exp_so3(&[0.2, -0.1, 0.3]),
            translation: [4.0, -2.0, 1.0],
        };
        let serial = linearize(&pairs, &pose, [3.0, 1.0, -2.0], false).unwrap();
        let parallel = linearize(&pairs, &pose, [3.0, 1.0, -2.0], true).unwrap();
        assert_eq!(bits(&serial), bits(&parallel));
        assert_eq!(
            serial.positive_weight_points,
            parallel.positive_weight_points
        );
    }

    #[test]
    fn invalid_shapes_and_weights_fail_at_first_input_even_with_zero_weight() {
        for parallel in [false, true] {
            let mut pairs = vec![pair(); 1025];
            pairs[7].source_shape[0][1] = 0.01;
            pairs[7].weight = 0.0;
            pairs[99].target_shape[2][2] = -1.0;
            pairs[777].weight = f64::NAN;
            let error = linearize(&pairs, &Rigid::IDENTITY, [0.0; 3], parallel).unwrap_err();
            assert_eq!(error.index, Some(7));
            assert!(error.reason.starts_with("source shape"));
        }
        for diagonal in [0.0, -1.0, f64::NAN, f64::INFINITY] {
            let mut input = pair();
            input.target_shape[1][1] = diagonal;
            let error = linearize(&[input], &Rigid::IDENTITY, [0.0; 3], false).unwrap_err();
            assert_eq!(error.index, Some(0));
            assert!(error.reason.starts_with("target shape"));
        }
    }

    #[test]
    fn invalid_common_frames_and_numeric_overflow_are_rejected() {
        let mut pose = Rigid::IDENTITY;
        pose.rotation[0][0] = -1.0;
        assert_eq!(
            linearize(&[], &pose, [0.0; 3], false).unwrap_err().index,
            None
        );
        assert_eq!(
            linearize(&[], &Rigid::IDENTITY, [f64::NAN, 0.0, 0.0], false)
                .unwrap_err()
                .index,
            None
        );
        let input = Correspondence {
            source: [1e308; 3],
            ..pair()
        };
        let error = linearize(&[input], &Rigid::IDENTITY, [0.0; 3], false).unwrap_err();
        assert!(error.reason.starts_with("nonfinite"));
        let input = Correspondence {
            source: [0.0; 3],
            target: [0.0; 3],
            weight: f64::MAX / 3.0,
            ..pair()
        };
        let error = linearize(&[input; 10], &Rigid::IDENTITY, [0.0; 3], true).unwrap_err();
        assert!(error.reason.starts_with("nonfinite accumulated"));
    }

    #[test]
    fn roundoff_asymmetry_is_canonicalized_without_an_eigenvalue_floor() {
        let mut input = pair();
        input.source_shape[0][1] = 1e-14;
        let mut transposed = input;
        transposed.source_shape[1][0] = 1e-14;
        transposed.source_shape[0][1] = 0.0;
        let a = linearize(&[input], &Rigid::IDENTITY, [0.0; 3], false).unwrap();
        let b = linearize(&[transposed], &Rigid::IDENTITY, [0.0; 3], false).unwrap();
        assert_eq!(bits(&a), bits(&b));
        let mut thin = pair();
        thin.source_shape[0][0] = 1e-18;
        thin.target_shape[0][0] = 1e-18;
        let result = linearize(&[thin], &Rigid::IDENTITY, thin.source, false).unwrap();
        assert_eq!(result.combined_shapes[0][0][0], 2e-18);
    }
}
