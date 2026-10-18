//! Explicit unit-weight derivative of transported-shape GICP residual scores.
//! Matches stay declared; no optimizer, likelihood normalization or state update.

use crate::gicp::{self, Correspondence, GicpError, GicpLinearization, Matrix3};
use crate::icp::Rigid;

#[derive(Debug, Clone, PartialEq)]
pub struct TransportedLinearization {
    /// Unchanged frozen-metric factor. Its normal matrix is not the exact
    /// Hessian of the transported residual score.
    pub frozen: GicpLinearization,
    pub metric_gradient_rotation: [f64; 3],
    pub objective_gradient: [f64; 6],
    pub objective_right_hand_side: [f64; 6],
}

fn error(index: Option<usize>, reason: &'static str) -> GicpError {
    GicpError { index, reason }
}

#[allow(clippy::needless_range_loop)]
fn row_gradient(
    pair: &Correspondence,
    pose: &Rigid,
    world: [f64; 3],
    combined: &Matrix3,
) -> Option<[f64; 3]> {
    // The original factor has already validated every point/shape and C.
    let source = gicp::canonical_shape(&pair.source_shape)?;
    let lower = gicp::cholesky(combined)?;
    let mut e = [0.; 3];
    for i in 0..3 {
        e[i] = (world[i] - pair.target[i] - (0..i).map(|j| lower[i][j] * e[j]).sum::<f64>())
            / lower[i][i];
    }
    for i in (0..3).rev() {
        e[i] = (e[i] - (i + 1..3).map(|j| lower[j][i] * e[j]).sum::<f64>()) / lower[i][i];
    }
    let rotated: Matrix3 = std::array::from_fn(|i| {
        std::array::from_fn(|j| (0..3).map(|k| pose.rotation[i][k] * source[k][j]).sum())
    });
    let mut shape = [[0.; 3]; 3];
    for i in 0..3 {
        for j in 0..=i {
            let value = (0..3).map(|k| rotated[i][k] * pose.rotation[j][k]).sum();
            shape[i][j] = value;
            shape[j][i] = value;
        }
    }
    let v: [f64; 3] = std::array::from_fn(|i| (0..3).map(|j| shape[i][j] * e[j]).sum());
    let gradient = [
        e[1] * v[2] - e[2] * v[1],
        e[2] * v[0] - e[0] * v[2],
        e[0] * v[1] - e[1] * v[0],
    ];
    if e.iter()
        .chain(shape.iter().flatten())
        .chain(&v)
        .chain(&gradient)
        .all(|v| v.is_finite())
    {
        Some(gradient)
    } else {
        None
    }
}

/// Differentiate .5 sum rᵀ(Ct + R Cs Rᵀ)⁻¹r at explicit fixed pairs.
///
/// The original frozen factor validates inputs first and retains every bit.
/// Then every external weight must be exactly one; no derivative of optional
/// robust weights or log determinant is silently omitted. The extra rotation
/// gradient is sum(e × (R Cs Rᵀ e)), e=C⁻¹r. Parallel work uses input-order
/// accumulation and preserves serial arithmetic/error order.
pub fn linearize(
    pairs: &[Correspondence],
    pose: &Rigid,
    center_world: [f64; 3],
    parallel: bool,
) -> Result<TransportedLinearization, GicpError> {
    let frozen = gicp::linearize(pairs, pose, center_world, parallel)?;
    for (i, pair) in pairs.iter().enumerate() {
        if pair.weight != 1. {
            return Err(error(
                Some(i),
                "transported derivative requires unit weights",
            ));
        }
    }
    let compute = |i: usize| {
        row_gradient(
            &pairs[i],
            pose,
            frozen.source_world[i],
            &frozen.combined_shapes[i],
        )
    };
    #[cfg(feature = "parallel")]
    let rows: Vec<_> = if parallel {
        use rayon::prelude::*;
        (0..pairs.len()).into_par_iter().map(compute).collect()
    } else {
        (0..pairs.len()).map(compute).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let rows: Vec<_> = {
        let _ = parallel;
        (0..pairs.len()).map(compute).collect()
    };
    let mut metric = [0.; 3];
    for (i, row) in rows.into_iter().enumerate() {
        let row = row.ok_or(error(Some(i), "nonfinite transported metric gradient"))?;
        for j in 0..3 {
            metric[j] += row[j];
        }
        if !metric.iter().all(|v| v.is_finite()) {
            return Err(error(
                Some(i),
                "nonfinite transported gradient accumulation",
            ));
        }
    }
    let mut gradient = frozen.right_hand_side.map(|v| -v);
    for j in 0..3 {
        gradient[j + 3] += metric[j];
    }
    if !gradient.iter().all(|v| v.is_finite()) {
        return Err(error(None, "nonfinite transported objective gradient"));
    }
    Ok(TransportedLinearization {
        frozen,
        metric_gradient_rotation: metric,
        objective_gradient: gradient,
        objective_right_hand_side: gradient.map(|v| -v),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    const ID: Matrix3 = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
    fn pair() -> Correspondence {
        Correspondence {
            source: [0.; 3],
            target: [-1., -1., 0.],
            source_shape: [[0.1, 0., 0.], [0., 1., 0.], [0., 0., 2.]],
            target_shape: ID,
            weight: 1.,
        }
    }
    #[test]
    fn closed_form_metric_gradient_preserves_frozen_factor() {
        let pairs = [pair()];
        let actual = linearize(&pairs, &Rigid::IDENTITY, [0.; 3], false).unwrap();
        assert_eq!(
            actual.frozen,
            gicp::linearize(&pairs, &Rigid::IDENTITY, [0.; 3], false).unwrap()
        );
        assert!((actual.metric_gradient_rotation[2] - 9. / 22.).abs() < 1e-15);
        assert_eq!(actual.metric_gradient_rotation[..2], [0., 0.]);
        assert_eq!(
            actual.objective_right_hand_side,
            actual.objective_gradient.map(|v| -v)
        );
    }
    #[test]
    fn all_six_gradient_components_agree_with_central_cost_differences() {
        let mut pair = pair();
        pair.source = [0.3, -0.5, 0.8];
        let center = [0.2, -0.1, 0.4];
        let actual = linearize(&[pair], &Rigid::IDENTITY, center, false).unwrap();
        for axis in 0..6 {
            let mut costs = [0.; 2];
            for (i, sign) in [1., -1.].into_iter().enumerate() {
                let mut delta = [0.; 6];
                delta[axis] = sign * 1e-5;
                let q = crate::pose_graph::exp_so3(&[delta[3], delta[4], delta[5]]);
                let pose = Rigid {
                    rotation: q,
                    translation: std::array::from_fn(|j| {
                        center[j] + delta[j] - (0..3).map(|k| q[j][k] * center[k]).sum::<f64>()
                    }),
                };
                costs[i] = gicp::linearize(&[pair], &pose, center, false).unwrap().cost;
            }
            assert!(((costs[0] - costs[1]) / 2e-5 - actual.objective_gradient[axis]).abs() < 2e-9);
        }
    }
    #[test]
    fn empty_isotropic_weights_validation_and_parallel_order_are_explicit() {
        let empty = linearize(&[], &Rigid::IDENTITY, [0.; 3], true).unwrap();
        assert_eq!(empty.metric_gradient_rotation, [0.; 3]);
        let isotropic = Correspondence {
            source_shape: ID,
            ..pair()
        };
        assert_eq!(
            linearize(&[isotropic], &Rigid::IDENTITY, [0.; 3], false)
                .unwrap()
                .metric_gradient_rotation,
            [0.; 3]
        );
        let pairs = vec![pair(); 1025];
        assert_eq!(
            linearize(&pairs, &Rigid::IDENTITY, [0.; 3], false).unwrap(),
            linearize(&pairs, &Rigid::IDENTITY, [0.; 3], true).unwrap()
        );
        let mut invalid = pairs;
        invalid[2].weight = 0.;
        invalid[7].source_shape[0][0] = -1.;
        assert_eq!(
            linearize(&invalid, &Rigid::IDENTITY, [0.; 3], true)
                .unwrap_err()
                .index,
            Some(7)
        );
        invalid[7] = pair();
        assert_eq!(
            linearize(&invalid, &Rigid::IDENTITY, [0.; 3], true)
                .unwrap_err()
                .index,
            Some(2)
        );
    }

    #[test]
    fn new_gradient_overflow_preserves_original_factor_and_first_row_error() {
        let shape = [[0.5e-100, 0., 0.], [0., 0.5e100, 0.], [0., 0., 0.5]];
        let large = Correspondence {
            source: [0.; 3],
            target: [-1e100, -1e200, 0.],
            source_shape: shape,
            target_shape: shape,
            weight: 1.,
        };
        assert!(gicp::linearize(&[large], &Rigid::IDENTITY, [0.; 3], false).is_ok());
        let mut pairs = vec![pair(); 1025];
        pairs[7] = large;
        pairs[333] = large;
        for parallel in [false, true] {
            let error = linearize(&pairs, &Rigid::IDENTITY, [0.; 3], parallel).unwrap_err();
            assert_eq!(error.index, Some(7));
            assert_eq!(error.reason, "nonfinite transported metric gradient");
        }
    }
}
