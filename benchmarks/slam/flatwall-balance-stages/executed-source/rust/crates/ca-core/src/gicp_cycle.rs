//! Bounded, opt-in recurrence witnesses for GICP outer baseline states.
//! A witness diagnoses failure; it neither repairs a cycle nor proves convergence.

use std::collections::VecDeque;

use crate::gicp::{GicpError, Matrix3};
use crate::gicp_registration::Iteration;
use crate::icp::Rigid;

#[derive(Debug, Clone, Copy)]
pub struct CycleGuardParams {
    /// Inspect periods 2 through this value, inclusive (2..=16).
    pub max_period: usize,
    /// Require this many full phase-by-phase returns (2..=4).
    pub confirmations: usize,
    pub translation_tolerance: f64,
    pub rotation_tolerance: f64,
    /// Per-shape Frobenius difference divided by the larger Frobenius norm.
    pub relative_shape_tolerance: f64,
    /// Absolute difference for each same-ID scalar weight.
    pub weight_tolerance: f64,
}

impl Default for CycleGuardParams {
    fn default() -> Self {
        Self {
            max_period: 8,
            confirmations: 2,
            translation_tolerance: 1e-9,
            rotation_tolerance: 1e-10,
            relative_shape_tolerance: 1e-9,
            weight_tolerance: 1e-12,
        }
    }
}

impl CycleGuardParams {
    pub fn validate(self) -> Result<(), GicpError> {
        if !(2..=16).contains(&self.max_period) || !(2..=4).contains(&self.confirmations) {
            return Err(error(
                "cycle max_period must be in 2..16 and confirmations in 2..4",
            ));
        }
        if self.bounds().iter().any(|v| !v.is_finite() || *v <= 0.) {
            return Err(error("cycle tolerances must be finite and positive"));
        }
        Ok(())
    }

    fn bounds(self) -> [f64; 4] {
        [
            self.translation_tolerance,
            self.rotation_tolerance,
            self.relative_shape_tolerance,
            self.weight_tolerance,
        ]
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct CycleReport {
    pub period: usize,
    pub confirmations: usize,
    /// One-based baseline iteration indices, inclusive.
    pub first_iteration: usize,
    pub last_iteration: usize,
    pub compared_state_pairs: usize,
    pub max_translation: f64,
    pub max_rotation: f64,
    pub max_relative_shape: f64,
    pub max_weight: f64,
}

struct State {
    pose: Rigid,
    source: Vec<usize>,
    target: Vec<usize>,
    shapes: Vec<Matrix3>,
    weights: Vec<f64>,
}

pub(crate) struct CycleTracker {
    params: CycleGuardParams,
    states: VecDeque<State>,
}

fn error(reason: &'static str) -> GicpError {
    GicpError {
        index: None,
        reason,
    }
}

fn norm(values: impl Iterator<Item = f64>) -> f64 {
    values.fold(0.0_f64, f64::hypot)
}

fn motion(a: &Rigid, b: &Rigid, center: [f64; 3]) -> Result<[f64; 2], GicpError> {
    // D = Rrel-I for rotations. Subtraction first avoids magnifying a rounded
    // Ra*Ra^T-I by a large source-coordinate lever arm for identical poses.
    let d: Matrix3 = std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            (0..3)
                .map(|k| (a.rotation[i][k] - b.rotation[i][k]) * b.rotation[j][k])
                .sum()
        })
    });
    let displacement = std::array::from_fn::<_, 3, _>(|i| {
        (a.translation[i] - b.translation[i])
            - (0..3)
                .map(|k| d[i][k] * (b.translation[k] - center[k]))
                .sum::<f64>()
    });
    let skew = [
        (d[2][1] - d[1][2]) / 2.,
        (d[0][2] - d[2][0]) / 2.,
        (d[1][0] - d[0][1]) / 2.,
    ];
    let values = [
        norm(displacement.into_iter()),
        norm(skew.into_iter()).atan2(1. + (d[0][0] + d[1][1] + d[2][2]) / 2.),
    ];
    if d.iter()
        .flatten()
        .chain(&displacement)
        .chain(&values)
        .all(|v| v.is_finite())
    {
        Ok(values)
    } else {
        Err(error("nonfinite cycle comparison"))
    }
}

fn differences(a: &State, b: &State, center: [f64; 3]) -> Result<Option<[f64; 4]>, GicpError> {
    // Literal ordered IDs, including the entire matched-source support.
    if a.source != b.source || a.target != b.target {
        return Ok(None);
    }
    let [translation, rotation] = motion(&a.pose, &b.pose, center)?;
    let mut shape = 0.0_f64;
    for (ca, cb) in a.shapes.iter().zip(&b.shapes) {
        let na = norm(ca.iter().flatten().copied());
        let nb = norm(cb.iter().flatten().copied());
        let nd = norm(
            ca.iter()
                .flatten()
                .zip(cb.iter().flatten())
                .map(|(a, b)| a - b),
        );
        let relative = nd / na.max(nb);
        if ![na, nb, nd, relative].iter().all(|v| v.is_finite()) {
            return Err(error("nonfinite cycle comparison"));
        }
        shape = shape.max(relative);
    }
    let mut weight = 0.0_f64;
    for (a, b) in a.weights.iter().zip(&b.weights) {
        let delta = (a - b).abs();
        if !delta.is_finite() {
            return Err(error("nonfinite cycle comparison"));
        }
        weight = weight.max(delta);
    }
    Ok(Some([translation, rotation, shape, weight]))
}

impl CycleTracker {
    pub(crate) fn new(params: CycleGuardParams) -> Self {
        Self {
            params,
            states: VecDeque::new(),
        }
    }

    pub(crate) fn observe(
        &mut self,
        item: &Iteration,
        iteration: usize,
        center: [f64; 3],
        convergence_translation: f64,
        convergence_rotation: f64,
    ) -> Result<Option<CycleReport>, GicpError> {
        let limit = self.params.max_period * (self.params.confirmations + 1);
        if self.states.len() == limit {
            self.states.pop_front();
        }
        self.states.push_back(State {
            pose: item.pose_before,
            source: item.source_indices.clone(),
            target: item.target_indices.clone(),
            shapes: item.combined_shapes.clone(),
            weights: item.weights.clone(),
        });
        let n = self.states.len();
        for period in 2..=self.params.max_period {
            let count = self.params.confirmations * period;
            if n < count + period {
                continue;
            }
            let mut maxima = [0.0_f64; 4];
            let mut repeated = true;
            for index in n - count..n {
                let Some(values) =
                    differences(&self.states[index], &self.states[index - period], center)?
                else {
                    repeated = false;
                    break;
                };
                if values
                    .iter()
                    .zip(self.params.bounds())
                    .any(|(v, bound)| *v > bound)
                {
                    repeated = false;
                    break;
                }
                for j in 0..4 {
                    maxima[j] = maxima[j].max(values[j]);
                }
            }
            if !repeated {
                continue;
            }
            let mut nonstationary = false;
            for index in n - period + 1..n {
                let [t, r] = motion(
                    &self.states[index].pose,
                    &self.states[index - 1].pose,
                    center,
                )?;
                nonstationary |= t > convergence_translation || r > convergence_rotation;
            }
            if nonstationary {
                return Ok(Some(CycleReport {
                    period,
                    confirmations: self.params.confirmations,
                    first_iteration: iteration - count - period + 1,
                    last_iteration: iteration,
                    compared_state_pairs: count,
                    max_translation: maxima[0],
                    max_rotation: maxima[1],
                    max_relative_shape: maxima[2],
                    max_weight: maxima[3],
                }));
            }
        }
        Ok(None)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn item(phase: usize) -> Iteration {
        let mut pose = Rigid::IDENTITY;
        pose.translation[0] = phase as f64 * 0.01;
        Iteration {
            pose_before: pose,
            pose_after: pose,
            source_indices: vec![0, 1, 2],
            target_indices: vec![0, phase + 1, 2],
            squared_distances: vec![0.; 3],
            weights: vec![1.; 3],
            combined_shapes: vec![[[0.1, 0., 0.], [0., 1., 0.], [0., 0., 2.]]; 3],
            normal_matrix: [[0.; 6]; 6],
            right_hand_side: [0.; 6],
            cost_before: 0.,
            cost_after: None,
            step: None,
            trial_scales: vec![],
            trial_costs: vec![],
            accepted_scale: None,
        }
    }

    fn replay(rows: &[Iteration], params: CycleGuardParams) -> Option<CycleReport> {
        let mut tracker = CycleTracker::new(params);
        for (i, row) in rows.iter().enumerate() {
            let report = tracker.observe(row, i + 1, [0.; 3], 1e-4, 1e-5).unwrap();
            assert!(tracker.states.len() <= params.max_period * (params.confirmations + 1));
            if report.is_some() {
                return report;
            }
        }
        None
    }

    #[test]
    fn requires_two_complete_phase_returns_within_the_bound() {
        for period in [2, 3, 5, 7, 8] {
            let rows: Vec<_> = (0..3 * period).map(|i| item(i % period)).collect();
            assert!(replay(&rows[..rows.len() - 1], Default::default()).is_none());
            let report = replay(&rows, Default::default()).unwrap();
            assert_eq!(
                (report.period, report.compared_state_pairs),
                (period, 2 * period)
            );
            assert_eq!(
                (report.first_iteration, report.last_iteration),
                (1, 3 * period)
            );
        }
        let rows: Vec<_> = (0..200).map(|i| item(i % 9)).collect();
        assert!(replay(&rows, Default::default()).is_none());
        let report = replay(
            &rows,
            CycleGuardParams {
                max_period: 9,
                ..Default::default()
            },
        )
        .unwrap();
        assert_eq!((report.period, report.last_iteration), (9, 27));
    }

    #[test]
    fn one_returning_phase_changed_support_metrics_or_weights_cannot_confirm() {
        for change in 0..6 {
            let mut rows: Vec<_> = (0..9).map(|i| item(i % 3)).collect();
            for (i, row) in rows.iter_mut().enumerate() {
                match change {
                    0 => row.pose_before.translation[1] = i as f64 * 0.001,
                    1 => row.combined_shapes[0][0][0] += i as f64 * 0.01,
                    2 => row.weights[0] -= i as f64 * 0.01,
                    3 => row.source_indices[0] = 10 + i,
                    4 => row.pose_before.translation[0] = i as f64 * 1e-12,
                    _ => row.pose_before.translation[0] = 0.,
                }
            }
            assert!(replay(&rows, Default::default()).is_none());
        }
        let mut rows: Vec<_> = (0..9).map(|i| item(i % 3)).collect();
        rows[1].target_indices[0] = 99;
        assert!(replay(&rows, Default::default()).is_none());
    }

    #[test]
    fn rolling_window_preserves_absolute_iteration_indices() {
        let mut rows: Vec<_> = (0..100).map(item).collect();
        rows.extend((0..9).map(|i| item(i % 3)));
        let report = replay(&rows, Default::default()).unwrap();
        assert_eq!((report.first_iteration, report.last_iteration), (101, 109));
    }

    #[test]
    fn centered_motion_is_world_and_source_frame_invariant() {
        let a = Rigid {
            rotation: crate::pose_graph::exp_so3(&[0.1, -0.2, 0.3]),
            translation: [2., -1., 0.5],
        };
        let b = Rigid {
            translation: [-0.5, 0.8, 1.],
            ..Rigid::IDENTITY
        };
        let center = [0.4, -0.7, 2.];
        let frame = Rigid {
            rotation: crate::pose_graph::exp_so3(&[-0.2, 0.3, 0.4]),
            translation: [1e6, -2e6, 3e6],
        };
        let expected = motion(&a, &b, center).unwrap();
        for actual in [
            motion(&a.compose(&frame), &b.compose(&frame), center).unwrap(),
            motion(&frame.compose(&a), &frame.compose(&b), frame.apply(&center)).unwrap(),
        ] {
            for (a, b) in actual.iter().zip(expected) {
                assert!((a - b).abs() < 2e-9);
            }
        }
    }

    #[test]
    fn invalid_config_and_nonfinite_arithmetic_are_explicit() {
        for params in [
            CycleGuardParams {
                max_period: 17,
                ..Default::default()
            },
            CycleGuardParams {
                confirmations: 1,
                ..Default::default()
            },
            CycleGuardParams {
                weight_tolerance: f64::NAN,
                ..Default::default()
            },
        ] {
            assert!(params.validate().is_err());
        }
        let a = Rigid {
            translation: [f64::MAX, 0., 0.],
            ..Rigid::IDENTITY
        };
        let b = Rigid {
            translation: [-f64::MAX, 0., 0.],
            ..Rigid::IDENTITY
        };
        assert_eq!(
            motion(&a, &b, [0.; 3]).unwrap_err().reason,
            "nonfinite cycle comparison"
        );
    }

    #[test]
    fn identical_poses_return_exact_zero_with_large_source_lever_arm() {
        let pose = Rigid {
            rotation: crate::pose_graph::exp_so3(&[0.2, -0.3, 0.4]),
            translation: [1e6, -2e6, 3e6],
        };
        assert_eq!(motion(&pose, &pose, [0.; 3]).unwrap(), [0., 0.]);
    }
}
