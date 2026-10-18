//! Opt-in iterative GICP on caller-prepared shapes and an immutable target.
//! Inner trials hold the baseline metric, weights and matches fixed. Rematching
//! starts a different surrogate; costs across outer iterations need not decrease.

use crate::gicp::{self, Correspondence, GicpError, Matrix3};
use crate::gicp_cycle::{CycleGuardParams, CycleReport, CycleTracker};
use crate::icp::Rigid;
use crate::kdtree::KdTree;

fn error(index: Option<usize>, reason: &'static str) -> GicpError {
    GicpError { index, reason }
}

/// Owns source/target pairs, weights and Cholesky factors of a frozen metric.
#[derive(Debug, Clone)]
pub struct FrozenMetric {
    source: Vec<[f64; 3]>,
    target: Vec<[f64; 3]>,
    lower: Vec<Matrix3>,
    weights: Vec<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct FrozenCost {
    pub cost: f64,
    pub mahalanobis_squared: Vec<f64>,
}

impl FrozenMetric {
    pub fn new(
        source: &[[f64; 3]],
        target: &[[f64; 3]],
        combined_shapes: &[Matrix3],
        weights: &[f64],
    ) -> Result<Self, GicpError> {
        if [target.len(), combined_shapes.len(), weights.len()]
            .iter()
            .any(|&n| n != source.len())
        {
            return Err(error(None, "frozen metric needs equal row counts"));
        }
        let mut lower = Vec::with_capacity(source.len());
        for i in 0..source.len() {
            if !source[i].iter().chain(&target[i]).all(|v| v.is_finite()) {
                return Err(error(Some(i), "source and target must be finite"));
            }
            if !weights[i].is_finite() || weights[i] < 0.0 {
                return Err(error(Some(i), "weight must be finite and nonnegative"));
            }
            let shape = gicp::canonical_shape(&combined_shapes[i]).ok_or(error(
                Some(i),
                "combined shape must be finite, symmetric and strictly positive definite",
            ))?;
            lower.push(gicp::cholesky(&shape).expect("validated SPD shape"));
        }
        Ok(Self {
            source: source.to_vec(),
            target: target.to_vec(),
            lower,
            weights: weights.to_vec(),
        })
    }

    pub fn evaluate(&self, pose: &Rigid, parallel: bool) -> Result<FrozenCost, GicpError> {
        if !gicp::proper_pose(pose) {
            return Err(error(None, "pose must be finite and proper rigid"));
        }
        let compute = |i: usize| {
            let world = pose.apply(&self.source[i]);
            let mut white = [0.0; 3];
            for j in 0..3 {
                let mut value = world[j] - self.target[i][j];
                for (k, &w) in white.iter().enumerate().take(j) {
                    value -= self.lower[i][j][k] * w;
                }
                white[j] = value / self.lower[i][j][j];
            }
            let distance: f64 = white.iter().map(|v| v * v).sum();
            let cost = 0.5 * self.weights[i] * distance;
            if !distance.is_finite() || !cost.is_finite() {
                Err(error(Some(i), "nonfinite frozen-metric residual"))
            } else {
                Ok((distance, cost))
            }
        };
        let rows = indexed(self.source.len(), parallel, compute);
        let mut result = FrozenCost {
            cost: 0.0,
            mahalanobis_squared: Vec::with_capacity(rows.len()),
        };
        for (i, row) in rows.into_iter().enumerate() {
            let (distance, cost) = row?;
            result.cost += cost;
            if !result.cost.is_finite() {
                return Err(error(Some(i), "nonfinite frozen-metric cost accumulation"));
            }
            result.mahalanobis_squared.push(distance);
        }
        Ok(result)
    }
}

fn indexed<T: Send, F: Fn(usize) -> T + Sync + Send>(n: usize, parallel: bool, f: F) -> Vec<T> {
    #[cfg(feature = "parallel")]
    if parallel {
        use rayon::prelude::*;
        return (0..n).into_par_iter().map(f).collect();
    }
    let _ = parallel;
    (0..n).map(f).collect()
}

#[derive(Debug, Clone, Copy)]
pub struct RegistrationParams {
    pub max_correspondence_distance: f64,
    pub max_iterations: usize,
    pub translation_tolerance: f64,
    pub rotation_tolerance: f64,
    /// Positive distance scale s: w=(s²/(s²+Euclidean distance²))².
    pub robust_scale: Option<f64>,
    pub parallel: bool,
    pub capture_history: bool,
    /// Explicit failure diagnosis; disabled by default and independent of history capture.
    pub cycle_guard: Option<CycleGuardParams>,
}

impl Default for RegistrationParams {
    fn default() -> Self {
        Self {
            max_correspondence_distance: 1.0,
            max_iterations: 30,
            translation_tolerance: 1e-4,
            rotation_tolerance: 1e-5,
            robust_scale: None,
            parallel: false,
            capture_history: false,
            cycle_guard: None,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Termination {
    ConvergedStep,
    InsufficientCorrespondences,
    SingularSystem,
    Nonfinite,
    NoDescent,
    MaxIterations,
    CycleDetected,
}

impl Termination {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::ConvergedStep => "converged_step",
            Self::InsufficientCorrespondences => "insufficient_correspondences",
            Self::SingularSystem => "singular_system",
            Self::Nonfinite => "nonfinite",
            Self::NoDescent => "no_descent",
            Self::MaxIterations => "max_iterations",
            Self::CycleDetected => "cycle_detected",
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Iteration {
    pub pose_before: Rigid,
    pub pose_after: Rigid,
    pub source_indices: Vec<usize>,
    pub target_indices: Vec<usize>,
    pub squared_distances: Vec<f64>,
    pub weights: Vec<f64>,
    pub combined_shapes: Vec<Matrix3>,
    pub normal_matrix: [[f64; 6]; 6],
    pub right_hand_side: [f64; 6],
    pub cost_before: f64,
    pub cost_after: Option<f64>,
    pub step: Option<[f64; 6]>,
    pub trial_scales: Vec<f64>,
    /// None identifies nonfinite trial arithmetic; no candidate was adopted.
    pub trial_costs: Vec<Option<f64>>,
    pub accepted_scale: Option<f64>,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Registration {
    pub pose: Rigid,
    pub termination: Termination,
    pub iterations: usize,
    pub accepted_steps: usize,
    /// Rematched and retransported at the returned pose, not a frozen trial cost.
    pub final_cost: Option<f64>,
    pub matched_source_indices: Vec<usize>,
    pub matched_target_indices: Vec<usize>,
    pub squared_distances: Vec<f64>,
    pub weights: Vec<f64>,
    pub failure_reason: Option<&'static str>,
    pub history: Vec<Iteration>,
    pub cycle: Option<CycleReport>,
}

/// Owned, immutable world-frame target for repeated GICP registration.
///
/// All rows are validated once, including unmatched rows. Original shape
/// coefficients are retained: the factor still canonicalizes each matched
/// shape exactly as in [`register`]. No support filtering or shape fallback is
/// performed. Point IDs and exact tree traversal ties retain input order.
#[derive(Debug)]
pub struct PreparedTarget {
    points: Vec<[f64; 3]>,
    shapes: Vec<Matrix3>,
    tree: Option<KdTree>,
}

impl PreparedTarget {
    pub fn new(target: &[[f64; 3]], target_shapes: &[Matrix3]) -> Result<Self, GicpError> {
        validate_rows(target, target_shapes, "invalid target point or shape")?;
        validate_target_capacity(target.len())?;
        Ok(Self {
            points: target.to_vec(),
            shapes: target_shapes.to_vec(),
            tree: KdTree::new(target),
        })
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Source points/shapes use sensor axes; target points/shapes use world axes.
    /// Every call validates its pose, parameters and source rows, then borrows
    /// the prepared target. Shared calls have no mutable target scratch state.
    pub fn register(
        &self,
        source: &[[f64; 3]],
        source_shapes: &[Matrix3],
        initial_pose: Rigid,
        center_world: [f64; 3],
        params: RegistrationParams,
    ) -> Result<Registration, GicpError> {
        validate_common(&initial_pose, &center_world, params)?;
        validate_rows(source, source_shapes, "invalid source point or shape")?;
        Ok(register_problem(
            Problem {
                source,
                target: &self.points,
                source_shapes,
                target_shapes: &self.shapes,
                tree: self.tree.as_ref(),
                params,
            },
            initial_pose,
            center_world,
        ))
    }
}

struct Matches {
    source: Vec<usize>,
    target: Vec<usize>,
    distances: Vec<f64>,
    pairs: Vec<Correspondence>,
}

struct Problem<'a> {
    source: &'a [[f64; 3]],
    target: &'a [[f64; 3]],
    source_shapes: &'a [Matrix3],
    target_shapes: &'a [Matrix3],
    tree: Option<&'a KdTree>,
    params: RegistrationParams,
}

impl Problem<'_> {
    fn matches(&self, pose: &Rigid) -> Result<Matches, GicpError> {
        let mut result = Matches {
            source: Vec::new(),
            target: Vec::new(),
            distances: Vec::new(),
            pairs: Vec::new(),
        };
        let Some(tree) = &self.tree else {
            return Ok(result);
        };
        let rows = indexed(self.source.len(), self.params.parallel, |i| {
            let world = pose.apply(&self.source[i]);
            if !world.iter().all(|v| v.is_finite()) {
                return Err(error(Some(i), "nonfinite transformed source"));
            }
            let nearest = tree.nearest(&world, None);
            if !nearest.distance_sq.is_finite() {
                return Err(error(Some(i), "nonfinite nearest squared distance"));
            }
            Ok(nearest)
        });
        for (i, row) in rows.into_iter().enumerate() {
            let hit = row?;
            if hit.distance_sq >= self.params.max_correspondence_distance.powi(2) {
                continue;
            }
            let weight = if let Some(scale) = self.params.robust_scale {
                let denominator = scale * scale + hit.distance_sq;
                if !denominator.is_finite() {
                    return Err(error(Some(i), "nonfinite robust weight denominator"));
                }
                (scale * scale / denominator).powi(2)
            } else {
                1.0
            };
            result.source.push(i);
            result.target.push(hit.index);
            result.distances.push(hit.distance_sq);
            result.pairs.push(Correspondence {
                source: self.source[i],
                target: self.target[hit.index],
                source_shape: self.source_shapes[i],
                target_shape: self.target_shapes[hit.index],
                weight,
            });
        }
        Ok(result)
    }
}

/// Left world rotation about a fixed world center; translation precedes rotation.
fn retract(pose: &Rigid, center: [f64; 3], step: [f64; 6], scale: f64) -> Rigid {
    let q = crate::pose_graph::exp_so3(&[scale * step[3], scale * step[4], scale * step[5]]);
    Rigid {
        rotation: std::array::from_fn(|i| {
            std::array::from_fn(|j| (0..3).map(|k| q[i][k] * pose.rotation[k][j]).sum())
        }),
        translation: std::array::from_fn(|i| {
            center[i]
                + scale * step[i]
                + (0..3)
                    .map(|k| q[i][k] * (pose.translation[k] - center[k]))
                    .sum::<f64>()
        }),
    }
}

#[allow(clippy::needless_range_loop)]
fn solve(
    h: [[f64; 6]; 6],
    rhs: [f64; 6],
    weights: f64,
    radius: f64,
) -> Result<[f64; 6], Termination> {
    if radius <= 0.0 {
        return Err(Termination::SingularSystem);
    }
    let d = [1.0, 1.0, 1.0, 1.0 / radius, 1.0 / radius, 1.0 / radius];
    let a: [[f64; 6]; 6] = std::array::from_fn(|i| {
        std::array::from_fn(|j| (h[i][j] / 2.0 + h[j][i] / 2.0) / weights * d[i] * d[j])
    });
    let mut b = std::array::from_fn::<_, 6, _>(|i| rhs[i] / weights * d[i]);
    if !a.iter().flatten().chain(&b).all(|v| v.is_finite()) {
        return Err(Termination::Nonfinite);
    }
    let max_diagonal = (0..6).map(|i| a[i][i]).fold(0.0, f64::max);
    let mut lower = [[0.0; 6]; 6];
    for i in 0..6 {
        for j in 0..=i {
            let value = a[i][j] - (0..j).map(|k| lower[i][k] * lower[j][k]).sum::<f64>();
            if !value.is_finite() {
                return Err(Termination::Nonfinite);
            }
            if i == j {
                if value <= 1e-12 * max_diagonal {
                    return Err(Termination::SingularSystem);
                }
                lower[i][j] = value.sqrt();
            } else {
                lower[i][j] = value / lower[j][j];
            }
        }
        b[i] = (b[i] - (0..i).map(|k| lower[i][k] * b[k]).sum::<f64>()) / lower[i][i];
    }
    let mut x = [0.0; 6];
    for i in (0..6).rev() {
        x[i] = (b[i] - (i + 1..6).map(|k| lower[k][i] * x[k]).sum::<f64>()) / lower[i][i];
    }
    let delta = std::array::from_fn(|i| d[i] * x[i]);
    if delta.iter().all(|v| v.is_finite()) {
        Ok(delta)
    } else {
        Err(Termination::Nonfinite)
    }
}

/// Register against a fixed global nearest-neighbor target; no hidden fallback.
#[allow(clippy::too_many_arguments)]
pub fn register(
    source: &[[f64; 3]],
    target: &[[f64; 3]],
    source_shapes: &[Matrix3],
    target_shapes: &[Matrix3],
    initial_pose: Rigid,
    center_world: [f64; 3],
    params: RegistrationParams,
) -> Result<Registration, GicpError> {
    validate_common(&initial_pose, &center_world, params)?;
    validate_rows(source, source_shapes, "invalid source point or shape")?;
    validate_rows(target, target_shapes, "invalid target point or shape")?;
    validate_target_capacity(target.len())?;
    let tree = KdTree::new(target);
    Ok(register_problem(
        Problem {
            source,
            target,
            source_shapes,
            target_shapes,
            tree: tree.as_ref(),
            params,
        },
        initial_pose,
        center_world,
    ))
}

fn validate_common(
    initial_pose: &Rigid,
    center_world: &[f64; 3],
    params: RegistrationParams,
) -> Result<(), GicpError> {
    if !gicp::proper_pose(initial_pose) || !center_world.iter().all(|v| v.is_finite()) {
        return Err(error(
            None,
            "initial_pose must be proper rigid and center_world finite",
        ));
    }
    if params.max_iterations == 0
        || [
            params.max_correspondence_distance,
            params.translation_tolerance,
            params.rotation_tolerance,
        ]
        .iter()
        .any(|v| !v.is_finite() || *v <= 0.0)
        || !params.max_correspondence_distance.powi(2).is_finite()
        || params.max_correspondence_distance.powi(2) == 0.0
        || params.robust_scale.is_some_and(|v| {
            !v.is_finite() || v <= 0.0 || !v.powi(2).is_finite() || v.powi(2) == 0.0
        })
    {
        return Err(error(
            None,
            "iterations, distances and tolerances must be finite and positive",
        ));
    }
    if let Some(cycle) = params.cycle_guard {
        cycle.validate()?;
        if cycle.translation_tolerance > 0.01 * params.translation_tolerance
            || cycle.rotation_tolerance > 0.01 * params.rotation_tolerance
        {
            return Err(error(
                None,
                "cycle pose tolerances must be <= 0.01 of convergence tolerances",
            ));
        }
    }
    Ok(())
}

fn validate_rows(
    points: &[[f64; 3]],
    shapes: &[Matrix3],
    reason: &'static str,
) -> Result<(), GicpError> {
    if points.len() != shapes.len() {
        return Err(error(None, "each point needs one shape"));
    }
    for (i, (point, shape)) in points.iter().zip(shapes).enumerate() {
        if !point.iter().all(|v| v.is_finite()) || gicp::canonical_shape(shape).is_none() {
            return Err(error(Some(i), reason));
        }
    }
    Ok(())
}

fn validate_target_capacity(count: usize) -> Result<(), GicpError> {
    if count > u32::MAX as usize {
        return Err(error(None, "target exceeds tree index capacity"));
    }
    Ok(())
}

fn register_problem(
    problem: Problem<'_>,
    initial_pose: Rigid,
    center_world: [f64; 3],
) -> Registration {
    let params = problem.params;
    let mut result = Registration {
        pose: initial_pose,
        termination: Termination::MaxIterations,
        iterations: 0,
        accepted_steps: 0,
        final_cost: None,
        matched_source_indices: Vec::new(),
        matched_target_indices: Vec::new(),
        squared_distances: Vec::new(),
        weights: Vec::new(),
        failure_reason: None,
        history: Vec::new(),
        cycle: None,
    };
    let mut cycle_tracker = params.cycle_guard.map(CycleTracker::new);
    for _ in 0..params.max_iterations {
        result.iterations += 1;
        let matches = match problem.matches(&result.pose) {
            Ok(value) => value,
            Err(e) => {
                result.termination = Termination::Nonfinite;
                result.failure_reason = Some(e.reason);
                break;
            }
        };
        let linear =
            match gicp::linearize(&matches.pairs, &result.pose, center_world, params.parallel) {
                Ok(value) => value,
                Err(e) => {
                    result.termination = Termination::Nonfinite;
                    result.failure_reason = Some(e.reason);
                    break;
                }
            };
        let mut item = Iteration {
            pose_before: result.pose,
            pose_after: result.pose,
            source_indices: matches.source,
            target_indices: matches.target,
            squared_distances: matches.distances,
            weights: matches.pairs.iter().map(|p| p.weight).collect(),
            combined_shapes: linear.combined_shapes,
            normal_matrix: linear.normal_matrix,
            right_hand_side: linear.right_hand_side,
            cost_before: linear.cost,
            cost_after: None,
            step: None,
            trial_scales: Vec::new(),
            trial_costs: Vec::new(),
            accepted_scale: None,
        };
        let outcome = (|| -> Result<(), Termination> {
            if linear.positive_weight_points < 3 {
                return Err(Termination::InsufficientCorrespondences);
            }
            let radius_sq = matches
                .pairs
                .iter()
                .zip(&linear.source_world)
                .map(|(p, q)| {
                    p.weight
                        * (0..3)
                            .map(|j| (q[j] - center_world[j]).powi(2))
                            .sum::<f64>()
                })
                .sum::<f64>()
                / linear.weight_sum;
            if !radius_sq.is_finite() {
                return Err(Termination::Nonfinite);
            }
            let delta = solve(
                linear.normal_matrix,
                linear.right_hand_side,
                linear.weight_sum,
                radius_sq.sqrt(),
            )?;
            item.step = Some(delta);
            let norm = |values: &[f64]| values.iter().fold(0.0_f64, |a, b| a.hypot(*b));
            if norm(&delta[..3]) <= params.translation_tolerance
                && norm(&delta[3..]) <= params.rotation_tolerance
            {
                return Err(Termination::ConvergedStep);
            }
            if let Some(tracker) = &mut cycle_tracker {
                match tracker.observe(
                    &item,
                    result.iterations,
                    center_world,
                    params.translation_tolerance,
                    params.rotation_tolerance,
                ) {
                    Ok(Some(report)) => {
                        result.cycle = Some(report);
                        result.failure_reason = Some("repeated nonstationary GICP outer states");
                        return Err(Termination::CycleDetected);
                    }
                    Ok(None) => {}
                    Err(e) => {
                        result.failure_reason = Some(e.reason);
                        return Err(Termination::Nonfinite);
                    }
                }
            }
            let frozen = FrozenMetric::new(
                &matches.pairs.iter().map(|p| p.source).collect::<Vec<_>>(),
                &matches.pairs.iter().map(|p| p.target).collect::<Vec<_>>(),
                &item.combined_shapes,
                &item.weights,
            )
            .map_err(|_| Termination::Nonfinite)?;
            let mut finite_trial = false;
            for scale in [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125] {
                let candidate = retract(&result.pose, center_world, delta, scale);
                let cost = frozen
                    .evaluate(&candidate, params.parallel)
                    .ok()
                    .map(|v| v.cost);
                finite_trial |= cost.is_some();
                item.trial_scales.push(scale);
                item.trial_costs.push(cost);
                if cost.is_some_and(|v| v < linear.cost) {
                    result.pose = candidate;
                    result.accepted_steps += 1;
                    item.pose_after = candidate;
                    item.cost_after = cost;
                    item.accepted_scale = Some(scale);
                    return Ok(());
                }
            }
            Err(if finite_trial {
                Termination::NoDescent
            } else {
                Termination::Nonfinite
            })
        })();
        if params.capture_history {
            result.history.push(item);
        }
        if let Err(termination) = outcome {
            result.termination = termination;
            break;
        }
    }
    match problem.matches(&result.pose).and_then(|matches| {
        let linear = gicp::linearize(&matches.pairs, &result.pose, center_world, params.parallel)?;
        Ok((matches, linear.cost))
    }) {
        Ok((matches, cost)) => {
            result.final_cost = Some(cost);
            result.matched_source_indices = matches.source;
            result.matched_target_indices = matches.target;
            result.squared_distances = matches.distances;
            result.weights = matches.pairs.iter().map(|p| p.weight).collect();
        }
        Err(e) => {
            result.termination = Termination::Nonfinite;
            result.failure_reason = Some(e.reason);
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    const ID: Matrix3 = [[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]];
    fn points() -> Vec<[f64; 3]> {
        vec![
            [0., 0., 0.],
            [2., 0., 0.],
            [0., 3., 0.],
            [0., 0., 4.],
            [2., 3., 4.],
        ]
    }
    fn run(source: &[[f64; 3]], target: &[[f64; 3]], params: RegistrationParams) -> Registration {
        register(
            source,
            target,
            &vec![ID; source.len()],
            &vec![ID; target.len()],
            Rigid::IDENTITY,
            [0.; 3],
            params,
        )
        .unwrap()
    }

    #[test]
    fn recovers_a_small_rigid_motion_and_preserves_parallel_history() {
        let source = points();
        let truth = Rigid {
            rotation: crate::pose_graph::exp_so3(&[0.03, -0.02, 0.04]),
            translation: [0.1, -0.05, 0.07],
        };
        let target: Vec<_> = source.iter().map(|p| truth.apply(p)).collect();
        let params = RegistrationParams {
            translation_tolerance: 1e-10,
            rotation_tolerance: 1e-10,
            capture_history: true,
            ..Default::default()
        };
        let result = run(&source, &target, params);
        assert_eq!(result.termination, Termination::ConvergedStep);
        assert!(result.accepted_steps >= 2);
        for (a, b) in result.pose.to_matrix().iter().zip(truth.to_matrix()) {
            assert!((a - b).abs() < 1e-9);
        }
        assert_eq!(
            result,
            run(
                &source,
                &target,
                RegistrationParams {
                    parallel: true,
                    ..params
                }
            )
        );
        assert!(
            result
                .history
                .iter()
                .filter(|i| i.accepted_scale.is_some())
                .all(|i| i.cost_after.unwrap() < i.cost_before)
        );
        let uncaptured = run(
            &source,
            &target,
            RegistrationParams {
                capture_history: false,
                ..params
            },
        );
        assert_eq!(uncaptured.pose, result.pose);
        assert!(uncaptured.history.is_empty());
    }

    #[test]
    fn no_support_singular_and_iteration_limit_are_distinct() {
        let p = points();
        assert_eq!(
            run(&p, &[], Default::default()).termination,
            Termination::InsufficientCorrespondences
        );
        assert_eq!(
            run(&[], &p, Default::default()).termination,
            Termination::InsufficientCorrespondences
        );
        let line = [[0., 0., 0.], [1., 0., 0.], [2., 0., 0.]];
        assert_eq!(
            run(&line, &line, Default::default()).termination,
            Termination::SingularSystem
        );
        let target: Vec<_> = p.iter().map(|p| [p[0] + 0.1, p[1], p[2]]).collect();
        let limited = run(
            &p,
            &target,
            RegistrationParams {
                max_iterations: 1,
                ..Default::default()
            },
        );
        assert_eq!(limited.termination, Termination::MaxIterations);
        assert_eq!(limited.accepted_steps, 1);
        assert!(limited.final_cost.unwrap() < 1e-25);
    }

    #[test]
    fn validates_unmatched_shapes_and_rejects_overflow_explicitly() {
        let source = points();
        let mut shapes = vec![ID; source.len()];
        shapes[2][0][0] = -1.;
        let e = register(
            &source,
            &[],
            &shapes,
            &[],
            Rigid::IDENTITY,
            [0.; 3],
            Default::default(),
        )
        .unwrap_err();
        assert_eq!(e.index, Some(2));
        assert!(e.reason.contains("source"));
        let result = run(&[[f64::MAX; 3]], &[[0.; 3]], Default::default());
        assert_eq!(result.termination, Termination::Nonfinite);
        assert!(result.final_cost.is_none());
        assert_eq!(result.pose, Rigid::IDENTITY);
    }

    #[test]
    fn frozen_metric_keeps_baseline_axes_and_validates_zero_weights() {
        let source = [[1., 2., 3.]];
        let target = [[0.; 3]];
        let shape = [[[0.5, 0., 0.], [0., 1., 0.], [0., 0., 2.]]];
        let frozen = FrozenMetric::new(&source, &target, &shape, &[2.]).unwrap();
        let result = frozen.evaluate(&Rigid::IDENTITY, false).unwrap();
        assert!((result.cost - 10.5).abs() < 1e-14);
        assert_eq!(result, frozen.evaluate(&Rigid::IDENTITY, true).unwrap());
        let mut bad = shape;
        bad[0][2][2] = 0.;
        assert_eq!(
            FrozenMetric::new(&source, &target, &bad, &[0.])
                .unwrap_err()
                .index,
            Some(0)
        );
        let pose = Rigid {
            rotation: crate::pose_graph::exp_so3(&[0., 0., 0.5]),
            translation: [0.; 3],
        };
        let world = pose.apply(&source[0]);
        let expected = world[0].powi(2) / 0.5 + world[1].powi(2) + world[2].powi(2) / 2.;
        assert!((frozen.evaluate(&pose, false).unwrap().cost - expected).abs() < 1e-13);
    }

    #[test]
    fn threshold_is_strict_and_tied_queries_are_repeatable() {
        let source = [[0., 0., 0.]];
        let target = [[1., 0., 0.], [-1., 0., 0.]];
        assert!(
            run(&source, &target, Default::default())
                .matched_source_indices
                .is_empty()
        );
        let params = RegistrationParams {
            max_correspondence_distance: 2.,
            capture_history: true,
            ..Default::default()
        };
        let result = run(&source, &target, params);
        assert_eq!(result.matched_source_indices, vec![0]);
        assert_eq!(result.squared_distances, vec![1.]);
        assert_eq!(
            result,
            run(
                &source,
                &target,
                RegistrationParams {
                    parallel: true,
                    ..params
                }
            )
        );
    }

    #[test]
    fn retraction_uses_fixed_center_and_full_step_convergence_only() {
        let pose = Rigid {
            translation: [8., -3., 1.],
            ..Rigid::IDENTITY
        };
        let center = [7., -1., 2.];
        let step = [0.2, -0.1, 0.3, 0., 0., std::f64::consts::FRAC_PI_2];
        let updated = retract(&pose, center, step, 1.);
        let expected = [9.2, -0.1, 1.3];
        for (a, b) in updated.translation.iter().zip(expected) {
            assert!((a - b).abs() < 1e-14);
        }
        assert!(gicp::proper_pose(&updated));
    }

    #[test]
    fn prepared_target_owns_inputs_and_preserves_all_history_on_reuse() {
        let source = points();
        let mut target: Vec<_> = source.iter().map(|p| [p[0] + 0.1, p[1], p[2]]).collect();
        let cs = vec![ID; source.len()];
        let mut ct = cs.clone();
        ct[0][0][1] = 2e-14;
        let params = RegistrationParams {
            capture_history: true,
            ..Default::default()
        };
        let expected =
            register(&source, &target, &cs, &ct, Rigid::IDENTITY, [0.; 3], params).unwrap();
        let prepared = PreparedTarget::new(&target, &ct).unwrap();
        assert_eq!(prepared.len(), target.len());
        assert!(!prepared.is_empty());
        target.fill([f64::NAN; 3]);
        ct.fill([[0.; 3]; 3]);
        for parallel in [false, true, false] {
            let actual = prepared
                .register(
                    &source,
                    &cs,
                    Rigid::IDENTITY,
                    [0.; 3],
                    RegistrationParams { parallel, ..params },
                )
                .unwrap();
            assert_eq!(actual, expected);
        }
    }

    #[test]
    fn prepared_target_validates_constructor_and_per_call_first_errors() {
        let source = points();
        let mut shapes = vec![ID; source.len()];
        shapes[2][0][0] = 0.;
        assert_eq!(
            PreparedTarget::new(&source, &shapes).unwrap_err(),
            error(Some(2), "invalid target point or shape")
        );
        assert_eq!(
            PreparedTarget::new(&source, &[]).unwrap_err(),
            error(None, "each point needs one shape")
        );
        let empty = PreparedTarget::new(&[], &[]).unwrap();
        assert!(empty.is_empty());
        let result = empty
            .register(&[], &[], Rigid::IDENTITY, [0.; 3], Default::default())
            .unwrap();
        assert_eq!(result, run(&[], &[], Default::default()));
        let err = empty
            .register(
                &source,
                &shapes,
                Rigid::IDENTITY,
                [0.; 3],
                Default::default(),
            )
            .unwrap_err();
        assert_eq!(err, error(Some(2), "invalid source point or shape"));
        let mut pose = Rigid::IDENTITY;
        pose.translation[0] = f64::NAN;
        assert_eq!(
            empty
                .register(&source, &shapes, pose, [0.; 3], Default::default())
                .unwrap_err()
                .index,
            None
        );
        #[cfg(target_pointer_width = "64")]
        assert!(validate_target_capacity(u32::MAX as usize + 1).is_err());
    }

    #[test]
    fn prepared_target_shared_threads_and_traversal_ties_are_identical() {
        fn send_sync<T: Send + Sync>() {}
        send_sync::<PreparedTarget>();
        let source = [[0., 0., 0.], [0., 2., 0.], [0., 0., 3.]];
        let target = [[1., 0., 0.], [-1., 0., 0.], [0., 2., 0.], [0., 0., 3.]];
        let prepared = PreparedTarget::new(&target, &[ID; 4]).unwrap();
        let params = RegistrationParams {
            max_correspondence_distance: 2.,
            capture_history: true,
            ..Default::default()
        };
        let expected = register(
            &source,
            &target,
            &[ID; 3],
            &[ID; 4],
            Rigid::IDENTITY,
            [0.; 3],
            params,
        )
        .unwrap();
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..4)
                .map(|_| {
                    scope.spawn(|| {
                        prepared
                            .register(&source, &[ID; 3], Rigid::IDENTITY, [0.; 3], params)
                            .unwrap()
                    })
                })
                .collect();
            for handle in handles {
                assert_eq!(handle.join().unwrap(), expected);
            }
        });
        let strict = RegistrationParams {
            max_correspondence_distance: 1.,
            ..params
        };
        assert_eq!(
            prepared
                .register(&source, &[ID; 3], Rigid::IDENTITY, [0.; 3], strict)
                .unwrap(),
            register(
                &source,
                &target,
                &[ID; 3],
                &[ID; 4],
                Rigid::IDENTITY,
                [0.; 3],
                strict
            )
            .unwrap()
        );
    }
}
