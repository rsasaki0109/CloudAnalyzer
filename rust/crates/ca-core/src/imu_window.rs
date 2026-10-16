//! Bounded batch optimization of an adjacent IMU chain with diagonal priors.
//!
//! Each node has 15 increments: right-local rotation, world velocity, world
//! position, gyro bias, accelerometer bias. A scaled block Cholesky LM solve
//! uses O(N) memory and work for fixed block size. Oldest-state marginalization
//! preserves a frozen linear Gaussian prior; initialization, gap handling,
//! extrinsics and bias reintegration remain caller policies.

use crate::imu::{ImuError, multiply, transpose};
use crate::imu_factor::{ImuFactor, ImuState, right_jacobian_inverse, validate_state};
use crate::pose_graph::{exp_so3, log_so3};

const DIM: usize = 15;
pub const MAX_STATES: usize = 32;
type Vector = [f64; DIM];
type Matrix = [[f64; DIM]; DIM];

#[derive(Debug, Clone, Copy)]
pub struct NavigationState {
    pub motion: ImuState,
    /// Gyroscope, then accelerometer bias, in the IMU frame.
    pub bias: [f64; 6],
}

impl NavigationState {
    fn validate(&self) -> Result<(), ImuError> {
        validate_state(&self.motion)?;
        finite(self.bias.into_iter())
    }

    fn retract(&self, delta: &Vector) -> Result<Self, ImuError> {
        let result = Self {
            motion: ImuState {
                rotation: multiply(
                    &self.motion.rotation,
                    &exp_so3(&[delta[0], delta[1], delta[2]]),
                ),
                velocity: std::array::from_fn(|i| self.motion.velocity[i] + delta[i + 3]),
                position: std::array::from_fn(|i| self.motion.position[i] + delta[i + 6]),
            },
            bias: std::array::from_fn(|i| self.bias[i] + delta[i + 9]),
        };
        result.validate()?;
        Ok(result)
    }
}

#[derive(Debug, Clone)]
struct Edge {
    factor: ImuFactor,
    inverse_bias_sigma: [f64; 6],
}

#[derive(Debug, Clone)]
struct Prior {
    index: usize,
    mean: NavigationState,
    // Zero weights are used only for the unobserved parts of a pose prior.
    inverse_sigma: Vector,
}

impl Prior {
    fn residual(&self, state: &NavigationState) -> Result<Vector, ImuError> {
        let rotation = log_so3(&multiply(
            &transpose(&self.mean.motion.rotation),
            &state.motion.rotation,
        ));
        let residual = std::array::from_fn(|i| {
            let raw = match i {
                0..3 => rotation[i],
                3..6 => state.motion.velocity[i - 3] - self.mean.motion.velocity[i - 3],
                6..9 => state.motion.position[i - 6] - self.mean.motion.position[i - 6],
                _ => state.bias[i - 9] - self.mean.bias[i - 9],
            };
            raw * self.inverse_sigma[i]
        });
        finite(residual.into_iter())?;
        Ok(residual)
    }

    fn jacobian(&self, state: &NavigationState) -> Result<Matrix, ImuError> {
        let rotation = log_so3(&multiply(
            &transpose(&self.mean.motion.rotation),
            &state.motion.rotation,
        ));
        if std::f64::consts::PI - rotation.iter().map(|v| v * v).sum::<f64>().sqrt() < 1e-6 {
            return Err(ImuError(
                "IMU window prior rotation is at the SO(3) logarithm branch cut".into(),
            ));
        }
        let jr = right_jacobian_inverse(rotation);
        Ok(std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                if i < 3 && j < 3 {
                    jr[i][j] * self.inverse_sigma[i]
                } else if i == j {
                    self.inverse_sigma[i]
                } else {
                    0.0
                }
            })
        }))
    }
}

#[derive(Debug, Clone, Copy)]
pub struct OptimizeParams {
    pub max_iterations: usize,
    pub initial_damping: f64,
    pub gradient_tolerance: f64,
    pub step_tolerance: f64,
    pub cost_tolerance: f64,
}

impl Default for OptimizeParams {
    fn default() -> Self {
        Self {
            max_iterations: 20,
            initial_damping: 1e-3,
            gradient_tolerance: 1e-8,
            step_tolerance: 1e-9,
            cost_tolerance: 1e-10,
        }
    }
}

#[derive(Debug, Clone)]
pub struct OptimizeReport {
    pub initial_cost: f64,
    pub final_cost: f64,
    pub iterations: usize,
    pub accepted_steps: usize,
    pub rejected_steps: usize,
    pub converged: bool,
    pub termination: &'static str,
    /// Initial cost, followed by strictly decreasing accepted costs.
    pub cost_history: Vec<f64>,
    /// Per-axis maximum |bias_i - integration_bias_i| over the IMU edges.
    /// This is diagnostic; it does not certify a bias linearization radius.
    pub max_abs_bias_correction: [f64; 6],
}

/// A frozen linearized Gaussian at the first retained state. Its residual is
/// square_root * Local(reference, current) + residual_offset. This preserves
/// all 15D correlations, without relinearizing discarded nonlinear factors.
#[derive(Debug, Clone)]
pub struct MarginalPrior {
    pub reference: NavigationState,
    pub square_root: Matrix,
    pub residual_offset: Vector,
}

impl MarginalPrior {
    fn residual(&self, state: &NavigationState) -> Result<Vector, ImuError> {
        let rotation = log_so3(&multiply(
            &transpose(&self.reference.motion.rotation),
            &state.motion.rotation,
        ));
        let local: Vector = std::array::from_fn(|i| match i {
            0..3 => rotation[i],
            3..6 => state.motion.velocity[i - 3] - self.reference.motion.velocity[i - 3],
            6..9 => state.motion.position[i - 6] - self.reference.motion.position[i - 6],
            _ => state.bias[i - 9] - self.reference.bias[i - 9],
        });
        let residual = std::array::from_fn(|r| {
            self.residual_offset[r]
                + (0..DIM)
                    .map(|c| self.square_root[r][c] * local[c])
                    .sum::<f64>()
        });
        finite(residual.into_iter())?;
        Ok(residual)
    }

    fn jacobian(&self, state: &NavigationState) -> Result<Matrix, ImuError> {
        let rotation = log_so3(&multiply(
            &transpose(&self.reference.motion.rotation),
            &state.motion.rotation,
        ));
        if std::f64::consts::PI - rotation.iter().map(|v| v * v).sum::<f64>().sqrt() < 1e-6 {
            return Err(ImuError(
                "IMU marginal prior rotation is at the SO(3) logarithm branch cut".into(),
            ));
        }
        let jr = right_jacobian_inverse(rotation);
        Ok(std::array::from_fn(|r| {
            std::array::from_fn(|c| {
                if c < 3 {
                    (0..3).map(|k| self.square_root[r][k] * jr[k][c]).sum()
                } else {
                    self.square_root[r][c]
                }
            })
        }))
    }
}

#[derive(Debug, Clone)]
pub struct MarginalizeReport {
    pub dropped_timestamp_ns: i64,
    pub retained_timestamp_ns: i64,
    pub removed_priors: usize,
    pub replaced_marginal_prior: bool,
    /// Add this to the new prior cost to recover the reduced linearized cost.
    /// The optimizer drops this state-independent constant (possibly tiny
    /// negative roundoff); costs across different windows are not comparable.
    pub dropped_linearized_constant: f64,
}

#[derive(Debug, Clone)]
pub struct ImuWindow {
    timestamps_ns: Vec<i64>,
    states: Vec<NavigationState>,
    gravity: [f64; 3],
    edges: Vec<Option<Edge>>,
    priors: Vec<Prior>,
    has_state_prior: bool,
    marginal_prior: Option<MarginalPrior>,
}

fn finite(values: impl Iterator<Item = f64>) -> Result<(), ImuError> {
    if values.into_iter().all(f64::is_finite) {
        Ok(())
    } else {
        Err(ImuError("IMU window inputs/results must be finite".into()))
    }
}

fn inverse_sigma(sigma: f64) -> Result<f64, ImuError> {
    let inverse = sigma.recip();
    let weight = inverse * inverse;
    if !sigma.is_finite() || sigma <= 0.0 || !weight.is_finite() || weight <= 0.0 {
        return Err(ImuError("IMU window standard deviations must be positive with finite, nonzero squared inverse weights".into()));
    }
    Ok(inverse)
}

fn half_squared(residual: &[f64]) -> f64 {
    0.5 * residual.iter().map(|x| x * x).sum::<f64>()
}

impl ImuWindow {
    pub fn new(
        timestamps_ns: Vec<i64>,
        states: Vec<NavigationState>,
        gravity: [f64; 3],
    ) -> Result<Self, ImuError> {
        if !(2..=MAX_STATES).contains(&states.len()) || timestamps_ns.len() != states.len() {
            return Err(ImuError(format!(
                "IMU window requires 2..={MAX_STATES} states and matching timestamps"
            )));
        }
        for pair in timestamps_ns.windows(2) {
            if pair[1].checked_sub(pair[0]).is_none_or(|dt| dt <= 0) {
                return Err(ImuError(
                    "IMU window timestamps must increase without integer overflow".into(),
                ));
            }
        }
        finite(gravity.into_iter())?;
        for state in &states {
            state.validate()?;
        }
        let edges = vec![None; states.len() - 1];
        Ok(Self {
            timestamps_ns,
            states,
            gravity,
            edges,
            priors: Vec::new(),
            has_state_prior: false,
            marginal_prior: None,
        })
    }

    pub fn states(&self) -> &[NavigationState] {
        &self.states
    }

    pub fn timestamps_ns(&self) -> &[i64] {
        &self.timestamps_ns
    }

    pub fn add_imu_factor(
        &mut self,
        index: usize,
        factor: ImuFactor,
        gyro_bias_random_walk: f64,
        accel_bias_random_walk: f64,
    ) -> Result<(), ImuError> {
        if index >= self.edges.len() {
            return Err(ImuError("IMU edge index is outside the window".into()));
        }
        if self.edges[index].is_some() {
            return Err(ImuError("IMU window already has this adjacent edge".into()));
        }
        self.edges[index] = Some(self.make_edge(
            factor,
            self.timestamps_ns[index],
            self.timestamps_ns[index + 1],
            gyro_bias_random_walk,
            accel_bias_random_walk,
        )?);
        Ok(())
    }

    fn make_edge(
        &self,
        factor: ImuFactor,
        start_ns: i64,
        end_ns: i64,
        gyro_bias_random_walk: f64,
        accel_bias_random_walk: f64,
    ) -> Result<Edge, ImuError> {
        let m = factor.measurement();
        if m.start_ns != start_ns || m.end_ns != end_ns {
            return Err(ImuError(
                "IMU factor exact timestamps must match adjacent window states".into(),
            ));
        }
        if self
            .edges
            .iter()
            .flatten()
            .any(|e| e.factor.measurement().frame_id != m.frame_id)
        {
            return Err(ImuError(
                "IMU window factors must use the same IMU frame".into(),
            ));
        }
        let dt_sqrt = m.delta_time_s.sqrt();
        // Check the densities themselves, too: a negative/overflowing density
        // must not become a seemingly valid interval standard deviation.
        inverse_sigma(gyro_bias_random_walk)?;
        inverse_sigma(accel_bias_random_walk)?;
        let gyro = inverse_sigma(gyro_bias_random_walk * dt_sqrt)?;
        let accel = inverse_sigma(accel_bias_random_walk * dt_sqrt)?;
        Ok(Edge {
            factor,
            inverse_bias_sigma: [gyro, gyro, gyro, accel, accel, accel],
        })
    }

    /// Atomically append one state and its adjacent IMU edge to a complete
    /// window. Marginalize first when the bounded window is full.
    pub fn append_state(
        &mut self,
        timestamp_ns: i64,
        state: NavigationState,
        factor: ImuFactor,
        gyro_bias_random_walk: f64,
        accel_bias_random_walk: f64,
    ) -> Result<(), ImuError> {
        self.check_complete()?;
        if self.states.len() == MAX_STATES {
            return Err(ImuError(
                "IMU window is full; marginalize before appending".into(),
            ));
        }
        state.validate()?;
        let start = *self.timestamps_ns.last().expect("nonempty window");
        if timestamp_ns.checked_sub(start).is_none_or(|dt| dt <= 0) {
            return Err(ImuError(
                "IMU window timestamps must increase without integer overflow".into(),
            ));
        }
        let edge = self.make_edge(
            factor,
            start,
            timestamp_ns,
            gyro_bias_random_walk,
            accel_bias_random_walk,
        )?;
        self.timestamps_ns.push(timestamp_ns);
        self.states.push(state);
        self.edges.push(Some(edge));
        Ok(())
    }

    pub fn marginal_prior(&self) -> Option<&MarginalPrior> {
        self.marginal_prior.as_ref()
    }

    /// Eliminate only factors touching the oldest state. Retained pose/state
    /// priors and later IMU edges stay exactly once, without being folded into
    /// the new prior. The Schur complement is undamped and computed with scaled
    /// Cholesky solves; nonpositive pivots fail without jitter or mutation.
    pub fn marginalize_oldest(&mut self) -> Result<MarginalizeReport, ImuError> {
        self.check_complete()?;
        if self.states.len() < 3 {
            return Err(ImuError(
                "marginalization must retain at least two states".into(),
            ));
        }
        if self.marginal_prior.is_none()
            && !self
                .priors
                .iter()
                .any(|p| p.index == 0 && p.inverse_sigma.iter().all(|w| *w > 0.0))
        {
            return Err(ImuError("marginalization requires a full state prior at the oldest state or an existing marginal prior".into()));
        }
        let local = Self {
            timestamps_ns: self.timestamps_ns[..2].to_vec(),
            states: self.states[..2].to_vec(),
            gravity: self.gravity,
            edges: vec![self.edges[0].clone()],
            priors: self
                .priors
                .iter()
                .filter(|p| p.index == 0)
                .cloned()
                .collect(),
            has_state_prior: true,
            marginal_prior: self.marginal_prior.clone(),
        };
        let normal = local.linearize(&local.states)?;
        let (prior, constant) =
            normal.eliminate_first(local.states[1], local.cost(&local.states)?)?;
        let report = MarginalizeReport {
            dropped_timestamp_ns: self.timestamps_ns[0],
            retained_timestamp_ns: self.timestamps_ns[1],
            removed_priors: local.priors.len(),
            replaced_marginal_prior: self.marginal_prior.is_some(),
            dropped_linearized_constant: constant,
        };
        // Every fallible operation is complete before committing changes.
        self.timestamps_ns.remove(0);
        self.states.remove(0);
        self.edges.remove(0);
        self.priors.retain(|p| p.index > 0);
        for prior in &mut self.priors {
            prior.index -= 1;
        }
        self.marginal_prior = Some(prior);
        Ok(report)
    }

    pub fn add_state_prior(
        &mut self,
        index: usize,
        mean: NavigationState,
        sigmas: Vector,
    ) -> Result<(), ImuError> {
        self.check_index(index)?;
        mean.validate()?;
        let mut weights = [0.0; DIM];
        for i in 0..DIM {
            weights[i] = inverse_sigma(sigmas[i])?;
        }
        self.priors.push(Prior {
            index,
            mean,
            inverse_sigma: weights,
        });
        self.has_state_prior = true;
        Ok(())
    }

    pub fn add_pose_prior(
        &mut self,
        index: usize,
        mut mean: ImuState,
        rotation_sigma: f64,
        position_sigma: f64,
    ) -> Result<(), ImuError> {
        self.check_index(index)?;
        mean.velocity = [0.0; 3];
        validate_state(&mean)?;
        let rotation = inverse_sigma(rotation_sigma)?;
        let position = inverse_sigma(position_sigma)?;
        let mut weights = [0.0; DIM];
        weights[..3].fill(rotation);
        weights[6..9].fill(position);
        self.priors.push(Prior {
            index,
            mean: NavigationState {
                motion: mean,
                bias: [0.0; 6],
            },
            inverse_sigma: weights,
        });
        Ok(())
    }

    fn check_index(&self, index: usize) -> Result<(), ImuError> {
        if index < self.states.len() {
            Ok(())
        } else {
            Err(ImuError("IMU prior index is outside the window".into()))
        }
    }

    fn check_complete(&self) -> Result<(), ImuError> {
        if self.edges.iter().any(Option::is_none) {
            return Err(ImuError(
                "IMU window requires every adjacent edge; split at missing intervals".into(),
            ));
        }
        if !self.has_state_prior {
            return Err(ImuError(
                "IMU window requires at least one full 15D state prior".into(),
            ));
        }
        Ok(())
    }

    fn cost(&self, states: &[NavigationState]) -> Result<f64, ImuError> {
        let mut cost = 0.0;
        for (i, edge) in self.edges.iter().enumerate() {
            let edge = edge.as_ref().expect("complete window");
            cost += edge
                .factor
                .evaluate(
                    &states[i].motion,
                    &states[i + 1].motion,
                    self.gravity,
                    states[i].bias,
                )?
                .cost;
            let residual: [f64; 6] = std::array::from_fn(|k| {
                (states[i + 1].bias[k] - states[i].bias[k]) * edge.inverse_bias_sigma[k]
            });
            cost += half_squared(&residual);
        }
        for prior in &self.priors {
            cost += half_squared(&prior.residual(&states[prior.index])?);
        }
        if let Some(prior) = &self.marginal_prior {
            cost += half_squared(&prior.residual(&states[0])?);
        }
        finite(std::iter::once(cost))?;
        Ok(cost)
    }

    fn linearize(&self, states: &[NavigationState]) -> Result<Normal, ImuError> {
        let mut normal = Normal::new(states.len());
        for (i, edge) in self.edges.iter().enumerate() {
            let edge = edge.as_ref().expect("complete window");
            let lin = edge.factor.linearize(
                &states[i].motion,
                &states[i + 1].motion,
                self.gravity,
                states[i].bias,
            )?;
            let ji: [[f64; DIM]; 9] = std::array::from_fn(|r| {
                std::array::from_fn(|c| {
                    if c < 9 {
                        lin.whitened_jacobian_i[r][c]
                    } else {
                        lin.whitened_jacobian_bias[r][c - 9]
                    }
                })
            });
            let jj: [[f64; DIM]; 9] = std::array::from_fn(|r| {
                std::array::from_fn(|c| {
                    if c < 9 {
                        lin.whitened_jacobian_j[r][c]
                    } else {
                        0.0
                    }
                })
            });
            normal.add_node(i, &ji, &lin.evaluation.whitened_residual);
            normal.add_node(i + 1, &jj, &lin.evaluation.whitened_residual);
            normal.add_cross(i, &ji, &jj);
            // Bias random walk has a diagonal six-row Jacobian.
            for k in 0..6 {
                let c = k + 9;
                let weight = edge.inverse_bias_sigma[k].powi(2);
                let g = (states[i + 1].bias[k] - states[i].bias[k]) * weight;
                normal.diagonal[i][c][c] += weight;
                normal.diagonal[i + 1][c][c] += weight;
                normal.below[i][c][c] -= weight;
                normal.gradient[i][c] -= g;
                normal.gradient[i + 1][c] += g;
            }
        }
        for prior in &self.priors {
            let state = &states[prior.index];
            normal.add_node(
                prior.index,
                &prior.jacobian(state)?,
                &prior.residual(state)?,
            );
        }
        if let Some(prior) = &self.marginal_prior {
            normal.add_node(
                0,
                &prior.jacobian(&states[0])?,
                &prior.residual(&states[0])?,
            );
        }
        finite(
            normal
                .diagonal
                .iter()
                .chain(&normal.below)
                .flatten()
                .flatten()
                .copied()
                .chain(normal.gradient.iter().flatten().copied()),
        )?;
        Ok(normal)
    }

    /// All changes are committed together after a finite result. An error leaves
    /// the window unchanged; a nonconverged report retains accepted progress.
    pub fn optimize(&mut self, params: &OptimizeParams) -> Result<OptimizeReport, ImuError> {
        self.check_complete()?;
        if !(1..=100).contains(&params.max_iterations)
            || !params.initial_damping.is_finite()
            || params.initial_damping <= 0.0
            || params.initial_damping > 1e16
            || [
                params.gradient_tolerance,
                params.step_tolerance,
                params.cost_tolerance,
            ]
            .into_iter()
            .any(|v| !v.is_finite() || v < 0.0)
        {
            return Err(ImuError("invalid IMU window optimizer parameters".into()));
        }
        let mut states = self.states.clone();
        let initial_cost = self.cost(&states)?;
        let mut cost = initial_cost;
        let mut damping = params.initial_damping;
        let mut report = OptimizeReport {
            initial_cost,
            final_cost: initial_cost,
            iterations: 0,
            accepted_steps: 0,
            rejected_steps: 0,
            converged: false,
            termination: "max_iterations",
            cost_history: vec![initial_cost],
            max_abs_bias_correction: [0.0; 6],
        };
        'iterations: for iteration in 0..params.max_iterations {
            report.iterations = iteration + 1;
            let normal = self.linearize(&states)?;
            let gradient = normal
                .gradient
                .iter()
                .flatten()
                .fold(0.0_f64, |m, v| m.max(v.abs()));
            if gradient <= params.gradient_tolerance {
                report.termination = "gradient";
                report.converged = true;
                break;
            }
            for _ in 0..12 {
                if let Some(delta) = normal.solve(damping) {
                    let step = delta.iter().flatten().fold(0.0_f64, |m, v| m.max(v.abs()));
                    let candidate = states
                        .iter()
                        .zip(&delta)
                        .map(|(state, d)| state.retract(d))
                        .collect::<Result<Vec<_>, _>>();
                    if let Ok(candidate) = candidate
                        && let Ok(next_cost) = self.cost(&candidate)
                        && next_cost < cost
                    {
                        let reduction = cost - next_cost;
                        let old_cost = cost;
                        cost = next_cost;
                        states = candidate;
                        report.accepted_steps += 1;
                        report.cost_history.push(cost);
                        damping = (damping * 0.3).max(1e-12);
                        if step <= params.step_tolerance
                            || reduction <= params.cost_tolerance * (1.0 + old_cost)
                        {
                            report.termination = if step <= params.step_tolerance {
                                "step"
                            } else {
                                "cost"
                            };
                            report.converged = true;
                            break 'iterations;
                        }
                        continue 'iterations;
                    }
                    // A large damping can make a step arbitrarily small.
                    // Do not label such a rejected step as convergence.
                }
                report.rejected_steps += 1;
                damping *= 10.0;
                if damping > 1e16 {
                    break;
                }
            }
            report.termination = "no_descent";
            break;
        }
        report.final_cost = self.cost(&states)?;
        for (i, edge) in self.edges.iter().enumerate() {
            let bias = edge
                .as_ref()
                .expect("complete window")
                .factor
                .measurement()
                .linearization_bias;
            for (k, &reference) in bias.iter().enumerate() {
                report.max_abs_bias_correction[k] =
                    report.max_abs_bias_correction[k].max((states[i].bias[k] - reference).abs());
            }
        }
        self.states = states;
        Ok(report)
    }
}

/// H[i,i], H[i+1,i], and J^T r. Nonadjacent blocks are exactly zero.
struct Normal {
    diagonal: Vec<Matrix>,
    below: Vec<Matrix>,
    gradient: Vec<Vector>,
}

impl Normal {
    fn new(n: usize) -> Self {
        Self {
            diagonal: vec![[[0.0; DIM]; DIM]; n],
            below: vec![[[0.0; DIM]; DIM]; n - 1],
            gradient: vec![[0.0; DIM]; n],
        }
    }

    fn add_node<const M: usize>(&mut self, index: usize, j: &[[f64; DIM]; M], r: &[f64; M]) {
        for a in 0..DIM {
            self.gradient[index][a] += (0..M).map(|k| j[k][a] * r[k]).sum::<f64>();
            for b in 0..DIM {
                self.diagonal[index][a][b] += (0..M).map(|k| j[k][a] * j[k][b]).sum::<f64>();
            }
        }
    }

    fn add_cross<const M: usize>(
        &mut self,
        index: usize,
        ji: &[[f64; DIM]; M],
        jj: &[[f64; DIM]; M],
    ) {
        for (a, row) in self.below[index].iter_mut().enumerate() {
            for (b, value) in row.iter_mut().enumerate() {
                *value += (0..M).map(|k| jj[k][a] * ji[k][b]).sum::<f64>();
            }
        }
    }

    fn eliminate_first(
        &self,
        reference: NavigationState,
        cost: f64,
    ) -> Result<(MarginalPrior, f64), ImuError> {
        let failure = || {
            ImuError("IMU marginalization requires finite positive-definite scaled blocks".into())
        };
        let scales: [Vector; 2] =
            std::array::from_fn(|i| std::array::from_fn(|k| self.diagonal[i][k][k].sqrt().recip()));
        if scales.iter().flatten().any(|s| !s.is_finite() || *s <= 0.0) {
            return Err(failure());
        }
        let first = std::array::from_fn(|r| {
            std::array::from_fn(|c| self.diagonal[0][r][c] * scales[0][r] * scales[0][c])
        });
        let lower = cholesky(&first).ok_or_else(failure)?;
        let cross: Matrix = std::array::from_fn(|r| {
            forward(
                &lower,
                std::array::from_fn(|c| self.below[0][r][c] * scales[1][r] * scales[0][c]),
            )
        });
        let first_rhs = forward(
            &lower,
            std::array::from_fn(|r| self.gradient[0][r] * scales[0][r]),
        );
        let mut schur = [[0.0; DIM]; DIM];
        // Fill one triangle and mirror it, avoiding asymmetric rounding.
        for r in 0..DIM {
            for c in 0..=r {
                let value = self.diagonal[1][r][c] * scales[1][r] * scales[1][c]
                    - (0..DIM).map(|k| cross[r][k] * cross[c][k]).sum::<f64>();
                schur[r][c] = value;
                schur[c][r] = value;
            }
        }
        let lower_schur = cholesky(&schur).ok_or_else(failure)?;
        let gradient = std::array::from_fn(|r| {
            self.gradient[1][r] * scales[1][r]
                - (0..DIM).map(|k| cross[r][k] * first_rhs[k]).sum::<f64>()
        });
        let residual_offset = forward(&lower_schur, gradient);
        let square_root: Matrix =
            std::array::from_fn(|r| std::array::from_fn(|c| lower_schur[c][r] / scales[1][c]));
        let constant = cost - half_squared(&first_rhs) - half_squared(&residual_offset);
        finite(
            square_root
                .iter()
                .flatten()
                .copied()
                .chain(residual_offset)
                .chain([constant]),
        )?;
        Ok((
            MarginalPrior {
                reference,
                square_root,
                residual_offset,
            },
            constant,
        ))
    }

    fn solve(&self, damping: f64) -> Option<Vec<Vector>> {
        let n = self.diagonal.len();
        let scales: Vec<Vector> = self
            .diagonal
            .iter()
            .map(|a| {
                std::array::from_fn(|k| {
                    if a[k][k] > 0.0 {
                        a[k][k].sqrt().recip()
                    } else {
                        f64::NAN
                    }
                })
            })
            .collect();
        if scales.iter().flatten().any(|v| !v.is_finite()) {
            return None;
        }
        let mut lower: Vec<Matrix> = Vec::with_capacity(n);
        let mut cross: Vec<Matrix> = Vec::with_capacity(n - 1);
        let mut rhs: Vec<Vector> = Vec::with_capacity(n);
        for i in 0..n {
            let mut a: Matrix = std::array::from_fn(|r| {
                std::array::from_fn(|c| {
                    self.diagonal[i][r][c] * scales[i][r] * scales[i][c]
                        + if r == c { damping } else { 0.0 }
                })
            });
            let mut b: Vector = std::array::from_fn(|r| -self.gradient[i][r] * scales[i][r]);
            if i > 0 {
                let scaled: Matrix = std::array::from_fn(|r| {
                    std::array::from_fn(|c| {
                        self.below[i - 1][r][c] * scales[i][r] * scales[i - 1][c]
                    })
                });
                // C = B L_prev^-T; solve one row at a time without inverses.
                let c: Matrix = scaled.map(|row| forward(&lower[i - 1], row));
                for r in 0..DIM {
                    b[r] -= (0..DIM).map(|k| c[r][k] * rhs[i - 1][k]).sum::<f64>();
                    for s in 0..DIM {
                        a[r][s] -= (0..DIM).map(|k| c[r][k] * c[s][k]).sum::<f64>();
                    }
                }
                cross.push(c);
            }
            let l = cholesky(&a)?;
            rhs.push(forward(&l, b));
            lower.push(l);
        }
        let mut solution = vec![[0.0; DIM]; n];
        for i in (0..n).rev() {
            let mut b = rhs[i];
            if i + 1 < n {
                for (r, value) in b.iter_mut().enumerate() {
                    *value -= (0..DIM)
                        .map(|k| cross[i][k][r] * solution[i + 1][k])
                        .sum::<f64>();
                }
            }
            solution[i] = backward(&lower[i], b);
        }
        for i in 0..n {
            for k in 0..DIM {
                solution[i][k] *= scales[i][k];
            }
        }
        solution
            .iter()
            .flatten()
            .all(|v| v.is_finite())
            .then_some(solution)
    }
}

fn cholesky(a: &Matrix) -> Option<Matrix> {
    let mut l = [[0.0; DIM]; DIM];
    for i in 0..DIM {
        for j in 0..=i {
            let v = a[i][j] - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
            if i == j {
                if !v.is_finite() || v <= 0.0 {
                    return None;
                }
                l[i][j] = v.sqrt();
            } else {
                l[i][j] = v / l[j][j];
            }
        }
    }
    Some(l)
}

fn forward(l: &Matrix, mut b: Vector) -> Vector {
    for i in 0..DIM {
        b[i] = (b[i] - (0..i).map(|k| l[i][k] * b[k]).sum::<f64>()) / l[i][i];
    }
    b
}

fn backward(l: &Matrix, mut b: Vector) -> Vector {
    for i in (0..DIM).rev() {
        b[i] = (b[i] - (i + 1..DIM).map(|k| l[k][i] * b[k]).sum::<f64>()) / l[i][i];
    }
    b
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bag::RawImuMessage;
    use crate::imu::{PreintegrationParams, preintegrate};

    fn state() -> NavigationState {
        NavigationState {
            motion: ImuState {
                rotation: exp_so3(&[0.3, -0.2, 0.5]),
                velocity: [0.2, -0.3, 0.1],
                position: [10.0, -2.0, 3.0],
            },
            bias: [0.01, -0.02, 0.03, 0.1, -0.05, 0.02],
        }
    }

    fn factor(start: i64, end: i64) -> ImuFactor {
        let samples: Vec<_> = (0..=40)
            .map(|i| RawImuMessage {
                stamp_ns: i * 10_000_000,
                stamp: i as f64 * 0.01,
                frame_id: "imu".into(),
                orientation: [0.0; 4],
                orientation_covariance: [-1.0; 9],
                angular_velocity: [0.2 + (i as f64).sin(), -0.3, 0.4],
                angular_velocity_covariance: [0.0; 9],
                acceleration: [0.5, -0.2 + (i as f64).cos(), 9.81],
                acceleration_covariance: [0.0; 9],
            })
            .collect();
        let p = PreintegrationParams {
            gyro_noise_density: 0.02,
            accel_noise_density: 0.15,
            gyro_bias: [0.0; 3],
            accel_bias: [0.0; 3],
            max_gap_s: 0.05,
        };
        ImuFactor::new(preintegrate(&samples, start, end, &p).unwrap()).unwrap()
    }

    // Independent dense Gaussian elimination with partial pivoting; production
    // uses only a chain of 15x15 Cholesky blocks.
    fn dense_solve(mut a: Vec<Vec<f64>>, mut b: Vec<f64>) -> Vec<f64> {
        let n = b.len();
        for k in 0..n {
            let pivot = (k..n)
                .max_by(|&i, &j| a[i][k].abs().total_cmp(&a[j][k].abs()))
                .unwrap();
            a.swap(k, pivot);
            b.swap(k, pivot);
            let pivot_row = a[k].clone();
            for i in k + 1..n {
                let ratio = a[i][k] / a[k][k];
                for (value, &reference) in a[i][k..].iter_mut().zip(&pivot_row[k..]) {
                    *value -= ratio * reference;
                }
                b[i] -= ratio * b[k];
            }
        }
        for i in (0..n).rev() {
            b[i] = (b[i] - (i + 1..n).map(|j| a[i][j] * b[j]).sum::<f64>()) / a[i][i];
        }
        b
    }

    #[test]
    fn scaled_block_solve_matches_independent_dense_elimination() {
        for n in [2, 5, 11] {
            let mut normal = Normal::new(n);
            let l: Vec<Matrix> = (0..n)
                .map(|i| {
                    std::array::from_fn(|r| {
                        std::array::from_fn(|c| {
                            if r == c {
                                1.5 + (i + r) as f64 * 0.01
                            } else if r > c {
                                ((i + r + c + 1) as f64).sin() * 0.05
                            } else {
                                0.0
                            }
                        })
                    })
                })
                .collect();
            let c: Vec<Matrix> = (0..n - 1)
                .map(|i| {
                    std::array::from_fn(|r| {
                        std::array::from_fn(|s| ((2 * i + r + s + 1) as f64).cos() * 0.02)
                    })
                })
                .collect();
            let units: Vec<Vector> = (0..n)
                .map(|i| std::array::from_fn(|k| 10.0_f64.powf(((i + k) % 9) as f64 / 2.0 - 2.0)))
                .collect();
            for i in 0..n {
                let mut a = multiply(&l[i], &transpose(&l[i]));
                if i > 0 {
                    let previous = multiply(&c[i - 1], &transpose(&c[i - 1]));
                    for r in 0..DIM {
                        for s in 0..DIM {
                            a[r][s] += previous[r][s];
                        }
                    }
                }
                for r in 0..DIM {
                    for s in 0..DIM {
                        normal.diagonal[i][r][s] = a[r][s] * units[i][r] * units[i][s];
                    }
                }
                normal.gradient[i] =
                    std::array::from_fn(|k| ((i + 2 * k + 1) as f64).sin() * units[i][k]);
                if i + 1 < n {
                    let b = multiply(&c[i], &transpose(&l[i]));
                    for r in 0..DIM {
                        for s in 0..DIM {
                            normal.below[i][r][s] = b[r][s] * units[i + 1][r] * units[i][s];
                        }
                    }
                }
            }
            for damping in [0.0, 1e-3, 0.7] {
                let mut dense = vec![vec![0.0; n * DIM]; n * DIM];
                for i in 0..n {
                    for r in 0..DIM {
                        for s in 0..DIM {
                            dense[i * DIM + r][i * DIM + s] = normal.diagonal[i][r][s];
                            if i + 1 < n {
                                dense[(i + 1) * DIM + r][i * DIM + s] = normal.below[i][r][s];
                                dense[i * DIM + s][(i + 1) * DIM + r] = normal.below[i][r][s];
                            }
                        }
                    }
                }
                for (i, row) in dense.iter_mut().enumerate() {
                    row[i] *= 1.0 + damping;
                }
                let expected = dense_solve(
                    dense,
                    normal.gradient.iter().flatten().map(|v| -v).collect(),
                );
                let actual = normal.solve(damping).unwrap();
                for (a, b) in actual.iter().flatten().zip(expected) {
                    assert!((a - b).abs() < 1e-8 * (1.0 + b.abs()), "{a} != {b}");
                }
            }
        }
    }

    #[test]
    fn marginalization_matches_dense_schur_without_double_counting_retained_priors() {
        let s = state();
        let states = (0..3)
            .map(|i| {
                s.retract(&std::array::from_fn(|k| {
                    ((i * DIM + k + 1) as f64).sin() * 0.02
                }))
                .unwrap()
            })
            .collect();
        let mut window =
            ImuWindow::new(vec![0, 200_000_000, 400_000_000], states, [0.0, 0.0, -9.81]).unwrap();
        window
            .add_imu_factor(0, factor(0, 200_000_000), 0.01, 0.1)
            .unwrap();
        window
            .add_imu_factor(1, factor(200_000_000, 400_000_000), 0.01, 0.1)
            .unwrap();
        window.add_state_prior(0, s, [0.1; DIM]).unwrap();
        for i in 0..3 {
            window.add_pose_prior(i, s.motion, 0.03, 0.2).unwrap();
        }
        let before = window.linearize(window.states()).unwrap();
        let before_cost = window.cost(window.states()).unwrap();
        let a: Vec<Vec<f64>> = before.diagonal[0].iter().map(|r| r.to_vec()).collect();
        let ag = dense_solve(a.clone(), before.gradient[0].to_vec());
        let columns: Vec<_> = (0..DIM)
            .map(|c| dense_solve(a.clone(), (0..DIM).map(|r| before.below[0][c][r]).collect()))
            .collect();
        let retained_states = window.states()[1..].to_vec();
        let report = window.marginalize_oldest().unwrap();
        let after = window.linearize(window.states()).unwrap();
        assert_eq!(report.dropped_timestamp_ns, 0);
        assert_eq!(report.retained_timestamp_ns, 200_000_000);
        assert_eq!(report.removed_priors, 2);
        assert!(!report.replaced_marginal_prior);
        for r in 0..DIM {
            let expected_g = before.gradient[1][r]
                - (0..DIM).map(|k| before.below[0][r][k] * ag[k]).sum::<f64>();
            assert!((after.gradient[0][r] - expected_g).abs() < 1e-8 * (1.0 + expected_g.abs()));
            for (c, column) in columns.iter().enumerate() {
                let expected = before.diagonal[1][r][c]
                    - (0..DIM)
                        .map(|k| before.below[0][r][k] * column[k])
                        .sum::<f64>();
                assert!((after.diagonal[0][r][c] - expected).abs() < 1e-8 * (1.0 + expected.abs()));
            }
        }
        assert_eq!(after.diagonal[1], before.diagonal[2]);
        assert_eq!(after.below[0], before.below[1]);
        assert_eq!(after.gradient[1], before.gradient[2]);
        let reduced_cost =
            before_cost - 0.5 * (0..DIM).map(|k| before.gradient[0][k] * ag[k]).sum::<f64>();
        let actual_cost =
            window.cost(window.states()).unwrap() + report.dropped_linearized_constant;
        assert!((actual_cost - reduced_cost).abs() < 1e-8 * (1.0 + reduced_cost.abs()));
        assert_eq!(
            format!("{:?}", window.states()),
            format!("{retained_states:?}")
        );
        let root = &window.marginal_prior().unwrap().square_root;
        assert!((0..DIM).any(|r| (r + 1..DIM).any(|c| root[r][c].abs() > 1e-6)));
    }

    #[test]
    fn dense_marginal_prior_jacobian_matches_right_retraction_and_rejects_branch_cut() {
        let reference = state();
        let prior = MarginalPrior {
            reference,
            square_root: std::array::from_fn(|r| {
                std::array::from_fn(|c| {
                    if r == c {
                        1.0 + r as f64 * 0.1
                    } else {
                        ((r + c + 1) as f64).sin() * 0.03
                    }
                })
            }),
            residual_offset: std::array::from_fn(|r| (r as f64).cos() * 0.1),
        };
        let current = reference
            .retract(&std::array::from_fn(|k| ((k + 1) as f64).sin() * 0.2))
            .unwrap();
        let j = prior.jacobian(&current).unwrap();
        for c in 0..DIM {
            let mut d = [0.0; DIM];
            d[c] = 1e-6;
            let plus = prior.residual(&current.retract(&d).unwrap()).unwrap();
            d[c] = -1e-6;
            let minus = prior.residual(&current.retract(&d).unwrap()).unwrap();
            for r in 0..DIM {
                assert!(((plus[r] - minus[r]) / 2e-6 - j[r][c]).abs() < 2e-8);
            }
        }
        let at_cut = reference
            .retract(&std::array::from_fn(|k| {
                if k == 0 { std::f64::consts::PI } else { 0.0 }
            }))
            .unwrap();
        assert!(prior.jacobian(&at_cut).is_err());
    }

    #[test]
    fn append_and_marginalization_errors_are_atomic_and_window_stays_bounded() {
        let s = state();
        let mut window = ImuWindow::new(vec![0, 200_000_000], vec![s; 2], [0.0; 3]).unwrap();
        window
            .add_imu_factor(0, factor(0, 200_000_000), 0.01, 0.1)
            .unwrap();
        window.add_state_prior(0, s, [0.1; DIM]).unwrap();
        let snapshot = format!("{window:?}");
        assert!(window.marginalize_oldest().is_err());
        assert!(
            window
                .append_state(400_000_000, s, factor(0, 200_000_000), 0.01, 0.1)
                .is_err()
        );
        assert_eq!(snapshot, format!("{window:?}"));
        for i in 2..MAX_STATES {
            let mut m = factor(0, 200_000_000).measurement().clone();
            m.start_ns = (i as i64 - 1) * 200_000_000;
            m.end_ns = i as i64 * 200_000_000;
            window
                .append_state(m.end_ns, s, ImuFactor::new(m).unwrap(), 0.01, 0.1)
                .unwrap();
        }
        let mut next = factor(0, 200_000_000).measurement().clone();
        next.start_ns = (MAX_STATES as i64 - 1) * 200_000_000;
        next.end_ns = MAX_STATES as i64 * 200_000_000;
        let snapshot = format!("{window:?}");
        assert!(
            window
                .append_state(
                    next.end_ns,
                    s,
                    ImuFactor::new(next.clone()).unwrap(),
                    0.01,
                    0.1
                )
                .is_err()
        );
        assert_eq!(snapshot, format!("{window:?}"));
        window.marginalize_oldest().unwrap();
        assert_eq!(window.states().len(), MAX_STATES - 1);
        window
            .append_state(next.end_ns, s, ImuFactor::new(next).unwrap(), 0.01, 0.1)
            .unwrap();
        assert_eq!(window.states().len(), MAX_STATES);
        let report = window.marginalize_oldest().unwrap();
        assert!(report.replaced_marginal_prior);
    }

    #[test]
    fn marginalization_rejects_unsupported_anchor_without_mutation() {
        let s = state();
        let mut window =
            ImuWindow::new(vec![0, 200_000_000, 400_000_000], vec![s; 3], [0.0; 3]).unwrap();
        window
            .add_imu_factor(0, factor(0, 200_000_000), 0.01, 0.1)
            .unwrap();
        window
            .add_imu_factor(1, factor(200_000_000, 400_000_000), 0.01, 0.1)
            .unwrap();
        window.add_state_prior(1, s, [0.1; DIM]).unwrap();
        let before = format!("{window:?}");
        assert!(
            window
                .marginalize_oldest()
                .unwrap_err()
                .0
                .contains("oldest state")
        );
        assert_eq!(before, format!("{window:?}"));
    }

    #[test]
    fn marginalization_scaling_preserves_mixed_units_and_rejects_singular_schur() {
        let mut normal = Normal::new(2);
        let l: Matrix = std::array::from_fn(|r| {
            std::array::from_fn(|c| {
                if r == c {
                    1.0 + r as f64 * 0.1
                } else if r > c {
                    ((r + c + 1) as f64).sin() * 0.02
                } else {
                    0.0
                }
            })
        });
        let cross: Matrix =
            std::array::from_fn(|r| std::array::from_fn(|c| ((2 * r + c + 1) as f64).cos() * 0.02));
        let a = multiply(&l, &transpose(&l));
        let b = multiply(&cross, &transpose(&l));
        let cc = multiply(&cross, &transpose(&cross));
        let units: [Vector; 2] = std::array::from_fn(|i| {
            std::array::from_fn(|k| 10.0_f64.powi(((i + k) % 9) as i32 - 4))
        });
        for r in 0..DIM {
            for c in 0..DIM {
                normal.diagonal[0][r][c] = a[r][c] * units[0][r] * units[0][c];
                normal.diagonal[1][r][c] = (a[r][c] + cc[r][c]) * units[1][r] * units[1][c];
                normal.below[0][r][c] = b[r][c] * units[1][r] * units[0][c];
            }
        }
        let (prior, _) = normal.eliminate_first(state(), 0.0).unwrap();
        let information = multiply(&transpose(&prior.square_root), &prior.square_root);
        for r in 0..DIM {
            for c in 0..DIM {
                let expected = a[r][c] * units[1][r] * units[1][c];
                assert!((information[r][c] - expected).abs() < 1e-10 * (1.0 + expected.abs()));
            }
        }
        for i in 0..2 {
            normal.diagonal[i] =
                std::array::from_fn(|r| std::array::from_fn(|c| f64::from(r == c)));
        }
        normal.below[0] = normal.diagonal[0];
        assert!(normal.eliminate_first(state(), 0.0).is_err());
    }

    #[test]
    fn full_and_pose_prior_jacobians_match_right_retraction_differences() {
        let mean = state();
        let delta = std::array::from_fn(|i| ((i + 1) as f64).sin() * 0.2);
        let current = mean.retract(&delta).unwrap();
        for pose_only in [false, true] {
            let weights = std::array::from_fn(|i| {
                if pose_only && !(i < 3 || (6..9).contains(&i)) {
                    0.0
                } else {
                    1.0 + i as f64 * 0.1
                }
            });
            let prior = Prior {
                index: 0,
                mean,
                inverse_sigma: weights,
            };
            let j = prior.jacobian(&current).unwrap();
            for c in 0..DIM {
                let mut d = [0.0; DIM];
                d[c] = 1e-6;
                let plus = prior.residual(&current.retract(&d).unwrap()).unwrap();
                d[c] = -1e-6;
                let minus = prior.residual(&current.retract(&d).unwrap()).unwrap();
                for r in 0..DIM {
                    assert!(((plus[r] - minus[r]) / 2e-6 - j[r][c]).abs() < 2e-8);
                }
            }
        }
    }

    #[test]
    fn rejects_wrong_absolute_interval_missing_edges_anchor_and_mixed_frames() {
        let s = state();
        let mut window =
            ImuWindow::new(vec![0, 200_000_000, 400_000_000], vec![s; 3], [0.0; 3]).unwrap();
        assert!(
            window
                .add_imu_factor(0, factor(10_000_000, 210_000_000), 0.01, 0.1)
                .is_err()
        );
        assert!(window.optimize(&OptimizeParams::default()).is_err());
        window
            .add_imu_factor(0, factor(0, 200_000_000), 0.01, 0.1)
            .unwrap();
        assert!(
            window
                .add_imu_factor(0, factor(0, 200_000_000), 0.01, 0.1)
                .is_err()
        );
        let mut other = factor(200_000_000, 400_000_000).measurement().clone();
        other.frame_id = "other".into();
        assert!(
            window
                .add_imu_factor(1, ImuFactor::new(other).unwrap(), 0.01, 0.1)
                .is_err()
        );
        window
            .add_imu_factor(1, factor(200_000_000, 400_000_000), 0.01, 0.1)
            .unwrap();
        window.add_pose_prior(0, s.motion, 0.1, 0.1).unwrap();
        assert!(window.optimize(&OptimizeParams::default()).is_err());
        for sigma in [0.0, -1.0, f64::NAN, f64::INFINITY, 1e-200, 1e200] {
            assert!(window.add_state_prior(0, s, [sigma; DIM]).is_err());
        }
        assert!(ImuWindow::new(vec![i64::MIN, i64::MAX], vec![s; 2], [0.0; 3]).is_err());
        assert!(ImuWindow::new(vec![0, 0], vec![s; 2], [0.0; 3]).is_err());
        let mut m = factor(0, 200_000_000).measurement().clone();
        m.end_ns += 1;
        assert!(ImuFactor::new(m).is_err());
    }

    #[test]
    fn joint_optimization_recovers_consistent_chain_and_bias_with_finite_priors() {
        let gravity = [0.1, -0.2, -9.81];
        let initial = state();
        let a = factor(0, 200_000_000);
        let b = factor(200_000_000, 400_000_000);
        let middle = NavigationState {
            motion: a.predict(&initial.motion, gravity, initial.bias).unwrap(),
            ..initial
        };
        let end = NavigationState {
            motion: b.predict(&middle.motion, gravity, initial.bias).unwrap(),
            ..initial
        };
        let truth = [initial, middle, end];
        let perturbed = truth
            .iter()
            .enumerate()
            .map(|(i, s)| {
                s.retract(&std::array::from_fn(|k| {
                    ((i * DIM + k + 1) as f64).sin() * 0.03
                }))
                .unwrap()
            })
            .collect();
        let mut window =
            ImuWindow::new(vec![0, 200_000_000, 400_000_000], perturbed, gravity).unwrap();
        window.add_imu_factor(0, a, 0.01, 0.1).unwrap();
        window.add_imu_factor(1, b, 0.01, 0.1).unwrap();
        window.add_state_prior(0, initial, [0.1; DIM]).unwrap();
        let report = window
            .optimize(&OptimizeParams {
                max_iterations: 60,
                ..Default::default()
            })
            .unwrap();
        assert!(report.converged, "{report:?}");
        assert!(report.final_cost < 1e-14, "{report:?}");
        assert!(report.cost_history.windows(2).all(|c| c[1] < c[0]));
        for (actual, reference) in window.states().iter().zip(truth) {
            for (a, b) in actual
                .motion
                .position
                .into_iter()
                .chain(actual.motion.velocity)
                .chain(actual.bias)
                .zip(
                    reference
                        .motion
                        .position
                        .into_iter()
                        .chain(reference.motion.velocity)
                        .chain(reference.bias),
                )
            {
                assert!((a - b).abs() < 1e-7);
            }
        }
    }

    #[test]
    fn errors_leave_states_unchanged_and_random_walk_uses_interval_variance() {
        let s = state();
        let mut window = ImuWindow::new(vec![0, 200_000_000], vec![s; 2], [0.0; 3]).unwrap();
        window
            .add_imu_factor(0, factor(0, 200_000_000), 0.01, 0.1)
            .unwrap();
        let edge = window.edges[0].as_ref().unwrap();
        assert!(
            (edge.inverse_bias_sigma[0].powi(2) - 1.0 / (0.01_f64.powi(2) * 0.2)).abs() < 1e-10
        );
        assert!((edge.inverse_bias_sigma[3].powi(2) - 1.0 / (0.1_f64.powi(2) * 0.2)).abs() < 1e-10);
        let mut mean = s;
        mean.motion.rotation = multiply(
            &s.motion.rotation,
            &exp_so3(&[std::f64::consts::PI, 0.0, 0.0]),
        );
        window.add_state_prior(0, mean, [0.1; DIM]).unwrap();
        assert!(window.optimize(&OptimizeParams::default()).is_err());
        for actual in window.states() {
            assert_eq!(actual.motion.rotation, s.motion.rotation);
            assert_eq!(actual.motion.velocity, s.motion.velocity);
            assert_eq!(actual.motion.position, s.motion.position);
            assert_eq!(actual.bias, s.bias);
        }
    }
}
