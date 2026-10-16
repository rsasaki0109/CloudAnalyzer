//! A preintegrated IMU factor for states at the IMU origin.
//!
//! State increments are [right-local rotation, world velocity, world position].
//! Residual order is [rotation, velocity, position]; bias order is [gyro, accel].
//! Gravity is supplied in the world frame. Extrinsics, bias random walk and an
//! optimizer/initialization policy are outside this factor.

use crate::imu::{ImuError, PreintegratedImu, multiply, right_jacobian, skew, transpose};
use crate::pose_graph::{exp_so3, log_so3};

type Mat3 = [[f64; 3]; 3];
type Mat9 = [[f64; 9]; 9];
type CorrectedDelta = (Mat3, [f64; 3], [f64; 3], [f64; 3]);

#[derive(Debug, Clone, Copy)]
pub struct ImuState {
    /// IMU to world rotation.
    pub rotation: Mat3,
    pub velocity: [f64; 3],
    pub position: [f64; 3],
}

#[derive(Debug, Clone)]
pub struct ImuEvaluation {
    pub residual: [f64; 9],
    pub whitened_residual: [f64; 9],
    /// One half of the squared Mahalanobis norm.
    pub cost: f64,
}

#[derive(Debug, Clone)]
pub struct ImuLinearization {
    pub evaluation: ImuEvaluation,
    pub jacobian_i: Mat9,
    pub jacobian_j: Mat9,
    pub jacobian_bias: [[f64; 6]; 9],
    pub jacobian_gravity: [[f64; 3]; 9],
    pub whitened_jacobian_i: Mat9,
    pub whitened_jacobian_j: Mat9,
    pub whitened_jacobian_bias: [[f64; 6]; 9],
    pub whitened_jacobian_gravity: [[f64; 3]; 9],
}

#[derive(Debug, Clone)]
pub struct ImuFactor {
    measurement: PreintegratedImu,
    /// Covariance = lower * lower^T. Cached once, no explicit inverse or jitter.
    lower: Mat9,
}

fn finite(values: impl Iterator<Item = f64>) -> Result<(), ImuError> {
    if values.into_iter().all(f64::is_finite) {
        Ok(())
    } else {
        Err(ImuError("IMU factor inputs/results must be finite".into()))
    }
}

fn validate_rotation(r: &Mat3) -> Result<(), ImuError> {
    finite(r.iter().flatten().copied())?;
    let rrt = multiply(r, &transpose(r));
    let determinant = r[0][0] * (r[1][1] * r[2][2] - r[1][2] * r[2][1])
        - r[0][1] * (r[1][0] * r[2][2] - r[1][2] * r[2][0])
        + r[0][2] * (r[1][0] * r[2][1] - r[1][1] * r[2][0]);
    if (determinant - 1.0).abs() > 1e-6
        || !determinant.is_finite()
        || (0..3).any(|i| (0..3).any(|j| (rrt[i][j] - f64::from(u8::from(i == j))).abs() > 1e-6))
    {
        return Err(ImuError(
            "IMU state rotations must be proper orthonormal matrices".into(),
        ));
    }
    Ok(())
}

pub(crate) fn validate_state(state: &ImuState) -> Result<(), ImuError> {
    validate_rotation(&state.rotation)?;
    finite(state.velocity.into_iter().chain(state.position))
}

fn rotate(r: &Mat3, v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| (0..3).map(|j| r[i][j] * v[j]).sum())
}

pub(crate) fn right_jacobian_inverse(v: [f64; 3]) -> Mat3 {
    let t2: f64 = v.iter().map(|x| x * x).sum();
    let coefficient = if t2 < 1e-8 {
        1.0 / 12.0 + t2 / 720.0 + t2 * t2 / 30240.0
    } else {
        let half = 0.5 * t2.sqrt();
        (1.0 - half / half.tan()) / t2
    };
    let k = skew(v);
    let k2 = multiply(&k, &k);
    std::array::from_fn(|i| {
        std::array::from_fn(|j| {
            f64::from(u8::from(i == j)) + 0.5 * k[i][j] + coefficient * k2[i][j]
        })
    })
}

fn cholesky(covariance: &Mat9) -> Result<Mat9, ImuError> {
    let error = || {
        ImuError("IMU factor covariance must be finite, symmetric and numerically positive definite; no regularization is added".into())
    };
    let mut scales = [0.0; 9];
    for i in 0..9 {
        if !covariance[i][i].is_finite() || covariance[i][i] <= 0.0 {
            return Err(error());
        }
        scales[i] = covariance[i][i].sqrt();
    }
    // Factor the correlation matrix so the pivot test is dimensionless.
    let mut correlation = [[0.0; 9]; 9];
    for i in 0..9 {
        for j in 0..9 {
            let a = covariance[i][j] / scales[i] / scales[j];
            let b = covariance[j][i] / scales[i] / scales[j];
            if !a.is_finite() || !b.is_finite() || (a - b).abs() > 1e-12 {
                return Err(error());
            }
            correlation[i][j] = a * 0.5 + b * 0.5;
        }
    }
    let mut lower = [[0.0; 9]; 9];
    for i in 0..9 {
        for j in 0..=i {
            let value = correlation[i][j] - (0..j).map(|k| lower[i][k] * lower[j][k]).sum::<f64>();
            if i == j {
                if !value.is_finite() || value <= 1e-12 {
                    return Err(error());
                }
                lower[i][j] = value.sqrt();
            } else {
                lower[i][j] = value / lower[j][j];
            }
        }
    }
    for i in 0..9 {
        for value in &mut lower[i][..=i] {
            *value *= scales[i];
        }
    }
    Ok(lower)
}

impl ImuFactor {
    pub fn new(measurement: PreintegratedImu) -> Result<Self, ImuError> {
        let duration = measurement
            .end_ns
            .checked_sub(measurement.start_ns)
            .filter(|dt| *dt > 0)
            .ok_or_else(|| ImuError("IMU factor requires increasing exact timestamps".into()))?;
        let expected_dt = duration as f64 * 1e-9;
        if !measurement.delta_time_s.is_finite()
            || measurement.delta_time_s <= 0.0
            || measurement.integration_steps == 0
            || (measurement.delta_time_s - expected_dt).abs() > 1e-12 * expected_dt
        {
            return Err(ImuError(
                "IMU factor requires a positive integration interval".into(),
            ));
        }
        validate_rotation(&measurement.delta_rotation)?;
        finite(
            measurement
                .delta_velocity
                .into_iter()
                .chain(measurement.delta_position)
                .chain(measurement.linearization_bias)
                .chain(measurement.bias_jacobian.iter().flatten().copied()),
        )?;
        let lower = cholesky(&measurement.covariance)?;
        Ok(Self { measurement, lower })
    }

    pub fn measurement(&self) -> &PreintegratedImu {
        &self.measurement
    }

    fn corrected(&self, bias: &[f64; 6]) -> Result<CorrectedDelta, ImuError> {
        finite(bias.iter().copied())?;
        let m = &self.measurement;
        let difference: [f64; 6] = std::array::from_fn(|i| bias[i] - m.linearization_bias[i]);
        let correction: [f64; 9] =
            std::array::from_fn(|i| (0..6).map(|j| m.bias_jacobian[i][j] * difference[j]).sum());
        finite(correction.into_iter())?;
        let theta = [correction[0], correction[1], correction[2]];
        let rotation = multiply(&m.delta_rotation, &exp_so3(&theta));
        let velocity = std::array::from_fn(|i| m.delta_velocity[i] + correction[i + 3]);
        let position = std::array::from_fn(|i| m.delta_position[i] + correction[i + 6]);
        finite(
            rotation
                .iter()
                .flatten()
                .copied()
                .chain(velocity)
                .chain(position),
        )?;
        Ok((rotation, velocity, position, theta))
    }

    pub fn predict(
        &self,
        initial: &ImuState,
        gravity: [f64; 3],
        bias: [f64; 6],
    ) -> Result<ImuState, ImuError> {
        validate_state(initial)?;
        finite(gravity.into_iter())?;
        let (rotation, velocity, position, _) = self.corrected(&bias)?;
        let dt = self.measurement.delta_time_s;
        let dv = rotate(&initial.rotation, velocity);
        let dp = rotate(&initial.rotation, position);
        let result = ImuState {
            rotation: multiply(&initial.rotation, &rotation),
            velocity: std::array::from_fn(|i| initial.velocity[i] + gravity[i] * dt + dv[i]),
            position: std::array::from_fn(|i| {
                initial.position[i] + initial.velocity[i] * dt + 0.5 * gravity[i] * dt * dt + dp[i]
            }),
        };
        validate_state(&result)?;
        Ok(result)
    }

    fn whiten<const N: usize>(&self, values: &[[f64; N]; 9]) -> Result<[[f64; N]; 9], ImuError> {
        let mut out = [[0.0; N]; 9];
        for i in 0..9 {
            for j in 0..N {
                out[i][j] = (values[i][j]
                    - (0..i).map(|k| self.lower[i][k] * out[k][j]).sum::<f64>())
                    / self.lower[i][i];
            }
        }
        finite(out.iter().flatten().copied())?;
        Ok(out)
    }

    pub fn evaluate(
        &self,
        initial: &ImuState,
        terminal: &ImuState,
        gravity: [f64; 3],
        bias: [f64; 6],
    ) -> Result<ImuEvaluation, ImuError> {
        validate_state(initial)?;
        validate_state(terminal)?;
        finite(gravity.into_iter())?;
        let (rotation, velocity, position, _) = self.corrected(&bias)?;
        let dt = self.measurement.delta_time_s;
        let ri_t = transpose(&initial.rotation);
        let rotation_error = log_so3(&multiply(
            &transpose(&rotation),
            &multiply(&ri_t, &terminal.rotation),
        ));
        let uv = rotate(
            &ri_t,
            std::array::from_fn(|i| terminal.velocity[i] - initial.velocity[i] - gravity[i] * dt),
        );
        let up = rotate(
            &ri_t,
            std::array::from_fn(|i| {
                terminal.position[i]
                    - initial.position[i]
                    - initial.velocity[i] * dt
                    - 0.5 * gravity[i] * dt * dt
            }),
        );
        let residual = std::array::from_fn(|i| {
            if i < 3 {
                rotation_error[i]
            } else if i < 6 {
                uv[i - 3] - velocity[i - 3]
            } else {
                up[i - 6] - position[i - 6]
            }
        });
        finite(residual.into_iter())?;
        let whitened = self.whiten(&residual.map(|v| [v]))?;
        let whitened_residual: [f64; 9] = whitened.map(|v| v[0]);
        let cost = 0.5 * whitened_residual.iter().map(|x| x * x).sum::<f64>();
        finite(std::iter::once(cost))?;
        Ok(ImuEvaluation {
            residual,
            whitened_residual,
            cost,
        })
    }

    pub fn linearize(
        &self,
        initial: &ImuState,
        terminal: &ImuState,
        gravity: [f64; 3],
        bias: [f64; 6],
    ) -> Result<ImuLinearization, ImuError> {
        let evaluation = self.evaluate(initial, terminal, gravity, bias)?;
        let residual = &evaluation.residual;
        let angle = residual[..3].iter().map(|v| v * v).sum::<f64>().sqrt();
        if std::f64::consts::PI - angle < 1e-6 {
            return Err(ImuError("IMU rotation residual is at the SO(3) logarithm branch cut; linearization is undefined".into()));
        }
        let (rotation, velocity, position, theta) = self.corrected(&bias)?;
        let ri_t = transpose(&initial.rotation);
        let jr_inv = right_jacobian_inverse([residual[0], residual[1], residual[2]]);
        let jl_inv = transpose(&jr_inv);
        let rr_i = multiply(&jl_inv, &transpose(&rotation));
        let jr_bias: [[f64; 6]; 3] = std::array::from_fn(|i| self.measurement.bias_jacobian[i]);
        let rr_bias = multiply(&multiply(&jl_inv, &right_jacobian(theta)), &jr_bias);
        let uv = std::array::from_fn(|i| residual[i + 3] + velocity[i]);
        let up = std::array::from_fn(|i| residual[i + 6] + position[i]);
        let rv_i = skew(uv);
        let rp_i = skew(up);
        let dt = self.measurement.delta_time_s;
        let mut ji = [[0.0; 9]; 9];
        let mut jj = [[0.0; 9]; 9];
        let mut jb = [[0.0; 6]; 9];
        let mut jg = [[0.0; 3]; 9];
        for i in 0..3 {
            for j in 0..3 {
                ji[i][j] = -rr_i[i][j];
                jj[i][j] = jr_inv[i][j];
                ji[i + 3][j] = rv_i[i][j];
                ji[i + 6][j] = rp_i[i][j];
                ji[i + 3][j + 3] = -ri_t[i][j];
                jj[i + 3][j + 3] = ri_t[i][j];
                ji[i + 6][j + 3] = -ri_t[i][j] * dt;
                ji[i + 6][j + 6] = -ri_t[i][j];
                jj[i + 6][j + 6] = ri_t[i][j];
                jg[i + 3][j] = -ri_t[i][j] * dt;
                jg[i + 6][j] = -0.5 * ri_t[i][j] * dt * dt;
            }
            for j in 0..6 {
                jb[i][j] = -rr_bias[i][j];
                jb[i + 3][j] = -self.measurement.bias_jacobian[i + 3][j];
                jb[i + 6][j] = -self.measurement.bias_jacobian[i + 6][j];
            }
        }
        Ok(ImuLinearization {
            evaluation,
            whitened_jacobian_i: self.whiten(&ji)?,
            whitened_jacobian_j: self.whiten(&jj)?,
            whitened_jacobian_bias: self.whiten(&jb)?,
            whitened_jacobian_gravity: self.whiten(&jg)?,
            jacobian_i: ji,
            jacobian_j: jj,
            jacobian_bias: jb,
            jacobian_gravity: jg,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bag::RawImuMessage;
    use crate::imu::{PreintegrationParams, preintegrate};

    fn factor() -> ImuFactor {
        let samples: Vec<_> = (0..=20)
            .map(|i| RawImuMessage {
                stamp_ns: i * 10_000_000,
                stamp: i as f64 * 0.01,
                frame_id: "imu".into(),
                orientation: [0.0; 4],
                orientation_covariance: [-1.0; 9],
                angular_velocity: [0.1 + (i as f64).sin(), -0.3, 0.4],
                angular_velocity_covariance: [0.0; 9],
                acceleration: [1.0, -0.2 + (i as f64).cos(), 9.81],
                acceleration_covariance: [0.0; 9],
            })
            .collect();
        let params = PreintegrationParams {
            gyro_noise_density: 0.02,
            accel_noise_density: 0.15,
            gyro_bias: [0.01, 0.02, -0.03],
            accel_bias: [0.1, -0.05, 0.02],
            max_gap_s: 0.05,
        };
        ImuFactor::new(preintegrate(&samples, 0, 200_000_000, &params).unwrap()).unwrap()
    }
    fn initial() -> ImuState {
        ImuState {
            rotation: exp_so3(&[0.3, -0.2, 0.5]),
            velocity: [0.2, -0.3, 0.1],
            position: [10.0, -2.0, 3.0],
        }
    }
    fn retract(state: &ImuState, k: usize, h: f64) -> ImuState {
        let mut out = *state;
        if k < 3 {
            let mut w = [0.0; 3];
            w[k] = h;
            out.rotation = multiply(&state.rotation, &exp_so3(&w));
        } else if k < 6 {
            out.velocity[k - 3] += h;
        } else {
            out.position[k - 6] += h;
        }
        out
    }
    #[test]
    fn prediction_has_zero_residual_for_nonidentity_pose_and_changed_bias() {
        let f = factor();
        let a = initial();
        let g = [0.1, -0.2, -9.81];
        let b = [0.012, 0.018, -0.028, 0.12, -0.03, 0.01];
        let prediction = f.predict(&a, g, b).unwrap();
        let evaluated = f.evaluate(&a, &prediction, g, b).unwrap();
        assert!(evaluated.residual.iter().all(|x| x.abs() < 1e-14));
        assert!(evaluated.cost < 1e-22);
    }
    #[test]
    fn analytic_factor_jacobians_match_central_differences_away_from_zero_error() {
        let f = factor();
        let a = initial();
        let g = [0.1, -0.2, -9.81];
        let b = [0.012, 0.018, -0.028, 0.12, -0.03, 0.01];
        let mut z = f.predict(&a, g, b).unwrap();
        z.rotation = multiply(&z.rotation, &exp_so3(&[0.3, -0.4, 0.2]));
        z.velocity[1] += 0.2;
        z.position[0] -= 0.1;
        let lin = f.linearize(&a, &z, g, b).unwrap();
        let h = 1e-6;
        for group in 0..4 {
            let width = [9, 9, 6, 3][group];
            for j in 0..width {
                let mut plus_b = b;
                let mut minus_b = b;
                let mut plus_g = g;
                let mut minus_g = g;
                let (mut plus_a, mut minus_a, mut plus_z, mut minus_z) = (a, a, z, z);
                match group {
                    0 => {
                        plus_a = retract(&a, j, h);
                        minus_a = retract(&a, j, -h);
                    }
                    1 => {
                        plus_z = retract(&z, j, h);
                        minus_z = retract(&z, j, -h);
                    }
                    2 => {
                        plus_b[j] += h;
                        minus_b[j] -= h;
                    }
                    _ => {
                        plus_g[j] += h;
                        minus_g[j] -= h;
                    }
                }
                let plus = f.evaluate(&plus_a, &plus_z, plus_g, plus_b).unwrap();
                let minus = f.evaluate(&minus_a, &minus_z, minus_g, minus_b).unwrap();
                for i in 0..9 {
                    let expected = match group {
                        0 => lin.jacobian_i[i][j],
                        1 => lin.jacobian_j[i][j],
                        2 => lin.jacobian_bias[i][j],
                        _ => lin.jacobian_gravity[i][j],
                    };
                    let actual = (plus.residual[i] - minus.residual[i]) / (2.0 * h);
                    assert!(
                        (actual - expected).abs() < 2e-7,
                        "group {group}, ({i},{j}): {actual} != {expected}"
                    );
                }
            }
        }
    }
    #[test]
    fn rejects_bad_states_nonfinite_inputs_and_singular_covariance() {
        let f = factor();
        let a = initial();
        let g = [0.0, 0.0, -9.81];
        let b = f.measurement.linearization_bias;
        for change in 0..3 {
            let mut bad = a;
            match change {
                0 => bad.rotation[0][0] = f64::NAN,
                1 => bad.rotation[0].iter_mut().for_each(|v| *v = -*v),
                _ => bad.position[0] = f64::INFINITY,
            }
            assert!(f.predict(&bad, g, b).is_err());
            assert!(f.evaluate(&a, &bad, g, b).is_err());
        }
        assert!(f.predict(&a, [f64::NAN; 3], b).is_err());
        assert!(f.predict(&a, g, [f64::INFINITY; 6]).is_err());
        let mut m = f.measurement.clone();
        m.covariance = [[0.0; 9]; 9];
        assert!(ImuFactor::new(m).is_err());
        let mut m = f.measurement.clone();
        m.covariance[0][1] += 0.1;
        assert!(ImuFactor::new(m).is_err());
    }
    #[test]
    fn logarithm_branch_cut_can_be_evaluated_but_not_linearized() {
        let f = factor();
        let a = initial();
        let g = [0.0, 0.0, -9.81];
        let b = f.measurement.linearization_bias;
        let mut z = f.predict(&a, g, b).unwrap();
        z.rotation = multiply(&z.rotation, &exp_so3(&[std::f64::consts::PI, 0.0, 0.0]));
        assert!(f.evaluate(&a, &z, g, b).unwrap().cost.is_finite());
        assert!(f.linearize(&a, &z, g, b).is_err());
    }
}
