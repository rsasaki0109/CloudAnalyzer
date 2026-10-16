//! IMU preintegration in the initial IMU frame, with fixed linearization biases.
//!
//! Measurements are held on [t_k, t_{k+1}); requested endpoints clip those intervals.
//! Gravity, initial velocity, extrinsics, bias random walks and clock calibration
//! belong to the eventual estimator and are not included here. Reported ROS
//! covariances determine availability only; the explicit noise densities below
//! define this discrete measurement noise model, including for unknown covariance.

use crate::bag::RawImuMessage;
use crate::pose_graph::exp_so3;

type Mat3 = [[f64; 3]; 3];
const IDENTITY: Mat3 = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]];

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct ImuError(pub String);

/// Noise densities: rad/s/sqrt(Hz), m/s^2/sqrt(Hz); biases: rad/s, m/s^2.
#[derive(Debug, Clone, Copy)]
pub struct PreintegrationParams {
    pub gyro_noise_density: f64,
    pub accel_noise_density: f64,
    pub gyro_bias: [f64; 3],
    pub accel_bias: [f64; 3],
    pub max_gap_s: f64,
}

#[derive(Debug, Clone)]
pub struct PreintegratedImu {
    /// Exact header-time endpoints of the integrated interval.
    pub start_ns: i64,
    pub end_ns: i64,
    pub delta_rotation: Mat3,
    pub delta_velocity: [f64; 3],
    pub delta_position: [f64; 3],
    pub delta_time_s: f64,
    pub integration_steps: usize,
    pub frame_id: String,
    /// Error order: right-local rotation, additive velocity, additive position.
    /// Velocity and position errors are expressed in the initial IMU frame.
    pub covariance: [[f64; 9]; 9],
    /// Rows follow covariance; columns are gyro bias then accelerometer bias.
    pub bias_jacobian: [[f64; 6]; 9],
    /// Bias used during integration: gyro then accelerometer, in the IMU frame.
    pub linearization_bias: [f64; 6],
}

pub(crate) fn skew([x, y, z]: [f64; 3]) -> Mat3 {
    [[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]]
}

pub(crate) fn multiply<const M: usize, const N: usize, const P: usize>(
    a: &[[f64; N]; M],
    b: &[[f64; P]; N],
) -> [[f64; P]; M] {
    std::array::from_fn(|i| std::array::from_fn(|j| (0..N).map(|k| a[i][k] * b[k][j]).sum()))
}

pub(crate) fn transpose<const M: usize, const N: usize>(a: &[[f64; N]; M]) -> [[f64; M]; N] {
    std::array::from_fn(|i| std::array::from_fn(|j| a[j][i]))
}

pub(crate) fn right_jacobian(theta: [f64; 3]) -> Mat3 {
    let t2 = theta.iter().map(|x| x * x).sum::<f64>();
    let (a, b) = if t2 < 1e-8 {
        (
            0.5 - t2 / 24.0 + t2 * t2 / 720.0,
            1.0 / 6.0 - t2 / 120.0 + t2 * t2 / 5040.0,
        )
    } else {
        let t = t2.sqrt();
        ((1.0 - t.cos()) / t2, (t - t.sin()) / (t2 * t))
    };
    let k = skew(theta);
    let k2 = multiply(&k, &k);
    std::array::from_fn(|i| std::array::from_fn(|j| IDENTITY[i][j] - a * k[i][j] + b * k2[i][j]))
}

fn elapsed(start: i64, end: i64) -> Result<f64, ImuError> {
    end.checked_sub(start)
        .filter(|dt| *dt > 0)
        .map(|dt| dt as f64 * 1e-9)
        .ok_or_else(|| ImuError("IMU times must increase without integer overflow".into()))
}

/// Preintegrates header-time samples without sorting, extrapolation or frame conversion.
/// At least two samples must bracket [start_ns, end_ns]. A missing final endpoint
/// measurement is permitted: its time closes the preceding held measurement.
pub fn preintegrate(
    samples: &[RawImuMessage],
    start_ns: i64,
    end_ns: i64,
    params: &PreintegrationParams,
) -> Result<PreintegratedImu, ImuError> {
    let total = elapsed(start_ns, end_ns)?;
    if !params.gyro_noise_density.is_finite()
        || params.gyro_noise_density < 0.0
        || !params.accel_noise_density.is_finite()
        || params.accel_noise_density < 0.0
        || !params.max_gap_s.is_finite()
        || params.max_gap_s <= 0.0
        || !params
            .gyro_bias
            .iter()
            .chain(&params.accel_bias)
            .all(|v| v.is_finite())
    {
        return Err(ImuError("IMU noise densities/biases must be finite, densities nonnegative and max_gap_s positive".into()));
    }
    if samples.len() < 2
        || samples[0].stamp_ns > start_ns
        || samples.last().unwrap().stamp_ns < end_ns
    {
        return Err(ImuError(
            "IMU samples do not bracket the requested interval".into(),
        ));
    }
    for pair in samples.windows(2) {
        elapsed(pair[0].stamp_ns, pair[1].stamp_ns)?;
    }
    let mut out = PreintegratedImu {
        start_ns,
        end_ns,
        delta_rotation: IDENTITY,
        delta_velocity: [0.0; 3],
        delta_position: [0.0; 3],
        delta_time_s: total,
        integration_steps: 0,
        frame_id: String::new(),
        covariance: [[0.0; 9]; 9],
        bias_jacobian: [[0.0; 6]; 9],
        linearization_bias: std::array::from_fn(|i| {
            if i < 3 {
                params.gyro_bias[i]
            } else {
                params.accel_bias[i - 3]
            }
        }),
    };
    for pair in samples.windows(2) {
        let (sample, next) = (&pair[0], &pair[1]);
        let left = start_ns.max(sample.stamp_ns);
        let right = end_ns.min(next.stamp_ns);
        if right <= left {
            continue;
        }
        if elapsed(sample.stamp_ns, next.stamp_ns)? > params.max_gap_s {
            return Err(ImuError(format!(
                "IMU sample gap exceeds max_gap_s at {}",
                sample.stamp_ns
            )));
        }
        if !sample.angular_velocity_status().is_usable()
            || !sample.acceleration_status().is_usable()
        {
            return Err(ImuError(format!(
                "IMU angular velocity or acceleration is missing/invalid at {}",
                sample.stamp_ns
            )));
        }
        if out.integration_steps == 0 {
            out.frame_id.clone_from(&sample.frame_id);
        }
        if sample.frame_id != out.frame_id || next.frame_id != out.frame_id {
            return Err(ImuError(
                "IMU frame_id changes within the integration interval".into(),
            ));
        }
        let dt = elapsed(left, right)?;
        let acceleration = std::array::from_fn(|i| sample.acceleration[i] - params.accel_bias[i]);
        let theta =
            std::array::from_fn(|i| (sample.angular_velocity[i] - params.gyro_bias[i]) * dt);
        let rotation = out.delta_rotation;
        let increment = exp_so3(&theta);
        let jr = right_jacobian(theta);
        let acceleration_hat = multiply(&rotation, &skew(acceleration));
        let rotated_acceleration: [f64; 3] =
            std::array::from_fn(|i| (0..3).map(|j| rotation[i][j] * acceleration[j]).sum());
        let mut f = [[0.0; 9]; 9];
        let mut g = [[0.0; 6]; 9];
        for i in 0..3 {
            for j in 0..3 {
                f[i][j] = increment[j][i];
                f[i + 3][j] = -acceleration_hat[i][j] * dt;
                f[i + 6][j] = -0.5 * acceleration_hat[i][j] * dt * dt;
                f[i + 3][j + 3] = IDENTITY[i][j];
                f[i + 6][j + 3] = IDENTITY[i][j] * dt;
                f[i + 6][j + 6] = IDENTITY[i][j];
                g[i][j] = jr[i][j] * dt;
                g[i + 3][j + 3] = rotation[i][j] * dt;
                g[i + 6][j + 3] = 0.5 * rotation[i][j] * dt * dt;
            }
        }
        let fj = multiply(&f, &out.bias_jacobian);
        out.bias_jacobian = std::array::from_fn(|i| std::array::from_fn(|j| fj[i][j] - g[i][j]));
        let fpf = multiply(&multiply(&f, &out.covariance), &transpose(&f));
        let variance = [
            params.gyro_noise_density.powi(2) / dt,
            params.accel_noise_density.powi(2) / dt,
        ];
        out.covariance = std::array::from_fn(|i| {
            std::array::from_fn(|j| {
                fpf[i][j]
                    + (0..6)
                        .map(|k| g[i][k] * variance[k / 3] * g[j][k])
                        .sum::<f64>()
            })
        });
        for (i, &a) in rotated_acceleration.iter().enumerate() {
            out.delta_position[i] += out.delta_velocity[i] * dt + 0.5 * a * dt * dt;
            out.delta_velocity[i] += a * dt;
        }
        out.delta_rotation = multiply(&rotation, &increment);
        out.integration_steps += 1;
    }
    if !out
        .delta_rotation
        .iter()
        .flatten()
        .chain(&out.delta_velocity)
        .chain(&out.delta_position)
        .chain(out.covariance.iter().flatten())
        .chain(out.bias_jacobian.iter().flatten())
        .all(|v| v.is_finite())
    {
        return Err(ImuError(
            "IMU preintegration produced nonfinite values".into(),
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pose_graph::log_so3;

    fn sample(stamp_ns: i64) -> RawImuMessage {
        RawImuMessage {
            stamp_ns,
            stamp: stamp_ns as f64 * 1e-9,
            frame_id: "imu".into(),
            orientation: [0.0; 4],
            orientation_covariance: [-1.0; 9],
            angular_velocity: [0.2, -0.3, 0.4],
            angular_velocity_covariance: [0.0; 9],
            acceleration: [1.0, -2.0, 9.81],
            acceleration_covariance: [0.0; 9],
        }
    }
    fn params() -> PreintegrationParams {
        PreintegrationParams {
            gyro_noise_density: 0.01,
            accel_noise_density: 0.1,
            gyro_bias: [0.03, -0.02, 0.01],
            accel_bias: [0.1, 0.2, -0.05],
            max_gap_s: 0.05,
        }
    }
    fn integrate(samples: &[RawImuMessage], p: &PreintegrationParams) -> PreintegratedImu {
        preintegrate(
            samples,
            samples[0].stamp_ns,
            samples.last().unwrap().stamp_ns,
            p,
        )
        .unwrap()
    }
    #[test]
    fn stationary_specific_force_keeps_gravity_for_the_estimator() {
        let mut data: Vec<_> = (0..=100).map(|i| sample(i * 10_000_000)).collect();
        for s in &mut data {
            s.angular_velocity = [0.0; 3];
            s.acceleration = [0.0, 0.0, 9.81];
        }
        let p = PreintegrationParams {
            gyro_bias: [0.0; 3],
            accel_bias: [0.0; 3],
            ..params()
        };
        let out = integrate(&data, &p);
        assert_eq!(out.delta_rotation, IDENTITY);
        assert!((out.delta_velocity[2] - 9.81).abs() < 1e-12);
        assert!((out.delta_position[2] - 4.905).abs() < 1e-12);
        assert_eq!(out.integration_steps, 100);
    }
    #[test]
    fn bias_jacobian_matches_independent_central_differences() {
        let data: Vec<_> = (0..=25)
            .map(|i| {
                let mut s = sample(i * 8_000_000);
                s.angular_velocity[0] += (i as f64 * 0.7).sin();
                s.acceleration[1] += (i as f64 * 0.3).cos();
                s
            })
            .collect();
        let p = params();
        let nominal = integrate(&data, &p);
        let eps = 1e-6;
        for j in 0..6 {
            let mut plus = p;
            let mut minus = p;
            if j < 3 {
                plus.gyro_bias[j] += eps;
                minus.gyro_bias[j] -= eps;
            } else {
                plus.accel_bias[j - 3] += eps;
                minus.accel_bias[j - 3] -= eps;
            }
            let a = integrate(&data, &plus);
            let b = integrate(&data, &minus);
            let ar = log_so3(&multiply(
                &transpose(&nominal.delta_rotation),
                &a.delta_rotation,
            ));
            let br = log_so3(&multiply(
                &transpose(&nominal.delta_rotation),
                &b.delta_rotation,
            ));
            for i in 0..9 {
                let numerical = if i < 3 {
                    (ar[i] - br[i]) / (2.0 * eps)
                } else if i < 6 {
                    (a.delta_velocity[i - 3] - b.delta_velocity[i - 3]) / (2.0 * eps)
                } else {
                    (a.delta_position[i - 6] - b.delta_position[i - 6]) / (2.0 * eps)
                };
                assert!(
                    (numerical - nominal.bias_jacobian[i][j]).abs() < 2e-8,
                    "({i},{j}): {numerical}"
                );
            }
        }
    }
    #[test]
    fn clips_exact_integer_times_and_does_not_use_the_final_measurement() {
        let epoch = 1_700_000_000_000_000_000;
        let mut data = vec![sample(epoch), sample(epoch + 100), sample(epoch + 200)];
        for s in &mut data {
            s.angular_velocity = [0.0; 3];
            s.acceleration = [2.0, 0.0, 0.0];
        }
        data[2].angular_velocity_covariance[0] = -1.0;
        let p = PreintegrationParams {
            gyro_bias: [0.0; 3],
            accel_bias: [0.0; 3],
            ..params()
        };
        let out = preintegrate(&data, epoch + 50, epoch + 200, &p).unwrap();
        assert!((out.delta_time_s - 150e-9).abs() < 1e-22);
        assert!((out.delta_velocity[0] - 300e-9).abs() < 1e-22);
        assert!((out.delta_position[0] - (150e-9_f64).powi(2)).abs() < 1e-28);
    }
    #[test]
    fn rejects_unavailable_measurements_gaps_frames_and_invalid_parameters() {
        let valid = vec![sample(0), sample(10_000_000), sample(20_000_000)];
        for change in 0..6 {
            let mut data = valid.clone();
            match change {
                0 => data[0].angular_velocity_covariance[0] = -1.0,
                1 => data[0].acceleration[0] = f64::NAN,
                2 => data[1].stamp_ns = 0,
                3 => data[1].frame_id = "other".into(),
                4 => data[2].stamp_ns = 100_000_000,
                _ => data[0].acceleration_covariance[0] = -2.0,
            }
            assert!(
                preintegrate(&data, 0, 20_000_000, &params()).is_err(),
                "case {change}"
            );
        }
        for (a, b) in [
            (-1, 20_000_000),
            (0, 20_000_001),
            (10, 10),
            (20, 10),
            (i64::MIN, i64::MAX),
        ] {
            assert!(preintegrate(&valid, a, b, &params()).is_err());
        }
        let bad = PreintegrationParams {
            gyro_noise_density: f64::NAN,
            ..params()
        };
        assert!(preintegrate(&valid, 0, 20_000_000, &bad).is_err());
    }
    #[test]
    fn covariance_matches_single_interval_closed_form() {
        let mut a = sample(0);
        a.angular_velocity = [0.0; 3];
        let p = PreintegrationParams {
            gyro_bias: [0.0; 3],
            ..params()
        };
        let out = integrate(&[a, sample(10_000_000)], &p);
        let dt = 0.01;
        for i in 0..3 {
            assert!((out.covariance[i][i] - p.gyro_noise_density.powi(2) * dt).abs() < 1e-18);
            assert!(
                (out.covariance[i + 3][i + 3] - p.accel_noise_density.powi(2) * dt).abs() < 1e-18
            );
            assert!(
                (out.covariance[i + 6][i + 6] - p.accel_noise_density.powi(2) * dt.powi(3) / 4.0)
                    .abs()
                    < 1e-18
            );
            assert!(
                (out.covariance[i + 3][i + 6] - p.accel_noise_density.powi(2) * dt.powi(2) / 2.0)
                    .abs()
                    < 1e-18
            );
        }
    }
}
