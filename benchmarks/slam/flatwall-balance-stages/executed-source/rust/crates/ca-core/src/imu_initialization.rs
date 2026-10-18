//! Metric LiDAR/IMU startup alignment with a declared fixed acceleration bias.
//! Uses no reported IMU orientation, stationary assumption or pose ground truth.

use crate::icp::Rigid;
use crate::imu::{
    ImuError, ImuSeries, PreintegratedImu, PreintegrationParams, multiply, transpose,
};
use crate::imu_factor::{ImuState, right_jacobian_inverse, validate_state};
use crate::pose_graph::log_so3;

type Mat3 = [[f64; 3]; 3];

#[derive(Clone, Copy)]
pub struct InitializationParams {
    pub gyro_bias: [f64; 3],
    pub accel_bias: [f64; 3],
    pub gravity_magnitude: f64,
    pub max_gravity_error: f64,
    pub max_rotation_rms: f64,
    pub max_velocity_rms: f64,
    pub max_gyro_bias_change: f64,
    pub min_singular_ratio: f64,
    pub max_gap_s: f64,
    pub max_iterations: usize,
    pub parallel: bool,
}

#[derive(Debug)]
pub struct ImuInitialization {
    pub timestamps_ns: Vec<i64>,
    pub imu_poses: Vec<Rigid>,
    pub velocities: Vec<[f64; 3]>,
    pub gravity: [f64; 3],
    /// Estimated gyro xyz followed by the caller's fixed SI acceleration bias.
    pub bias: [f64; 6],
    pub gyro_iterations: usize,
    pub gyro_singular_ratio: f64,
    pub gravity_error: f64,
    pub rotation_rms: f64,
    pub velocity_rms: f64,
    pub frame_id: String,
    pub acceleration_scale: f64,
}

fn finite(values: impl Iterator<Item = f64>) -> Result<(), ImuError> {
    if values.into_iter().all(f64::is_finite) {
        Ok(())
    } else {
        Err(ImuError(
            "IMU initialization inputs/results must be finite".into(),
        ))
    }
}

fn rotate(r: &Mat3, v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| (0..3).map(|j| r[i][j] * v[j]).sum())
}

fn norm(v: [f64; 3]) -> f64 {
    v.into_iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn validate_pose(pose: &Rigid) -> Result<(), ImuError> {
    validate_state(&ImuState {
        rotation: pose.rotation,
        position: pose.translation,
        velocity: [0.0; 3],
    })
}

// Scale first. No damping or jitter may turn missing sensitivity into a solution.
fn solve_gyro(h: Mat3, gradient: [f64; 3], minimum: f64) -> Result<([f64; 3], f64), ImuError> {
    let scale = h.iter().flatten().fold(0.0_f64, |m, &v| m.max(v.abs()));
    if !scale.is_finite() || scale <= 0.0 {
        return Err(ImuError(
            "IMU initialization gyro sensitivity is rank deficient".into(),
        ));
    }
    let h = h.map(|row| row.map(|v| v / scale));
    let (eigenvalues, _) = crate::icp::symmetric_eigen(h);
    let low = eigenvalues.into_iter().fold(f64::INFINITY, f64::min);
    let high = eigenvalues.into_iter().fold(0.0_f64, f64::max);
    let ratio = (low / high).sqrt();
    if !ratio.is_finite() || ratio < minimum {
        return Err(ImuError(
            "IMU initialization gyro sensitivity is rank deficient".into(),
        ));
    }
    let mut l = [[0.0; 3]; 3];
    for i in 0..3 {
        for j in 0..=i {
            let v = h[i][j] - (0..j).map(|k| l[i][k] * l[j][k]).sum::<f64>();
            if i == j {
                if v <= 0.0 || !v.is_finite() {
                    return Err(ImuError(
                        "IMU initialization gyro sensitivity is not positive definite".into(),
                    ));
                }
                l[i][j] = v.sqrt();
            } else {
                l[i][j] = v / l[j][j];
            }
        }
    }
    let mut rhs = [0.0; 3];
    for i in 0..3 {
        rhs[i] = (-gradient[i] / scale - (0..i).map(|j| l[i][j] * rhs[j]).sum::<f64>()) / l[i][i];
    }
    let mut step = [0.0; 3];
    for i in (0..3).rev() {
        step[i] = (rhs[i] - (i + 1..3).map(|j| l[j][i] * step[j]).sum::<f64>()) / l[i][i];
    }
    finite(step.into_iter())?;
    Ok((step, ratio))
}

fn rotation_system(
    poses: &[Rigid],
    measurements: &[PreintegratedImu],
) -> Result<(Mat3, [f64; 3], f64), ImuError> {
    let mut h = [[0.0; 3]; 3];
    let mut gradient = [0.0; 3];
    let mut cost = 0.0;
    for (i, m) in measurements.iter().enumerate() {
        let observed = multiply(&transpose(&poses[i].rotation), &poses[i + 1].rotation);
        let residual = log_so3(&multiply(&transpose(&m.delta_rotation), &observed));
        let left_inverse = transpose(&right_jacobian_inverse(residual));
        let bias_jacobian: Mat3 =
            std::array::from_fn(|r| std::array::from_fn(|c| m.bias_jacobian[r][c]));
        let jacobian = multiply(&left_inverse, &bias_jacobian).map(|row| row.map(|v| -v));
        for row in 0..3 {
            cost += residual[row] * residual[row];
            for col in 0..3 {
                gradient[col] += jacobian[row][col] * residual[row];
                for other in 0..3 {
                    h[col][other] += jacobian[row][col] * jacobian[row][other];
                }
            }
        }
    }
    finite(h.iter().flatten().copied().chain(gradient).chain([cost]))?;
    Ok((h, gradient, cost))
}

fn integrate(
    series: &ImuSeries,
    times: &[i64],
    bias: [f64; 3],
    p: &InitializationParams,
) -> Result<Vec<PreintegratedImu>, ImuError> {
    let params = PreintegrationParams {
        gyro_noise_density: 0.0,
        accel_noise_density: 0.0,
        gyro_bias: bias,
        accel_bias: p.accel_bias,
        max_gap_s: p.max_gap_s,
    };
    let edge = |(i, pair): (usize, &[i64])| {
        series
            .preintegrate(pair[0], pair[1], &params)
            .map_err(|e| ImuError(format!("IMU initialization interval {i}: {}", e.0)))
    };
    let results: Vec<_> = if p.parallel {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            times.par_windows(2).enumerate().map(edge).collect()
        }
        #[cfg(not(feature = "parallel"))]
        {
            return Err(ImuError(
                "parallel IMU initialization requires the parallel feature".into(),
            ));
        }
    } else {
        times.windows(2).enumerate().map(edge).collect()
    };
    // Indexed collection and scalar extraction preserve the first failing edge.
    let measurements = results.into_iter().collect::<Result<Vec<_>, _>>()?;
    if measurements
        .iter()
        .any(|m| m.frame_id != measurements[0].frame_id)
    {
        return Err(ImuError(
            "IMU initialization requires one covered IMU frame".into(),
        ));
    }
    Ok(measurements)
}

impl ImuSeries {
    /// Batch startup at the last reference, using metric LiDAR poses in one world.
    /// Acceleration bias is supplied and fixed; no stationary assumption or pose fit.
    pub fn initialize_lidar(
        &self,
        times: &[i64],
        lidar_poses: &[Rigid],
        imu_from_lidar: &Rigid,
        p: &InitializationParams,
    ) -> Result<ImuInitialization, ImuError> {
        if !(3..=4096).contains(&times.len()) || times.len() != lidar_poses.len() {
            return Err(ImuError(
                "IMU initialization needs 3..4096 matched timestamps and LiDAR poses".into(),
            ));
        }
        for pair in times.windows(2) {
            if pair[1].checked_sub(pair[0]).is_none_or(|dt| dt <= 0) {
                return Err(ImuError(
                    "IMU initialization timestamps must strictly increase without overflow".into(),
                ));
            }
        }
        finite(p.gyro_bias.into_iter().chain(p.accel_bias))?;
        if !p.gravity_magnitude.is_finite()
            || p.gravity_magnitude <= 0.0
            || [p.max_gravity_error, p.max_rotation_rms, p.max_velocity_rms]
                .into_iter()
                .any(|v| !v.is_finite() || v < 0.0)
            || !p.max_gyro_bias_change.is_finite()
            || p.max_gyro_bias_change <= 0.0
            || !p.min_singular_ratio.is_finite()
            || p.min_singular_ratio <= 0.0
            || p.min_singular_ratio > 1.0
            || !p.max_gap_s.is_finite()
            || p.max_gap_s <= 0.0
            || !(1..=100).contains(&p.max_iterations)
        {
            return Err(ImuError(
                "invalid IMU initialization limits or iteration settings".into(),
            ));
        }
        validate_pose(imu_from_lidar)?;
        let mut poses = Vec::with_capacity(times.len());
        let mut relative_positions = Vec::with_capacity(times.len());
        let imu_rotation_from_lidar = transpose(&imu_from_lidar.rotation);
        let mut first_offset = [0.0; 3];
        for (i, pose) in lidar_poses.iter().enumerate() {
            validate_pose(pose)?;
            let rotation = multiply(&pose.rotation, &imu_rotation_from_lidar);
            let offset = rotate(&rotation, imu_from_lidar.translation);
            if i == 0 {
                first_offset = offset;
            }
            let relative: [f64; 3] = std::array::from_fn(|k| {
                (pose.translation[k] - lidar_poses[0].translation[k])
                    + (first_offset[k] - offset[k])
            });
            let imu_pose = Rigid {
                rotation,
                translation: std::array::from_fn(|k| pose.translation[k] - offset[k]),
            };
            validate_pose(&imu_pose)?;
            finite(relative.into_iter())?;
            poses.push(imu_pose);
            relative_positions.push(relative);
        }
        let mut bias = p.gyro_bias;
        let mut measurements = integrate(self, times, bias, p)?;
        let mut gyro_iterations = 0;
        let mut converged = false;
        let mut singular_ratio = 0.0;
        for _ in 0..p.max_iterations {
            let (h, gradient, cost) = rotation_system(&poses, &measurements)?;
            let (step, ratio) = solve_gyro(h, gradient, p.min_singular_ratio)?;
            singular_ratio = ratio;
            let predicted_decrease = -0.5 * (0..3).map(|i| step[i] * gradient[i]).sum::<f64>();
            // Only the full undamped step can establish numerical stationarity.
            // A tiny backtracking trial is never used as a convergence test.
            if step.into_iter().fold(0.0_f64, |m, v| m.max(v.abs())) <= 1e-10
                || (predicted_decrease >= 0.0 && predicted_decrease <= 8.0 * f64::EPSILON * cost)
            {
                converged = true;
                break;
            }
            let mut accepted = None;
            for attempt in 0..10 {
                let scale = 0.5_f64.powi(attempt);
                let candidate = std::array::from_fn(|i| bias[i] + scale * step[i]);
                if (0..3).any(|i| {
                    !candidate[i].is_finite()
                        || (candidate[i] - p.gyro_bias[i]).abs() > p.max_gyro_bias_change
                }) {
                    continue;
                }
                let next = integrate(self, times, candidate, p)?;
                if rotation_system(&poses, &next)?.2 < cost {
                    accepted = Some((candidate, next));
                    break;
                }
            }
            if let Some((next_bias, next_measurements)) = accepted {
                bias = next_bias;
                measurements = next_measurements;
                gyro_iterations += 1;
            } else {
                return Err(ImuError(
                    "IMU initialization gyro fit has no descent within the declared bias limit"
                        .into(),
                ));
            }
        }
        if !converged {
            return Err(ImuError(
                "IMU initialization gyro fit did not converge".into(),
            ));
        }
        let mut averages = Vec::with_capacity(measurements.len());
        let mut world_dv = Vec::with_capacity(measurements.len());
        for (i, m) in measurements.iter().enumerate() {
            let dp = rotate(&poses[i].rotation, m.delta_position);
            world_dv.push(rotate(&poses[i].rotation, m.delta_velocity));
            averages.push(std::array::from_fn::<_, 3, _>(|k| {
                (relative_positions[i + 1][k] - relative_positions[i][k] - dp[k]) / m.delta_time_s
            }));
        }
        let mut gravity = [0.0; 3];
        let mut weight = 0.0;
        let mut equations = Vec::with_capacity(times.len() - 2);
        for i in 0..measurements.len() - 1 {
            let alpha = 0.5 * (measurements[i].delta_time_s + measurements[i + 1].delta_time_s);
            let y: [f64; 3] =
                std::array::from_fn(|k| averages[i + 1][k] - averages[i][k] - world_dv[i][k]);
            for k in 0..3 {
                gravity[k] += alpha * y[k];
            }
            weight += alpha * alpha;
            equations.push((alpha, y));
        }
        for g in &mut gravity {
            *g /= weight;
        }
        let mut velocities: Vec<[f64; 3]> = averages
            .iter()
            .zip(&measurements)
            .map(|(v, m)| std::array::from_fn(|k| v[k] - 0.5 * gravity[k] * m.delta_time_s))
            .collect();
        let last = measurements.len() - 1;
        velocities.push(std::array::from_fn(|k| {
            velocities[last][k] + gravity[k] * measurements[last].delta_time_s + world_dv[last][k]
        }));
        let rotation_rms =
            (rotation_system(&poses, &measurements)?.2 / measurements.len() as f64).sqrt();
        let velocity_rms = (equations
            .iter()
            .map(|(alpha, y)| {
                (0..3)
                    .map(|k| (y[k] - alpha * gravity[k]).powi(2))
                    .sum::<f64>()
            })
            .sum::<f64>()
            / equations.len() as f64)
            .sqrt();
        let gravity_error = (norm(gravity) - p.gravity_magnitude).abs();
        finite(
            gravity
                .into_iter()
                .chain(velocities.iter().flatten().copied())
                .chain([rotation_rms, velocity_rms, gravity_error, singular_ratio]),
        )?;
        let mut rejected = Vec::new();
        for (name, value, limit) in [
            ("gravity_error", gravity_error, p.max_gravity_error),
            ("rotation_rms", rotation_rms, p.max_rotation_rms),
            ("velocity_rms", velocity_rms, p.max_velocity_rms),
        ] {
            if value > limit {
                rejected.push(format!("{name} {value:.9e} exceeds {limit:.9e}"));
            }
        }
        if !rejected.is_empty() {
            return Err(ImuError(format!(
                "IMU initialization rejected: {}",
                rejected.join("; ")
            )));
        }
        Ok(ImuInitialization {
            timestamps_ns: times.to_vec(),
            imu_poses: poses,
            velocities,
            gravity,
            bias: [
                bias[0],
                bias[1],
                bias[2],
                p.accel_bias[0],
                p.accel_bias[1],
                p.accel_bias[2],
            ],
            gyro_iterations,
            gyro_singular_ratio: singular_ratio,
            gravity_error,
            rotation_rms,
            velocity_rms,
            frame_id: measurements[0].frame_id.clone(),
            acceleration_scale: self.acceleration_scale(),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bag::RawImuMessage;
    use crate::pose_graph::exp_so3;

    const EPOCH: i64 = 1_700_000_000_000_000_123;
    const TRUE_GYRO: [f64; 3] = [0.01, -0.02, 0.015];

    fn params() -> InitializationParams {
        InitializationParams {
            gyro_bias: [0.0; 3],
            accel_bias: [0.12, -0.06, 0.2],
            gravity_magnitude: 9.80665,
            max_gravity_error: 1e-7,
            max_rotation_rms: 1e-8,
            max_velocity_rms: 1e-7,
            max_gyro_bias_change: 0.2,
            min_singular_ratio: 1e-6,
            max_gap_s: 0.02,
            max_iterations: 15,
            parallel: false,
        }
    }

    fn samples(rotation: Mat3, scale: f64, p: &InitializationParams) -> Vec<RawImuMessage> {
        let force = rotate(&transpose(&rotation), [0.0, 0.0, 9.80665]);
        (0..65)
            .map(|i| RawImuMessage {
                stamp_ns: EPOCH + i * 7_812_500,
                stamp: 0.0,
                frame_id: "imu/body".into(),
                orientation: [0.0; 4],
                orientation_covariance: [-1.0; 9],
                angular_velocity: TRUE_GYRO,
                angular_velocity_covariance: [0.0; 9],
                acceleration: std::array::from_fn(|k| (force[k] + p.accel_bias[k]) / scale),
                acceleration_covariance: [0.0; 9],
            })
            .collect()
    }

    fn poses(rotation: Mat3, origin: [f64; 3], extrinsic: &Rigid) -> (Vec<i64>, Vec<Rigid>) {
        let times: Vec<_> = (0..5).map(|i| EPOCH + i * 125_000_000).collect();
        let poses = (0..5)
            .map(|i| {
                Rigid {
                    rotation,
                    translation: [
                        origin[0] + i as f64 * 0.125,
                        origin[1] + i as f64 * 0.0625,
                        origin[2] - i as f64 * 0.03125,
                    ],
                }
                .compose(extrinsic)
            })
            .collect();
        (times, poses)
    }

    fn close(actual: &[f64], expected: &[f64], tolerance: f64) {
        assert!(
            actual
                .iter()
                .zip(expected)
                .all(|(a, b)| (a - b).abs() <= tolerance),
            "{actual:?} != {expected:?}"
        );
    }

    #[test]
    fn moving_start_with_full_extrinsics_scale_and_fixed_accel_bias_is_recovered() {
        let p = params();
        let rotation = exp_so3(&[0.3, -0.2, 0.1]);
        let extrinsic = Rigid {
            rotation: exp_so3(&[-0.1, 0.2, 0.3]),
            translation: [0.5, -0.3, 0.2],
        };
        for scale in [1.0, 9.80665] {
            let series =
                ImuSeries::new_with_acceleration_scale(samples(rotation, scale, &p), scale)
                    .unwrap();
            let (times, lidar) = poses(rotation, [10.0, -20.0, 3.0], &extrinsic);
            let out = series
                .initialize_lidar(&times, &lidar, &extrinsic, &p)
                .unwrap();
            close(&out.gravity, &[0.0, 0.0, -9.80665], 1e-9);
            close(&out.bias[..3], &TRUE_GYRO, 1e-9);
            assert_eq!(&out.bias[3..], &p.accel_bias);
            assert_eq!(out.timestamps_ns, times);
            assert_eq!(out.frame_id, "imu/body");
            assert_eq!(out.acceleration_scale, scale);
            for (i, velocity) in out.velocities.iter().enumerate() {
                close(velocity, &[1.0, 0.5, -0.25], 1e-9);
                close(
                    &out.imu_poses[i].translation,
                    &[
                        10.0 + i as f64 * 0.125,
                        -20.0 + i as f64 * 0.0625,
                        3.0 - i as f64 * 0.03125,
                    ],
                    1e-12,
                );
            }
        }
    }

    #[test]
    fn origin_is_removed_before_lever_arm_arithmetic() {
        let p = params();
        let series = ImuSeries::new(samples(Rigid::IDENTITY.rotation, 1.0, &p)).unwrap();
        let extrinsic = Rigid {
            rotation: exp_so3(&[0.1, -0.2, 0.3]),
            translation: [0.5, -0.25, 0.125],
        };
        let (times, a) = poses(Rigid::IDENTITY.rotation, [0.0; 3], &extrinsic);
        let (_, b) = poses(Rigid::IDENTITY.rotation, [1e12, -2e12, 3e12], &extrinsic);
        let a = series.initialize_lidar(&times, &a, &extrinsic, &p).unwrap();
        let b = series.initialize_lidar(&times, &b, &extrinsic, &p).unwrap();
        assert_eq!(a.gravity.map(f64::to_bits), b.gravity.map(f64::to_bits));
        assert_eq!(a.bias.map(f64::to_bits), b.bias.map(f64::to_bits));
        for (a, b) in a.velocities.iter().zip(b.velocities) {
            assert_eq!(a.map(f64::to_bits), b.map(f64::to_bits));
        }
    }

    #[test]
    fn invalid_times_shapes_poses_and_limits_are_rejected() {
        let p = params();
        let series = ImuSeries::new(samples(Rigid::IDENTITY.rotation, 1.0, &p)).unwrap();
        let (times, lidar) = poses(Rigid::IDENTITY.rotation, [0.0; 3], &Rigid::IDENTITY);
        assert!(
            series
                .initialize_lidar(&times[..2], &lidar[..2], &Rigid::IDENTITY, &p)
                .is_err()
        );
        assert!(
            series
                .initialize_lidar(&times[..3], &lidar, &Rigid::IDENTITY, &p)
                .is_err()
        );
        let mut bad = times.clone();
        bad[2] = bad[1];
        assert!(
            series
                .initialize_lidar(&bad, &lidar, &Rigid::IDENTITY, &p)
                .is_err()
        );
        let mut bad = lidar.clone();
        bad[0].rotation[0][0] = -1.0;
        assert!(
            series
                .initialize_lidar(&times, &bad, &Rigid::IDENTITY, &p)
                .is_err()
        );
        let mut bad = p;
        bad.accel_bias[1] = f64::NAN;
        assert!(
            series
                .initialize_lidar(&times, &lidar, &Rigid::IDENTITY, &bad)
                .is_err()
        );
        bad = p;
        bad.min_singular_ratio = 0.0;
        assert!(
            series
                .initialize_lidar(&times, &lidar, &Rigid::IDENTITY, &bad)
                .is_err()
        );
    }

    #[test]
    fn gravity_and_dynamic_consistency_limits_are_enforced() {
        let p = params();
        let series = ImuSeries::new(samples(Rigid::IDENTITY.rotation, 1.0, &p)).unwrap();
        let (times, lidar) = poses(Rigid::IDENTITY.rotation, [0.0; 3], &Rigid::IDENTITY);
        let mut bad = p;
        bad.gravity_magnitude = 12.0;
        assert!(
            series
                .initialize_lidar(&times, &lidar, &Rigid::IDENTITY, &bad)
                .unwrap_err()
                .0
                .contains("gravity_error")
        );
        let mut inconsistent = lidar.clone();
        inconsistent[2].translation[0] += 0.1;
        assert!(
            series
                .initialize_lidar(&times, &inconsistent, &Rigid::IDENTITY, &p)
                .unwrap_err()
                .0
                .contains("velocity_rms")
        );
        inconsistent = lidar;
        inconsistent[2].rotation = exp_so3(&[0.05, 0.0, 0.0]);
        assert!(
            series
                .initialize_lidar(&times, &inconsistent, &Rigid::IDENTITY, &p)
                .unwrap_err()
                .0
                .contains("rotation_rms")
        );
    }

    #[test]
    fn unobservable_full_turn_gyro_sensitivity_is_not_regularized() {
        let p = params();
        let mut raw = samples(Rigid::IDENTITY.rotation, 1.0, &p);
        for sample in &mut raw {
            sample.angular_velocity = [0.0, 0.0, 16.0 * std::f64::consts::PI];
        }
        let series = ImuSeries::new(raw).unwrap();
        let (times, lidar) = poses(Rigid::IDENTITY.rotation, [0.0; 3], &Rigid::IDENTITY);
        assert!(
            series
                .initialize_lidar(&times, &lidar, &Rigid::IDENTITY, &p)
                .unwrap_err()
                .0
                .contains("rank deficient")
        );
    }

    #[test]
    fn finite_gyro_iteration_and_bias_limits_cannot_be_ignored() {
        let p = params();
        let series = ImuSeries::new(samples(Rigid::IDENTITY.rotation, 1.0, &p)).unwrap();
        let (times, lidar) = poses(Rigid::IDENTITY.rotation, [0.0; 3], &Rigid::IDENTITY);
        let mut bad = p;
        bad.max_iterations = 1;
        assert!(
            series
                .initialize_lidar(&times, &lidar, &Rigid::IDENTITY, &bad)
                .unwrap_err()
                .0
                .contains("did not converge")
        );
        bad = p;
        bad.max_gyro_bias_change = 1e-12;
        assert!(
            series
                .initialize_lidar(&times, &lidar, &Rigid::IDENTITY, &bad)
                .unwrap_err()
                .0
                .contains("bias limit")
        );
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn four_thread_edges_preserve_scalar_results_and_first_error() {
        let p = params();
        let series = ImuSeries::new(samples(Rigid::IDENTITY.rotation, 1.0, &p)).unwrap();
        let (times, lidar) = poses(Rigid::IDENTITY.rotation, [0.0; 3], &Rigid::IDENTITY);
        let a = series
            .initialize_lidar(&times, &lidar, &Rigid::IDENTITY, &p)
            .unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let b = pool.install(|| {
            assert_eq!(rayon::current_num_threads(), 4);
            series
                .initialize_lidar(
                    &times,
                    &lidar,
                    &Rigid::IDENTITY,
                    &InitializationParams {
                        parallel: true,
                        ..p
                    },
                )
                .unwrap()
        });
        assert_eq!(a.gravity.map(f64::to_bits), b.gravity.map(f64::to_bits));
        assert_eq!(a.bias.map(f64::to_bits), b.bias.map(f64::to_bits));
        for (a, b) in a.velocities.iter().zip(b.velocities) {
            assert_eq!(a.map(f64::to_bits), b.map(f64::to_bits));
        }
        let mut raw = samples(Rigid::IDENTITY.rotation, 1.0, &p);
        raw[16].angular_velocity_covariance[0] = -1.0;
        raw[32].angular_velocity_covariance[0] = -1.0;
        let series = ImuSeries::new(raw).unwrap();
        let serial = series
            .initialize_lidar(&times, &lidar, &Rigid::IDENTITY, &p)
            .unwrap_err()
            .0;
        let parallel = pool
            .install(|| {
                series.initialize_lidar(
                    &times,
                    &lidar,
                    &Rigid::IDENTITY,
                    &InitializationParams {
                        parallel: true,
                        ..p
                    },
                )
            })
            .unwrap_err()
            .0;
        assert_eq!(serial, parallel);
        assert!(serial.contains("interval 1"));
    }
}
