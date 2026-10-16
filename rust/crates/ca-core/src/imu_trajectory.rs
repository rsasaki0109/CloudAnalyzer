//! Fixed-anchor nominal IMU motion and independent per-point compensation.
//!
//! No covariance integration, calibration, initialization or LiDAR estimation.
//! Held measurements and clipped intervals match `imu::preintegrate`.

use crate::icp::Rigid;
use crate::imu::{ImuError, ImuSeries, multiply, transpose};
use crate::imu_factor::{ImuState, validate_state};
use crate::pose_graph::exp_so3;

type Mat3 = [[f64; 3]; 3];

/// Gravity is in world axes; gyro/acceleration biases are in IMU axes and SI.
#[derive(Debug, Clone, Copy)]
pub struct TrajectoryParams {
    pub gravity: [f64; 3],
    pub gyro_bias: [f64; 3],
    pub accel_bias: [f64; 3],
    pub max_gap_s: f64,
}

struct Knot {
    stamp_ns: i64,
    // Position is a world-axis displacement from the initial position, avoiding
    // large world-origin additions in every integration/deskew operation.
    state: ImuState,
    angular_velocity: [f64; 3],
    acceleration_world: [f64; 3],
}

/// Immutable nominal motion over an explicitly bracketed interval. Owns only
/// covered knots; queries do not access the source series or recording.
pub struct ImuTrajectory {
    knots: Vec<Knot>,
    position_origin: [f64; 3],
    frame_id: String,
    acceleration_scale: f64,
}

impl ImuSeries {
    pub fn trajectory(
        &self,
        start_ns: i64,
        end_ns: i64,
        initial: ImuState,
        params: &TrajectoryParams,
    ) -> Result<ImuTrajectory, ImuError> {
        validate_state(&initial)?;
        duration(start_ns, end_ns)?;
        if !params.max_gap_s.is_finite()
            || params.max_gap_s <= 0.0
            || !params
                .gravity
                .iter()
                .chain(&params.gyro_bias)
                .chain(&params.accel_bias)
                .all(|v| v.is_finite())
        {
            return Err(ImuError(
                "IMU trajectory gravity/bias must be finite and max_gap_s positive".into(),
            ));
        }
        let samples = self.samples();
        if start_ns < samples[0].stamp_ns || end_ns > samples.last().unwrap().stamp_ns {
            return Err(ImuError(
                "IMU samples do not bracket the trajectory interval".into(),
            ));
        }
        let left = samples.partition_point(|s| s.stamp_ns <= start_ns) - 1;
        let right = samples.partition_point(|s| s.stamp_ns < end_ns);
        let frame_id = samples[left].frame_id.clone();
        let mut state = ImuState {
            position: [0.0; 3],
            ..initial
        };
        let mut knots = Vec::with_capacity(right - left + 1);
        for pair in samples[left..=right].windows(2) {
            let (sample, next) = (&pair[0], &pair[1]);
            let begin = start_ns.max(sample.stamp_ns);
            let end = end_ns.min(next.stamp_ns);
            if duration(sample.stamp_ns, next.stamp_ns)? > params.max_gap_s {
                return Err(ImuError(format!(
                    "IMU sample gap exceeds max_gap_s at {}",
                    sample.stamp_ns
                )));
            }
            if sample.frame_id != frame_id || next.frame_id != frame_id {
                return Err(ImuError(
                    "IMU frame_id changes within the trajectory interval".into(),
                ));
            }
            if !sample.angular_velocity_status().is_usable()
                || !sample.acceleration_status().is_usable()
            {
                return Err(ImuError(format!(
                    "IMU angular velocity or acceleration is missing/invalid at {}",
                    sample.stamp_ns
                )));
            }
            let angular_velocity =
                std::array::from_fn(|i| sample.angular_velocity[i] - params.gyro_bias[i]);
            let acceleration = std::array::from_fn(|i| {
                let measured = if self.acceleration_scale() == 1.0 {
                    sample.acceleration[i]
                } else {
                    sample.acceleration[i] * self.acceleration_scale()
                };
                measured - params.accel_bias[i]
            });
            let rotated = rotate(&state.rotation, acceleration);
            let acceleration_world = std::array::from_fn(|i| rotated[i] + params.gravity[i]);
            if !angular_velocity
                .iter()
                .chain(&acceleration_world)
                .all(|v| v.is_finite())
            {
                return Err(ImuError(
                    "IMU trajectory corrected measurements must be finite".into(),
                ));
            }
            let knot = Knot {
                stamp_ns: begin,
                state,
                angular_velocity,
                acceleration_world,
            };
            state = advance(&knot, duration(begin, end)?);
            validate_state(&state)?;
            knots.push(knot);
        }
        knots.push(Knot {
            stamp_ns: end_ns,
            state,
            angular_velocity: [0.0; 3],
            acceleration_world: [0.0; 3],
        });
        Ok(ImuTrajectory {
            knots,
            position_origin: initial.position,
            frame_id,
            acceleration_scale: self.acceleration_scale(),
        })
    }
}

fn duration(start: i64, end: i64) -> Result<f64, ImuError> {
    end.checked_sub(start)
        .filter(|d| *d > 0)
        .map(|d| d as f64 * 1e-9)
        .ok_or_else(|| {
            ImuError("IMU trajectory times must increase without integer overflow".into())
        })
}

fn rotate(rotation: &Mat3, point: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| (0..3).map(|j| rotation[i][j] * point[j]).sum())
}

fn advance(knot: &Knot, dt: f64) -> ImuState {
    let state = &knot.state;
    ImuState {
        rotation: multiply(
            &state.rotation,
            &exp_so3(&knot.angular_velocity.map(|w| w * dt)),
        ),
        velocity: std::array::from_fn(|i| state.velocity[i] + knot.acceleration_world[i] * dt),
        position: std::array::from_fn(|i| {
            state.position[i] + state.velocity[i] * dt + 0.5 * knot.acceleration_world[i] * dt * dt
        }),
    }
}

struct Reference {
    stamp_ns: i64,
    state: ImuState,
    world_to_imu: Mat3,
    imu_from_lidar: Rigid,
    imu_to_lidar: Mat3,
}

impl ImuTrajectory {
    pub fn start_ns(&self) -> i64 {
        self.knots[0].stamp_ns
    }
    pub fn end_ns(&self) -> i64 {
        self.knots.last().unwrap().stamp_ns
    }
    /// Source IMU frame, independent of the caller-declared LiDAR coordinates.
    pub fn frame_id(&self) -> &str {
        &self.frame_id
    }
    pub fn acceleration_scale(&self) -> f64 {
        self.acceleration_scale
    }
    pub fn knot_count(&self) -> usize {
        self.knots.len()
    }
    pub fn timestamps_ns(&self) -> impl Iterator<Item = i64> + '_ {
        self.knots.iter().map(|k| k.stamp_ns)
    }

    fn contains(&self, stamp_ns: i64) -> bool {
        stamp_ns >= self.start_ns() && stamp_ns <= self.end_ns()
    }

    // Only called after coverage validation. Differences fit i64 because the
    // constructor checks the entire interval before building any knots.
    fn relative_state(&self, stamp_ns: i64) -> ImuState {
        let index = self.knots.partition_point(|k| k.stamp_ns <= stamp_ns) - 1;
        let knot = &self.knots[index];
        if stamp_ns == knot.stamp_ns {
            knot.state
        } else {
            advance(knot, (stamp_ns - knot.stamp_ns) as f64 * 1e-9)
        }
    }

    pub fn state(&self, stamp_ns: i64) -> Result<ImuState, ImuError> {
        if !self.contains(stamp_ns) {
            return Err(ImuError(
                "IMU trajectory does not cover the requested timestamp".into(),
            ));
        }
        let mut state = self.relative_state(stamp_ns);
        state.position = std::array::from_fn(|i| state.position[i] + self.position_origin[i]);
        validate_state(&state)?;
        Ok(state)
    }

    /// Map LiDAR coordinates at `to_ns` into LiDAR coordinates at `from_ns`.
    /// Subtracts trajectory displacements before adding any world origin.
    pub fn relative_lidar_pose(
        &self,
        from_ns: i64,
        to_ns: i64,
        imu_from_lidar: &Rigid,
    ) -> Result<Rigid, ImuError> {
        let reference = self.prepare(&[], &[], from_ns, imu_from_lidar)?;
        if !self.contains(to_ns) {
            return Err(ImuError(
                "IMU trajectory does not cover the destination timestamp".into(),
            ));
        }
        if from_ns == to_ns {
            return Ok(Rigid::IDENTITY);
        }
        let state = self.relative_state(to_ns);
        let pose = Rigid {
            rotation: multiply(
                &reference.imu_to_lidar,
                &multiply(
                    &reference.world_to_imu,
                    &multiply(&state.rotation, &imu_from_lidar.rotation),
                ),
            ),
            translation: self.transform(&[0.0; 3], to_ns, &reference),
        };
        validate_state(&ImuState {
            rotation: pose.rotation,
            position: pose.translation,
            velocity: [0.0; 3],
        })
        .map_err(|e| ImuError(format!("relative LiDAR pose: {}", e.0)))?;
        Ok(pose)
    }

    fn prepare(
        &self,
        points: &[[f64; 3]],
        times: &[i64],
        reference_ns: i64,
        imu_from_lidar: &Rigid,
    ) -> Result<Reference, ImuError> {
        if points.len() != times.len() {
            return Err(ImuError(
                "deskew needs one integer timestamp per point".into(),
            ));
        }
        validate_state(&ImuState {
            rotation: imu_from_lidar.rotation,
            position: imu_from_lidar.translation,
            velocity: [0.0; 3],
        })
        .map_err(|e| ImuError(format!("IMU-from-LiDAR transform: {}", e.0)))?;
        if !self.contains(reference_ns) {
            return Err(ImuError(
                "IMU trajectory does not cover the reference timestamp".into(),
            ));
        }
        for (index, (point, &time)) in points.iter().zip(times).enumerate() {
            if !point.iter().all(|v| v.is_finite()) {
                return Err(ImuError(format!(
                    "deskew point at index {index} is not finite"
                )));
            }
            if !self.contains(time) {
                return Err(ImuError(format!(
                    "IMU trajectory does not cover point timestamp at index {index}"
                )));
            }
        }
        let state = self.relative_state(reference_ns);
        validate_state(&state)?;
        Ok(Reference {
            stamp_ns: reference_ns,
            state,
            world_to_imu: transpose(&state.rotation),
            imu_from_lidar: *imu_from_lidar,
            imu_to_lidar: transpose(&imu_from_lidar.rotation),
        })
    }

    fn transform(&self, point: &[f64; 3], stamp_ns: i64, reference: &Reference) -> [f64; 3] {
        if stamp_ns == reference.stamp_ns {
            return *point;
        }
        let state = self.relative_state(stamp_ns);
        let rotated = rotate(&state.rotation, reference.imu_from_lidar.apply(point));
        let relative_world =
            std::array::from_fn(|i| rotated[i] + (state.position[i] - reference.state.position[i]));
        let reference_imu = rotate(&reference.world_to_imu, relative_world);
        rotate(
            &reference.imu_to_lidar,
            std::array::from_fn(|i| reference_imu[i] - reference.imu_from_lidar.translation[i]),
        )
    }

    /// LiDAR points at individual timestamps into LiDAR coordinates at reference_ns.
    /// The extrinsic maps input LiDAR coordinates into the IMU frame. No input
    /// sorting, extrapolation, attribute filtering or covariance integration.
    pub fn deskew(
        &self,
        points: &[[f64; 3]],
        times: &[i64],
        reference_ns: i64,
        imu_from_lidar: &Rigid,
    ) -> Result<Vec<[f64; 3]>, ImuError> {
        let reference = self.prepare(points, times, reference_ns, imu_from_lidar)?;
        let output = points
            .iter()
            .zip(times)
            .map(|(p, &t)| self.transform(p, t, &reference))
            .collect();
        checked_output(output)
    }

    /// Indexed point map: each point uses the same scalar arithmetic as deskew.
    #[cfg(feature = "parallel")]
    pub fn deskew_parallel(
        &self,
        points: &[[f64; 3]],
        times: &[i64],
        reference_ns: i64,
        imu_from_lidar: &Rigid,
    ) -> Result<Vec<[f64; 3]>, ImuError> {
        use rayon::prelude::*;
        let reference = self.prepare(points, times, reference_ns, imu_from_lidar)?;
        let output = points
            .par_iter()
            .zip(times.par_iter())
            .map(|(p, &t)| self.transform(p, t, &reference))
            .collect();
        checked_output(output)
    }
}

fn checked_output(output: Vec<[f64; 3]>) -> Result<Vec<[f64; 3]>, ImuError> {
    if let Some(index) = output.iter().position(|p| !p.iter().all(|v| v.is_finite())) {
        return Err(ImuError(format!(
            "deskew result at index {index} is not finite"
        )));
    }
    Ok(output)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bag::RawImuMessage;
    use crate::imu::PreintegrationParams;

    const EPOCH: i64 = 1_700_000_000_000_000_123;
    fn initial() -> ImuState {
        ImuState {
            rotation: Rigid::IDENTITY.rotation,
            position: [0.0; 3],
            velocity: [0.0; 3],
        }
    }
    fn params() -> TrajectoryParams {
        TrajectoryParams {
            gravity: [0.0; 3],
            gyro_bias: [0.0; 3],
            accel_bias: [0.0; 3],
            max_gap_s: 0.05,
        }
    }
    fn samples() -> Vec<RawImuMessage> {
        (0..21)
            .map(|i| RawImuMessage {
                stamp_ns: EPOCH + i * 10_000_000,
                stamp: 0.0,
                frame_id: "imu".into(),
                orientation: [0.0; 4],
                orientation_covariance: [-1.0; 9],
                angular_velocity: [0.2, -0.1, 0.3],
                angular_velocity_covariance: [0.0; 9],
                acceleration: [0.5, -0.2, 1.0],
                acceleration_covariance: [0.0; 9],
            })
            .collect()
    }
    fn bits(points: &[[f64; 3]]) -> Vec<u64> {
        points.iter().flatten().map(|v| v.to_bits()).collect()
    }
    fn close(a: &[f64], b: &[f64], tolerance: f64) {
        assert!(
            a.iter().zip(b).all(|(a, b)| (a - b).abs() <= tolerance),
            "{a:?} != {b:?}"
        );
    }

    #[test]
    fn clipped_fixed_anchor_states_match_existing_scaled_preintegration() {
        let series = ImuSeries::new_with_acceleration_scale(samples(), 9.80665).unwrap();
        let initial = ImuState {
            rotation: exp_so3(&[0.3, -0.2, 0.1]),
            velocity: [1.0, -2.0, 0.5],
            position: [10.0, -20.0, 3.0],
        };
        let p = TrajectoryParams {
            gravity: [-0.1, 0.3, -9.7],
            gyro_bias: [0.01, -0.02, 0.03],
            accel_bias: [0.1, -0.2, 0.05],
            ..params()
        };
        let start = EPOCH + 13;
        let end = EPOCH + 200_000_000 - 21;
        let trajectory = series.trajectory(start, end, initial, &p).unwrap();
        assert_eq!(trajectory.start_ns(), start);
        assert_eq!(trajectory.end_ns(), end);
        assert_eq!(trajectory.knot_count(), 21);
        assert_eq!(trajectory.frame_id(), "imu");
        assert_eq!(trajectory.acceleration_scale(), 9.80665);
        close(
            &trajectory.state(start).unwrap().position,
            &initial.position,
            0.0,
        );
        for stamp in [start + 1, EPOCH + 10_000_000, EPOCH + 93_456_789, end] {
            let m = series
                .preintegrate(
                    start,
                    stamp,
                    &PreintegrationParams {
                        gyro_noise_density: 0.02,
                        accel_noise_density: 0.15,
                        gyro_bias: p.gyro_bias,
                        accel_bias: p.accel_bias,
                        max_gap_s: p.max_gap_s,
                    },
                )
                .unwrap();
            let state = trajectory.state(stamp).unwrap();
            let rotation = multiply(&initial.rotation, &m.delta_rotation);
            let dv = rotate(&initial.rotation, m.delta_velocity);
            let dp = rotate(&initial.rotation, m.delta_position);
            let dt = m.delta_time_s;
            close(&state.rotation.concat(), &rotation.concat(), 1e-12);
            close(
                &state.velocity,
                &std::array::from_fn::<_, 3, _>(|i| {
                    initial.velocity[i] + p.gravity[i] * dt + dv[i]
                }),
                1e-11,
            );
            close(
                &state.position,
                &std::array::from_fn::<_, 3, _>(|i| {
                    initial.position[i]
                        + initial.velocity[i] * dt
                        + 0.5 * p.gravity[i] * dt * dt
                        + dp[i]
                }),
                1e-11,
            );
        }
    }

    #[test]
    fn closed_form_rotation_lever_arm_and_unsorted_point_times() {
        let mut raw = samples();
        for s in &mut raw {
            s.angular_velocity = [0.0, 0.0, 1.0];
            s.acceleration = [0.0; 3];
        }
        let series = ImuSeries::new(raw).unwrap();
        let trajectory = series
            .trajectory(EPOCH, EPOCH + 200_000_000, initial(), &params())
            .unwrap();
        let extrinsic = Rigid {
            rotation: exp_so3(&[0.2, -0.1, 0.3]),
            translation: [0.5, -0.3, 0.2],
        };
        let points = [
            [1.0, 2.0, 3.0],
            [-0.0, 1.0, 2.0],
            [4.0, 5.0, 6.0],
            [1.0, 2.0, 3.0],
        ];
        let times = [EPOCH + 200_000_000, EPOCH, EPOCH + 43_456_789, EPOCH];
        let output = trajectory
            .deskew(&points, &times, EPOCH + 200_000_000, &extrinsic)
            .unwrap();
        assert_eq!(bits(&output[..1]), bits(&points[..1]));
        for i in 1..points.len() {
            let dt = (times[i] - (EPOCH + 200_000_000)) as f64 * 1e-9;
            let imu = extrinsic.apply(&points[i]);
            let rotated = rotate(&exp_so3(&[0.0, 0.0, dt]), imu);
            let expected = rotate(
                &transpose(&extrinsic.rotation),
                std::array::from_fn(|j| rotated[j] - extrinsic.translation[j]),
            );
            close(&output[i], &expected, 1e-12);
        }
    }

    #[test]
    fn stationary_gravity_and_constant_velocity_have_expected_compensation() {
        let mut raw = samples();
        for s in &mut raw {
            s.angular_velocity = [0.0; 3];
            s.acceleration = [0.0, 0.0, 1.0];
        }
        let series = ImuSeries::new_with_acceleration_scale(raw, 9.80665).unwrap();
        let p = TrajectoryParams {
            gravity: [0.0, 0.0, -9.80665],
            ..params()
        };
        let initial = ImuState {
            velocity: [1.0, -2.0, 0.5],
            ..initial()
        };
        let t = series
            .trajectory(EPOCH, EPOCH + 200_000_000, initial, &p)
            .unwrap();
        let points = [[1.0, 2.0, 3.0]; 3];
        let times = [EPOCH, EPOCH + 200_000_000, EPOCH + 23_456_789];
        let output = t
            .deskew(&points, &times, times[1], &Rigid::IDENTITY)
            .unwrap();
        for i in 0..3 {
            let dt = (times[i] - times[1]) as f64 * 1e-9;
            close(
                &output[i],
                &std::array::from_fn::<_, 3, _>(|j| points[i][j] + initial.velocity[j] * dt),
                1e-12,
            );
        }
    }

    #[test]
    fn covered_segments_validate_gaps_frames_and_availability_only_where_used() {
        let mut raw = samples();
        raw[0].angular_velocity_covariance[0] = -1.0;
        raw[20].acceleration_covariance[0] = -1.0; // Final time closes the preceding valid held sample.
        let series = ImuSeries::new(raw.clone()).unwrap();
        assert!(
            series
                .trajectory(EPOCH, EPOCH + 20_000_000, initial(), &params())
                .is_err()
        );
        series
            .trajectory(
                EPOCH + 10_000_000,
                EPOCH + 200_000_000,
                initial(),
                &params(),
            )
            .unwrap();
        raw[10].frame_id = "other".into();
        assert!(
            ImuSeries::new(raw)
                .unwrap()
                .trajectory(
                    EPOCH + 10_000_000,
                    EPOCH + 200_000_000,
                    initial(),
                    &params()
                )
                .is_err()
        );
        let series = ImuSeries::new(samples()).unwrap();
        assert!(
            series
                .trajectory(
                    EPOCH + 1,
                    EPOCH + 2,
                    initial(),
                    &TrajectoryParams {
                        max_gap_s: 0.001,
                        ..params()
                    }
                )
                .is_err()
        );
        for p in [
            TrajectoryParams {
                gravity: [f64::NAN; 3],
                ..params()
            },
            TrajectoryParams {
                max_gap_s: 0.0,
                ..params()
            },
            TrajectoryParams {
                gyro_bias: [f64::INFINITY; 3],
                ..params()
            },
        ] {
            assert!(
                series
                    .trajectory(EPOCH, EPOCH + 10_000_000, initial(), &p)
                    .is_err()
            );
        }
        for (a, b) in [
            (EPOCH - 1, EPOCH + 1),
            (EPOCH, EPOCH + 200_000_001),
            (EPOCH, EPOCH),
        ] {
            assert!(series.trajectory(a, b, initial(), &params()).is_err());
        }
        let mut bad = initial();
        bad.rotation[0][0] = 2.0;
        assert!(
            series
                .trajectory(EPOCH, EPOCH + 10_000_000, bad, &params())
                .is_err()
        );
    }

    #[test]
    fn queries_and_deskew_reject_invalid_inputs_and_nonfinite_results() {
        let series = ImuSeries::new(samples()).unwrap();
        let t = series
            .trajectory(EPOCH, EPOCH + 200_000_000, initial(), &params())
            .unwrap();
        for stamp in [EPOCH - 1, EPOCH + 200_000_001] {
            assert!(t.state(stamp).is_err());
        }
        assert!(
            t.deskew(&[], &[], EPOCH, &Rigid::IDENTITY)
                .unwrap()
                .is_empty()
        );
        assert!(t.deskew(&[], &[], EPOCH - 1, &Rigid::IDENTITY).is_err());
        assert!(t.deskew(&[[0.0; 3]], &[], EPOCH, &Rigid::IDENTITY).is_err());
        assert!(
            t.deskew(&[[0.0; 3]], &[EPOCH - 1], EPOCH, &Rigid::IDENTITY)
                .unwrap_err()
                .0
                .contains("index 0")
        );
        assert!(
            t.deskew(&[[f64::NAN; 3]], &[EPOCH], EPOCH, &Rigid::IDENTITY)
                .unwrap_err()
                .0
                .contains("index 0")
        );
        let mut bad = Rigid::IDENTITY;
        bad.rotation[0][0] = -1.0;
        assert!(t.deskew(&[], &[], EPOCH, &bad).is_err());
        assert!(
            t.deskew(
                &[[1.0; 3], [f64::MAX; 3]],
                &[EPOCH; 2],
                t.end_ns(),
                &Rigid::IDENTITY
            )
            .unwrap_err()
            .0
            .contains("index 1")
        );
        let mut raw = samples();
        raw[0].acceleration = [f64::MAX; 3];
        assert!(
            ImuSeries::new_with_acceleration_scale(raw, 9.80665)
                .unwrap()
                .trajectory(EPOCH, EPOCH + 10_000_000, initial(), &params())
                .is_err()
        );
    }

    #[test]
    fn large_world_origin_cancels_before_point_arithmetic() {
        let series = ImuSeries::new(samples()).unwrap();
        let a = series
            .trajectory(EPOCH, EPOCH + 200_000_000, initial(), &params())
            .unwrap();
        let b = series
            .trajectory(
                EPOCH,
                EPOCH + 200_000_000,
                ImuState {
                    position: [1e12, -2e12, 3e12],
                    ..initial()
                },
                &params(),
            )
            .unwrap();
        let points = [[1.0, 2.0, 3.0]; 3];
        let times = [EPOCH, EPOCH + 31_234_567, EPOCH + 200_000_000];
        assert_eq!(
            bits(
                &a.deskew(&points, &times, times[2], &Rigid::IDENTITY)
                    .unwrap()
            ),
            bits(
                &b.deskew(&points, &times, times[2], &Rigid::IDENTITY)
                    .unwrap()
            )
        );
    }

    #[test]
    fn relative_lidar_pose_has_full_extrinsics_and_no_world_origin_loss() {
        let mut raw = samples();
        for sample in &mut raw {
            sample.angular_velocity = [0.0, 0.0, 2.0];
            sample.acceleration = [0.0; 3];
        }
        let series = ImuSeries::new(raw).unwrap();
        let initial = ImuState {
            rotation: exp_so3(&[0.2, -0.1, 0.3]),
            velocity: [1.0, -0.5, 0.3],
            position: [10.0, -20.0, 3.0],
        };
        let p = TrajectoryParams {
            gravity: [0.0; 3],
            ..params()
        };
        let a = series
            .trajectory(EPOCH, EPOCH + 200_000_000, initial, &p)
            .unwrap();
        let b = series
            .trajectory(
                EPOCH,
                EPOCH + 200_000_000,
                ImuState {
                    position: [1e12, -2e12, 3e12],
                    ..initial
                },
                &p,
            )
            .unwrap();
        let extrinsic = Rigid {
            rotation: exp_so3(&[-0.2, 0.3, 0.1]),
            translation: [0.5, -0.3, 0.2],
        };
        let absolute = |stamp| {
            let dt = (stamp - EPOCH) as f64 * 1e-9;
            Rigid {
                rotation: multiply(&initial.rotation, &exp_so3(&[0.0, 0.0, 2.0 * dt])),
                translation: std::array::from_fn(|k| {
                    initial.position[k] + initial.velocity[k] * dt
                }),
            }
            .compose(&extrinsic)
        };
        let x = EPOCH + 31_234_567;
        let y = EPOCH + 170_456_789;
        for (from, to) in [(x, y), (y, x), (x, x)] {
            let expected = crate::pose_graph::inverse(&absolute(from)).compose(&absolute(to));
            let actual = a.relative_lidar_pose(from, to, &extrinsic).unwrap();
            for (actual, expected) in actual.to_matrix().iter().zip(expected.to_matrix()) {
                assert!((actual - expected).abs() < 1e-12);
            }
            assert_eq!(
                actual.to_matrix().map(f64::to_bits),
                b.relative_lidar_pose(from, to, &extrinsic)
                    .unwrap()
                    .to_matrix()
                    .map(f64::to_bits)
            );
        }
        assert_eq!(
            a.relative_lidar_pose(x, x, &extrinsic).unwrap(),
            Rigid::IDENTITY
        );
    }

    #[test]
    fn relative_lidar_pose_rejects_uncovered_times_and_invalid_extrinsics() {
        let t = ImuSeries::new(samples())
            .unwrap()
            .trajectory(EPOCH, EPOCH + 200_000_000, initial(), &params())
            .unwrap();
        assert!(
            t.relative_lidar_pose(EPOCH - 1, EPOCH, &Rigid::IDENTITY)
                .is_err()
        );
        assert!(
            t.relative_lidar_pose(EPOCH, t.end_ns() + 1, &Rigid::IDENTITY)
                .is_err()
        );
        let mut bad = Rigid::IDENTITY;
        bad.rotation[0][0] = -1.0;
        assert!(t.relative_lidar_pose(EPOCH, EPOCH, &bad).is_err());
        bad = Rigid::IDENTITY;
        bad.translation[1] = f64::NAN;
        assert!(t.relative_lidar_pose(EPOCH, EPOCH, &bad).is_err());
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn four_thread_point_map_matches_serial_bits_and_error_order() {
        let series = ImuSeries::new(samples()).unwrap();
        let t = series
            .trajectory(EPOCH, EPOCH + 200_000_000, initial(), &params())
            .unwrap();
        let points: Vec<_> = (0..4096).map(|i| [i as f64 * 0.01, 2.0, -3.0]).collect();
        let mut times: Vec<_> = (0..4096)
            .map(|i| EPOCH + (i * 7919 % 200_000_001) as i64)
            .collect();
        let extrinsic = Rigid {
            rotation: exp_so3(&[0.1, 0.3, -0.2]),
            translation: [0.5, -0.2, 0.1],
        };
        let expected = t.deskew(&points, &times, t.end_ns(), &extrinsic).unwrap();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let actual = pool.install(|| {
            assert_eq!(rayon::current_num_threads(), 4);
            t.deskew_parallel(&points, &times, t.end_ns(), &extrinsic)
                .unwrap()
        });
        assert_eq!(bits(&expected), bits(&actual));
        times[7] = EPOCH - 1;
        times[19] = t.end_ns() + 1;
        let serial = t
            .deskew(&points, &times, t.end_ns(), &extrinsic)
            .unwrap_err()
            .0;
        let parallel = pool.install(|| {
            t.deskew_parallel(&points, &times, t.end_ns(), &extrinsic)
                .unwrap_err()
                .0
        });
        assert_eq!(serial, parallel);
        assert!(serial.contains("index 7"));
    }
}
