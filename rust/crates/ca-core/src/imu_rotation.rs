//! Gyro-only point compensation with a fixed IMU origin within each scan.
//!
//! This model does not estimate IMU-origin translation, velocity, gravity or
//! bias. It includes the LiDAR lever arm under rotation and needs no usable
//! acceleration or reported orientation. It is separate from full nominal
//! inertial motion so an unavailable physical initial state is not fabricated.

use crate::icp::Rigid;
use crate::imu::{ImuError, ImuSeries, multiply, transpose};
use crate::imu_factor::{ImuState, validate_rotation, validate_state};
use crate::pose_graph::exp_so3;

type Mat3 = [[f64; 3]; 3];

#[derive(Debug, Clone, Copy)]
pub struct RotationParams {
    /// Declared fixed body gyro bias, in rad/s; never fitted by this model.
    pub gyro_bias: [f64; 3],
    pub max_gap_s: f64,
}

#[derive(Debug, Clone)]
struct Knot {
    stamp_ns: i64,
    rotation: Mat3,
    angular_velocity: [f64; 3],
}

/// Owned gyro knots. Rotations map IMU coordinates to the identity start gauge.
/// No recording access, acceleration scaling, covariance or physical pose state.
#[derive(Debug, Clone)]
pub struct ImuRotationTrajectory {
    knots: Vec<Knot>,
    frame_id: String,
    gyro_bias: [f64; 3],
}

fn duration(start: i64, end: i64) -> Result<f64, ImuError> {
    end.checked_sub(start)
        .filter(|&dt| dt > 0)
        .map(|dt| dt as f64 * 1e-9)
        .ok_or_else(|| {
            ImuError("rotation trajectory needs a positive, representable interval".into())
        })
}

fn advance(knot: &Knot, dt: f64) -> Mat3 {
    multiply(
        &knot.rotation,
        &exp_so3(&knot.angular_velocity.map(|w| w * dt)),
    )
}

fn rotate(r: &Mat3, v: [f64; 3]) -> [f64; 3] {
    std::array::from_fn(|i| (0..3).map(|j| r[i][j] * v[j]).sum())
}

impl ImuSeries {
    /// Integrate only usable gyro, held on [t_k, t_(k+1)), with exact clipped
    /// integer timestamp differences. The final sample closes the last interval.
    pub fn rotation_trajectory(
        &self,
        start_ns: i64,
        end_ns: i64,
        params: &RotationParams,
    ) -> Result<ImuRotationTrajectory, ImuError> {
        duration(start_ns, end_ns)?;
        if !params.gyro_bias.iter().all(|v| v.is_finite())
            || !params.max_gap_s.is_finite()
            || params.max_gap_s <= 0.0
        {
            return Err(ImuError(
                "rotation trajectory needs finite gyro bias and a finite positive max_gap_s".into(),
            ));
        }
        let samples = self.samples();
        if start_ns < samples[0].stamp_ns || end_ns > samples[samples.len() - 1].stamp_ns {
            return Err(ImuError(
                "IMU samples must bracket the complete rotation interval".into(),
            ));
        }
        let left = samples.partition_point(|s| s.stamp_ns <= start_ns) - 1;
        let right = samples.partition_point(|s| s.stamp_ns < end_ns);
        let frame_id = samples[left].frame_id.clone();
        let mut rotation = Rigid::IDENTITY.rotation;
        let mut knots = Vec::with_capacity(right - left + 1);
        for pair in samples[left..=right].windows(2) {
            let (sample, next) = (&pair[0], &pair[1]);
            let begin = start_ns.max(sample.stamp_ns);
            let end = end_ns.min(next.stamp_ns);
            if duration(sample.stamp_ns, next.stamp_ns)? > params.max_gap_s {
                return Err(ImuError(
                    "IMU rotation trajectory interval exceeds max_gap_s".into(),
                ));
            }
            if sample.frame_id != frame_id || next.frame_id != frame_id {
                return Err(ImuError(
                    "IMU frame_id changes inside rotation trajectory".into(),
                ));
            }
            if !sample.angular_velocity_status().is_usable() {
                return Err(ImuError(
                    "IMU rotation trajectory requires usable gyro measurements".into(),
                ));
            }
            let angular_velocity =
                std::array::from_fn(|i| sample.angular_velocity[i] - params.gyro_bias[i]);
            if !angular_velocity.iter().all(|v| v.is_finite()) {
                return Err(ImuError("corrected gyro is not finite".into()));
            }
            let knot = Knot {
                stamp_ns: begin,
                rotation,
                angular_velocity,
            };
            rotation = advance(&knot, duration(begin, end)?);
            validate_rotation(&rotation)?;
            knots.push(knot);
        }
        knots.push(Knot {
            stamp_ns: end_ns,
            rotation,
            angular_velocity: [0.0; 3],
        });
        Ok(ImuRotationTrajectory {
            knots,
            frame_id,
            gyro_bias: params.gyro_bias,
        })
    }
}

struct Reference {
    stamp_ns: i64,
    gauge_to_imu: Mat3,
    imu_from_lidar: Rigid,
    imu_to_lidar: Mat3,
}

impl ImuRotationTrajectory {
    pub fn start_ns(&self) -> i64 {
        self.knots[0].stamp_ns
    }
    pub fn end_ns(&self) -> i64 {
        self.knots[self.knots.len() - 1].stamp_ns
    }
    pub fn frame_id(&self) -> &str {
        &self.frame_id
    }
    pub fn gyro_bias(&self) -> [f64; 3] {
        self.gyro_bias
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
    fn at(&self, stamp_ns: i64) -> Mat3 {
        let index = self.knots.partition_point(|k| k.stamp_ns <= stamp_ns) - 1;
        let knot = &self.knots[index];
        if stamp_ns == knot.stamp_ns {
            knot.rotation
        } else {
            advance(knot, (stamp_ns - knot.stamp_ns) as f64 * 1e-9)
        }
    }
    /// IMU-to-start-gauge rotation; not an absolute attitude estimate.
    pub fn rotation(&self, stamp_ns: i64) -> Result<Mat3, ImuError> {
        if !self.contains(stamp_ns) {
            return Err(ImuError(
                "IMU rotation trajectory does not cover the requested timestamp".into(),
            ));
        }
        let rotation = self.at(stamp_ns);
        validate_rotation(&rotation)?;
        Ok(rotation)
    }
    fn prepare(
        &self,
        points: &[[f64; 3]],
        times: &[i64],
        reference_ns: i64,
        extrinsic: &Rigid,
    ) -> Result<Reference, ImuError> {
        if points.len() != times.len() {
            return Err(ImuError(
                "deskew needs one integer timestamp per point".into(),
            ));
        }
        validate_state(&ImuState {
            rotation: extrinsic.rotation,
            position: extrinsic.translation,
            velocity: [0.0; 3],
        })
        .map_err(|e| ImuError(format!("IMU-from-LiDAR transform: {}", e.0)))?;
        if !self.contains(reference_ns) {
            return Err(ImuError(
                "IMU rotation trajectory does not cover the reference timestamp".into(),
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
                    "IMU rotation trajectory does not cover point timestamp at index {index}"
                )));
            }
        }
        Ok(Reference {
            stamp_ns: reference_ns,
            gauge_to_imu: transpose(&self.rotation(reference_ns)?),
            imu_from_lidar: *extrinsic,
            imu_to_lidar: transpose(&extrinsic.rotation),
        })
    }
    fn transform(&self, point: &[f64; 3], time: i64, reference: &Reference) -> [f64; 3] {
        if time == reference.stamp_ns {
            return *point;
        }
        let gauge = rotate(&self.at(time), reference.imu_from_lidar.apply(point));
        let reference_imu = rotate(&reference.gauge_to_imu, gauge);
        rotate(
            &reference.imu_to_lidar,
            std::array::from_fn(|i| reference_imu[i] - reference.imu_from_lidar.translation[i]),
        )
    }
    /// Point coordinates at reference_ns under the fixed IMU-origin assumption.
    /// Includes the full extrinsic lever arm; leaves IMU translation unmodeled.
    pub fn deskew(
        &self,
        points: &[[f64; 3]],
        times: &[i64],
        reference_ns: i64,
        imu_from_lidar: &Rigid,
    ) -> Result<Vec<[f64; 3]>, ImuError> {
        let reference = self.prepare(points, times, reference_ns, imu_from_lidar)?;
        checked_output(
            points
                .iter()
                .zip(times)
                .map(|(p, &t)| self.transform(p, t, &reference))
                .collect(),
        )
    }
    /// Each indexed point uses the same scalar arithmetic as the serial map.
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
        checked_output(
            points
                .par_iter()
                .zip(times.par_iter())
                .map(|(p, &t)| self.transform(p, t, &reference))
                .collect(),
        )
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

    const EPOCH: i64 = 1_700_000_000_000_000_123;
    fn samples() -> Vec<RawImuMessage> {
        (0..21)
            .map(|i| RawImuMessage {
                stamp_ns: EPOCH + i * 10_000_000,
                stamp: f64::NAN,
                frame_id: "imu".into(),
                orientation: [f64::NAN; 4],
                orientation_covariance: [-1.0; 9],
                angular_velocity: [0.2 + i as f64 * 0.01, -0.1, 0.3],
                angular_velocity_covariance: [0.0; 9],
                acceleration: [f64::NAN; 3],
                acceleration_covariance: [-1.0; 9],
            })
            .collect()
    }
    fn params() -> RotationParams {
        RotationParams {
            gyro_bias: [0.01, -0.02, 0.03],
            max_gap_s: 0.05,
        }
    }
    fn close(a: &[f64], b: &[f64]) {
        assert!(
            a.iter().zip(b).all(|(a, b)| (a - b).abs() < 1e-12),
            "{a:?} != {b:?}"
        );
    }
    fn bits(points: &[[f64; 3]]) -> Vec<u64> {
        points.iter().flatten().map(|v| v.to_bits()).collect()
    }

    #[test]
    fn clipped_noncommuting_gyro_queries_ignore_acceleration_attitude_and_scale() {
        let raw = samples();
        let p = params();
        let start = EPOCH + 13;
        let end = EPOCH + 200_000_000 - 21;
        let trajectory = ImuSeries::new_with_acceleration_scale(raw.clone(), 9.80665)
            .unwrap()
            .rotation_trajectory(start, end, &p)
            .unwrap();
        assert_eq!(trajectory.start_ns(), start);
        assert_eq!(trajectory.end_ns(), end);
        assert_eq!(trajectory.frame_id(), "imu");
        assert_eq!(trajectory.gyro_bias(), p.gyro_bias);
        assert_eq!(trajectory.knot_count(), 21);
        assert_eq!(trajectory.timestamps_ns().next(), Some(start));
        assert_eq!(
            trajectory.rotation(start).unwrap(),
            Rigid::IDENTITY.rotation
        );
        for stamp in [end, start + 1, EPOCH + 43_456_789, start] {
            let mut expected = Rigid::IDENTITY.rotation;
            for pair in raw.windows(2) {
                let begin = start.max(pair[0].stamp_ns);
                let finish = stamp.min(pair[1].stamp_ns);
                if finish > begin {
                    let dt = (finish - begin) as f64 * 1e-9;
                    expected = multiply(
                        &expected,
                        &exp_so3(&std::array::from_fn(|i| {
                            (pair[0].angular_velocity[i] - p.gyro_bias[i]) * dt
                        })),
                    );
                }
            }
            close(
                &trajectory.rotation(stamp).unwrap().concat(),
                &expected.concat(),
            );
        }
        let mut changed = raw;
        for s in &mut changed {
            s.acceleration = [f64::MAX; 3];
            s.orientation = [0.0; 4];
        }
        let other = ImuSeries::new(changed)
            .unwrap()
            .rotation_trajectory(start, end, &p)
            .unwrap();
        assert_eq!(
            trajectory.rotation(end).unwrap(),
            other.rotation(end).unwrap()
        );
    }

    #[test]
    fn full_lever_arm_closed_form_and_unsorted_repeated_point_times() {
        let mut raw = samples();
        for s in &mut raw {
            s.angular_velocity = [0.0, 0.0, 1.0];
        }
        let p = RotationParams {
            gyro_bias: [0.0; 3],
            ..params()
        };
        let trajectory = ImuSeries::new(raw)
            .unwrap()
            .rotation_trajectory(EPOCH, EPOCH + 200_000_000, &p)
            .unwrap();
        let extrinsic = Rigid {
            rotation: exp_so3(&[0.2, -0.1, 0.3]),
            translation: [0.5, -0.3, 0.2],
        };
        let points = [[-0.0, 1.0, 2.0], [1.0, 2.0, 3.0], [0.0; 3], [1.0, 2.0, 3.0]];
        let times = [
            trajectory.end_ns(),
            EPOCH + 43_456_789,
            EPOCH,
            EPOCH + 43_456_789,
        ];
        let output = trajectory
            .deskew(&points, &times, times[0], &extrinsic)
            .unwrap();
        assert_eq!(bits(&output[..1]), bits(&points[..1]));
        for i in 1..points.len() {
            let dt = (times[i] - times[0]) as f64 * 1e-9;
            let q = rotate(&exp_so3(&[0.0, 0.0, dt]), extrinsic.apply(&points[i]));
            let expected = rotate(
                &transpose(&extrinsic.rotation),
                std::array::from_fn(|j| q[j] - extrinsic.translation[j]),
            );
            close(&output[i], &expected);
        }
        assert_eq!(output[1], output[3]);
        assert!(output[2].iter().any(|v| v.abs() > 0.01));
    }

    #[test]
    fn only_integrated_left_gyro_is_required_but_closing_frame_and_whole_gap_are_checked() {
        let mut raw = samples();
        raw[0].angular_velocity_covariance[0] = -1.0;
        raw[20].angular_velocity_covariance[0] = -1.0;
        let start = EPOCH + 10_000_000;
        let end = EPOCH + 200_000_000;
        let series = ImuSeries::new(raw.clone()).unwrap();
        assert!(series.rotation_trajectory(EPOCH, end, &params()).is_err());
        series.rotation_trajectory(start, end, &params()).unwrap();
        raw[20].frame_id = "other".into();
        assert!(
            ImuSeries::new(raw)
                .unwrap()
                .rotation_trajectory(start, end, &params())
                .is_err()
        );
        let series = ImuSeries::new(samples()).unwrap();
        assert!(
            series
                .rotation_trajectory(
                    EPOCH + 1,
                    EPOCH + 2,
                    &RotationParams {
                        max_gap_s: 0.001,
                        ..params()
                    }
                )
                .is_err()
        );
    }

    #[test]
    fn constructor_rejects_invalid_intervals_parameters_and_nonfinite_integration() {
        let series = ImuSeries::new(samples()).unwrap();
        for (start, end) in [
            (EPOCH, EPOCH),
            (EPOCH + 1, EPOCH),
            (EPOCH - 1, EPOCH + 1),
            (EPOCH, EPOCH + 200_000_001),
            (i64::MIN, i64::MAX),
        ] {
            assert!(series.rotation_trajectory(start, end, &params()).is_err());
        }
        for p in [
            RotationParams {
                gyro_bias: [f64::INFINITY; 3],
                ..params()
            },
            RotationParams {
                max_gap_s: 0.0,
                ..params()
            },
            RotationParams {
                max_gap_s: f64::NAN,
                ..params()
            },
        ] {
            assert!(
                series
                    .rotation_trajectory(EPOCH, EPOCH + 10_000_000, &p)
                    .is_err()
            );
        }
        for bad in [f64::NAN, f64::MAX] {
            let mut raw = samples();
            raw[0].angular_velocity = [bad; 3];
            assert!(
                ImuSeries::new(raw)
                    .unwrap()
                    .rotation_trajectory(EPOCH, EPOCH + 10_000_000, &params())
                    .is_err()
            );
        }
        let mut raw = samples();
        raw[0].angular_velocity = [f64::MAX; 3];
        assert!(
            ImuSeries::new(raw)
                .unwrap()
                .rotation_trajectory(
                    EPOCH,
                    EPOCH + 1,
                    &RotationParams {
                        gyro_bias: [-f64::MAX; 3],
                        ..params()
                    }
                )
                .unwrap_err()
                .0
                .contains("corrected gyro")
        );
    }

    #[test]
    fn single_nanosecond_interval_preserves_integer_epoch_difference() {
        let trajectory = ImuSeries::new(samples())
            .unwrap()
            .rotation_trajectory(EPOCH + 13, EPOCH + 14, &params())
            .unwrap();
        let expected = exp_so3(&[0.19e-9, -0.08e-9, 0.27e-9]);
        close(
            &trajectory.rotation(EPOCH + 14).unwrap().concat(),
            &expected.concat(),
        );
        assert_eq!(
            trajectory.timestamps_ns().collect::<Vec<_>>(),
            vec![EPOCH + 13, EPOCH + 14]
        );
    }

    #[test]
    fn point_queries_and_calibration_reject_errors_in_input_order() {
        let t = ImuSeries::new(samples())
            .unwrap()
            .rotation_trajectory(EPOCH, EPOCH + 200_000_000, &params())
            .unwrap();
        assert!(t.rotation(EPOCH - 1).is_err());
        assert!(t.rotation(t.end_ns() + 1).is_err());
        assert!(
            t.deskew(&[], &[], EPOCH, &Rigid::IDENTITY)
                .unwrap()
                .is_empty()
        );
        assert!(t.deskew(&[], &[], EPOCH - 1, &Rigid::IDENTITY).is_err());
        assert!(t.deskew(&[[0.0; 3]], &[], EPOCH, &Rigid::IDENTITY).is_err());
        let mut extrinsic = Rigid::IDENTITY;
        extrinsic.rotation[0][0] = -1.0;
        assert!(t.deskew(&[], &[], EPOCH, &extrinsic).is_err());
        extrinsic = Rigid::IDENTITY;
        extrinsic.translation[0] = f64::NAN;
        assert!(t.deskew(&[], &[], EPOCH, &extrinsic).is_err());
        let points = [[0.0; 3], [f64::NAN; 3], [0.0; 3]];
        assert!(
            t.deskew(&points, &[EPOCH; 3], EPOCH, &Rigid::IDENTITY)
                .unwrap_err()
                .0
                .contains("index 1")
        );
        assert!(
            t.deskew(&points, &[EPOCH - 1; 3], EPOCH, &Rigid::IDENTITY)
                .unwrap_err()
                .0
                .contains("index 0")
        );
        assert!(
            t.deskew(&[[f64::MAX; 3]], &[EPOCH], t.end_ns(), &Rigid::IDENTITY)
                .unwrap_err()
                .0
                .contains("result at index 0")
        );
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn indexed_parallel_output_and_first_error_match_serial_bits() {
        let t = ImuSeries::new(samples())
            .unwrap()
            .rotation_trajectory(EPOCH, EPOCH + 200_000_000, &params())
            .unwrap();
        let extrinsic = Rigid {
            rotation: exp_so3(&[0.2, -0.1, 0.3]),
            translation: [0.5, -0.3, 0.2],
        };
        let points: Vec<_> = (0..2049).map(|i| [i as f64 * 0.01, -0.0, 2.0]).collect();
        let mut times: Vec<_> = (0..points.len())
            .map(|i| EPOCH + ((i * 7919) % 200_000_001) as i64)
            .collect();
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(4)
            .build()
            .unwrap();
        let serial = t.deskew(&points, &times, t.end_ns(), &extrinsic).unwrap();
        let parallel = pool.install(|| {
            t.deskew_parallel(&points, &times, t.end_ns(), &extrinsic)
                .unwrap()
        });
        assert_eq!(bits(&serial), bits(&parallel));
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
