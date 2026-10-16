//! ROS recordings for Python: topics, scans, IMU and trajectory messages,
//! without a ROS install (ROS 1 bags, MCAP, rosbag2 SQLite files and folders).

use ca_core::bag::{self, Encoding};
use ca_core::rosbag2::{Recording, RecordingCursor};
use numpy::ndarray::{Array2, Array3};
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

/// Per IMU message: stamps (N,), up directions (N, 3) and accelerations (N, 3).
type ImuArrays<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
);

/// Per pose: stamps (N,), positions (N, 3) and orientations (N, 4: x, y, z, w).
type PoseArrays<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
);

fn err(e: bag::BagError) -> PyErr {
    PyValueError::new_err(e.0)
}

fn rows<'py>(py: Python<'py>, flat: Vec<f64>, width: usize) -> Bound<'py, PyArray2<f64>> {
    let n = flat.len() / width;
    Array2::from_shape_vec((n, width), flat)
        .expect("rows of the width")
        .into_pyarray(py)
}

/// A ROS 1 bag (`.bag`), an MCAP file, a rosbag2 SQLite file (`.db3`) or a
/// rosbag2 folder, opened for reading.
#[pyclass]
pub struct BagReader {
    path: String,
    topics: Vec<(String, String, u64)>,
    time_range: Option<(f64, f64)>,
}

/// The messages of some topics of a recording, oldest first.
#[pyclass(unsendable)]
pub struct BagMessages {
    recording: Recording,
    cursor: RecordingCursor,
}

/// A PointCloud2 decoded: exact header time/frame, points (N, 3), optional
/// intensity and normalized per-point time. `stamp` retains floating seconds.
#[pyclass]
pub struct Scan {
    #[pyo3(get)]
    pub stamp_ns: i64,
    #[pyo3(get)]
    pub stamp: f64,
    #[pyo3(get)]
    pub frame_id: String,
    positions: Vec<[f64; 3]>,
    intensity: Option<Vec<f32>>,
    time: Option<Vec<f32>>,
}

#[pymethods]
impl Scan {
    fn positions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        rows(py, self.positions.iter().flatten().copied().collect(), 3)
    }

    fn intensity<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f32>>> {
        self.intensity.as_ref().map(|v| v.clone().into_pyarray(py))
    }

    fn time<'py>(&self, py: Python<'py>) -> Option<Bound<'py, PyArray1<f32>>> {
        self.time.as_ref().map(|v| v.clone().into_pyarray(py))
    }

    fn __len__(&self) -> usize {
        self.positions.len()
    }
}

impl BagReader {
    fn integrated_imu(
        &self,
        py: Python<'_>,
        start_ns: i64,
        end_ns: i64,
        params: ca_core::imu::PreintegrationParams,
        topic: Option<String>,
    ) -> PyResult<ca_core::imu::PreintegratedImu> {
        let samples = self.raw_imus(py, topic)?;
        py.detach(|| ca_core::imu::preintegrate(&samples, start_ns, end_ns, &params))
            .map_err(|e| PyValueError::new_err(e.0))
    }

    fn raw_imus(&self, py: Python<'_>, topic: Option<String>) -> PyResult<Vec<bag::RawImuMessage>> {
        let Some(topic) = self.pick(bag::IMU, topic, "pick one")? else {
            return Ok(Vec::new());
        };
        let mut recording = Recording::open(&self.path).map_err(err)?;
        let mut cursor = recording.cursor(&[topic.as_str()]);
        py.detach(|| -> Result<Vec<bag::RawImuMessage>, bag::BagError> {
            let mut out = Vec::new();
            while let Some(m) = cursor.next(&mut recording) {
                let m = m?;
                out.push(bag::decode_imu_raw(&m.data, m.encoding)?);
            }
            Ok(out)
        })
        .map_err(err)
    }

    /// The one topic of `kind`, or `topic` when given (an error names the
    /// topics otherwise); None when the recording has no such topic and none was asked for.
    fn pick(&self, kind: &str, topic: Option<String>, hint: &str) -> PyResult<Option<String>> {
        let matching: Vec<&str> = self
            .topics
            .iter()
            .filter(|(_, k, _)| k == kind)
            .map(|(name, _, _)| name.as_str())
            .collect();
        match topic {
            Some(t) => {
                if !matching.contains(&t.as_str()) {
                    let names: Vec<&str> = self.topics.iter().map(|(n, _, _)| n.as_str()).collect();
                    return Err(PyValueError::new_err(format!(
                        "{kind} topic not found: {t}. Available topics: {}",
                        names.join(", ")
                    )));
                }
                Ok(Some(t))
            }
            None => match matching.as_slice() {
                [] => Ok(None),
                [one] => Ok(Some(one.to_string())),
                many => Err(PyValueError::new_err(format!(
                    "Multiple {kind} topics found in bag; {hint}. Candidates: {}",
                    many.join(", ")
                ))),
            },
        }
    }
}

#[pymethods]
impl BagReader {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let recording = Recording::open(path).map_err(err)?;
        Ok(BagReader {
            path: path.to_string(),
            time_range: recording.time_range(),
            topics: recording
                .topics()
                .iter()
                .map(|t| (t.name.clone(), t.kind.clone(), t.count))
                .collect(),
        })
    }

    /// Every topic: (name, type as ROS 1 names it, message count).
    fn topics(&self) -> Vec<(String, String, u64)> {
        self.topics.clone()
    }

    /// When the first and the last message were recorded (seconds), when the recording's index says.
    fn time_range(&self) -> Option<(f64, f64)> {
        self.time_range
    }

    /// The messages of `topics`, oldest first (see `BagMessages`).
    fn messages(&self, topics: Vec<String>) -> PyResult<BagMessages> {
        let names: Vec<&str> = topics.iter().map(|t| t.as_str()).collect();
        let recording = Recording::open(&self.path).map_err(err)?;
        let cursor = recording.cursor(&names);
        Ok(BagMessages { recording, cursor })
    }

    /// The PointCloud2 messages of `topic` (the one PointCloud2 topic when
    /// None; an error names them when there are several), oldest first,
    /// decoded as `Scan`s.
    #[pyo3(signature = (topic=None))]
    fn scans(&self, topic: Option<String>) -> PyResult<BagMessages> {
        let Some(topic) = self.pick(bag::POINT_CLOUD, topic, "use --pointcloud-topic")? else {
            return Err(PyValueError::new_err(
                "No sensor_msgs/msg/PointCloud2 topic found in bag. Use --pointcloud-topic to select a topic.",
            ));
        };
        self.messages(vec![topic])
    }

    /// The Imu messages of `topic` (the one Imu topic when None): per message its
    /// stamp, its up direction in its frame from its orientation (NaN when it gives
    /// none) and its linear acceleration, as (N,), (N, 3) and (N, 3) arrays.
    #[pyo3(signature = (topic=None))]
    fn imu<'py>(&self, py: Python<'py>, topic: Option<String>) -> PyResult<ImuArrays<'py>> {
        let topic = self.pick(bag::IMU, topic, "pick one")?;
        let mut messages: Vec<bag::ImuMessage> = Vec::new();
        if let Some(topic) = topic {
            let mut recording = Recording::open(&self.path).map_err(err)?;
            let mut cursor = recording.cursor(&[topic.as_str()]);
            messages = py
                .detach(|| -> Result<Vec<bag::ImuMessage>, bag::BagError> {
                    let mut out = Vec::new();
                    while let Some(m) = cursor.next(&mut recording) {
                        let m = m?;
                        out.push(bag::decode_imu(&m.data, m.encoding)?);
                    }
                    Ok(out)
                })
                .map_err(err)?;
        }
        let stamps: Vec<f64> = messages.iter().map(|m| m.stamp).collect();
        let ups: Vec<f64> = messages
            .iter()
            .flat_map(|m| m.up.unwrap_or([f64::NAN; 3]))
            .collect();
        let accel: Vec<f64> = messages.iter().flat_map(|m| m.acceleration).collect();
        Ok((
            stamps.into_pyarray(py),
            rows(py, ups, 3),
            rows(py, accel, 3),
        ))
    }

    /// Unmodified IMU fields, exact header timestamps and measurement statuses.
    /// Covariance arrays have shape (N, 3, 3); zeros mean unknown, not certainty.
    #[pyo3(signature = (topic=None))]
    fn imu_raw<'py>(&self, py: Python<'py>, topic: Option<String>) -> PyResult<Bound<'py, PyDict>> {
        let samples = self.raw_imus(py, topic)?;
        let result = PyDict::new(py);
        result.set_item(
            "timestamps_ns",
            samples
                .iter()
                .map(|s| s.stamp_ns)
                .collect::<Vec<_>>()
                .into_pyarray(py),
        )?;
        result.set_item(
            "timestamps_s",
            samples
                .iter()
                .map(|s| s.stamp)
                .collect::<Vec<_>>()
                .into_pyarray(py),
        )?;
        result.set_item(
            "frame_ids",
            samples
                .iter()
                .map(|s| s.frame_id.as_str())
                .collect::<Vec<_>>(),
        )?;
        result.set_item(
            "orientation_xyzw",
            rows(py, samples.iter().flat_map(|s| s.orientation).collect(), 4),
        )?;
        result.set_item(
            "angular_velocity_rad_s",
            rows(
                py,
                samples.iter().flat_map(|s| s.angular_velocity).collect(),
                3,
            ),
        )?;
        result.set_item(
            "linear_acceleration_m_s2",
            rows(py, samples.iter().flat_map(|s| s.acceleration).collect(), 3),
        )?;
        for (name, values) in [
            (
                "orientation_covariance",
                samples
                    .iter()
                    .flat_map(|s| s.orientation_covariance)
                    .collect::<Vec<_>>(),
            ),
            (
                "angular_velocity_covariance",
                samples
                    .iter()
                    .flat_map(|s| s.angular_velocity_covariance)
                    .collect(),
            ),
            (
                "linear_acceleration_covariance",
                samples
                    .iter()
                    .flat_map(|s| s.acceleration_covariance)
                    .collect(),
            ),
        ] {
            result.set_item(
                name,
                Array3::from_shape_vec((samples.len(), 3, 3), values)
                    .expect("3x3 covariances")
                    .into_pyarray(py),
            )?;
        }
        result.set_item(
            "orientation_status",
            samples
                .iter()
                .map(|s| s.orientation_status().as_str())
                .collect::<Vec<_>>(),
        )?;
        result.set_item(
            "angular_velocity_status",
            samples
                .iter()
                .map(|s| s.angular_velocity_status().as_str())
                .collect::<Vec<_>>(),
        )?;
        result.set_item(
            "linear_acceleration_status",
            samples
                .iter()
                .map(|s| s.acceleration_status().as_str())
                .collect::<Vec<_>>(),
        )?;
        Ok(result)
    }

    /// Relative IMU motion on an integer header-time interval, with held measurements.
    /// Requires explicit noise densities. Bias order: gyro then accelerometer.
    /// Does not remove gravity or apply LiDAR extrinsics. Error order: right-local
    /// rotation, additive velocity, additive position in the initial IMU frame.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (start_ns, end_ns, *, gyro_noise_density, accel_noise_density, max_gap_s=0.05, bias=None, topic=None))]
    fn preintegrate_imu<'py>(
        &self,
        py: Python<'py>,
        start_ns: i64,
        end_ns: i64,
        gyro_noise_density: f64,
        accel_noise_density: f64,
        max_gap_s: f64,
        bias: Option<[f64; 6]>,
        topic: Option<String>,
    ) -> PyResult<Bound<'py, PyDict>> {
        let params = crate::imu::params(gyro_noise_density, accel_noise_density, max_gap_s, bias);
        let out = self.integrated_imu(py, start_ns, end_ns, params, topic)?;
        crate::imu::measurement_dict(py, &out)
    }

    /// Construct a reusable, covariance-weighted IMU factor for this interval.
    #[allow(clippy::too_many_arguments)]
    #[pyo3(signature = (start_ns, end_ns, *, gyro_noise_density, accel_noise_density, max_gap_s=0.05, bias=None, topic=None))]
    fn imu_factor(
        &self,
        py: Python<'_>,
        start_ns: i64,
        end_ns: i64,
        gyro_noise_density: f64,
        accel_noise_density: f64,
        max_gap_s: f64,
        bias: Option<[f64; 6]>,
        topic: Option<String>,
    ) -> PyResult<crate::imu::ImuFactor> {
        let params = crate::imu::params(gyro_noise_density, accel_noise_density, max_gap_s, bias);
        crate::imu::ImuFactor::new(self.integrated_imu(py, start_ns, end_ns, params, topic)?)
    }

    /// The poses of a trajectory topic (`nav_msgs/Odometry`, `geometry_msgs/PoseStamped`,
    /// or `tf2_msgs/TFMessage` with the child `frame`): the one such topic when
    /// `topic` is None. Per pose its stamp, position and orientation (x, y, z, w),
    /// as (N,), (N, 3) and (N, 4) arrays, and the topic and type read.
    #[pyo3(signature = (topic=None, frame=None))]
    fn poses<'py>(
        &self,
        py: Python<'py>,
        topic: Option<String>,
        frame: Option<String>,
    ) -> PyResult<(PoseArrays<'py>, String, String)> {
        let kind_of = |name: &str| {
            self.topics
                .iter()
                .find(|(n, _, _)| n == name)
                .map(|(_, k, _)| k.clone())
        };
        let topic = match topic {
            Some(t) => match kind_of(&t) {
                Some(k) if k == bag::POSE_STAMPED || k == bag::ODOMETRY || k == bag::TF => t,
                Some(k) => {
                    return Err(PyValueError::new_err(format!(
                        "Unsupported trajectory message type on {t}: {}",
                        k.replacen('/', "/msg/", 1)
                    )));
                }
                None => {
                    let names: Vec<&str> = self.topics.iter().map(|(n, _, _)| n.as_str()).collect();
                    return Err(PyValueError::new_err(format!(
                        "Trajectory topic not found: {t}. Available topics: {}",
                        names.join(", ")
                    )));
                }
            },
            None => {
                let direct: Vec<&str> = self
                    .topics
                    .iter()
                    .filter(|(_, k, _)| k == bag::POSE_STAMPED || k == bag::ODOMETRY)
                    .map(|(n, _, _)| n.as_str())
                    .collect();
                match direct.as_slice() {
                    [] => {
                        return Err(PyValueError::new_err(
                            "No supported trajectory topic found in bag. Supported types: geometry_msgs/msg/PoseStamped, nav_msgs/msg/Odometry. Use --topic to select a topic.",
                        ));
                    }
                    [one] => one.to_string(),
                    many => {
                        return Err(PyValueError::new_err(format!(
                            "Multiple supported trajectory topics found in bag; use --topic. Candidates: {}",
                            many.join(", ")
                        )));
                    }
                }
            }
        };
        let kind = kind_of(&topic).unwrap_or_default();
        if kind == bag::TF && frame.is_none() {
            return Err(PyValueError::new_err(format!(
                "Topic {topic} carries tf2_msgs/msg/TFMessage; pass --frame with the child frame id"
            )));
        }
        let mut recording = Recording::open(&self.path).map_err(err)?;
        let mut cursor = recording.cursor(&[topic.as_str()]);
        let frame = frame.unwrap_or_default();
        let poses = py
            .detach(|| -> Result<Vec<bag::PoseMessage>, bag::BagError> {
                let mut out = Vec::new();
                while let Some(m) = cursor.next(&mut recording) {
                    let m = m?;
                    if kind == bag::TF {
                        out.extend(bag::decode_tf(&m.data, m.encoding, &frame)?);
                    } else if kind == bag::ODOMETRY {
                        out.push(bag::decode_odometry(&m.data, m.encoding)?);
                    } else {
                        out.push(bag::decode_pose_stamped(&m.data, m.encoding)?);
                    }
                }
                Ok(out)
            })
            .map_err(err)?;
        let stamps: Vec<f64> = poses.iter().map(|p| p.stamp).collect();
        let positions: Vec<f64> = poses.iter().flat_map(|p| p.position).collect();
        let orientations: Vec<f64> = poses.iter().flat_map(|p| p.orientation).collect();
        Ok((
            (
                stamps.into_pyarray(py),
                rows(py, positions, 3),
                rows(py, orientations, 4),
            ),
            topic,
            kind.replacen('/', "/msg/", 1),
        ))
    }
}

#[pymethods]
impl BagMessages {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// How far through the recording the read is, 0 to 1.
    fn progress(&self) -> f64 {
        self.cursor.progress(&self.recording)
    }

    /// The next message as a `Scan` for a PointCloud2, else as
    /// (topic, stamp, encoding, bytes).
    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let (recording, cursor) = (&mut self.recording, &mut self.cursor);
        let Some(m) = py
            .detach(|| cursor.next(recording))
            .transpose()
            .map_err(err)?
        else {
            return Ok(None);
        };
        let kind = self
            .recording
            .topics()
            .iter()
            .find(|t| t.name == m.topic)
            .map(|t| t.kind.clone())
            .unwrap_or_default();
        if kind == bag::POINT_CLOUD {
            let cloud = bag::decode_point_cloud2(&m.data, m.encoding).map_err(err)?;
            let scan = Scan {
                stamp_ns: cloud.stamp_ns,
                stamp: cloud.stamp,
                frame_id: cloud.frame_id,
                positions: cloud.positions,
                intensity: cloud.intensity,
                time: cloud.time,
            };
            return Ok(Some(Py::new(py, scan)?.into_any()));
        }
        let encoding = match m.encoding {
            Encoding::Ros1 => "ros1",
            Encoding::Cdr => "cdr",
        };
        Ok(Some(
            (
                m.topic,
                m.time,
                encoding,
                pyo3::types::PyBytes::new(py, &m.data),
            )
                .into_pyobject(py)?
                .into_any()
                .unbind(),
        ))
    }
}
