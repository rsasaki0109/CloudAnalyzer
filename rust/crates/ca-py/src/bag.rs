//! ROS bags for Python: topics, scans and IMU messages, without a ROS install.

use ca_core::bag::{self, Bag, Cursor, Encoding, ImuMessage};
use numpy::ndarray::Array2;
use numpy::{IntoPyArray, PyArray1, PyArray2};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// Per IMU message: stamps (N,), up directions (N, 3) and accelerations (N, 3).
type ImuArrays<'py> = (
    Bound<'py, PyArray1<f64>>,
    Bound<'py, PyArray2<f64>>,
    Bound<'py, PyArray2<f64>>,
);

fn err(e: bag::BagError) -> PyErr {
    PyValueError::new_err(e.0)
}

/// A ROS 1 bag (`.bag`) or MCAP file, opened for reading.
#[pyclass]
pub struct BagReader {
    path: String,
    topics: Vec<(String, String, u64)>,
    time_range: Option<(f64, f64)>,
}

/// The messages of some topics of a bag, oldest first.
#[pyclass]
pub struct BagMessages {
    bag: Bag<std::fs::File>,
    cursor: Cursor,
}

/// A PointCloud2 decoded: its stamp (seconds), points (N, 3), intensity (N,)
/// when it has one and the time of each point within the scan, 0 to 1, when it has that.
#[pyclass]
pub struct Scan {
    #[pyo3(get)]
    pub stamp: f64,
    positions: Vec<[f64; 3]>,
    intensity: Option<Vec<f32>>,
    time: Option<Vec<f32>>,
}

#[pymethods]
impl Scan {
    fn positions<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray2<f64>> {
        let flat: Vec<f64> = self.positions.iter().flatten().copied().collect();
        Array2::from_shape_vec((self.positions.len(), 3), flat)
            .expect("three per point")
            .into_pyarray(py)
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

#[pymethods]
impl BagReader {
    #[new]
    fn new(path: &str) -> PyResult<Self> {
        let bag = Bag::open(path).map_err(err)?;
        Ok(BagReader {
            path: path.to_string(),
            time_range: bag.time_range(),
            topics: bag
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

    /// When the first and the last message were recorded (seconds), when the bag's index says.
    fn time_range(&self) -> Option<(f64, f64)> {
        self.time_range
    }

    /// The messages of `topics`, oldest first (see `BagMessages`).
    fn messages(&self, topics: Vec<String>) -> PyResult<BagMessages> {
        let names: Vec<&str> = topics.iter().map(|t| t.as_str()).collect();
        Ok(BagMessages {
            bag: Bag::open(&self.path).map_err(err)?,
            cursor: Cursor::new(&names),
        })
    }

    /// The PointCloud2 messages of `topic` (the one PointCloud2 topic when
    /// None; an error names them when there are several), oldest first,
    /// decoded as `Scan`s.
    #[pyo3(signature = (topic=None))]
    fn scans(&self, topic: Option<String>) -> PyResult<BagMessages> {
        let clouds: Vec<&(String, String, u64)> = self
            .topics
            .iter()
            .filter(|(_, kind, _)| kind == bag::POINT_CLOUD)
            .collect();
        let topic = match topic {
            Some(t) => {
                if !clouds.iter().any(|(name, _, _)| *name == t) {
                    let names: Vec<&str> = self.topics.iter().map(|(n, _, _)| n.as_str()).collect();
                    return Err(PyValueError::new_err(format!(
                        "PointCloud2 topic not found: {t}. Available topics: {}",
                        names.join(", ")
                    )));
                }
                t
            }
            None => match clouds.as_slice() {
                [] => {
                    return Err(PyValueError::new_err(
                        "No sensor_msgs/msg/PointCloud2 topic found in bag. Use --pointcloud-topic to select a topic.",
                    ));
                }
                [one] => one.0.clone(),
                many => {
                    let names: Vec<&str> = many.iter().map(|(n, _, _)| n.as_str()).collect();
                    return Err(PyValueError::new_err(format!(
                        "Multiple PointCloud2 topics found in bag; use --pointcloud-topic. Candidates: {}",
                        names.join(", ")
                    )));
                }
            },
        };
        self.messages(vec![topic])
    }

    /// The Imu messages of `topic` (the one Imu topic when None): per message its
    /// stamp, its up direction in its frame from its orientation (NaN when it gives
    /// none) and its linear acceleration, as (N,), (N, 3) and (N, 3) arrays.
    #[pyo3(signature = (topic=None))]
    fn imu<'py>(&self, py: Python<'py>, topic: Option<String>) -> PyResult<ImuArrays<'py>> {
        let imus: Vec<&str> = self
            .topics
            .iter()
            .filter(|(_, kind, _)| kind == bag::IMU)
            .map(|(name, _, _)| name.as_str())
            .collect();
        let topic = match topic {
            Some(t) => {
                if !imus.contains(&t.as_str()) {
                    return Err(PyValueError::new_err(format!("Imu topic not found: {t}")));
                }
                t
            }
            None => match imus.as_slice() {
                [] => String::new(),
                [one] => one.to_string(),
                many => {
                    return Err(PyValueError::new_err(format!(
                        "Several Imu topics in the bag; pick one: {}",
                        many.join(", ")
                    )));
                }
            },
        };
        let mut messages: Vec<ImuMessage> = Vec::new();
        if !topic.is_empty() {
            let mut bag = Bag::open(&self.path).map_err(err)?;
            messages = py
                .detach(|| -> Result<Vec<ImuMessage>, bag::BagError> {
                    bag.messages(&[topic.as_str()])
                        .map(|m| {
                            let m = m?;
                            bag::decode_imu(&m.data, m.encoding)
                        })
                        .collect()
                })
                .map_err(err)?;
        }
        let n = messages.len();
        let stamps: Vec<f64> = messages.iter().map(|m| m.stamp).collect();
        let ups: Vec<f64> = messages
            .iter()
            .flat_map(|m| m.up.unwrap_or([f64::NAN; 3]))
            .collect();
        let accel: Vec<f64> = messages.iter().flat_map(|m| m.acceleration).collect();
        Ok((
            stamps.into_pyarray(py),
            Array2::from_shape_vec((n, 3), ups)
                .expect("three per message")
                .into_pyarray(py),
            Array2::from_shape_vec((n, 3), accel)
                .expect("three per message")
                .into_pyarray(py),
        ))
    }
}

#[pymethods]
impl BagMessages {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    /// The next message as a `Scan` for a PointCloud2, else as
    /// (topic, stamp, encoding, bytes).
    /// How far through the bag the read is, 0 to 1.
    fn progress(&self) -> f64 {
        self.cursor.progress(&self.bag)
    }

    fn __next__(&mut self, py: Python<'_>) -> PyResult<Option<Py<PyAny>>> {
        let (bag, cursor) = (&mut self.bag, &mut self.cursor);
        let Some(m) = py.detach(|| cursor.next(bag)).transpose().map_err(err)? else {
            return Ok(None);
        };
        let kind = self
            .bag
            .topics()
            .iter()
            .find(|t| t.name == m.topic)
            .map(|t| t.kind.clone())
            .unwrap_or_default();
        if kind == bag::POINT_CLOUD {
            let cloud = bag::decode_point_cloud2(&m.data, m.encoding).map_err(err)?;
            let scan = Scan {
                stamp: cloud.stamp,
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
