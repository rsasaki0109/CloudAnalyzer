//! ROS bags in the browser: the core's reader over a File, read a slice at
//! a time through a JavaScript callback (`FileReaderSync` in a worker).

use ca_core::bag::{self, Bag, Cursor, Encoding, ImuMessage, Message};
use ca_core::{Attribute, AttributeValues, INTENSITY, PointCloud};
use std::io::{Read, Seek, SeekFrom};
use wasm_bindgen::prelude::*;

use crate::Cloud;

/// A file read through `read(at, length) -> Uint8Array`, in windows of
/// some megabytes so that the many small reads of a bag's records cost few
/// calls into JavaScript.
struct JsSource {
    read: js_sys::Function,
    size: u64,
    at: u64,
    window_at: u64,
    window: Vec<u8>,
}

/// How much is read at once.
const WINDOW: usize = 8 << 20;

impl Read for JsSource {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        let n = buf.len().min((self.size - self.at.min(self.size)) as usize);
        if n == 0 {
            return Ok(0);
        }
        let inside = self.at >= self.window_at
            && self.at + n as u64 <= self.window_at + self.window.len() as u64;
        if !inside {
            let want = n.max(WINDOW).min((self.size - self.at) as usize);
            let got = self
                .read
                .call2(
                    &JsValue::NULL,
                    &JsValue::from(self.at as f64),
                    &JsValue::from(want as f64),
                )
                .map_err(|_| std::io::Error::other("reading the file failed"))?;
            self.window = js_sys::Uint8Array::from(got).to_vec();
            self.window_at = self.at;
            if self.window.len() < n {
                return Err(std::io::Error::other("the file ends early"));
            }
        }
        let start = (self.at - self.window_at) as usize;
        buf[..n].copy_from_slice(&self.window[start..start + n]);
        self.at += n as u64;
        Ok(n)
    }
}

impl Seek for JsSource {
    fn seek(&mut self, pos: SeekFrom) -> std::io::Result<u64> {
        let at = match pos {
            SeekFrom::Start(a) => a as i64,
            SeekFrom::End(d) => self.size as i64 + d,
            SeekFrom::Current(d) => self.at as i64 + d,
        };
        if at < 0 {
            return Err(std::io::Error::other("seek before the start"));
        }
        self.at = at as u64;
        Ok(self.at)
    }
}

fn err(e: bag::BagError) -> JsError {
    JsError::new(&e.0)
}

/// A ROS 1 bag or MCAP file open for reading.
#[wasm_bindgen]
pub struct BagFile {
    bag: Bag<JsSource>,
    cursor: Option<Cursor>,
}

/// One message of a bag.
#[wasm_bindgen]
pub struct BagMessage {
    inner: Message,
}

#[wasm_bindgen]
impl BagFile {
    /// Open a file of `size` bytes that `read(at, length)` gives slices of.
    #[wasm_bindgen(constructor)]
    pub fn new(size: f64, read: js_sys::Function) -> Result<BagFile, JsError> {
        let source = JsSource {
            read,
            size: size as u64,
            at: 0,
            window_at: 0,
            window: Vec::new(),
        };
        Ok(BagFile {
            bag: Bag::from_reader(source).map_err(err)?,
            cursor: None,
        })
    }

    /// The topics' names, with `topicKinds` (as ROS 1 names the types) and `topicCounts`.
    #[wasm_bindgen(js_name = topicNames)]
    pub fn topic_names(&self) -> Vec<String> {
        self.bag.topics().iter().map(|t| t.name.clone()).collect()
    }

    #[wasm_bindgen(js_name = topicKinds)]
    pub fn topic_kinds(&self) -> Vec<String> {
        self.bag.topics().iter().map(|t| t.kind.clone()).collect()
    }

    #[wasm_bindgen(js_name = topicCounts)]
    pub fn topic_counts(&self) -> Vec<f64> {
        self.bag.topics().iter().map(|t| t.count as f64).collect()
    }

    /// Start reading the messages of `topics`, oldest first, with `next`.
    pub fn start(&mut self, topics: Vec<String>) {
        let names: Vec<&str> = topics.iter().map(|t| t.as_str()).collect();
        self.cursor = Some(Cursor::new(&names));
    }

    /// The next message, or undefined at the end.
    #[wasm_bindgen(js_name = next)]
    pub fn next_message(&mut self) -> Result<Option<BagMessage>, JsError> {
        let Some(cursor) = &mut self.cursor else {
            return Ok(None);
        };
        cursor
            .next(&mut self.bag)
            .transpose()
            .map(|m| m.map(|inner| BagMessage { inner }))
            .map_err(err)
    }

    /// How far through the bag the read is, 0 to 1.
    pub fn progress(&self) -> f64 {
        self.cursor.as_ref().map_or(0.0, |c| c.progress(&self.bag))
    }
}

#[wasm_bindgen]
impl BagMessage {
    #[wasm_bindgen(getter)]
    pub fn topic(&self) -> String {
        self.inner.topic.clone()
    }

    /// When it was recorded (seconds).
    #[wasm_bindgen(getter)]
    pub fn time(&self) -> f64 {
        self.inner.time
    }

    /// As a PointCloud2: the cloud (finite points, with `intensity` and the
    /// per-point `time` within the scan when the message has them) and,
    /// through `stamp`, the header's time.
    pub fn scan(&self) -> Result<Cloud, JsError> {
        let points =
            bag::decode_point_cloud2(&self.inner.data, self.inner.encoding).map_err(err)?;
        let mut inner = PointCloud {
            positions: points.positions,
            ..PointCloud::default()
        };
        if let Some(values) = points.intensity {
            inner.attributes.push(Attribute {
                name: INTENSITY.to_string(),
                values: AttributeValues::F32(values),
            });
        }
        if let Some(values) = points.time {
            inner.attributes.push(Attribute {
                name: ca_core::odometry::TIME.to_string(),
                values: AttributeValues::F32(values),
            });
        }
        Ok(Cloud::unindexed(inner))
    }

    /// A PointCloud2's header time (seconds).
    pub fn stamp(&self) -> Result<f64, JsError> {
        // The header comes first in either layout: decode just it.
        Ok(
            bag::decode_point_cloud2(&self.inner.data, self.inner.encoding)
                .map_err(err)?
                .stamp,
        )
    }

    /// As an Imu: `[stamp, up x, up y, up z, acceleration x, y, z]`, the up
    /// direction (world up in the IMU's frame) NaN when it gives no orientation.
    pub fn imu(&self) -> Result<Vec<f64>, JsError> {
        let m = bag::decode_imu(&self.inner.data, self.inner.encoding).map_err(err)?;
        let up = m.up.unwrap_or([f64::NAN; 3]);
        Ok(vec![
            m.stamp,
            up[0],
            up[1],
            up[2],
            m.acceleration[0],
            m.acceleration[1],
            m.acceleration[2],
        ])
    }

    /// Whether the message is laid out as ROS 1 (`true`) or CDR.
    #[wasm_bindgen(getter, js_name = ros1)]
    pub fn ros1(&self) -> bool {
        self.inner.encoding == Encoding::Ros1
    }
}

/// Per time in `times`, the up direction from `imu` (seven numbers a
/// message, as `BagMessage::imu` gives them): three numbers each, NaN where
/// no message is within `window` seconds (see `ca_core::bag::ups_at`).
#[wasm_bindgen(js_name = upsAt)]
pub fn ups_at(imu: &[f64], times: &[f64], window: f64) -> Vec<f64> {
    let messages: Vec<ImuMessage> = imu
        .as_chunks::<7>()
        .0
        .iter()
        .map(|m| ImuMessage {
            stamp: m[0],
            up: (!m[1].is_nan()).then_some([m[1], m[2], m[3]]),
            acceleration: [m[4], m[5], m[6]],
        })
        .collect();
    bag::ups_at(&messages, times, window)
        .into_iter()
        .flat_map(|up| up.unwrap_or([f64::NAN; 3]))
        .collect()
}
