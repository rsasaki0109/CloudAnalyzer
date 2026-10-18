//! E57 (ASTM E2807) reader and writer, on top of the `e57` crate.
//!
//! Every scan (`data3D`) is read with its pose applied, spherical coordinates
//! converted, and points without a valid position skipped. Intensity is
//! normalized by the file's limits to `0..=65535` (the LAS range, so the
//! values mean the same as for other formats); colors to 8 bits. Scans are
//! merged into one cloud with a [`SOURCE`] attribute holding the scan index,
//! like a merge, so a multi-scan file can be split back per scan.

use std::io::Cursor;

use e57::{E57Reader, E57Writer, Record, RecordDataType, RecordName, RecordValue};

use super::IoError;
use crate::merge::SOURCE;
use crate::{Attribute, AttributeValues, INTENSITY, PointCloud};

const FORMAT: &str = "E57";
const INTENSITY_MAX: f32 = u16::MAX as f32;

fn error(e: e57::Error) -> IoError {
    IoError::header(FORMAT, e.to_string())
}

fn open(bytes: &[u8]) -> Result<E57Reader<Cursor<&[u8]>>, IoError> {
    E57Reader::new(Cursor::new(bytes)).map_err(error)
}

/// Total records over all scans (invalid points included).
pub(crate) fn announced_points(bytes: &[u8]) -> Option<u64> {
    let reader = open(bytes).ok()?;
    Some(reader.pointclouds().iter().map(|pc| pc.records).sum())
}

/// Name of every scan, in file order (`None` when the scan has none).
pub fn scan_names(bytes: &[u8]) -> Result<Vec<Option<String>>, IoError> {
    Ok(open(bytes)?
        .pointclouds()
        .into_iter()
        .map(|pc| pc.name.filter(|n| !n.trim().is_empty()))
        .collect())
}

/// Where each attribute sits in a scan's records, with its value range.
struct Layout {
    xyz: Option<[usize; 3]>,
    xyz_state: Option<usize>,
    /// Range, azimuth, elevation.
    spherical: Option<[usize; 3]>,
    spherical_state: Option<usize>,
    intensity: Option<(usize, Option<(f64, f64)>)>,
    intensity_invalid: Option<usize>,
    rgb: Option<[(usize, (f64, f64)); 3]>,
    rgb_invalid: Option<usize>,
    /// Row-major rotation of the pose.
    rotation: [[f64; 3]; 3],
    translation: [f64; 3],
}

impl Layout {
    fn new(pc: &e57::PointCloud) -> Self {
        let find = |name: RecordName| pc.prototype.iter().position(|r| r.name == name);
        let all = |names: [RecordName; 3]| -> Option<[usize; 3]> {
            let [a, b, c] = names.map(find);
            Some([a?, b?, c?])
        };
        let range = |i: usize, limits: Option<(&Option<RecordValue>, &Option<RecordValue>)>| {
            limits
                .and_then(|(lo, hi)| limit_range(pc, i, lo, hi))
                .or_else(|| type_range(&pc.prototype[i].data_type))
        };
        let intensity = find(RecordName::Intensity).map(|i| {
            let limits = pc.intensity_limits.as_ref();
            (
                i,
                range(i, limits.map(|l| (&l.intensity_min, &l.intensity_max))),
            )
        });
        let rgb = all([
            RecordName::ColorRed,
            RecordName::ColorGreen,
            RecordName::ColorBlue,
        ])
        .map(|idx| {
            let l = pc.color_limits.as_ref();
            let limits = [
                l.map(|l| (&l.red_min, &l.red_max)),
                l.map(|l| (&l.green_min, &l.green_max)),
                l.map(|l| (&l.blue_min, &l.blue_max)),
            ];
            // Unbounded float colors: take the usual unit range.
            std::array::from_fn(|k| (idx[k], range(idx[k], limits[k]).unwrap_or((0.0, 1.0))))
        });
        let (rotation, translation) = match &pc.transform {
            Some(t) => {
                let q = &t.rotation;
                let n = (q.w * q.w + q.x * q.x + q.y * q.y + q.z * q.z).sqrt();
                let (w, x, y, z) = (q.w / n, q.x / n, q.y / n, q.z / n);
                let rotation = [
                    [
                        1.0 - 2.0 * (y * y + z * z),
                        2.0 * (x * y - w * z),
                        2.0 * (x * z + w * y),
                    ],
                    [
                        2.0 * (x * y + w * z),
                        1.0 - 2.0 * (x * x + z * z),
                        2.0 * (y * z - w * x),
                    ],
                    [
                        2.0 * (x * z - w * y),
                        2.0 * (y * z + w * x),
                        1.0 - 2.0 * (x * x + y * y),
                    ],
                ];
                let d = &t.translation;
                (rotation, [d.x, d.y, d.z])
            }
            None => (
                [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                [0.0; 3],
            ),
        };
        Self {
            xyz: all([
                RecordName::CartesianX,
                RecordName::CartesianY,
                RecordName::CartesianZ,
            ]),
            xyz_state: find(RecordName::CartesianInvalidState),
            spherical: all([
                RecordName::SphericalRange,
                RecordName::SphericalAzimuth,
                RecordName::SphericalElevation,
            ]),
            spherical_state: find(RecordName::SphericalInvalidState),
            intensity,
            intensity_invalid: find(RecordName::IsIntensityInvalid),
            rgb,
            rgb_invalid: find(RecordName::IsColorInvalid),
            rotation,
            translation,
        }
    }

    /// The posed position, or `None` for a point without a valid one (an
    /// invalid state of 1 means a direction only, 2 no position at all).
    fn position(&self, values: &[f64]) -> Option<[f64; 3]> {
        let valid = |state: Option<usize>| state.is_none_or(|i| values[i] == 0.0);
        let local = if let Some([x, y, z]) = self.xyz.filter(|_| valid(self.xyz_state)) {
            [values[x], values[y], values[z]]
        } else if let Some([r, a, e]) = self.spherical.filter(|_| valid(self.spherical_state)) {
            let (range, azimuth, elevation) = (values[r], values[a], values[e]);
            [
                range * elevation.cos() * azimuth.cos(),
                range * elevation.cos() * azimuth.sin(),
                range * elevation.sin(),
            ]
        } else {
            return None;
        };
        let m = &self.rotation;
        Some(std::array::from_fn(|k| {
            m[k][0] * local[0] + m[k][1] * local[1] + m[k][2] * local[2] + self.translation[k]
        }))
    }
}

/// A range from a scan's intensity or color limits (a scaled integer limit
/// is scaled like the record's values).
fn limit_range(
    pc: &e57::PointCloud,
    record: usize,
    min: &Option<RecordValue>,
    max: &Option<RecordValue>,
) -> Option<(f64, f64)> {
    let dt = &pc.prototype[record].data_type;
    let value = |v: &RecordValue| match v {
        RecordValue::ScaledInteger(i) if !matches!(dt, RecordDataType::ScaledInteger { .. }) => {
            Some(*i as f64)
        }
        _ => v.to_f64(dt).ok(),
    };
    let (lo, hi) = (value(min.as_ref()?)?, value(max.as_ref()?)?);
    (hi > lo).then_some((lo, hi))
}

/// The range a record type can hold, when it is bounded.
fn type_range(dt: &RecordDataType) -> Option<(f64, f64)> {
    let (lo, hi) = match *dt {
        RecordDataType::Integer { min, max } => (min as f64, max as f64),
        RecordDataType::ScaledInteger {
            min,
            max,
            scale,
            offset,
        } => (min as f64 * scale + offset, max as f64 * scale + offset),
        RecordDataType::Single {
            min: Some(min),
            max: Some(max),
        } => (min as f64, max as f64),
        RecordDataType::Double {
            min: Some(min),
            max: Some(max),
        } => (min, max),
        _ => return None,
    };
    (hi > lo).then_some((lo, hi))
}

fn unit(v: f64, (lo, hi): (f64, f64)) -> f64 {
    ((v - lo) / (hi - lo)).clamp(0.0, 1.0)
}

/// Read all scans, keeping every `keep_every`-th record (counted over the
/// whole file, so thinning is even across scans).
///
/// Uses the crate's raw reader: its simple reader fails on a data packet
/// that completes no point, which its own writer produces for tiny scans.
pub(crate) fn read(bytes: &[u8], keep_every: usize) -> Result<PointCloud, IoError> {
    let mut reader = open(bytes)?;
    let scans = reader.pointclouds();
    let has_color = scans.iter().any(|pc| pc.has_color());
    let has_intensity = scans.iter().any(|pc| pc.has_intensity());
    // Scan indices only fit a `u8` attribute up to 256 scans.
    let tag_source = scans.len() > 1 && scans.len() <= 256;

    let mut cloud = PointCloud::default();
    let mut colors = Vec::new();
    let mut intensity = Vec::new();
    let mut source = Vec::new();
    let mut record = 0usize;
    let mut values = Vec::new();
    for (index, pc) in scans.iter().enumerate() {
        let layout = Layout::new(pc);
        for raw in reader.pointcloud_raw(pc).map_err(error)? {
            let raw = raw.map_err(error)?;
            let keep = record.is_multiple_of(keep_every);
            record += 1;
            if !keep {
                continue;
            }
            values.clear();
            for (v, r) in raw.iter().zip(&pc.prototype) {
                values.push(v.to_f64(&r.data_type).map_err(error)?);
            }
            let Some(position) = layout.position(&values) else {
                continue;
            };
            cloud.positions.push(position);
            let flagged = |i: Option<usize>| i.is_some_and(|i| values[i] != 0.0);
            if has_color {
                colors.push(match layout.rgb {
                    Some(rgb) if !flagged(layout.rgb_invalid) => {
                        rgb.map(|(i, range)| (unit(values[i], range) * 255.0).round() as u8)
                    }
                    _ => [0; 3],
                });
            }
            if has_intensity {
                intensity.push(match layout.intensity {
                    Some((i, range)) if !flagged(layout.intensity_invalid) => match range {
                        Some(range) => (unit(values[i], range) * INTENSITY_MAX as f64) as f32,
                        // No known range: keep the value as stored.
                        None => values[i] as f32,
                    },
                    _ => 0.0,
                });
            }
            if tag_source {
                source.push(index as u8);
            }
        }
    }
    if has_color {
        cloud.colors = Some(colors);
    }
    if has_intensity {
        cloud.attributes.push(Attribute {
            name: INTENSITY.into(),
            values: AttributeValues::F32(intensity),
        });
    }
    if tag_source {
        cloud.attributes.push(Attribute {
            name: SOURCE.into(),
            values: AttributeValues::U8(source),
        });
    }
    Ok(cloud)
}

/// Single-scan E57 with `double` XYZ, the intensity (as `float`, limits
/// `0..=65535` unless the values exceed them) and 8-bit RGB when present.
pub fn write_e57(cloud: &PointCloud) -> Result<Vec<u8>, String> {
    let intensity = match cloud.attribute(INTENSITY).map(|a| &a.values) {
        Some(AttributeValues::F32(v)) => Some(v.clone()),
        Some(AttributeValues::U8(v)) => Some(v.iter().map(|&x| x as f32).collect()),
        None => None,
    };
    let mut prototype = vec![
        Record::CARTESIAN_X_F64,
        Record::CARTESIAN_Y_F64,
        Record::CARTESIAN_Z_F64,
    ];
    if let Some(values) = &intensity {
        let (lo, hi) = values
            .iter()
            .fold((0f32, INTENSITY_MAX), |(lo, hi), &v| (lo.min(v), hi.max(v)));
        prototype.push(Record {
            name: RecordName::Intensity,
            data_type: RecordDataType::Single {
                min: Some(lo),
                max: Some(hi),
            },
        });
    }
    if cloud.colors.is_some() {
        prototype.extend([
            Record::COLOR_RED_U8,
            Record::COLOR_GREEN_U8,
            Record::COLOR_BLUE_U8,
        ]);
    }

    let guid = guid(cloud);
    let mut out = Cursor::new(Vec::new());
    let mut writer = E57Writer::new(&mut out, &guid).map_err(|e| e.to_string())?;
    let mut scan = writer
        .add_pointcloud(&format!("{guid}-scan"), prototype)
        .map_err(|e| e.to_string())?;
    scan.set_name(Some("CloudAnalyzer".into()));
    for (i, p) in cloud.positions.iter().enumerate() {
        let mut values: Vec<RecordValue> = p.iter().map(|&v| RecordValue::Double(v)).collect();
        if let Some(v) = &intensity {
            values.push(RecordValue::Single(v[i]));
        }
        if let Some(colors) = &cloud.colors {
            values.extend(colors[i].map(|c| RecordValue::Integer(c as i64)));
        }
        scan.add_point(values).map_err(|e| e.to_string())?;
    }
    scan.finalize().map_err(|e| e.to_string())?;
    writer.finalize().map_err(|e| e.to_string())?;
    drop(writer);
    Ok(out.into_inner())
}

/// A GUID-shaped id derived from the points (FNV-1a), since the browser build
/// has no random source here; different clouds almost surely get different ids.
fn guid(cloud: &PointCloud) -> String {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325;
    let mut mix = |bytes: &[u8]| {
        for &b in bytes {
            h = (h ^ b as u64).wrapping_mul(0x0100_0000_01b3);
        }
    };
    mix(&(cloud.len() as u64).to_le_bytes());
    for p in cloud
        .positions
        .iter()
        .step_by(cloud.len().div_ceil(4096).max(1))
    {
        for v in p {
            mix(&v.to_le_bytes());
        }
    }
    let lo = h.rotate_left(29) ^ 0x9e37_79b9_7f4a_7c15;
    format!(
        "{{{:08x}-{:04x}-{:04x}-{:04x}-{:012x}}}",
        h >> 32,
        (h >> 16) & 0xffff,
        h & 0xffff,
        lo >> 48,
        lo & 0xffff_ffff_ffff
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> PointCloud {
        PointCloud {
            positions: vec![[1.0, 2.0, 3.0], [-4.5, 0.25, 100.0], [0.0, 0.0, 0.0]],
            colors: Some(vec![[255, 0, 10], [0, 128, 255], [7, 7, 7]]),
            attributes: vec![Attribute {
                name: INTENSITY.into(),
                values: AttributeValues::F32(vec![0.0, 1000.0, 65535.0]),
            }],
        }
    }

    #[test]
    fn round_trips_xyz_intensity_and_rgb() {
        let cloud = sample();
        let bytes = write_e57(&cloud).unwrap();
        let back = crate::read("out.e57", &bytes).unwrap();
        assert_eq!(back, cloud);
        assert_eq!(scan_names(&bytes).unwrap(), [Some("CloudAnalyzer".into())]);
        assert_eq!(announced_points(&bytes), Some(3));
    }

    #[test]
    fn writes_plain_xyz() {
        let cloud = PointCloud {
            positions: vec![[1.0, 2.0, 3.0]],
            ..Default::default()
        };
        let back = crate::read("a.e57", &write_e57(&cloud).unwrap()).unwrap();
        assert_eq!(back, cloud);
    }

    /// Two posed scans, one spherical with an invalid point, built with the
    /// `e57` crate's writer.
    fn two_scans() -> Vec<u8> {
        let mut out = Cursor::new(Vec::new());
        let mut writer = E57Writer::new(&mut out, "{file}").unwrap();

        let mut scan = writer
            .add_pointcloud(
                "{a}",
                vec![
                    Record::CARTESIAN_X_F64,
                    Record::CARTESIAN_Y_F64,
                    Record::CARTESIAN_Z_F64,
                    Record::CARTESIAN_INVALID_STATE,
                    Record {
                        name: RecordName::Intensity,
                        data_type: RecordDataType::Integer { min: 0, max: 2047 },
                    },
                ],
            )
            .unwrap();
        scan.set_name(Some("north".into()));
        // A quarter turn about z, then a shift along x.
        let h = std::f64::consts::FRAC_1_SQRT_2;
        scan.set_transform(Some(e57::Transform {
            rotation: e57::Quaternion {
                w: h,
                x: 0.0,
                y: 0.0,
                z: h,
            },
            translation: e57::Translation {
                x: 10.0,
                y: 0.0,
                z: 0.0,
            },
        }));
        for (x, state, i) in [(1.0, 0, 2047), (2.0, 2, 0), (3.0, 0, 0)] {
            scan.add_point(vec![
                RecordValue::Double(x),
                RecordValue::Double(0.0),
                RecordValue::Double(0.0),
                RecordValue::Integer(state),
                RecordValue::Integer(i),
            ])
            .unwrap();
        }
        scan.finalize().unwrap();

        let mut scan = writer
            .add_pointcloud(
                "{b}",
                vec![
                    Record::SPHERICAL_RANGE_F64,
                    Record::SPHERICAL_AZIMUTH_F64,
                    Record::SPHERICAL_ELEVATION_F64,
                    Record::SPHERICAL_INVALID_STATE,
                    Record::COLOR_RED_U8,
                    Record::COLOR_GREEN_U8,
                    Record::COLOR_BLUE_U8,
                ],
            )
            .unwrap();
        for (range, elevation, state) in [(2.0, 0.0, 0), (5.0, std::f64::consts::FRAC_PI_2, 0)] {
            scan.add_point(vec![
                RecordValue::Double(range),
                RecordValue::Double(0.0),
                RecordValue::Double(elevation),
                RecordValue::Integer(state),
                RecordValue::Integer(255),
                RecordValue::Integer(0),
                RecordValue::Integer(51),
            ])
            .unwrap();
        }
        scan.finalize().unwrap();
        writer.finalize().unwrap();
        drop(writer);
        out.into_inner()
    }

    #[test]
    fn merges_posed_scans_and_skips_invalid_points() {
        let bytes = two_scans();
        let cloud = crate::read("scans.e57", &bytes).unwrap();
        let expected = [
            [10.0, 1.0, 0.0],
            [10.0, 3.0, 0.0],
            [2.0, 0.0, 0.0],
            [0.0, 0.0, 5.0],
        ];
        assert_eq!(cloud.len(), expected.len());
        for (p, e) in cloud.positions.iter().zip(expected) {
            assert!((0..3).all(|k| (p[k] - e[k]).abs() < 1e-9), "{p:?} vs {e:?}");
        }
        assert_eq!(
            cloud.colors.as_deref(),
            Some(&[[0, 0, 0], [0, 0, 0], [255, 0, 51], [255, 0, 51]][..])
        );
        assert_eq!(
            cloud.attribute(INTENSITY).unwrap().values,
            AttributeValues::F32(vec![65535.0, 0.0, 0.0, 0.0])
        );
        assert_eq!(
            cloud.attribute(SOURCE).unwrap().values,
            AttributeValues::U8(vec![0, 0, 1, 1])
        );
        assert_eq!(scan_names(&bytes).unwrap(), [Some("north".into()), None]);
        assert_eq!(announced_points(&bytes), Some(5));

        let thinned = crate::io::read_thinned("scans.e57", &bytes, 2).unwrap();
        // Records 0, 2 (scan 0) and 4 (scan 1); record 2 is valid.
        assert_eq!(thinned.len(), 3);
    }

    #[test]
    fn reads_the_committed_fixture() {
        let bytes = include_bytes!("../../../../../web/e2e/fixtures/scans.e57");
        let cloud = crate::read("scans.e57", bytes).unwrap();
        assert_eq!(cloud.len(), 3000);
        assert!(cloud.colors.is_some());
        assert_eq!(scan_names(bytes).unwrap().len(), 2);
    }

    #[test]
    fn rejects_garbage() {
        assert!(crate::read("x.e57", b"ASTM-E57 but not really").is_err());
    }
}
