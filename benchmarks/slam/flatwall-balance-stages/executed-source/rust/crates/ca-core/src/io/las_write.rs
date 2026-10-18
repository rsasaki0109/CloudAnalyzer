//! LAS 1.4 / LAZ writer.
//!
//! Points are written as format 6 (or 7 with RGB): the 1.4 formats give the
//! class a whole byte, so codes above 31 survive. Scalar fields and any
//! attribute other than intensity and classification (normals, M3C2 results)
//! become `float` extra bytes, described by an Extra Bytes VLR, which
//! CloudCompare, PDAL and LAStools load as scalar fields.

use std::io::{Cursor, Seek, SeekFrom};

use super::write::{ScalarField, check};
use crate::{AttributeValues, CLASSIFICATION, INTENSITY, PointCloud};

const HEADER_LEN: usize = 375;
const VLR_HEADER_LEN: usize = 54;
const EXTRA_BYTES_DESCRIPTOR_LEN: usize = 192;
/// Finest scale tried: 0.1 µm is beyond any scanner's precision.
const FINEST_EXPONENT: i32 = 7;

/// Scale and offset for one axis. A cloud read from LAS sits on the grid of
/// its original scale, so the coarsest power of ten on which every
/// coordinate lies is kept (a 0.01 file is written at 0.01, exactly). Other
/// clouds get the finest power of ten whose integers still fit an `i32`.
fn quantization(cloud: &PointCloud, axis: usize) -> (f64, f64) {
    let (lo, hi) = cloud
        .positions
        .iter()
        .fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), p| {
            (lo.min(p[axis]), hi.max(p[axis]))
        });
    if !lo.is_finite() {
        return (0.001, 0.0);
    }
    let offset = lo.floor();
    let mut scale = 1.0;
    for exponent in 0..=FINEST_EXPONENT {
        let s = 10f64.powi(-exponent);
        if (hi - offset) / s > i32::MAX as f64 {
            break;
        }
        scale = s;
        let on_grid = cloud.positions.iter().all(|p| {
            let q = (p[axis] - offset) / s;
            (q - q.round()).abs() < 1e-3
        });
        if on_grid {
            break;
        }
    }
    (scale, offset)
}

/// A value of an attribute as `f32`, whatever its storage.
fn value(values: &AttributeValues, i: usize) -> f32 {
    match values {
        AttributeValues::F32(v) => v[i],
        AttributeValues::U8(v) => v[i] as f32,
    }
}

fn vlr_header(user_id: &str, record_id: u16, len: usize) -> Vec<u8> {
    let mut h = vec![0u8; VLR_HEADER_LEN];
    h[2..2 + user_id.len()].copy_from_slice(user_id.as_bytes());
    h[18..20].copy_from_slice(&record_id.to_le_bytes());
    h[20..22].copy_from_slice(&(len as u16).to_le_bytes());
    h
}

/// Extra Bytes VLR (`LASF_Spec`, record 4): one `float` descriptor per field.
fn extra_bytes_vlr(names: &[&str]) -> Vec<u8> {
    let mut out = vlr_header("LASF_Spec", 4, names.len() * EXTRA_BYTES_DESCRIPTOR_LEN);
    for name in names {
        let mut d = [0u8; EXTRA_BYTES_DESCRIPTOR_LEN];
        d[2] = 9; // float
        // The name field holds 32 bytes; cut longer names on a char boundary.
        let mut end = name.len().min(32);
        while !name.is_char_boundary(end) {
            end -= 1;
        }
        d[4..4 + end].copy_from_slice(&name.as_bytes()[..end]);
        out.extend_from_slice(&d);
    }
    out
}

/// LAS 1.4 point format 6/7 (LAZ when `compress`), with the cloud's
/// intensity, classification and colors, and every scalar field and other
/// attribute as a `float` extra byte field.
pub fn write_las(
    cloud: &PointCloud,
    scalars: &[ScalarField],
    compress: bool,
) -> Result<Vec<u8>, String> {
    check(cloud, scalars)?;
    let n = cloud.len();
    let format: u8 = if cloud.colors.is_some() { 7 } else { 6 };
    let base_len = if cloud.colors.is_some() { 36 } else { 30 };
    let intensity = cloud.attribute(INTENSITY).map(|a| &a.values);
    let classification = cloud.attribute(CLASSIFICATION).map(|a| &a.values);
    let others: Vec<_> = cloud
        .attributes
        .iter()
        .filter(|a| a.name != INTENSITY && a.name != CLASSIFICATION)
        .collect();
    let extra_names: Vec<&str> = others
        .iter()
        .map(|a| a.name.as_str())
        .chain(scalars.iter().map(|s| s.name))
        .collect();
    let record_len = base_len + 4 * extra_names.len();

    let mut vlrs = Vec::new();
    let mut vlr_count = 0u32;
    if !extra_names.is_empty() {
        vlrs.extend(extra_bytes_vlr(&extra_names));
        vlr_count += 1;
    }
    let laz_vlr = if compress {
        let items = laz::LazItemRecordBuilder::default_for_point_format_id(
            format,
            (record_len - base_len) as u16,
        )
        .map_err(|e| e.to_string())?;
        let vlr = laz::LazVlrBuilder::new(items).build();
        let mut payload = Vec::new();
        vlr.write_to(&mut payload).map_err(|e| e.to_string())?;
        vlrs.extend(vlr_header("laszip encoded", 22204, payload.len()));
        vlrs.extend(payload);
        vlr_count += 1;
        Some(vlr)
    } else {
        None
    };

    // Header (filled in last, once the bounds are known), VLRs, records.
    let data_offset = HEADER_LEN + vlrs.len();
    let mut out = Vec::with_capacity(data_offset + n * record_len);
    out.resize(HEADER_LEN, 0);
    out.extend(vlrs);
    let quant: [(f64, f64); 3] = std::array::from_fn(|axis| quantization(cloud, axis));
    let (mut lo, mut hi) = ([i32::MAX; 3], [i32::MIN; 3]);
    for (i, p) in cloud.positions.iter().enumerate() {
        for axis in 0..3 {
            let (scale, offset) = quant[axis];
            // `as` saturates, which only matters for non-finite coordinates.
            let v = ((p[axis] - offset) / scale).round() as i32;
            lo[axis] = lo[axis].min(v);
            hi[axis] = hi[axis].max(v);
            out.extend_from_slice(&v.to_le_bytes());
        }
        let intensity = intensity.map_or(0.0, |v| value(v, i));
        out.extend_from_slice(&(intensity.round().clamp(0.0, 65535.0) as u16).to_le_bytes());
        out.push(0x11); // return 1 of 1
        out.push(0); // flags, channel, scan direction, edge
        out.push(
            classification
                .map_or(0.0, |v| value(v, i))
                .clamp(0.0, 255.0) as u8,
        );
        // User data, scan angle, point source id, GPS time.
        out.extend_from_slice(&[0u8; 13]);
        if let Some(colors) = &cloud.colors {
            // 8-bit to 16-bit by `v * 257`, so 255 becomes 65535.
            for c in colors[i] {
                out.extend_from_slice(&(c as u16 * 257).to_le_bytes());
            }
        }
        for a in &others {
            out.extend_from_slice(&value(&a.values, i).to_le_bytes());
        }
        for s in scalars {
            out.extend_from_slice(&s.values[i].to_le_bytes());
        }
    }
    if let Some(vlr) = laz_vlr {
        let mut cursor = Cursor::new(out[..data_offset].to_vec());
        cursor.seek(SeekFrom::End(0)).map_err(|e| e.to_string())?;
        let mut compressor =
            laz::LasZipCompressor::new(&mut cursor, vlr).map_err(|e| e.to_string())?;
        compressor
            .compress_many(&out[data_offset..])
            .map_err(|e| e.to_string())?;
        compressor.done().map_err(|e| e.to_string())?;
        drop(compressor);
        out = cursor.into_inner();
    }

    let h = &mut out[..HEADER_LEN];
    h[..4].copy_from_slice(b"LASF");
    // Global encoding: WKT bit, required with point formats 6-10.
    h[6..8].copy_from_slice(&0x10u16.to_le_bytes());
    h[24] = 1;
    h[25] = 4;
    h[26..31].copy_from_slice(b"OTHER");
    h[58..75].copy_from_slice(b"CloudAnalyzer Web");
    h[94..96].copy_from_slice(&(HEADER_LEN as u16).to_le_bytes());
    h[96..100].copy_from_slice(&(data_offset as u32).to_le_bytes());
    h[100..104].copy_from_slice(&vlr_count.to_le_bytes());
    // LASzip marks a compressed file by setting bit 7 of the format id.
    h[104] = if compress { format | 0x80 } else { format };
    h[105..107].copy_from_slice(&(record_len as u16).to_le_bytes());
    // The legacy point counts stay 0, as formats 6-10 require.
    for axis in 0..3 {
        let (scale, offset) = quant[axis];
        let (min, max) = if n == 0 {
            (0.0, 0.0)
        } else {
            (
                lo[axis] as f64 * scale + offset,
                hi[axis] as f64 * scale + offset,
            )
        };
        h[131 + 8 * axis..139 + 8 * axis].copy_from_slice(&scale.to_le_bytes());
        h[155 + 8 * axis..163 + 8 * axis].copy_from_slice(&offset.to_le_bytes());
        h[179 + 16 * axis..187 + 16 * axis].copy_from_slice(&max.to_le_bytes());
        h[187 + 16 * axis..195 + 16 * axis].copy_from_slice(&min.to_le_bytes());
    }
    h[247..255].copy_from_slice(&(n as u64).to_le_bytes());
    // Every point is its own first return.
    h[255..263].copy_from_slice(&(n as u64).to_le_bytes());
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Attribute;

    /// A UTM-sized cloud on a 1 cm grid, as read from a typical LAS file.
    fn cloud() -> PointCloud {
        let positions = (0..3000)
            .map(|i| {
                let k = i as f64;
                [
                    368_000.0 + (k * 7.0) * 0.01,
                    3_955_000.0 - k * 0.01,
                    40.0 + (i % 100) as f64 * 0.01,
                ]
            })
            .collect();
        PointCloud {
            positions,
            colors: Some((0..3000).map(|i| [(i % 256) as u8, 7, 255]).collect()),
            attributes: vec![
                Attribute {
                    name: INTENSITY.into(),
                    values: AttributeValues::F32((0..3000).map(|i| (i * 20) as f32).collect()),
                },
                Attribute {
                    name: CLASSIFICATION.into(),
                    values: AttributeValues::U8((0..3000).map(|i| (i % 70) as u8).collect()),
                },
            ],
        }
    }

    fn f64_at(b: &[u8], at: usize) -> f64 {
        f64::from_le_bytes(b[at..at + 8].try_into().unwrap())
    }

    fn assert_same_points(back: &PointCloud, c: &PointCloud) {
        assert_eq!(back.len(), c.len());
        for (a, b) in back.positions.iter().zip(&c.positions) {
            for axis in 0..3 {
                assert!((a[axis] - b[axis]).abs() < 1e-8, "{a:?} vs {b:?}");
            }
        }
        assert_eq!(back.colors, c.colors);
        assert_eq!(back.attributes, c.attributes);
    }

    #[test]
    fn las_roundtrips_through_the_reader() {
        let c = cloud();
        let bytes = write_las(&c, &[], false).unwrap();
        assert_eq!(&bytes[..4], b"LASF");
        assert_eq!((bytes[25], bytes[104]), (4, 7));
        // The 1 cm grid is found, and the offset sits below the data.
        assert_eq!(f64_at(&bytes, 131), 0.01);
        assert_eq!(f64_at(&bytes, 147), 0.01);
        assert_eq!(f64_at(&bytes, 155), 368_000.0);
        assert_eq!(f64_at(&bytes, 187), 368_000.0); // min x
        assert_eq!(bytes.len(), HEADER_LEN + 3000 * 36);
        let back = super::super::read("out.las", &bytes).unwrap();
        assert_same_points(&back, &c);
        // Writing what was read gives the same file: nothing drifts.
        assert_eq!(write_las(&back, &[], false).unwrap(), bytes);
    }

    #[test]
    fn laz_roundtrips_through_the_reader() {
        let c = cloud();
        let las = write_las(&c, &[], false).unwrap();
        let laz = write_las(&c, &[], true).unwrap();
        assert_eq!(laz[104], 0x87);
        assert!(laz.len() < las.len() / 2, "{} vs {}", laz.len(), las.len());
        let back = super::super::read("out.laz", &laz).unwrap();
        assert_same_points(&back, &c);
    }

    #[test]
    fn scalars_and_attributes_become_extra_bytes() {
        let mut c = cloud();
        c.colors = None;
        c.attributes.push(Attribute {
            name: "nx".into(),
            values: AttributeValues::F32(vec![0.5; 3000]),
        });
        let distances: Vec<f32> = (0..3000).map(|i| i as f32 * -0.25).collect();
        let fields = [ScalarField {
            name: "C2C_distance",
            values: &distances,
        }];
        for compress in [false, true] {
            let bytes = write_las(&c, &fields, compress).unwrap();
            assert_eq!(bytes[104] & 0x3f, 6);
            assert_eq!(u16::from_le_bytes([bytes[105], bytes[106]]), 30 + 8);
            // The Extra Bytes VLR comes first and describes both floats.
            let vlr = &bytes[HEADER_LEN..];
            assert!(vlr[2..].starts_with(b"LASF_Spec"));
            assert_eq!(u16::from_le_bytes([vlr[18], vlr[19]]), 4);
            assert_eq!(u16::from_le_bytes([vlr[20], vlr[21]]), 2 * 192);
            let first = &vlr[VLR_HEADER_LEN..];
            assert_eq!(first[2], 9);
            assert!(first[4..].starts_with(b"nx\0"));
            assert!(first[192 + 4..].starts_with(b"C2C_distance\0"));

            let back = super::super::read("out.las", &bytes).unwrap();
            assert_eq!(back.attribute(INTENSITY), c.attribute(INTENSITY));
            assert_eq!(back.attribute(CLASSIFICATION), c.attribute(CLASSIFICATION));
            if !compress {
                let data = u32::from_le_bytes(bytes[96..100].try_into().unwrap()) as usize;
                let record = &bytes[data + 2999 * 38..data + 3000 * 38];
                assert_eq!(f32::from_le_bytes(record[30..34].try_into().unwrap()), 0.5);
                assert_eq!(
                    f32::from_le_bytes(record[34..38].try_into().unwrap()),
                    distances[2999]
                );
            }
        }
    }

    #[test]
    fn off_grid_clouds_get_a_fine_scale() {
        // Float coordinates from a PLY: no decimal grid, small extent.
        let c = PointCloud {
            positions: vec![[0.123_456_7, -0.2, 1.0 / 3.0], [0.2, 0.05, 0.1]],
            colors: None,
            attributes: Vec::new(),
        };
        let bytes = write_las(&c, &[], false).unwrap();
        assert_eq!(f64_at(&bytes, 131), 1e-7);
        let back = super::super::read("out.las", &bytes).unwrap();
        for (a, b) in back.positions.iter().zip(&c.positions) {
            for axis in 0..3 {
                assert!((a[axis] - b[axis]).abs() <= 0.51e-7);
            }
        }
        // Default intensity and class are written as 0.
        assert_eq!(
            back.attribute(CLASSIFICATION).unwrap().values,
            AttributeValues::U8(vec![0, 0])
        );

        // A wide extent still fits the 32-bit integers.
        let wide = PointCloud {
            positions: vec![[1.0 / 3.0, 0.0, 0.0], [2_000.0, 0.0, 0.0]],
            ..c
        };
        let bytes = write_las(&wide, &[], false).unwrap();
        assert_eq!(f64_at(&bytes, 131), 1e-6);
    }
}
