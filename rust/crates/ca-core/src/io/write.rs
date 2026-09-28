//! Point cloud writers (binary PLY and CSV) with optional scalar fields.

use std::fmt::Write as _;

use crate::{AttributeValues, PointCloud};

/// A named per-point value, e.g. a C2C distance.
pub struct ScalarField<'a> {
    pub name: &'a str,
    pub values: &'a [f32],
}

pub(super) fn check(cloud: &PointCloud, scalars: &[ScalarField]) -> Result<(), String> {
    for s in scalars {
        if s.values.len() != cloud.len() {
            return Err(format!(
                "scalar field {} has {} values for {} points",
                s.name,
                s.values.len(),
                cloud.len()
            ));
        }
        if s.name.is_empty() || s.name.contains(char::is_whitespace) {
            return Err(format!("invalid scalar field name {:?}", s.name));
        }
    }
    Ok(())
}

/// Binary little-endian PLY: `double x y z`, `uchar red green blue` when the
/// cloud has colors, the cloud's attributes (`float` or `uchar`, under their
/// own names), then one `float` property per scalar field (named
/// `scalar_<name>`, which CloudCompare loads as a scalar field).
pub fn write_ply(cloud: &PointCloud, scalars: &[ScalarField]) -> Result<Vec<u8>, String> {
    check(cloud, scalars)?;
    let mut header =
        String::from("ply\nformat binary_little_endian 1.0\ncomment CloudAnalyzer Web\n");
    let _ = writeln!(header, "element vertex {}", cloud.len());
    header.push_str("property double x\nproperty double y\nproperty double z\n");
    if cloud.colors.is_some() {
        header.push_str("property uchar red\nproperty uchar green\nproperty uchar blue\n");
    }
    let mut attribute_bytes = 0;
    for a in &cloud.attributes {
        let (kind, size) = match a.values {
            AttributeValues::F32(_) => ("float", 4),
            AttributeValues::U8(_) => ("uchar", 1),
        };
        let _ = writeln!(header, "property {kind} {}", a.name);
        attribute_bytes += size;
    }
    for s in scalars {
        let _ = writeln!(header, "property float scalar_{}", s.name);
    }
    header.push_str("end_header\n");

    let stride =
        24 + if cloud.colors.is_some() { 3 } else { 0 } + attribute_bytes + 4 * scalars.len();
    let mut out = Vec::with_capacity(header.len() + stride * cloud.len());
    out.extend_from_slice(header.as_bytes());
    for (i, p) in cloud.positions.iter().enumerate() {
        for v in p {
            out.extend_from_slice(&v.to_le_bytes());
        }
        if let Some(colors) = &cloud.colors {
            out.extend_from_slice(&colors[i]);
        }
        for a in &cloud.attributes {
            match &a.values {
                AttributeValues::F32(v) => out.extend_from_slice(&v[i].to_le_bytes()),
                AttributeValues::U8(v) => out.push(v[i]),
            }
        }
        for s in scalars {
            out.extend_from_slice(&s.values[i].to_le_bytes());
        }
    }
    Ok(out)
}

/// CSV with a header row: `x,y,z[,r,g,b][,<scalar>...]`. Coordinates use
/// the shortest representation that round-trips.
pub fn write_csv(cloud: &PointCloud, scalars: &[ScalarField]) -> Result<Vec<u8>, String> {
    check(cloud, scalars)?;
    let mut out = String::from("x,y,z");
    if cloud.colors.is_some() {
        out.push_str(",r,g,b");
    }
    for a in &cloud.attributes {
        out.push(',');
        out.push_str(&a.name);
    }
    for s in scalars {
        out.push(',');
        out.push_str(s.name);
    }
    out.push('\n');
    for (i, p) in cloud.positions.iter().enumerate() {
        let _ = write!(out, "{},{},{}", p[0], p[1], p[2]);
        if let Some(colors) = &cloud.colors {
            let [r, g, b] = colors[i];
            let _ = write!(out, ",{r},{g},{b}");
        }
        for a in &cloud.attributes {
            let _ = match &a.values {
                AttributeValues::F32(v) => write!(out, ",{}", v[i]),
                AttributeValues::U8(v) => write!(out, ",{}", v[i]),
            };
        }
        for s in scalars {
            let _ = write!(out, ",{}", s.values[i]);
        }
        out.push('\n');
    }
    Ok(out.into_bytes())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cloud() -> PointCloud {
        PointCloud {
            positions: vec![[368_000.125, 3_955_000.5, 40.0], [1.0, -2.0, 3.25]],
            colors: Some(vec![[255, 0, 10], [1, 2, 3]]),
            attributes: Vec::new(),
        }
    }

    #[test]
    fn ply_roundtrips_through_the_reader() {
        let distances = [0.5f32, -1.25];
        let bytes = write_ply(
            &cloud(),
            &[ScalarField {
                name: "C2C_distance",
                values: &distances,
            }],
        )
        .unwrap();
        let text = String::from_utf8_lossy(&bytes[..300]);
        assert!(text.contains("property float scalar_C2C_distance"));
        let back = super::super::ply::read(&bytes).unwrap();
        assert_eq!(back.positions, cloud().positions);
        assert_eq!(back.colors, cloud().colors);
        // The scalar is the last 4 bytes of each 31-byte record.
        let body = &bytes[bytes.len() - 2 * 31..];
        assert_eq!(f32::from_le_bytes(body[27..31].try_into().unwrap()), 0.5);
        assert_eq!(f32::from_le_bytes(body[58..62].try_into().unwrap()), -1.25);
    }

    #[test]
    fn csv_has_header_and_full_precision() {
        let bytes = write_csv(
            &cloud(),
            &[ScalarField {
                name: "C2M_distance",
                values: &[0.25, 2.0],
            }],
        )
        .unwrap();
        let text = String::from_utf8(bytes).unwrap();
        assert_eq!(
            text,
            "x,y,z,r,g,b,C2M_distance\n368000.125,3955000.5,40,255,0,10,0.25\n1,-2,3.25,1,2,3,2\n"
        );
    }

    #[test]
    fn attributes_roundtrip_through_ply_and_appear_in_csv() {
        let mut c = cloud();
        c.attributes = vec![
            crate::Attribute {
                name: crate::INTENSITY.into(),
                values: AttributeValues::F32(vec![100.0, 7.5]),
            },
            crate::Attribute {
                name: crate::CLASSIFICATION.into(),
                values: AttributeValues::U8(vec![2, 6]),
            },
        ];
        let back = super::super::ply::read(&write_ply(&c, &[]).unwrap()).unwrap();
        assert_eq!(back.attributes, c.attributes);
        let csv = String::from_utf8(write_csv(&c, &[]).unwrap()).unwrap();
        assert!(csv.starts_with(
            "x,y,z,r,g,b,intensity,classification\n368000.125,3955000.5,40,255,0,10,100,2\n"
        ));
    }

    #[test]
    fn rejects_mismatched_scalars() {
        let err = write_ply(
            &cloud(),
            &[ScalarField {
                name: "d",
                values: &[1.0],
            }],
        );
        assert!(err.is_err());
        let err = write_csv(
            &cloud(),
            &[ScalarField {
                name: "bad name",
                values: &[1.0, 2.0],
            }],
        );
        assert!(err.is_err());
    }
}
