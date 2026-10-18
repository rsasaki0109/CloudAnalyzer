//! KITTI Velodyne scans (`.bin`): headerless little-endian `f32` records of
//! x, y, z and reflectance, kept as intensity.

use super::IoError;
use crate::{Attribute, AttributeValues, INTENSITY, PointCloud};

const RECORD: usize = 16;

pub(crate) fn read(bytes: &[u8]) -> Result<PointCloud, IoError> {
    if !bytes.len().is_multiple_of(RECORD) {
        return Err(IoError::Truncated("KITTI .bin"));
    }
    let n = bytes.len() / RECORD;
    let mut positions = Vec::with_capacity(n);
    let mut intensity = Vec::with_capacity(n);
    for r in bytes.as_chunks::<RECORD>().0 {
        let f = |k: usize| f32::from_le_bytes(r[4 * k..4 * k + 4].try_into().unwrap());
        positions.push([f(0), f(1), f(2)].map(f64::from));
        intensity.push(f(3));
    }
    Ok(PointCloud {
        positions,
        colors: None,
        attributes: vec![Attribute {
            name: INTENSITY.into(),
            values: AttributeValues::F32(intensity),
        }],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_records() {
        let mut bytes = Vec::new();
        for v in [1.0f32, 2.0, 3.0, 0.5, -4.0, 0.0, 1.5, 0.25] {
            bytes.extend(v.to_le_bytes());
        }
        let cloud = read(&bytes).unwrap();
        assert_eq!(cloud.positions, vec![[1.0, 2.0, 3.0], [-4.0, 0.0, 1.5]]);
        assert_eq!(
            cloud.attribute(INTENSITY).unwrap().values,
            AttributeValues::F32(vec![0.5, 0.25])
        );
        assert!(read(&bytes[..20]).is_err());
    }
}
