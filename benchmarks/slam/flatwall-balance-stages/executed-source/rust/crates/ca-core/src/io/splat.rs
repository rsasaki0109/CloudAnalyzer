//! `.splat` reader: the headerless Gaussian Splatting format of the
//! antimatter15 web viewer. Each 32-byte record holds the center (`f32` x3),
//! linear scales (`f32` x3), RGBA (`u8` x4, alpha = opacity) and a rotation
//! (`u8` x4), all little-endian. Only centers, colors, opacity and size are
//! kept, like for a 3DGS PLY.

use super::IoError;
use crate::{Attribute, AttributeValues, OPACITY, PointCloud, SPLAT_SIZE};

const RECORD: usize = 32;

pub(crate) fn read(bytes: &[u8]) -> Result<PointCloud, IoError> {
    if !bytes.len().is_multiple_of(RECORD) {
        return Err(IoError::Truncated("SPLAT"));
    }
    let n = bytes.len() / RECORD;
    let mut positions = Vec::with_capacity(n);
    let mut colors = Vec::with_capacity(n);
    let mut opacity = Vec::with_capacity(n);
    let mut size = Vec::with_capacity(n);
    for r in bytes.as_chunks::<RECORD>().0 {
        let f = |k: usize| f32::from_le_bytes(r[4 * k..4 * k + 4].try_into().unwrap());
        positions.push([f(0), f(1), f(2)].map(f64::from));
        // Same measure as for a PLY: the largest standard deviation.
        size.push(f(3).max(f(4)).max(f(5)));
        colors.push([r[24], r[25], r[26]]);
        opacity.push(f32::from(r[27]) / 255.0);
    }
    Ok(PointCloud {
        positions,
        colors: Some(colors),
        attributes: vec![
            Attribute {
                name: OPACITY.into(),
                values: AttributeValues::F32(opacity),
            },
            Attribute {
                name: SPLAT_SIZE.into(),
                values: AttributeValues::F32(size),
            },
        ],
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_records() {
        let mut bytes = Vec::new();
        for (p, s, rgba) in [
            ([1.0f32, 2.0, 3.0], [0.1f32, 0.3, 0.2], [255u8, 128, 0, 255]),
            ([-1.0, 0.5, 0.0], [2.0, 1.0, 1.0], [0, 0, 10, 51]),
        ] {
            for v in p.into_iter().chain(s) {
                bytes.extend(v.to_le_bytes());
            }
            bytes.extend(rgba);
            bytes.extend([128u8, 128, 128, 255]);
        }
        let cloud = read(&bytes).unwrap();
        assert_eq!(cloud.positions, vec![[1.0, 2.0, 3.0], [-1.0, 0.5, 0.0]]);
        assert_eq!(cloud.colors, Some(vec![[255, 128, 0], [0, 0, 10]]));
        assert_eq!(
            cloud.attribute(OPACITY).unwrap().values,
            AttributeValues::F32(vec![1.0, 0.2])
        );
        assert_eq!(
            cloud.attribute(SPLAT_SIZE).unwrap().values,
            AttributeValues::F32(vec![0.3, 2.0])
        );
        assert!(matches!(read(&bytes[..40]), Err(IoError::Truncated(_))));
    }
}
