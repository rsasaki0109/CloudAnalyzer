//! Binary scalar decoding shared by the PLY and PCD readers.

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum Scalar {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    I64,
    U64,
    F32,
    F64,
}

impl Scalar {
    pub(crate) fn size(self) -> usize {
        match self {
            Self::I8 | Self::U8 => 1,
            Self::I16 | Self::U16 => 2,
            Self::I32 | Self::U32 | Self::F32 => 4,
            Self::I64 | Self::U64 | Self::F64 => 8,
        }
    }

    pub(crate) fn is_float(self) -> bool {
        matches!(self, Self::F32 | Self::F64)
    }

    /// Decode one value from the start of `b`, which must hold `self.size()` bytes.
    pub(crate) fn decode(self, b: &[u8], little_endian: bool) -> f64 {
        macro_rules! num {
            ($t:ty) => {{
                let arr = b[..size_of::<$t>()].try_into().unwrap();
                (if little_endian {
                    <$t>::from_le_bytes(arr)
                } else {
                    <$t>::from_be_bytes(arr)
                }) as f64
            }};
        }
        match self {
            Self::I8 => num!(i8),
            Self::U8 => num!(u8),
            Self::I16 => num!(i16),
            Self::U16 => num!(u16),
            Self::I32 => num!(i32),
            Self::U32 => num!(u32),
            Self::I64 => num!(i64),
            Self::U64 => num!(u64),
            Self::F32 => num!(f32),
            Self::F64 => num!(f64),
        }
    }

    /// Raw little/big-endian bits as `u32`, for packed RGB fields.
    pub(crate) fn bits_u32(b: &[u8], little_endian: bool) -> u32 {
        let arr = b[..4].try_into().unwrap();
        if little_endian {
            u32::from_le_bytes(arr)
        } else {
            u32::from_be_bytes(arr)
        }
    }
}

/// Convert a color channel to 8 bits: floats are assumed to be in `[0, 1]`,
/// 16-bit integers are scaled down, everything else is clamped.
pub(crate) fn color_channel(value: f64, kind: Scalar) -> u8 {
    let v = match kind {
        Scalar::F32 | Scalar::F64 => value * 255.0,
        Scalar::U16 | Scalar::I16 => value / 257.0,
        _ => value,
    };
    v.round().clamp(0.0, 255.0) as u8
}

/// Unpack `0x00RRGGBB`.
pub(crate) fn unpack_rgb(bits: u32) -> [u8; 3] {
    [(bits >> 16) as u8, (bits >> 8) as u8, bits as u8]
}
