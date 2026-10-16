//! Explicit source-time conversion, without floating-point epoch arithmetic.

/// Physical units declared by the consumer; never inferred from a field name.
#[derive(Debug, Clone, Copy)]
pub enum TimeUnit {
    Seconds,
    Milliseconds,
    Microseconds,
    Nanoseconds,
}

impl TimeUnit {
    fn nanoseconds(self) -> u64 {
        match self {
            Self::Seconds => 1_000_000_000,
            Self::Milliseconds => 1_000_000,
            Self::Microseconds => 1_000,
            Self::Nanoseconds => 1,
        }
    }
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum PointTimeError {
    #[error("point time value at index {0} is not finite")]
    NonFinite(usize),
    #[error("point time at index {0} is outside signed int64 nanoseconds")]
    OutOfRange(usize),
}

/// Converts each represented binary value to the nearest integer nanosecond,
/// ties to even, then adds the integer origin. Pass zero for absolute values or
/// an explicitly selected header/timebase for offsets. All arithmetic after
/// binary decomposition is integer; no intermediate f64 unit multiplication or
/// epoch addition can introduce another rounding. Input order is preserved.
///
/// Nonfinite values or a final timestamp outside i64 are errors. An offset may
/// exceed i64 when adding the origin brings the final timestamp into range.
pub fn timestamps_ns(
    values: &[f64],
    unit: TimeUnit,
    origin_ns: i64,
) -> Result<Vec<i64>, PointTimeError> {
    values
        .iter()
        .enumerate()
        .map(|(index, &value)| {
            if !value.is_finite() {
                return Err(PointTimeError::NonFinite(index));
            }
            let magnitude = rounded_magnitude(value, unit.nanoseconds())
                .ok_or(PointTimeError::OutOfRange(index))?;
            let offset = if value.is_sign_negative() {
                -i128::from(magnitude)
            } else {
                i128::from(magnitude)
            };
            i64::try_from(i128::from(origin_ns) + offset)
                .map_err(|_| PointTimeError::OutOfRange(index))
        })
        .collect()
}

/// The f64 significand has at most 53 bits; the largest scale adds 30 bits,
/// so their product fits exactly in u128. Magnitudes larger than u64::MAX
/// cannot yield an i64 result after adding any i64 origin.
fn rounded_magnitude(value: f64, scale: u64) -> Option<u64> {
    let bits = value.to_bits();
    let encoded_exponent = ((bits >> 52) & 0x7ff) as i32;
    let fraction = bits & 0x000f_ffff_ffff_ffff;
    let (significand, exponent) = if encoded_exponent == 0 {
        (fraction, -1074)
    } else {
        ((1 << 52) | fraction, encoded_exponent - 1023 - 52)
    };
    let scaled = u128::from(significand) * u128::from(scale);
    let rounded = if exponent >= 0 {
        if exponent >= 64 || scaled > u128::from(u64::MAX) >> exponent {
            return None;
        }
        scaled << exponent
    } else {
        let shift = (-exponent) as u32;
        if shift >= 128 {
            0
        } else {
            let whole = scaled >> shift;
            let remainder = scaled & ((1u128 << shift) - 1);
            let half = 1u128 << (shift - 1);
            whole + u128::from(remainder > half || (remainder == half && whole & 1 != 0))
        }
    };
    u64::try_from(rounded).ok()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn exact_integer_offsets_keep_epoch_precision_and_source_order() {
        let header = 1_700_000_000_000_000_123;
        let values = [0.0, 4167.0, 1.0, 1.0, f64::from(u32::MAX)];
        assert_eq!(
            timestamps_ns(&values, TimeUnit::Nanoseconds, header).unwrap(),
            [
                header,
                header + 4167,
                header + 1,
                header + 1,
                header + i64::from(u32::MAX)
            ]
        );
        for (unit, scale) in [
            (TimeUnit::Seconds, 1_000_000_000),
            (TimeUnit::Milliseconds, 1_000_000),
            (TimeUnit::Microseconds, 1_000),
            (TimeUnit::Nanoseconds, 1),
        ] {
            assert_eq!(
                timestamps_ns(&[-2.0, 0.0, 3.0], unit, 7).unwrap(),
                [7 - 2 * scale, 7, 7 + 3 * scale]
            );
        }
    }

    #[test]
    fn floating_values_round_once_from_the_exact_binary_source() {
        let values = [-2.5, -1.5, -0.5, -0.0, 0.5, 1.5, 2.5];
        assert_eq!(
            timestamps_ns(&values, TimeUnit::Nanoseconds, 1).unwrap(),
            [-1, -1, 1, 1, 1, 3, 3]
        );
        assert_eq!(
            timestamps_ns(&[1_700_000_000.000_000_2], TimeUnit::Seconds, 0).unwrap(),
            [1_700_000_000_000_000_238]
        );
        // This stored binary value is just above half a nanosecond, although
        // multiplying it in f64 rounds to exactly 0.5 before integer rounding.
        assert_eq!(
            timestamps_ns(&[5e-10, -5e-10], TimeUnit::Seconds, 0).unwrap(),
            [1, -1]
        );
        assert_eq!(
            timestamps_ns(
                &[f64::from_bits(1), -f64::MIN_POSITIVE],
                TimeUnit::Seconds,
                0
            )
            .unwrap(),
            [0, 0]
        );
    }

    #[test]
    fn checks_final_range_after_integer_origin_addition() {
        assert_eq!(
            timestamps_ns(&[0.0], TimeUnit::Nanoseconds, i64::MAX).unwrap(),
            [i64::MAX]
        );
        assert_eq!(
            timestamps_ns(&[-9_223_372_036_854_775_808.0], TimeUnit::Nanoseconds, 0).unwrap(),
            [i64::MIN]
        );
        assert_eq!(
            timestamps_ns(
                &[9_223_372_036_854_775_808.0],
                TimeUnit::Nanoseconds,
                i64::MIN
            )
            .unwrap(),
            [0]
        );
        assert_eq!(
            timestamps_ns(
                &[-9_223_372_036_854_775_808.0],
                TimeUnit::Nanoseconds,
                i64::MAX
            )
            .unwrap(),
            [-1]
        );
        for (value, origin) in [
            (1.0, i64::MAX),
            (-1.0, i64::MIN),
            (9_223_372_036_854_775_808.0, 0),
            (f64::MAX, i64::MIN),
            (-f64::MAX, i64::MAX),
        ] {
            assert_eq!(
                timestamps_ns(&[0.0, value], TimeUnit::Nanoseconds, origin),
                Err(PointTimeError::OutOfRange(1))
            );
        }
    }

    #[test]
    fn rejects_nonfinite_values_and_accepts_empty_input() {
        for value in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
            assert_eq!(
                timestamps_ns(&[0.0, value], TimeUnit::Seconds, 0),
                Err(PointTimeError::NonFinite(1))
            );
        }
        assert!(
            timestamps_ns(&[], TimeUnit::Seconds, i64::MIN)
                .unwrap()
                .is_empty()
        );
    }
}
