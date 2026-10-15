//! Two independent Jacobi decompositions, preserving each lane's scalar order.

type Matrix = [[f64; 3]; 3];
pub(crate) type Eigen = ([f64; 3], Matrix);

#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
mod lane {
    use core::arch::wasm32::*;
    pub type F = v128;
    pub type Mask = v128;
    #[inline]
    pub fn pair(a: f64, b: f64) -> F {
        f64x2_replace_lane::<1>(f64x2_splat(a), b)
    }
    #[inline]
    pub fn get<const I: usize>(a: F) -> f64 {
        f64x2_extract_lane::<I>(a)
    }
    #[inline]
    pub fn add(a: F, b: F) -> F {
        f64x2_add(a, b)
    }
    #[inline]
    pub fn sub(a: F, b: F) -> F {
        f64x2_sub(a, b)
    }
    #[inline]
    pub fn mul(a: F, b: F) -> F {
        f64x2_mul(a, b)
    }
    #[inline]
    pub fn div(a: F, b: F) -> F {
        f64x2_div(a, b)
    }
    #[inline]
    pub fn sqrt(a: F) -> F {
        f64x2_sqrt(a)
    }
    #[inline]
    pub fn abs(a: F) -> F {
        f64x2_abs(a)
    }
    #[inline]
    pub fn max(a: F, b: F) -> F {
        f64x2_max(a, b)
    }
    #[inline]
    pub fn lt(a: F, b: F) -> Mask {
        f64x2_lt(a, b)
    }
    #[inline]
    pub fn le(a: F, b: F) -> Mask {
        f64x2_le(a, b)
    }
    #[inline]
    pub fn eq(a: F, b: F) -> Mask {
        f64x2_eq(a, b)
    }
    #[inline]
    pub fn and(a: Mask, b: Mask) -> Mask {
        v128_and(a, b)
    }
    #[inline]
    pub fn not(a: Mask) -> Mask {
        v128_not(a)
    }
    #[inline]
    pub fn bits(a: Mask) -> u32 {
        i64x2_bitmask(a) as u32
    }
    #[inline]
    pub fn select(mask: Mask, yes: F, no: F) -> F {
        v128_bitselect(yes, no, mask)
    }
    #[inline]
    pub fn sign(a: F) -> F {
        v128_or(v128_and(a, i64x2_splat(i64::MIN)), f64x2_splat(1.0))
    }
}

// Portable lanes exercise the same masks/rotation schedule in native tests.
#[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
mod lane {
    pub type F = [f64; 2];
    pub type Mask = [bool; 2];
    #[inline]
    pub fn pair(a: f64, b: f64) -> F {
        [a, b]
    }
    #[inline]
    pub fn get<const I: usize>(a: F) -> f64 {
        a[I]
    }
    #[inline]
    pub fn add(a: F, b: F) -> F {
        [a[0] + b[0], a[1] + b[1]]
    }
    #[inline]
    pub fn sub(a: F, b: F) -> F {
        [a[0] - b[0], a[1] - b[1]]
    }
    #[inline]
    pub fn mul(a: F, b: F) -> F {
        [a[0] * b[0], a[1] * b[1]]
    }
    #[inline]
    pub fn div(a: F, b: F) -> F {
        [a[0] / b[0], a[1] / b[1]]
    }
    #[inline]
    pub fn sqrt(a: F) -> F {
        [a[0].sqrt(), a[1].sqrt()]
    }
    #[inline]
    pub fn abs(a: F) -> F {
        [a[0].abs(), a[1].abs()]
    }
    #[inline]
    pub fn max(a: F, b: F) -> F {
        [a[0].max(b[0]), a[1].max(b[1])]
    }
    #[inline]
    pub fn lt(a: F, b: F) -> Mask {
        [a[0] < b[0], a[1] < b[1]]
    }
    #[inline]
    pub fn le(a: F, b: F) -> Mask {
        [a[0] <= b[0], a[1] <= b[1]]
    }
    #[inline]
    pub fn eq(a: F, b: F) -> Mask {
        [a[0] == b[0], a[1] == b[1]]
    }
    #[inline]
    pub fn and(a: Mask, b: Mask) -> Mask {
        [a[0] && b[0], a[1] && b[1]]
    }
    #[inline]
    pub fn not(a: Mask) -> Mask {
        [!a[0], !a[1]]
    }
    #[inline]
    pub fn bits(a: Mask) -> u32 {
        a[0] as u32 | (a[1] as u32) << 1
    }
    #[inline]
    pub fn select(mask: Mask, yes: F, no: F) -> F {
        [
            if mask[0] { yes[0] } else { no[0] },
            if mask[1] { yes[1] } else { no[1] },
        ]
    }
    #[inline]
    pub fn sign(a: F) -> F {
        [a[0].signum(), a[1].signum()]
    }
}

use lane::*;
#[inline]
fn splat(a: f64) -> F {
    pair(a, a)
}

/// Unsafe magnitudes/nonfinite matrices keep the original individual solves.
pub(crate) fn solve(matrices: [Matrix; 2]) -> Option<[Eigen; 2]> {
    if matrices
        .iter()
        .flatten()
        .flatten()
        .any(|x| !x.is_finite() || x.abs() > 1e150)
    {
        return None;
    }
    let mut a: [F; 9] =
        std::array::from_fn(|i| pair(matrices[0][i / 3][i % 3], matrices[1][i / 3][i % 3]));
    let mut v = std::array::from_fn(|i| splat(if i / 3 == i % 3 { 1.0 } else { 0.0 }));
    let mut active = eq(splat(0.0), splat(0.0));
    for _ in 0..50 {
        let mut off = splat(0.0);
        for i in [1, 2, 3, 5, 6, 7] {
            off = add(off, mul(a[i], a[i]));
        }
        let mut scale = splat(0.0);
        for i in [0, 4, 8] {
            scale = add(scale, mul(a[i], a[i]));
        }
        let done = le(off, mul(splat(1e-30), max(scale, splat(f64::MIN_POSITIVE))));
        active = and(active, not(done));
        if bits(active) == 0 {
            break;
        }
        rotate::<0, 1>(&mut a, &mut v, active);
        rotate::<0, 2>(&mut a, &mut v, active);
        rotate::<1, 2>(&mut a, &mut v, active);
    }
    Some([
        (
            std::array::from_fn(|i| get::<0>(a[i * 3 + i])),
            std::array::from_fn(|r| std::array::from_fn(|c| get::<0>(v[r * 3 + c]))),
        ),
        (
            std::array::from_fn(|i| get::<1>(a[i * 3 + i])),
            std::array::from_fn(|r| std::array::from_fn(|c| get::<1>(v[r * 3 + c]))),
        ),
    ])
}

#[inline]
fn rotate<const P: usize, const Q: usize>(a: &mut [F; 9], v: &mut [F; 9], active: Mask) {
    let turn = and(active, not(lt(abs(a[P * 3 + Q]), splat(1e-300))));
    match bits(turn) {
        0 => {}
        3 => rotation::<P, Q, false>(a, v, turn),
        _ => rotation::<P, Q, true>(a, v, turn),
    }
}

#[inline]
fn put<const MASKED: bool>(target: &mut F, value: F, turn: Mask) {
    *target = if MASKED {
        select(turn, value, *target)
    } else {
        value
    };
}

#[inline]
fn rotation<const P: usize, const Q: usize, const MASKED: bool>(
    a: &mut [F; 9],
    v: &mut [F; 9],
    turn: Mask,
) {
    let theta = div(
        sub(a[Q * 3 + Q], a[P * 3 + P]),
        mul(splat(2.0), a[P * 3 + Q]),
    );
    let t = select(
        eq(theta, splat(0.0)),
        splat(1.0),
        div(
            sign(theta),
            add(abs(theta), sqrt(add(mul(theta, theta), splat(1.0)))),
        ),
    );
    let c = div(splat(1.0), sqrt(add(mul(t, t), splat(1.0))));
    let s = mul(t, c);
    for r in 0..3 {
        let (akp, akq) = (a[r * 3 + P], a[r * 3 + Q]);
        put::<MASKED>(&mut a[r * 3 + P], sub(mul(c, akp), mul(s, akq)), turn);
        put::<MASKED>(&mut a[r * 3 + Q], add(mul(s, akp), mul(c, akq)), turn);
    }
    let row_p = [a[P * 3], a[P * 3 + 1], a[P * 3 + 2]];
    let row_q = [a[Q * 3], a[Q * 3 + 1], a[Q * 3 + 2]];
    for k in 0..3 {
        put::<MASKED>(
            &mut a[P * 3 + k],
            sub(mul(c, row_p[k]), mul(s, row_q[k])),
            turn,
        );
        put::<MASKED>(
            &mut a[Q * 3 + k],
            add(mul(s, row_p[k]), mul(c, row_q[k])),
            turn,
        );
    }
    for r in 0..3 {
        let (vp, vq) = (v[r * 3 + P], v[r * 3 + Q]);
        put::<MASKED>(&mut v[r * 3 + P], sub(mul(c, vp), mul(s, vq)), turn);
        put::<MASKED>(&mut v[r * 3 + Q], add(mul(s, vp), mul(c, vq)), turn);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn paired_jacobi_preserves_each_scalar_eigen_bit_and_independent_stops() {
        let mut seed = 0x7710_9018_1721_3309_u64;
        for scale in [0.0, 1e-160, 1e-150, 1.0, 1e100, 1e150] {
            for step in 0..256 {
                let matrices = std::array::from_fn(|lane| {
                    let mut m = [[0.0; 3]; 3];
                    for (r, c) in [(0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2)] {
                        seed ^= seed << 13;
                        seed ^= seed >> 7;
                        seed ^= seed << 17;
                        let value = if lane == 0 && step % 3 == 0 && r != c {
                            0.0
                        } else {
                            ((seed >> 11) as f64 / (1_u64 << 53) as f64 * 2.0 - 1.0) * scale
                        };
                        m[r][c] = value;
                        m[c][r] = value;
                    }
                    m
                });
                let paired = solve(matrices).unwrap();
                for lane in 0..2 {
                    let expected = crate::icp::symmetric_eigen(matrices[lane]);
                    assert_eq!(
                        expected.0.map(f64::to_bits),
                        paired[lane].0.map(f64::to_bits)
                    );
                    assert_eq!(
                        expected.1.map(|r| r.map(f64::to_bits)),
                        paired[lane].1.map(|r| r.map(f64::to_bits))
                    );
                }
            }
        }
        for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, 1e151, -1e151] {
            let mut matrices = [[[0.0; 3]; 3]; 2];
            matrices[1][0][2] = bad;
            assert!(solve(matrices).is_none());
        }
    }
}
