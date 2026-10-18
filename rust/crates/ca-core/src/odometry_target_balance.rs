//! Experimental all-pair inverse-target-multiplicity normal equations.

use super::{LocalMap, Mat6, Rigid, Vec6, VoxelHasher};
use std::collections::HashMap;
use std::hash::BuildHasherDefault;

fn coordinate_key(point: [f64; 3]) -> [u64; 3] {
    point.map(|v| if v == 0.0 { 0 } else { v.to_bits() })
}

/// Exact numeric-coordinate equality, with signed zero normalized.
pub(crate) fn multiplicities(targets: &[[f64; 3]]) -> Vec<usize> {
    multiplicities_by(targets, |point| *point)
}

/// Count targets directly from pair records, preserving their input order.
pub(super) fn multiplicities_by<T>(items: &[T], point: impl Fn(&T) -> [f64; 3]) -> Vec<usize> {
    let mut index: HashMap<[u64; 3], usize, BuildHasherDefault<VoxelHasher>> = HashMap::default();
    let mut frequencies = Vec::<usize>::new();
    let mut counts = Vec::with_capacity(items.len());
    for item in items {
        let next = frequencies.len();
        let id = *index.entry(coordinate_key(point(item))).or_insert_with(|| {
            frequencies.push(0);
            next
        });
        frequencies[id] += 1;
        counts.push(id);
    }
    // Reuse the output buffer: first coordinate IDs, then completed counts.
    // The second pass uses dense IDs and needs no additional hash lookups.
    for id in &mut counts {
        *id = frequencies[*id];
    }
    counts
}

struct Pair {
    world: [f64; 3],
    target: [f64; 3],
    distance_sq: f64,
}

pub(super) fn equations(
    map: &LocalMap,
    source: &[[f64; 3]],
    total: &Rigid,
    center: [f64; 3],
    threshold_sq: f64,
    kernel: f64,
) -> (Mat6, Vec6) {
    let matched = |p: &[f64; 3]| {
        let world = total.apply(p);
        let (target, distance_sq) = map.nearest(&world)?;
        (distance_sq <= threshold_sq).then_some(Pair {
            world,
            target,
            distance_sq,
        })
    };
    #[cfg(feature = "parallel")]
    let pairs: Vec<_> = {
        use rayon::prelude::*;
        source.par_iter().filter_map(matched).collect()
    };
    #[cfg(not(feature = "parallel"))]
    let pairs: Vec<_> = source.iter().filter_map(matched).collect();
    let counts = multiplicities_by(&pairs, |pair| pair.target);
    let chunk = |(offset, pairs): (usize, &[Pair])| {
        let mut h = [[0.0; 6]; 6];
        let mut b = [0.0; 6];
        for (i, pair) in pairs.iter().enumerate() {
            let q: [f64; 3] = std::array::from_fn(|k| pair.world[k] - center[k]);
            let residual: [f64; 3] = std::array::from_fn(|k| pair.world[k] - pair.target[k]);
            let weight = kernel * kernel
                / (kernel + pair.distance_sq).powi(2)
                / counts[offset * 256 + i] as f64;
            let rows = [
                [1.0, 0.0, 0.0, 0.0, q[2], -q[1]],
                [0.0, 1.0, 0.0, -q[2], 0.0, q[0]],
                [0.0, 0.0, 1.0, q[1], -q[0], 0.0],
            ];
            for (j, r) in rows.iter().zip(residual) {
                for a in 0..6 {
                    b[a] -= weight * j[a] * r;
                    for k in 0..6 {
                        h[a][k] += weight * j[a] * j[k];
                    }
                }
            }
        }
        (h, b)
    };
    let add = |(mut h, mut b): (Mat6, Vec6), (other_h, other_b): (Mat6, Vec6)| {
        for i in 0..6 {
            b[i] += other_b[i];
            for j in 0..6 {
                h[i][j] += other_h[i][j];
            }
        }
        (h, b)
    };
    #[cfg(feature = "parallel")]
    {
        use rayon::prelude::*;
        pairs
            .par_chunks(256)
            .enumerate()
            .map(chunk)
            .reduce(|| ([[0.0; 6]; 6], [0.0; 6]), add)
    }
    #[cfg(not(feature = "parallel"))]
    {
        pairs
            .chunks(256)
            .enumerate()
            .map(chunk)
            .fold(([[0.0; 6]; 6], [0.0; 6]), add)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pending_registration_retains_its_weighting_policy() {
        use super::super::{Odometry, OdometryParams};
        let mut odometry = Odometry::new(OdometryParams {
            min_range: 0.0,
            ..OdometryParams::default()
        });
        let scan = crate::PointCloud {
            positions: (0..12)
                .map(|i| [3.0 * (i % 4) as f64, 3.0 * (i / 4) as f64, (i % 3) as f64])
                .collect(),
            ..crate::PointCloud::default()
        };
        odometry.register(&scan);
        assert!(!odometry.target_balance());
        let pending = odometry.begin(&scan);
        odometry.set_target_balance(true);
        assert!(!pending.target_balance);
        assert!(odometry.begin(&scan).target_balance);
    }

    fn compare(a: (Mat6, Vec6), b: (Mat6, Vec6)) {
        for (x, y) in
            a.0.iter()
                .flatten()
                .chain(&a.1)
                .zip(b.0.iter().flatten().chain(&b.1))
        {
            assert!((x - y).abs() <= 1e-10, "{x} != {y}");
        }
    }

    #[test]
    fn coordinate_counts_are_exact_and_normalize_signed_zero() {
        assert_eq!(
            multiplicities(&[[0.0, 0.0, 0.0], [-0.0, 0.0, 0.0], [1e-12, 0.0, 0.0]]),
            vec![2, 2, 1]
        );
        assert!(multiplicities(&[]).is_empty());
    }

    #[test]
    fn pair_counts_preserve_order_and_distinct_nearby_coordinates() {
        let close = f64::from_bits(1.0_f64.to_bits() + 1);
        let targets = [
            [1.0, 0.0, 0.0],
            [-0.0, 0.0, 0.0],
            [close, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, -0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -0.0],
        ];
        let pairs: Vec<_> = targets
            .iter()
            .map(|&target| Pair {
                world: [9.0; 3],
                target,
                distance_sq: 0.0,
            })
            .collect();
        assert_eq!(
            multiplicities_by(&pairs, |pair| pair.target),
            [3, 3, 1, 3, 3, 3, 3]
        );
        assert_eq!(multiplicities(&targets), [3, 3, 1, 3, 3, 3, 3]);
    }

    #[test]
    fn resized_count_tables_match_numeric_equality_in_input_order() {
        let targets: Vec<_> = (0..2051)
            .map(|i| {
                let group = (i * 37) % 521;
                [group as f64, (group % 17) as f64, -(group as f64)]
            })
            .collect();
        // Numeric equality is an independent reference for these finite points.
        let expected: Vec<_> = targets
            .iter()
            .map(|point| targets.iter().filter(|other| *other == point).count())
            .collect();
        assert_eq!(multiplicities(&targets), expected);
    }

    #[test]
    fn repeated_identical_pairs_do_not_gain_target_influence() {
        let mut map = LocalMap::new(1.0, 20);
        map.insert(&[[0.0; 3], [3.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 3.0]]);
        let source = vec![
            [0.125, 0.0, 0.0],
            [3.125, 0.0, 0.0],
            [0.0, 3.125, 0.0],
            [0.0, 0.0, 3.125],
        ];
        let baseline = equations(&map, &source, &Rigid::IDENTITY, [0.0; 3], 1.0, 1.0);
        compare(
            baseline,
            map.normal_equations(&source, &Rigid::IDENTITY, [0.0; 3], 1.0, 1.0),
        );
        let mut duplicated = source.clone();
        duplicated.extend(std::iter::repeat_n(source[0], 600));
        compare(
            baseline,
            equations(&map, &duplicated, &Rigid::IDENTITY, [0.0; 3], 1.0, 1.0),
        );
        assert!(
            (map.normal_equations(&duplicated, &Rigid::IDENTITY, [0.0; 3], 1.0, 1.0)
                .0[0][0]
                - baseline.0[0][0])
                .abs()
                > 100.0
        );
    }

    #[test]
    fn only_threshold_accepted_pairs_count_and_zero_support_is_explicit() {
        let mut map = LocalMap::new(1.0, 20);
        map.insert(&[[0.0; 3]]);
        let one = equations(
            &map,
            &[[0.125, 0.0, 0.0]],
            &Rigid::IDENTITY,
            [0.0; 3],
            0.25,
            1.0,
        );
        compare(
            one,
            equations(
                &map,
                &[[0.125, 0.0, 0.0], [0.75, 0.0, 0.0]],
                &Rigid::IDENTITY,
                [0.0; 3],
                0.25,
                1.0,
            ),
        );
        compare(
            ([[0.0; 6]; 6], [0.0; 6]),
            equations(&map, &[[50.0; 3]], &Rigid::IDENTITY, [0.0; 3], 1.0, 1.0),
        );
    }
}
