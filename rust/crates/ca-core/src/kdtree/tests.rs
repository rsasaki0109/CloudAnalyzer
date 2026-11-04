use super::*;

#[test]
fn knn_ancestral_bounds_keep_hit_bits_and_order() {
    for count in [1, 17, 257, 2049] {
        for mode in 0..7 {
            let mut points = pseudo_random(count, 73);
            for (i, p) in points.iter_mut().enumerate() {
                *p = match mode {
                    1 => [p[0] + 368000.0, p[1] + 3955000.0, p[2] + 40.0],
                    2 => p.map(|v| v * 1e-160),
                    3 => p.map(|v| v * 1e155),
                    4 => [1.0, -0.0, 0.0],
                    5 => [(i % 13) as f64, (i / 13) as f64, 0.0],
                    6 if i % 17 == 0 => [f64::NAN, f64::INFINITY, f64::NEG_INFINITY],
                    _ => *p,
                };
            }
            let tree = KdTree::new(&points).unwrap();
            let queries: Vec<_> = points
                .iter()
                .step_by((count / 17).max(1))
                .copied()
                .chain([
                    [0.0; 3],
                    [f64::NAN, 0.0, 0.0],
                    [f64::INFINITY, -0.0, 1.0],
                    [1e155; 3],
                ])
                .collect();
            for k in [0, 1, 3, 12, 24, 64, count + 2] {
                for q in &queries {
                    for simd in [false, true] {
                        let (mut old, mut bounded) = (Vec::new(), Vec::new());
                        tree.nearest_k_into_with_options(q, k, &mut old, simd, false);
                        tree.nearest_k_into_with_options(q, k, &mut bounded, simd, true);
                        let bits = |v: &[(usize, f64)]| {
                            v.iter().map(|&(i, d)| (i, d.to_bits())).collect::<Vec<_>>()
                        };
                        assert_eq!(
                            bits(&old),
                            bits(&bounded),
                            "count={count}, mode={mode}, k={k}, q={q:?}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn knn_shift_preserves_swap_insert_bits_and_tie_order() {
    fn swapped(found: &mut Vec<(usize, f64)>, k: usize, slot: usize, d: f64) {
        if found.len() < k || d < found[found.len() - 1].1 {
            if found.len() < k {
                found.push((slot, d));
            } else {
                found[k - 1] = (slot, d);
            }
            let mut j = found.len() - 1;
            while j > 0 && found[j - 1].1 > d {
                found.swap(j - 1, j);
                j -= 1;
            }
        }
    }
    let specials = [
        0.0,
        -0.0,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::from_bits(0x7ff0_0000_0000_0123),
        f64::from_bits(0xfff8_0000_0000_0456),
        f64::MIN_POSITIVE,
        f64::MAX,
    ];
    for k in [1, 3, 12, 24, 64, 128] {
        for mode in 0..5 {
            let (mut expected, mut actual) = (Vec::new(), Vec::new());
            let mut random = 17u64;
            for i in 0..1024 {
                random ^= random << 13;
                random ^= random >> 7;
                random ^= random << 17;
                let d = match mode {
                    0 => i as f64,
                    1 => (1024 - i) as f64,
                    2 => (i % 7) as f64,
                    3 => (random % 1024) as f64 / 13.0,
                    _ => specials[(random as usize) % specials.len()],
                };
                swapped(&mut expected, k, i, d);
                insert_knn(&mut actual, k, i, d);
                let bits = |v: &[(usize, f64)]| {
                    v.iter()
                        .map(|&(slot, d)| (slot, d.to_bits()))
                        .collect::<Vec<_>>()
                };
                assert_eq!(
                    bits(&actual),
                    bits(&expected),
                    "k={k}, mode={mode}, step={i}"
                );
            }
        }
    }
}

#[test]
fn distance_only_queries_keep_tree_slot_order_and_original_distance_bits() {
    for count in [1, 17, 257] {
        for mode in 0..6 {
            let points: Vec<_> = (0..count)
                .map(|i| {
                    let p = [(i % 11) as f64, ((i / 11) % 7) as f64, (i % 3) as f64];
                    match mode {
                        1 => p.map(|v| v + 3955000.0),
                        2 => [1.0, -0.0, 0.0],
                        3 => p.map(|v| v * 1e155),
                        4 => p.map(|v| v * 1e-160),
                        5 if i == 0 => [f64::NAN, f64::INFINITY, f64::NEG_INFINITY],
                        _ => p,
                    }
                })
                .collect();
            let tree = KdTree::new(&points).unwrap();
            let bytes = tree.storage_bytes();
            let mut hits = Vec::new();
            for q in points.iter().step_by(17).copied().chain([
                [0.5; 3],
                [-0.0; 3],
                [f64::NAN; 3],
                [f64::INFINITY; 3],
            ]) {
                for k in [0, 1, 3, 12, 24, 64, count + 2] {
                    let expected: Vec<_> = tree
                        .nearest_k(&q, k)
                        .into_iter()
                        .map(|(i, d)| (i, d.to_bits()))
                        .collect();
                    for simd in [false, true] {
                        for bounds in [false, true] {
                            tree.nearest_k_slots_into_with_options(&q, k, &mut hits, simd, bounds);
                            let actual: Vec<_> = hits
                                .iter()
                                .map(|&(slot, d)| (tree.indices[slot] as usize, d.to_bits()))
                                .collect();
                            assert_eq!(expected, actual);
                            assert_eq!(tree.storage_bytes(), bytes);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn exact_snapshots_preserve_slots_bits_certificates_and_owned_data() {
    for count in [1, 15, 16, 17, 31, 33, 257, 4097] {
        for coincident in [false, true] {
            let points: Vec<_> = (0..count)
                .map(|i| {
                    if coincident {
                        [6_378_137.000_001, -0.0, 1.0]
                    } else {
                        [
                            6_378_137.0 + (i % 47) as f64 * 0.000_001,
                            (i % 23) as f64 * 0.1,
                            if i % 2 == 0 { -0.0 } else { 0.0 },
                        ]
                    }
                })
                .collect();
            let tree = CandidateKdTree::new(&points).unwrap();
            let mut bytes = tree.snapshot();
            let restored = CandidateKdTree::from_snapshot(&bytes, count).unwrap();
            assert_eq!(restored.snapshot(), bytes);
            bytes.fill(0);
            for (p, q) in points.iter().zip(restored.original_points()) {
                assert_eq!(p.map(f64::to_bits), q.map(f64::to_bits));
            }
            assert_eq!(
                restored.original_point(0).unwrap().map(f64::to_bits),
                points[0].map(f64::to_bits)
            );
            assert_eq!(restored.original_point(count), None);
            for i in 0..37 {
                let q = [
                    points[i % count][0] + 0.000_003,
                    (i % 11) as f64 * 0.09,
                    0.01,
                ];
                for slot in [0, count as u32 - 1, count as u32 / 2, u32::MAX] {
                    for bounds in [false, true] {
                        for simd in [false, true] {
                            let (a, ac) = tree.nearest_with_options(&q, slot, bounds, simd);
                            let (b, bc) = restored.nearest_with_options(&q, slot, bounds, simd);
                            assert_eq!(a.distance_sq.to_bits(), b.distance_sq.to_bits());
                            assert_eq!((a.index, a.slot, ac), (b.index, b.slot, bc));
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn exact_snapshots_preserve_nonfinite_payloads_and_full_search_behavior() {
    let points: Vec<_> = (0..65)
        .map(|i| {
            [
                i as f64,
                match i % 5 {
                    0 => f64::from_bits(0x7ff8_0000_0000_0123),
                    1 => f64::NEG_INFINITY,
                    2 => f64::INFINITY,
                    3 => -0.0,
                    _ => 0.0,
                },
                (i % 3) as f64,
            ]
        })
        .collect();
    let tree = CandidateKdTree::new(&points).unwrap();
    let bytes = tree.snapshot();
    let restored = CandidateKdTree::from_snapshot(&bytes, 65).unwrap();
    assert_eq!(restored.snapshot(), bytes);
    for p in &points {
        for slot in [0, 64, u32::MAX] {
            let (a, ac) = tree.nearest_bounded(p, slot);
            let (b, bc) = restored.nearest_bounded(p, slot);
            assert_eq!(a.distance_sq.to_bits(), b.distance_sq.to_bits());
            assert_eq!((a.slot, ac), (b.slot, bc));
            assert!(!bc);
        }
    }
}

#[test]
fn reused_validated_extrema_preserve_hits_certificates_and_snapshot_bits() {
    for count in [1, 16, 17, 65, 257, 2049] {
        for mode in 0..5 {
            let mut points = pseudo_random(count, 97);
            for (i, p) in points.iter_mut().enumerate() {
                match mode {
                    0 => *p = [368_000.0 + p[0], 3_955_000.0 + p[1], p[2] * 1e-6],
                    1 => *p = [if i % 2 == 0 { -0.0 } else { 0.0 }; 3],
                    2 => *p = p.map(|v| v * f64::from_bits(1)),
                    3 => *p = p.map(|v| v * 1e150),
                    _ => p[2] = if i % 2 == 0 { -0.0 } else { 0.0 },
                }
            }
            let tree = CandidateKdTree::new(&points).unwrap();
            let bytes = tree.snapshot();
            let old =
                CandidateKdTree::from_snapshot_with_reused_bounds(&bytes, count, false).unwrap();
            let reused =
                CandidateKdTree::from_snapshot_with_reused_bounds(&bytes, count, true).unwrap();
            assert_eq!(reused.snapshot(), bytes);
            assert_eq!(reused.bounds.len(), reused.tree.nodes.len());
            for q in points.iter().step_by(17).copied().chain([
                [0.0; 3],
                [-0.0; 3],
                [f64::from_bits(1); 3],
                [f64::MAX; 3],
                [f64::NAN; 3],
                [f64::INFINITY; 3],
            ]) {
                for slot in [0, count as u32 / 2, count as u32 - 1, u32::MAX] {
                    for bounds in [false, true] {
                        for simd in [false, true] {
                            let (a, ac) = old.nearest_with_options(&q, slot, bounds, simd);
                            let (b, bc) = reused.nearest_with_options(&q, slot, bounds, simd);
                            assert_eq!(a.distance_sq.to_bits(), b.distance_sq.to_bits());
                            assert_eq!((a.index, a.slot, ac), (b.index, b.slot, bc));
                        }
                    }
                }
            }
        }
    }
    let points = [
        [f64::NAN, -0.0, 1.0],
        [f64::INFINITY; 3],
        [-0.0; 3],
        [f64::from_bits(0xfff8_0000_0000_0123), 1.0, 2.0],
        [f64::NEG_INFINITY; 3],
    ];
    let bytes = CandidateKdTree::new(&points).unwrap().snapshot();
    let old =
        CandidateKdTree::from_snapshot_with_reused_bounds(&bytes, points.len(), false).unwrap();
    let reused =
        CandidateKdTree::from_snapshot_with_reused_bounds(&bytes, points.len(), true).unwrap();
    assert!(!reused.finite);
    assert_eq!(reused.snapshot(), bytes);
    for q in points {
        let (a, ac) = old.nearest_bounded(&q, 0);
        let (b, bc) = reused.nearest_bounded(&q, 0);
        assert_eq!(a.distance_sq.to_bits(), b.distance_sq.to_bits());
        assert_eq!((a.slot, ac), (b.slot, bc));
    }
}

#[test]
fn exact_snapshots_reject_invalid_headers_indices_topology_and_partitions() {
    for reuse_bounds in [false, true] {
        let tree = CandidateKdTree::new(&(0..65).map(|i| [i as f64, 0.0, 0.0]).collect::<Vec<_>>())
            .unwrap();
        let bytes = tree.snapshot();
        let point_end = 16 + 65 * 24;
        let node_start = point_end + 65 * 4;
        assert!(
            CandidateKdTree::from_snapshot_with_reused_bounds(&bytes, 64, reuse_bounds).is_err()
        );
        for len in [0, 3, 15, 16, point_end, bytes.len() - 1] {
            assert!(
                CandidateKdTree::from_snapshot_with_reused_bounds(&bytes[..len], 65, reuse_bounds)
                    .is_err()
            );
        }
        let mut padded = bytes.clone();
        let node_count = read_u32(&bytes[12..16]) as usize;
        padded.resize(bytes.len() + (129 - node_count) * 24, 0);
        padded[12..16].copy_from_slice(&129_u32.to_le_bytes());
        assert!(
            CandidateKdTree::from_snapshot_with_reused_bounds(&padded, 65, reuse_bounds).is_err()
        );
        let mut extra = bytes.clone();
        extra.push(0);
        assert!(
            CandidateKdTree::from_snapshot_with_reused_bounds(&extra, 65, reuse_bounds).is_err()
        );
        for (at, value) in [
            (0, 0),
            (4, 2),
            (8, 0),
            (8, u32::MAX),
            (12, 0),
            (12, u32::MAX),
            (point_end, 65),
            (point_end + 4, read_u32(&bytes[point_end..point_end + 4])),
            (node_start, 1),
            (node_start + 4, 64),
            (node_start + 8, 0),
            (node_start + 8, u32::MAX),
            (node_start + 12, 3),
            (node_start + 12, 256),
            (node_start + 24, 1),
            (node_start + 24 + 4, 65),
        ] {
            let mut corrupt = bytes.clone();
            corrupt[at..at + 4].copy_from_slice(&value.to_le_bytes());
            assert!(
                CandidateKdTree::from_snapshot_with_reused_bounds(&corrupt, 65, reuse_bounds)
                    .is_err(),
                "word at {at}: {value}"
            );
        }
        let mut split = bytes.clone();
        split[node_start + 16..node_start + 24].copy_from_slice(&(-1.0_f64).to_le_bytes());
        assert!(
            CandidateKdTree::from_snapshot_with_reused_bounds(&split, 65, reuse_bounds).is_err()
        );
        let mut partition = bytes.clone();
        partition[16..24].copy_from_slice(&1000.0_f64.to_le_bytes());
        assert!(
            CandidateKdTree::from_snapshot_with_reused_bounds(&partition, 65, reuse_bounds)
                .is_err()
        );
        let leaf = tree.tree.nodes.iter().position(|n| n.axis == LEAF).unwrap();
        for (offset, value) in [(8, 1_u32), (16, 1)] {
            let mut corrupt = bytes.clone();
            let at = node_start + leaf * 24 + offset;
            corrupt[at..at + 4].copy_from_slice(&value.to_le_bytes());
            assert!(
                CandidateKdTree::from_snapshot_with_reused_bounds(&corrupt, 65, reuse_bounds)
                    .is_err()
            );
        }
    }
}

#[test]
fn radius_slots_and_import_preserve_original_hit_coordinates_in_order() {
    for count in [1, 17, 65, 2057] {
        let mut points = pseudo_random(count, 731);
        points[0] = [-0.0, 0.0, f64::MIN_POSITIVE];
        if count > 17 {
            points[17] = [f64::from_bits(0x7ff8_0000_0000_0123), f64::INFINITY, 0.0];
        }
        let tree = KdTree::new(&points).unwrap();
        let restored = KdTree::from_snapshot(&tree.snapshot(), count).unwrap();
        for query in [[0.0; 3], [0.3, -0.1, 0.2], [f64::NAN; 3]] {
            for radius in [-1.0, 0.0, 0.5, f64::INFINITY, f64::NAN] {
                for bounds in [false, true] {
                    let mut original = Vec::new();
                    tree.within_with_bounds(&query, radius, &mut original, bounds);
                    let expected: Vec<_> = original
                        .iter()
                        .map(|&i| points[i].map(f64::to_bits))
                        .collect();
                    for indexed in [&tree, &restored] {
                        let mut slots = Vec::new();
                        indexed.within_slots_with_bounds(&query, radius, &mut slots, bounds);
                        let actual: Vec<_> = slots
                            .iter()
                            .map(|&i| indexed.tree_points()[i].map(f64::to_bits))
                            .collect();
                        assert_eq!(actual, expected);
                    }
                }
            }
        }
    }
}

fn brute(points: &[[f64; 3]], q: &[f64; 3]) -> f64 {
    points
        .iter()
        .map(|p| (0..3).map(|a| (p[a] - q[a]).powi(2)).sum::<f64>())
        .fold(f64::INFINITY, f64::min)
}

fn pseudo_random(n: usize, seed: u64) -> Vec<[f64; 3]> {
    let mut s = seed;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    (0..n)
        .map(|_| [next() * 10.0, next() * 10.0, (next() * 4.0).floor()])
        .collect()
}

#[test]
fn matches_brute_force() {
    let points = pseudo_random(2000, 7);
    let tree = KdTree::new(&points).unwrap();
    for q in pseudo_random(300, 11) {
        let hit = tree.nearest(&q, None);
        assert_eq!(hit.distance_sq, brute(&points, &q));
        let p = points[hit.index];
        let d: f64 = (0..3).map(|a| (p[a] - q[a]).powi(2)).sum();
        assert_eq!(d, hit.distance_sq);
    }
}

#[test]
fn any_guess_yields_the_nearest() {
    let points = pseudo_random(500, 3);
    let tree = KdTree::new(&points).unwrap();
    let far = tree.nearest(&[100.0, 100.0, 100.0], None);
    for q in pseudo_random(50, 5) {
        assert_eq!(tree.nearest(&q, Some(far)).distance_sq, brute(&points, &q));
    }
}

#[test]
fn paired_nearest_keeps_scalar_hits_and_guess_order() {
    for count in [1, 2, 3, 15, 16, 17, 31, 33, 257, 2000] {
        let mut points = pseudo_random(count, 53);
        if count > 2 {
            points[1] = points[0];
            points[2] = points[0].map(|v| v + 1e-9);
        }
        let tree = KdTree::new(&points).unwrap();
        let mut scalar = None;
        let mut vector = None;
        for q in points.iter().copied().chain(pseudo_random(100, 57)) {
            let a = tree.nearest_with_simd(&q, scalar, false);
            let b = tree.nearest_with_simd(&q, vector, true);
            assert_eq!(a.index, b.index);
            assert_eq!(a.slot, b.slot);
            assert_eq!(a.distance_sq.to_bits(), b.distance_sq.to_bits());
            assert_eq!(b.distance_sq, brute(&points, &q));
            scalar = Some(a);
            vector = Some(b);
        }
    }
}

#[test]
fn nearest_radius_hints_keep_seed_ties_distance_bits_and_fallbacks() {
    let mut tightened = 0;
    for count in [1, 17, 33, 257, 2000] {
        for scale in [1.0, 1e-160, 1e155] {
            let mut points = pseudo_random(count, 53);
            for p in &mut points {
                *p = p.map(|v| v * scale);
            }
            if count > 2 {
                points[1] = points[0];
                points[2] = [0.0, -0.0, 0.0];
            }
            let tree = KdTree::new(&points).unwrap();
            let hints = NearestRadiusHints::new(&tree);
            let far = tree.nearest(&[100.0 * scale; 3], None);
            for q in points
                .iter()
                .take(19)
                .copied()
                .chain(
                    pseudo_random(19, 57)
                        .into_iter()
                        .map(|p| p.map(|v| v * scale)),
                )
                .chain([
                    [f64::NAN, 0.0, 0.0],
                    [f64::INFINITY, -0.0, 0.0],
                    [f64::MAX; 3],
                ])
            {
                for guess in [None, Some(far)] {
                    let expected = tree.nearest(&q, guess);
                    for candidate in [expected.slot, 0, count as u32 - 1, u32::MAX] {
                        for simd in [false, true] {
                            let original = tree.nearest_with_radius_layout::<false>(
                                &q, guess, candidate, &hints, simd,
                            );
                            let shared = tree.nearest_with_radius_layout::<true>(
                                &q, guess, candidate, &hints, simd,
                            );
                            assert_eq!(
                                (
                                    original.0.index,
                                    original.0.slot,
                                    original.0.distance_sq.to_bits(),
                                    original.1
                                ),
                                (
                                    shared.0.index,
                                    shared.0.slot,
                                    shared.0.distance_sq.to_bits(),
                                    shared.1
                                )
                            );
                            let (hit, tighter) = shared;
                            assert_eq!(
                                (hit.index, hit.slot, hit.distance_sq.to_bits()),
                                (
                                    expected.index,
                                    expected.slot,
                                    expected.distance_sq.to_bits()
                                )
                            );
                            tightened += usize::from(tighter);
                        }
                    }
                }
            }
        }
    }
    assert!(tightened > 0);
    for bad in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut points = pseudo_random(65, 11);
        points[2][0] = bad;
        let tree = KdTree::new(&points).unwrap();
        let hints = NearestRadiusHints::new(&tree);
        for q in pseudo_random(13, 17) {
            let expected = tree.nearest(&q, None);
            let (hit, tighter) =
                tree.nearest_with_radius_hint(&q, None, expected.slot, &hints, false);
            assert!(!tighter);
            assert_eq!(
                (hit.index, hit.slot, hit.distance_sq.to_bits()),
                (
                    expected.index,
                    expected.slot,
                    expected.distance_sq.to_bits()
                )
            );
        }
    }
}

#[test]
fn paired_leaf_scan_preserves_ties_nonfinite_values_and_odd_tails() {
    let points = [
        [1.0, 0.0, 0.0],
        [-1.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [f64::NAN, 0.0, 0.0],
        [f64::INFINITY, 1.0, 0.0],
        [1e300, 0.0, 0.0],
        [0.0, 0.0, -0.0],
    ];
    for end in 1..=points.len() as u32 {
        for q in [
            [0.0; 3],
            [2.0; 3],
            [1e-160; 3],
            [1e300; 3],
            [f64::NAN; 3],
            [f64::INFINITY; 3],
        ] {
            for seed in [0.0, 1.0, f64::INFINITY, f64::NAN] {
                let mut a = Nearest {
                    index: 0,
                    slot: u32::MAX,
                    distance_sq: seed,
                };
                let mut b = a;
                scan_candidate_leaf(&points, 0, end, &q, &mut a, false);
                scan_nearest_leaf(&points, 0, end, &q, &mut b);
                assert_eq!(a.slot, b.slot);
                assert_eq!(a.distance_sq.to_bits(), b.distance_sq.to_bits());
            }
        }
    }
}

#[test]
fn lane_minima_select_first_improvement_and_retain_tied_or_nan_seeds() {
    for (seed, distances, slots, expected_distance, expected_slot) in [
        (2.0, [1.0, 1.0], [4, 3], 1.0, 3),
        (2.0, [1.0, 1.0], [2, 3], 1.0, 2),
        (1.0, [1.0, 1.0], [2, 3], 1.0, 99),
        (2.0, [1.0, 0.5], [2, 3], 0.5, 3),
        (f64::INFINITY, [1.0, f64::INFINITY], [2, 3], 1.0, 2),
        (f64::INFINITY, [f64::INFINITY; 2], [2, 3], f64::INFINITY, 99),
        (f64::NAN, [0.0, 1.0], [2, 3], f64::NAN, 99),
        (1.0, [f64::NAN, 0.5], [2, 3], 0.5, 3),
    ] {
        let mut best = Nearest {
            index: 0,
            slot: 99,
            distance_sq: seed,
        };
        select_leaf_minimum(&mut best, distances, slots);
        assert_eq!(best.slot, expected_slot);
        assert_eq!(best.distance_sq.to_bits(), expected_distance.to_bits());
    }
}

#[test]
fn nearest_k_matches_brute_force() {
    let points = pseudo_random(1500, 13);
    let tree = KdTree::new(&points).unwrap();
    for q in pseudo_random(40, 17) {
        let got = tree.nearest_k(&q, 7);
        let mut all: Vec<f64> = points
            .iter()
            .map(|p| (0..3).map(|a| (p[a] - q[a]).powi(2)).sum())
            .collect();
        all.sort_by(f64::total_cmp);
        assert_eq!(
            got.iter().map(|g| g.1).collect::<Vec<_>>(),
            all[..7].to_vec()
        );
    }
    assert_eq!(tree.nearest_k(&[0.0; 3], 5000).len(), 1500);
}

#[test]
fn within_matches_brute_force() {
    let points = pseudo_random(2000, 21);
    let tree = KdTree::new(&points).unwrap();
    let mut found = Vec::new();
    for q in pseudo_random(30, 23) {
        tree.within(&q, 1.3, &mut found);
        found.sort_unstable();
        let expected: Vec<usize> = (0..points.len())
            .filter(|&i| (0..3).map(|a| (points[i][a] - q[a]).powi(2)).sum::<f64>() <= 1.3 * 1.3)
            .collect();
        assert_eq!(found, expected);
    }
}

#[test]
fn combined_radius_bounds_preserve_unsorted_hit_order_and_boundaries() {
    let mut samples = pseudo_random(2057, 31);
    samples.extend([[0.0; 3], [-0.0; 3], [3.0, 4.0, 0.0], [-3.0, -4.0, 0.0]]);
    for scale in [1e-150, 1.0, 1e150] {
        for offset in [0.0, 368_000.0] {
            let points: Vec<_> = samples
                .iter()
                .map(|p| p.map(|v| offset + v * scale))
                .collect();
            let tree = KdTree::new(&points).unwrap();
            let mut original = vec![usize::MAX];
            let mut bounded = vec![usize::MAX];
            for q in points
                .iter()
                .step_by(23)
                .chain(std::iter::once(&[offset; 3]))
            {
                let boundary = distance_sq(&points[37], q).sqrt();
                for radius in [
                    -1.0,
                    0.0,
                    scale * 0.01,
                    scale,
                    scale * 5.0,
                    boundary,
                    boundary.next_down(),
                    boundary.next_up(),
                    f64::INFINITY,
                    f64::NAN,
                ] {
                    tree.within(q, radius, &mut original);
                    tree.within_with_bounds(q, radius, &mut bounded, true);
                    assert_eq!(
                        bounded, original,
                        "scale {scale}, offset {offset}, radius {radius}"
                    );
                }
            }
        }
    }
    // Preserve the old path's behavior for nonfinite coordinates/queries,
    // squared-radius overflow and subnormal underflow as well.
    let points: Vec<_> = samples
        .into_iter()
        .chain([
            [f64::NAN, 1.0, 2.0],
            [f64::INFINITY, 0.0, 0.0],
            [f64::NEG_INFINITY, 0.0, 0.0],
            [f64::MAX; 3],
            [f64::from_bits(1); 3],
        ])
        .collect();
    let tree = KdTree::new(&points).unwrap();
    for q in points.iter().step_by(13).chain(points.iter().rev().take(5)) {
        for radius in [
            0.0,
            f64::from_bits(1),
            1.0,
            5.0,
            1e200,
            f64::INFINITY,
            f64::NAN,
        ] {
            let mut original = Vec::new();
            let mut bounded = Vec::new();
            tree.within(q, radius, &mut original);
            tree.within_with_bounds(q, radius, &mut bounded, true);
            assert_eq!(bounded, original);
        }
    }
}

#[test]
fn handles_duplicates_and_tiny_inputs() {
    let points = vec![[1.0, 2.0, 3.0]; 100];
    let tree = KdTree::new(&points).unwrap();
    assert_eq!(tree.nearest(&[1.0, 2.0, 4.0], None).distance_sq, 1.0);
    let single = KdTree::new(&[[0.0; 3]]).unwrap();
    assert_eq!(single.nearest(&[3.0, 4.0, 0.0], None).distance_sq, 25.0);
    assert!(KdTree::new(&[]).is_none());
}

#[test]
fn radius_visitor_preserves_hit_order_and_leaf_arithmetic() {
    for count in [1, 16, 17, 257, 2049] {
        for scale in [1e-150, 1.0, 1e150] {
            let mut points = pseudo_random(count, 83);
            for p in &mut points {
                *p = p.map(|v| v * scale);
            }
            points.extend([
                [0.0; 3],
                [-0.0; 3],
                [3.0 * scale, 4.0 * scale, 0.0],
                [f64::NAN, 1.0, 2.0],
                [f64::INFINITY, 0.0, 0.0],
                [f64::MAX; 3],
                [f64::from_bits(1); 3],
            ]);
            let tree = KdTree::new(&points).unwrap();
            for q in points.iter().step_by(71).chain(points.iter().rev().take(7)) {
                let boundary = distance_sq(&points[0], q).sqrt();
                for radius in [
                    -1.0,
                    -0.0,
                    f64::from_bits(1),
                    scale * 0.1,
                    scale,
                    boundary.next_down(),
                    boundary,
                    boundary.next_up(),
                    1e200,
                    f64::INFINITY,
                    f64::NAN,
                ] {
                    for bounds in [false, true] {
                        let mut expected = Vec::new();
                        tree.within_slots_with_bounds(q, radius, &mut expected, bounds);
                        let mut actual = Vec::new();
                        tree.visit_within_with_bounds(q, radius, bounds, &mut |slot, v, d| {
                            let p = tree.tree_points()[slot];
                            assert_eq!(
                                v.map(f64::to_bits),
                                [p[0] - q[0], p[1] - q[1], p[2] - q[2]].map(f64::to_bits)
                            );
                            assert_eq!(d.to_bits(), distance_sq(&p, q).to_bits());
                            actual.push(slot);
                        });
                        assert_eq!(actual, expected);
                    }
                }
            }
        }
    }
}

#[test]
fn shared_radius_search_preserves_each_independent_slot_sequence() {
    for count in [1, 16, 17, 257, 2049] {
        for scale in [1e-150, 1.0, 1e150] {
            let mut points = pseudo_random(count, 83);
            for p in &mut points {
                *p = p.map(|v| v * scale);
            }
            points.extend([
                [0.0; 3],
                [-0.0; 3],
                [3.0 * scale, 4.0 * scale, 0.0],
                [f64::NAN, 1.0, 2.0],
                [f64::INFINITY, 0.0, 0.0],
                [f64::NEG_INFINITY, 0.0, 0.0],
                [f64::MAX; 3],
                [f64::from_bits(1); 3],
            ]);
            let tree = KdTree::new(&points).unwrap();
            for q in points.iter().step_by(37).chain(points.iter().rev().take(8)) {
                let boundary = distance_sq(&points[0], q).sqrt();
                let radii = [
                    -1.0,
                    -0.0,
                    f64::from_bits(1),
                    scale * 0.1,
                    scale,
                    boundary.next_down(),
                    boundary,
                    boundary.next_up(),
                    1e200,
                    f64::INFINITY,
                    f64::NAN,
                ];
                for bounds in [false, true] {
                    for (i, &radius) in radii.iter().enumerate() {
                        for other in [radius, radii[(i + 3) % radii.len()]] {
                            let (mut a, mut b, mut combined_a, mut combined_b) =
                                (Vec::new(), Vec::new(), vec![usize::MAX], vec![usize::MAX]);
                            tree.within_slots_with_bounds(q, radius, &mut a, bounds);
                            tree.within_slots_with_bounds(q, other, &mut b, bounds);
                            tree.within_two_slots_with_bounds(
                                q,
                                [radius, other],
                                [&mut combined_a, &mut combined_b],
                                bounds,
                            );
                            assert_eq!(combined_a, a);
                            assert_eq!(combined_b, b);
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn untrusted_candidates_preserve_f64_distances() {
    let points: Vec<_> = pseudo_random(2000, 17)
        .into_iter()
        .map(|p| [p[0] + 368_000.0, p[1] + 3_955_000.0, p[2] * 1e-6])
        .collect();
    let tree = KdTree::new(&points).unwrap();
    for q in points.iter().step_by(13) {
        let q = [q[0] + 1e-4, q[1] - 2e-4, q[2] + 1e-7];
        let expected = tree.nearest(&q, None).distance_sq;
        for seed in [0, 19, 1999, 2000, u32::MAX] {
            assert_eq!(tree.nearest_seeded(&q, seed).distance_sq, expected);
        }
    }
    let (packed, nodes) = tree.candidate_buffers([368_000.0, 3_955_000.0, 0.0]);
    assert_eq!(packed.len(), 4 * points.len());
    assert_eq!(nodes.len(), 5 * tree.nodes.len());
    for (i, p) in tree.points.iter().enumerate() {
        assert_eq!(packed[4 * i], (p[0] - 368_000.0) as f32);
    }
}

#[test]
fn certified_candidates_match_brute_force_with_any_seed() {
    let points = pseudo_random(2049, 73);
    let tree = CandidateKdTree::new(&points).unwrap();
    let mut certified = 0;
    let mut full = 0;
    for q in pseudo_random(300, 79) {
        let expected = brute(&points, &q);
        let correct = tree.tree.nearest(&q, None).slot;
        for seed in [0, 511, 2048, 2049, u32::MAX, correct] {
            let (hit, fast) = tree.nearest(&q, seed);
            assert_eq!(hit.distance_sq, expected);
            assert_eq!(distance_sq(&points[hit.index], &q), expected);
            let (bounded, _) = tree.nearest_bounded(&q, seed);
            assert_eq!(bounded.distance_sq, expected);
            assert_eq!(distance_sq(&points[bounded.index], &q), expected);
            certified += usize::from(fast);
            full += usize::from(!fast);
        }
    }
    assert!(certified > 0);
    assert!(full > 0);
}

#[test]
fn certificates_preserve_boundaries_large_coordinates_and_duplicates() {
    let mut points: Vec<_> = (0..4096)
        .map(|i| {
            [
                368_000.0 + (i % 64) as f64 * 0.1,
                3_955_000.0 + (i / 64) as f64 * 0.1,
                40.0,
            ]
        })
        .collect();
    points.extend_from_within(..17);
    let tree = CandidateKdTree::new(&points).unwrap();
    for leaf in &tree.leaves {
        // On either side of every certificate face, and exactly on it.
        for axis in 0..3 {
            for face in [leaf.lower[axis], leaf.upper[axis]] {
                if !face.is_finite() {
                    continue;
                }
                for coordinate in [
                    face - 1e-6,
                    face.next_down(),
                    face,
                    face.next_up(),
                    face + 1e-6,
                ] {
                    let mut q = tree.tree.points[leaf.start as usize];
                    q[axis] = coordinate;
                    q[2] += 9e-6;
                    assert_eq!(
                        tree.nearest(&q, leaf.start).0.distance_sq,
                        tree.tree.nearest(&q, None).distance_sq
                    );
                    assert_eq!(
                        tree.nearest_bounded(&q, leaf.start).0.distance_sq,
                        tree.tree.nearest(&q, None).distance_sq
                    );
                }
            }
        }
    }
    // A distant/wrong seed must recover the nearer leaf via full search.
    let q = [368_000.001, 3_955_000.002, 40.000_009];
    let (hit, fast) = tree.nearest(&q, 4095);
    assert!(!fast);
    assert_eq!(hit.distance_sq, brute(&points, &q));

    let duplicates = CandidateKdTree::new(&vec![[6_378_137.000_01, 0.0, 0.0]; 20_000]).unwrap();
    let q = [6_378_137.000_009, 0.0, 0.0];
    let (hit, fast) = duplicates.nearest(&q, 19_999);
    assert!(fast);
    assert_eq!(hit.distance_sq, distance_sq(&duplicates.tree.points[0], &q));
    // Overflow cannot be used as a certificate.
    assert!(!duplicates.nearest(&[1e300; 3], 0).1);
    assert!(!duplicates.nearest_bounded(&[1e300; 3], 0).1);
}

#[test]
fn sibling_boxes_certify_diagonal_distance_and_recover_wrong_leaves() {
    let points: Vec<_> = (0..32).map(|i| [i as f64, (i % 2) as f64, 0.0]).collect();
    let tree = CandidateKdTree::new(&points).unwrap();
    let query = [15.5, 10.0, 0.0];
    let slot = tree.tree.nearest(&query, None).slot;
    assert!(!tree.nearest(&query, slot).1);
    let (hit, certified) = tree.nearest_bounded(&query, slot);
    assert!(certified);
    assert_eq!(hit.distance_sq, brute(&points, &query));
    // Equal box bound is safe, as are its adjacent representable values.
    for x in [query[0].next_down(), query[0], query[0].next_up()] {
        for seed in 0..32 {
            let q = [x, 10.0, 0.0];
            assert_eq!(
                tree.nearest_bounded(&q, seed).0.distance_sq,
                brute(&points, &q)
            );
        }
    }
    // Exercise searches spanning multiple ancestor siblings, with gaps,
    // coincident subtrees, negative axes and near-overflow coordinates.
    for scale in [1e-150, 1.0, 1e150] {
        let positions: Vec<_> = (0..257)
            .map(|i| {
                [
                    ((i % 17) as f64 - 8.0) * scale,
                    ((i / 17) as f64 - 7.0) * scale,
                    (i % 3) as f64 * scale,
                ]
            })
            .collect();
        let tree = CandidateKdTree::new(&positions).unwrap();
        for q in [
            [-8.3 * scale, -7.7 * scale, -scale],
            [9.1 * scale, 7.8 * scale, 2.4 * scale],
            [0.1 * scale; 3],
        ] {
            let expected = brute(&positions, &q);
            for seed in [0, 17, 128, 256, u32::MAX] {
                assert_eq!(tree.nearest_bounded(&q, seed).0.distance_sq, expected);
            }
        }
    }
}
