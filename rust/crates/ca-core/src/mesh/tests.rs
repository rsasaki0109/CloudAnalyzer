use super::*;

#[cfg(feature = "parallel")]
#[test]
fn projection_dot_pairs_preserve_ordered_bits() {
    let values = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        f64::MAX,
        -f64::MAX,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::from_bits(0x7ff8_0000_0000_1234),
        f64::from_bits(0xfff8_0000_0000_5678),
        3955000.0,
        1e154,
    ];
    let check = |a, b, v| {
        let (x, y) = projection_dot_pair(a, b, v);
        assert_eq!(x.to_bits(), dot(a, v).to_bits());
        assert_eq!(y.to_bits(), dot(b, v).to_bits());
    };
    for axis in 0..3 {
        for &x in &values {
            for &y in &values {
                for &z in &values {
                    let (mut a, mut b, mut v) =
                        ([0.5, -1.5, 2.5], [-2.0, 3.0, -4.0], [1.25, 2.25, -3.25]);
                    a[axis] = x;
                    b[axis] = y;
                    v[axis] = z;
                    check(a, b, v);
                }
            }
        }
    }
    let mut state = 53u64;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        f64::from_bits(state)
    };
    for _ in 0..16384 {
        check(
            std::array::from_fn(|_| next()),
            std::array::from_fn(|_| next()),
            std::array::from_fn(|_| next()),
        );
    }
}

#[cfg(feature = "parallel")]
#[test]
fn paired_projection_preserves_region_and_extreme_point_bits() {
    for seed in 1..=8 {
        for (scale, origin) in [
            (1.0, [0.0; 3]),
            (0.01, [368000.0, 3955000.0, 40.0]),
            (1e-160, [0.0; 3]),
            (1e60, [0.0; 3]),
            (1e140, [0.0; 3]),
            (1e200, [0.0; 3]),
        ] {
            let vertices: Vec<_> = pseudo_random(12, seed)
                .into_iter()
                .map(|p| std::array::from_fn(|a| p[a] * scale + origin[a]))
                .collect();
            let faces = [
                [0, 1, 2],
                [2, 1, 0],
                [3, 4, 5],
                [6, 7, 8],
                [9, 10, 11],
                [0, 0, 0],
                [0, 1, 0],
            ];
            let mut queries: Vec<_> = pseudo_random(32, seed + 99)
                .into_iter()
                .map(|p| std::array::from_fn(|a| p[a] * scale + origin[a]))
                .collect();
            queries.extend(vertices.iter().take(3));
            queries.extend([
                [f64::from_bits(0x7ff8_0000_0000_1234), 0.0, 0.0],
                [f64::INFINITY; 3],
                [f64::NEG_INFINITY; 3],
                [1e308; 3],
                [0.0, -0.0, 0.0],
                [-0.0, 0.0, -0.0],
            ]);
            for face in faces {
                let [a, b, c] = face.map(|i| vertices[i]);
                for &p in &queries {
                    assert_eq!(
                        closest_point_on_triangle(p, a, b, c).map(f64::to_bits),
                        closest_point_with_paired_dots(p, a, b, c).map(f64::to_bits),
                    );
                }
            }
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn paired_box_bounds_preserve_scalar_bits() {
    let values = [
        0.0,
        -0.0,
        1.0,
        -1.0,
        f64::from_bits(1),
        -f64::from_bits(1),
        f64::MIN_POSITIVE,
        -f64::MIN_POSITIVE,
        f64::MAX,
        -f64::MAX,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::from_bits(0x7ff8_0000_0000_1234),
        f64::from_bits(0xfff8_0000_0000_5678),
        3955000.0,
        1e154,
    ];
    let check = |left: &BvhNode, right: &BvhNode, p| {
        let (a, b) = box_distance_sq_pair(left, right, p);
        assert_eq!(a.to_bits(), box_distance_sq(left, p).to_bits());
        assert_eq!(b.to_bits(), box_distance_sq(right, p).to_bits());
        let (a, b) = box_distance_sq_pair_clamped(left, right, p);
        assert_eq!(a.to_bits(), box_distance_sq(left, p).to_bits());
        assert_eq!(b.to_bits(), box_distance_sq(right, p).to_bits());
    };
    let mut left = BvhNode {
        min: [-3.0, -5.0, -7.0],
        max: [2.0, 4.0, 6.0],
        start: 0,
        count: 0,
    };
    let mut right = BvhNode {
        min: [0.0, 1.0, 2.0],
        max: [3.0, 4.0, 5.0],
        ..left
    };
    let (initial_left, initial_right) = (left, right);
    for axis in 0..3 {
        left = initial_left;
        right = initial_right;
        for &lo in &values {
            for &hi in &values {
                for &other_lo in &values {
                    for &other_hi in &values {
                        for &q in &values {
                            left.min[axis] = lo;
                            left.max[axis] = hi;
                            right.min[axis] = other_lo;
                            right.max[axis] = other_hi;
                            let mut p = [0.5, 1.5, 2.5];
                            p[axis] = q;
                            check(&left, &right, p);
                        }
                    }
                }
            }
        }
    }
    let mut state = 53u64;
    let mut next = || {
        state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
        f64::from_bits(state)
    };
    for _ in 0..16384 {
        left.min = std::array::from_fn(|_| next());
        left.max = std::array::from_fn(|_| next());
        right.min = std::array::from_fn(|_| next());
        right.max = std::array::from_fn(|_| next());
        check(&left, &right, std::array::from_fn(|_| next()));
    }
}

fn node_bits(nodes: &[BvhNode]) -> Vec<([u64; 3], [u64; 3], u32, u32)> {
    nodes
        .iter()
        .map(|n| {
            (
                n.min.map(f64::to_bits),
                n.max.map(f64::to_bits),
                n.start,
                n.count,
            )
        })
        .collect()
}

fn test_build(mesh: &TriangleMesh, merge: bool) -> (Vec<BvhNode>, Vec<(u32, [f64; 3])>) {
    let mut items: Vec<_> = (0..mesh.triangles.len() as u32)
        .map(|t| {
            let [a, b, c] = mesh.corners(t as usize);
            (t, std::array::from_fn(|i| (a[i] + b[i] + c[i]) / 3.0))
        })
        .collect();
    let mut nodes = Vec::with_capacity(2 * items.len() / BVH_LEAF + 1);
    build(mesh, &mut items, 0, &mut nodes, merge);
    (nodes, items)
}

#[test]
fn merged_bounds_preserve_every_node_bit_partition_and_capacity() {
    for seed in 1..=8 {
        for n in [1, 3, 4, 5, 31, 129, 1025] {
            for (scale, origin) in [
                (1.0, [10.0; 3]),
                (0.01, [368000.0, 3955000.0, 40.0]),
                (1e-160, [0.0; 3]),
                (1e140, [0.0; 3]),
            ] {
                let vertices = pseudo_random(n * 3, seed)
                    .into_iter()
                    .map(|p| std::array::from_fn(|a| (p[a] + 10.0) * scale + origin[a]))
                    .collect();
                let mut triangles: Vec<_> = (0..n as u32)
                    .map(|t| [3 * t, 3 * t + 1, 3 * t + 2])
                    .collect();
                triangles.extend([[0, 0, 0], [2, 1, 0]]);
                let mesh = TriangleMesh {
                    vertices,
                    triangles,
                };
                assert!(can_merge_bounds(&mesh));
                let (old, a) = test_build(&mesh, false);
                let (new, b) = test_build(&mesh, true);
                assert_eq!(node_bits(&old), node_bits(&new));
                assert_eq!(
                    a.iter()
                        .map(|&(t, c)| (t, c.map(f64::to_bits)))
                        .collect::<Vec<_>>(),
                    b.iter()
                        .map(|&(t, c)| (t, c.map(f64::to_bits)))
                        .collect::<Vec<_>>()
                );
                assert_eq!(old.capacity(), new.capacity());
                assert_eq!(a.capacity(), b.capacity());
            }
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn preorder_node_count_matches_recursive_leaf_policy() {
    fn count(n: usize) -> usize {
        if n <= BVH_LEAF {
            1
        } else {
            1 + count(n / 2) + count(n - n / 2)
        }
    }
    for n in (1..=8193).chain([65_535, 65_536, 65_537, 131_071, 131_072, 131_073]) {
        assert_eq!(bvh_node_count(n), count(n), "triangles={n}");
    }
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_builder_preserves_node_bits_partitions_and_capacity() {
    for threads in [1, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        for n in [
            1, 4, 5, 9, 10, 1025, 32_767, 32_768, 32_769, 65_535, 65_536, 65_537,
        ] {
            for mode in 0..3 {
                let mut vertices: Vec<[f64; 3]> = pseudo_random(n * 3, 19)
                    .into_iter()
                    .map(|p| [p[0] + 368000.0, p[1] + 3955000.0, p[2] + 40.0])
                    .collect();
                if mode == 1 {
                    vertices[0][2] = -0.0;
                } else if mode == 2 {
                    vertices[0] = [f64::NAN, f64::INFINITY, f64::NEG_INFINITY];
                }
                let mesh = TriangleMesh {
                    vertices,
                    triangles: (0..n as u32)
                        .map(|t| {
                            if t % 37 == 0 {
                                [3 * t; 3]
                            } else if t % 19 == 0 {
                                [3 * t + 2, 3 * t + 1, 3 * t]
                            } else {
                                [3 * t, 3 * t + 1, 3 * t + 2]
                            }
                        })
                        .collect(),
                };
                let merge = can_merge_bounds(&mesh);
                let (old, a) = test_build(&mesh, merge);
                let mut b: Vec<_> = (0..mesh.triangles.len() as u32)
                    .map(|t| {
                        let [a, b, c] = mesh.corners(t as usize);
                        (t, std::array::from_fn(|i| (a[i] + b[i] + c[i]) / 3.0))
                    })
                    .collect();
                let mut new = Vec::with_capacity(2 * b.len() / BVH_LEAF + 1);
                new.resize(bvh_node_count(b.len()), empty_node());
                pool.install(|| {
                    build_parallel(
                        &mesh,
                        &mut b,
                        0,
                        0,
                        &mut new,
                        merge,
                        USE_MESH_PAR_CENTROID_BOUNDS,
                    )
                });
                assert_eq!(node_bits(&old), node_bits(&new), "n={n}, mode={mode}");
                assert_eq!(
                    a.iter()
                        .map(|&(t, c)| (t, c.map(f64::to_bits)))
                        .collect::<Vec<_>>(),
                    b.iter()
                        .map(|&(t, c)| (t, c.map(f64::to_bits)))
                        .collect::<Vec<_>>()
                );
                assert_eq!(old.capacity(), new.capacity());
                assert_eq!(a.capacity(), b.capacity());
            }
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_centroid_bounds_preserve_extrema_and_axis_bits() {
    let pools: Vec<_> = [1, 2, 4]
        .map(|threads| {
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
        })
        .into_iter()
        .collect();
    let bits = |(lo, hi): ([f64; 3], [f64; 3])| (lo.map(f64::to_bits), hi.map(f64::to_bits));
    let axis = |(lo, hi): ([f64; 3], [f64; 3])| {
        (0..3)
            .max_by(|&a, &b| (hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b])))
            .unwrap()
    };
    for n in [
        0, 1, 16_383, 16_384, 16_385, 262_143, 262_144, 262_145, 524_287, 524_288, 524_289,
    ] {
        for mode in 0..6 {
            let mut points = pseudo_random(n, 53);
            for (i, p) in points.iter_mut().enumerate() {
                let sign = if i % 2 == 0 { 1.0 } else { -1.0 };
                *p = match mode {
                    0 => [p[0] + 368000.0, p[1] + 3955000.0, p[2] + 40.0],
                    1 => [1.0, 2.0, 3.0],
                    2 => [sign * 7.0; 3],
                    3 => [sign * 0.0, -sign * 0.0, sign],
                    4 => match i % 4 {
                        0 => [
                            f64::from_bits(0x7ff8_0000_0000_1234),
                            f64::INFINITY,
                            f64::NEG_INFINITY,
                        ],
                        1 => [1.0, -2.0, 3.0],
                        2 => [
                            f64::NEG_INFINITY,
                            f64::from_bits(0x7ff8_0000_0000_5678),
                            f64::INFINITY,
                        ],
                        _ => [f64::NAN; 3],
                    },
                    _ => [sign * 1e308, -sign * 1e308, sign * f64::from_bits(1)],
                };
            }
            let items: Vec<_> = points
                .into_iter()
                .enumerate()
                .map(|(i, p)| (i as u32, p))
                .collect();
            let before = centroid_bounds_serial(&items);
            let item_capacity = items.capacity();
            for pool in &pools {
                let after = pool.install(|| centroid_bounds_parallel(&items));
                assert_eq!(bits(before), bits(after), "n={n} mode={mode}");
                assert_eq!(axis(before), axis(after));
                assert_eq!(items.capacity(), item_capacity);
            }
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn parallel_centroid_scans_preserve_full_nodes_partitions_and_capacity() {
    for threads in [1, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        for n in [262_143, 262_144, 262_145, 524_288] {
            for mode in 0..2 {
                let mut vertices: Vec<[f64; 3]> = pseudo_random(n * 3, 19)
                    .into_iter()
                    .map(|p| [p[0] + 368000.0, p[1] + 3955000.0, p[2] + 40.0])
                    .collect();
                if mode == 1 {
                    // Nonzero finite corners whose original centroid is zero.
                    vertices[3][0] = 5.0;
                    vertices[4][0] = -10.0;
                    vertices[5][0] = 5.0;
                }
                let mesh = TriangleMesh {
                    vertices,
                    triangles: (0..n as u32)
                        .map(|t| {
                            if t % 37 == 0 {
                                [3 * t; 3]
                            } else if t % 19 == 0 {
                                [3 * t + 2, 3 * t + 1, 3 * t]
                            } else {
                                [3 * t, 3 * t + 1, 3 * t + 2]
                            }
                        })
                        .collect(),
                };
                let merge = can_merge_bounds(&mesh);
                let (old, a) = test_build(&mesh, merge);
                let mut b: Vec<_> = (0..mesh.triangles.len() as u32)
                    .map(|t| {
                        let [a, b, c] = mesh.corners(t as usize);
                        (t, std::array::from_fn(|i| (a[i] + b[i] + c[i]) / 3.0))
                    })
                    .collect();
                let mut new = Vec::with_capacity(2 * b.len() / BVH_LEAF + 1);
                new.resize(bvh_node_count(b.len()), empty_node());
                pool.install(|| build_parallel(&mesh, &mut b, 0, 0, &mut new, merge, true));
                assert_eq!(node_bits(&old), node_bits(&new), "n={n}, mode={mode}");
                assert_eq!(
                    a.iter()
                        .map(|&(t, c)| (t, c.map(f64::to_bits)))
                        .collect::<Vec<_>>(),
                    b.iter()
                        .map(|&(t, c)| (t, c.map(f64::to_bits)))
                        .collect::<Vec<_>>()
                );
                assert_eq!(old.capacity(), new.capacity());
                assert_eq!(a.capacity(), b.capacity());
            }
        }
    }
}

#[test]
fn zero_and_nonfinite_meshes_keep_original_bounds() {
    for value in [0.0, -0.0, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        let mut vertices = pseudo_random(39, 7);
        vertices[0][0] = value;
        let mesh = TriangleMesh {
            vertices,
            triangles: (0..13).map(|t| [3 * t, 3 * t + 1, 3 * t + 2]).collect(),
        };
        assert!(!can_merge_bounds(&mesh));
        let (old, items) = test_build(&mesh, false);
        let storage = BvhStorage::new(&mesh).unwrap();
        assert_eq!(node_bits(&old), node_bits(&storage.nodes));
        assert_eq!(
            items.iter().map(|&(t, _)| t).collect::<Vec<_>>(),
            storage.order
        );
    }
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
        .map(|_| {
            [
                next() * 10.0 - 5.0,
                next() * 10.0 - 5.0,
                next() * 10.0 - 5.0,
            ]
        })
        .collect()
}

#[test]
fn query_order_reuse_preserves_bits_and_lru_budget() {
    let mesh = PreparedMesh::new(TriangleMesh {
        vertices: vec![[0.0; 3], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        triangles: vec![[0, 1, 2], [2, 1, 0]],
    })
    .unwrap();
    let width = std::mem::size_of::<[f64; 3]>() + std::mem::size_of::<usize>();
    let mut cache = MeshQueryOrder::new(3 * width * 2);
    let queries = |offset| vec![[0.25, 0.25, offset], [0.5, 0.5, offset], [2.0, 1.0, offset]];
    let bits = |values: Vec<f64>| values.into_iter().map(f64::to_bits).collect::<Vec<_>>();
    for (offset, expected_hit) in [
        (0.1, false),
        (0.2, false),
        (0.1, true),
        (0.3, false),
        (0.1, true),
        (0.2, false),
    ] {
        for signed in [false, true] {
            let points = queries(offset);
            assert_eq!(
                bits(cache.distances(&mesh, points.clone(), signed)),
                bits(mesh.distances(&points, signed))
            );
            assert_eq!(cache.hit(), expected_hit || signed);
            assert!(cache.retained_bytes() <= 6 * width);
        }
    }
    cache.clear();
    assert!(!cache.hit());
    assert_eq!(cache.retained_bytes(), 0);
    assert_eq!(cache.entry_count(), 0);
    let mut limited = MeshQueryOrder::new(width);
    assert_eq!(
        bits(limited.distances(&mesh, queries(0.1), true)),
        bits(mesh.distances(&queries(0.1), true))
    );
    assert!(!limited.hit());
    assert_eq!(limited.retained_bytes(), 0);
    assert!(limited.distances(&mesh, Vec::new(), false).is_empty());
    let mut eight = MeshQueryOrder::new(usize::MAX);
    for i in 0..9 {
        eight.distances(&mesh, queries(i as f64), false);
    }
    assert_eq!(eight.entry_count(), 8);
    eight.distances(&mesh, queries(0.0), true);
    assert!(!eight.hit());
}

#[test]
fn query_order_keys_preserve_signed_zero_and_nan_payloads() {
    let points = vec![[0.0, -0.0, f64::from_bits(0x7ff8_0000_0000_0001)]];
    let entry = QueryOrder {
        points: points.clone().into_boxed_slice(),
        order: vec![0].into_boxed_slice(),
    };
    assert!(entry.matches(&points));
    assert!(!entry.matches(&[[0.0, 0.0, points[0][2]]]));
    assert!(!entry.matches(&[[0.0, -0.0, f64::from_bits(0x7ff8_0000_0000_0002)]]));
    assert!(!entry.matches(&[]));
}

#[test]
fn cached_stack_bounds_keep_hit_bits_order_and_signed_ties() {
    let bits = |h: MeshHit| {
        (
            h.triangle,
            h.point.map(f64::to_bits),
            h.distance_sq.to_bits(),
        )
    };
    for seed in 1..=12 {
        for (scale, origin) in [
            (1.0, [0.0; 3]),
            (0.01, [368000.0, 3955000.0, 40.0]),
            (1e-160, [0.0; 3]),
            (1e60, [0.0; 3]),
            (1e140, [0.0; 3]),
        ] {
            let vertices: Vec<_> = pseudo_random(291, seed)
                .into_iter()
                .map(|p| std::array::from_fn(|a| p[a] * scale + origin[a]))
                .collect();
            let mut triangles: Vec<_> = (0..97u32).map(|t| [3 * t, 3 * t + 1, 3 * t + 2]).collect();
            triangles.extend((0..17u32).map(|t| [3 * t + 2, 3 * t + 1, 3 * t]));
            triangles.extend([[0, 0, 0], [0, 1, 0]]);
            let mesh = TriangleMesh {
                vertices,
                triangles,
            };
            let bvh = MeshBvh::new(&mesh).unwrap();
            let mut queries: Vec<_> = pseudo_random(103, seed + 100)
                .into_iter()
                .map(|p| std::array::from_fn(|a| p[a] * scale + origin[a]))
                .collect();
            queries.extend(mesh.vertices.iter().take(7));
            let (mut old_stack, mut new_stack) = (Vec::new(), Vec::new());
            for with_guess in [false, true] {
                let mut guess = None;
                for &p in &queries {
                    let old = bvh.nearest_with(p, guess, &mut old_stack);
                    let new = bvh.nearest_with_cached_bounds(p, guess, &mut new_stack);
                    assert_eq!(bits(old), bits(new));
                    if old.triangle != usize::MAX {
                        let signed = |h: MeshHit| {
                            let d = h.distance_sq.sqrt();
                            if dot(sub(p, h.point), mesh.normal(h.triangle)) < 0.0 {
                                -d
                            } else {
                                d
                            }
                        };
                        assert_eq!(signed(old).to_bits(), signed(new).to_bits());
                        guess = with_guess.then_some(old.triangle);
                    }
                }
            }
            for p in [
                [f64::NAN, 0.0, 0.0],
                [f64::INFINITY; 3],
                [f64::NEG_INFINITY; 3],
            ] {
                assert_eq!(
                    bits(bvh.nearest_with(p, None, &mut old_stack)),
                    bits(bvh.nearest_with_cached_bounds(p, None, &mut new_stack))
                );
            }
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn unique_seed_preserves_hit_bits_signs_and_missing_queries() {
    let bits = |h: MeshHit| {
        (
            h.triangle,
            h.point.map(f64::to_bits),
            h.distance_sq.to_bits(),
        )
    };
    for seed in 1..=8 {
        for (scale, origin) in [
            (1.0, [0.0; 3]),
            (0.01, [368000.0, 3955000.0, 40.0]),
            (1e-160, [0.0; 3]),
            (1e60, [0.0; 3]),
            (1e140, [0.0; 3]),
            (1e200, [0.0; 3]),
        ] {
            let mut vertices: Vec<_> = pseudo_random(291, seed)
                .into_iter()
                .map(|p| std::array::from_fn(|a| p[a] * scale + origin[a]))
                .collect();
            if seed % 4 == 0 {
                vertices[0] = [f64::NAN, f64::INFINITY, -0.0];
            }
            let mut triangles: Vec<_> = (0..97u32).map(|t| [3 * t, 3 * t + 1, 3 * t + 2]).collect();
            triangles.extend((0..17u32).map(|t| [3 * t + 2, 3 * t + 1, 3 * t]));
            triangles.extend([[0, 0, 0], [0, 1, 0]]);
            let mesh = TriangleMesh {
                vertices,
                triangles,
            };
            let bvh = MeshBvh::new(&mesh).unwrap();
            let mut queries: Vec<_> = pseudo_random(103, seed + 100)
                .into_iter()
                .map(|p| std::array::from_fn(|a| p[a] * scale + origin[a]))
                .collect();
            queries.extend(mesh.vertices.iter().take(7));
            queries.extend([
                [f64::from_bits(0x7ff8_0000_0000_1234), 0.0, 0.0],
                [f64::INFINITY; 3],
                [f64::NEG_INFINITY; 3],
                [1e308; 3],
                [0.0, -0.0, 0.0],
                [-0.0, 0.0, -0.0],
            ]);
            let (mut old_stack, mut new_stack) = (Vec::new(), Vec::new());
            for policy in 0..4 {
                let mut previous = None;
                for (i, &p) in queries.iter().enumerate() {
                    let guess = match policy {
                        0 => None,
                        1 => previous,
                        2 => Some(i % mesh.triangles.len()),
                        _ => Some(0),
                    };
                    let old = bvh.nearest_with_cached_bounds(p, guess, &mut old_stack);
                    let new = bvh.nearest_with_cached_bounds_unique(p, guess, &mut new_stack);
                    assert_eq!(bits(old), bits(new));
                    if old.triangle != usize::MAX {
                        let signed = |h: MeshHit| {
                            let d = h.distance_sq.sqrt();
                            if dot(sub(p, h.point), mesh.normal(h.triangle)) < 0.0 {
                                -d
                            } else {
                                d
                            }
                        };
                        assert_eq!(signed(old).to_bits(), signed(new).to_bits());
                        previous = Some(old.triangle);
                    } else {
                        previous = None;
                    }
                    assert_eq!(old_stack.capacity(), new_stack.capacity());
                }
            }
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn indexed_corners_preserve_hit_bits_signs_and_missing_queries() {
    let bits = |h: MeshHit| {
        (
            h.triangle,
            h.point.map(f64::to_bits),
            h.distance_sq.to_bits(),
        )
    };
    for seed in 1..=8 {
        for (scale, origin) in [
            (1.0, [0.0; 3]),
            (0.01, [368000.0, 3955000.0, 40.0]),
            (1e-160, [0.0; 3]),
            (1e60, [0.0; 3]),
            (1e140, [0.0; 3]),
            (1e200, [0.0; 3]),
        ] {
            let mut vertices: Vec<_> = pseudo_random(291, seed)
                .into_iter()
                .map(|p| std::array::from_fn(|a| p[a] * scale + origin[a]))
                .collect();
            if seed % 4 == 0 {
                vertices[0] = [f64::NAN, f64::INFINITY, -0.0];
            }
            let mut triangles: Vec<_> = (0..97u32).map(|t| [3 * t, 3 * t + 1, 3 * t + 2]).collect();
            triangles.extend((0..17u32).map(|t| [3 * t + 2, 3 * t + 1, 3 * t]));
            triangles.extend([[0, 0, 0], [0, 1, 0]]);
            let mesh = TriangleMesh {
                vertices,
                triangles,
            };
            let bvh = MeshBvh::new(&mesh).unwrap();
            let indexed = IndexedMeshBvh::new(&mesh).unwrap();
            assert_eq!(
                node_bits(&bvh.storage.nodes),
                node_bits(&indexed.storage.nodes)
            );
            assert_eq!(bvh.storage.order, indexed.storage.order);
            assert_eq!(
                bvh.storage.nodes.capacity(),
                indexed.storage.nodes.capacity()
            );
            assert_eq!(
                bvh.storage.order.capacity(),
                indexed.storage.order.capacity()
            );
            assert_eq!(bvh.storage.corners.capacity(), mesh.triangles.len());
            assert_eq!(indexed.storage.corners.capacity(), 0);
            assert_eq!(
                bvh.storage.storage_bytes() - indexed.storage.storage_bytes(),
                72 * mesh.triangles.len()
            );
            for (slot, &t) in indexed.storage.order.iter().enumerate() {
                assert_eq!(
                    bvh.storage.corners[slot].map(|p| p.map(f64::to_bits)),
                    mesh.corners(t as usize).map(|p| p.map(f64::to_bits))
                );
            }
            let mut queries: Vec<_> = pseudo_random(103, seed + 100)
                .into_iter()
                .map(|p| std::array::from_fn(|a| p[a] * scale + origin[a]))
                .collect();
            queries.extend(mesh.vertices.iter().take(7));
            queries.extend([
                [f64::from_bits(0x7ff8_0000_0000_1234), 0.0, 0.0],
                [f64::INFINITY; 3],
                [f64::NEG_INFINITY; 3],
                [1e308; 3],
                [0.0, -0.0, 0.0],
                [-0.0, 0.0, -0.0],
            ]);
            let (mut old_stack, mut new_stack) = (Vec::new(), Vec::new());
            for policy in 0..4 {
                let mut previous = None;
                for (i, &p) in queries.iter().enumerate() {
                    let guess = match policy {
                        0 => None,
                        1 => previous,
                        2 => Some(i % mesh.triangles.len()),
                        _ => Some(0),
                    };
                    let old = bvh.nearest_with_cached_bounds(p, guess, &mut old_stack);
                    let new = indexed.nearest(p, guess, &mut new_stack);
                    assert_eq!(bits(old), bits(new));
                    if old.triangle != usize::MAX {
                        let signed = |h: MeshHit| {
                            let d = h.distance_sq.sqrt();
                            if dot(sub(p, h.point), mesh.normal(h.triangle)) < 0.0 {
                                -d
                            } else {
                                d
                            }
                        };
                        assert_eq!(signed(old).to_bits(), signed(new).to_bits());
                        previous = Some(old.triangle);
                    } else {
                        previous = None;
                    }
                    assert_eq!(old_stack.capacity(), new_stack.capacity());
                }
            }
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn indexed_corners_preserve_large_preparation_nodes_order_and_payload() {
    for threads in [1, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        for n in [65_535, 65_536, 65_537] {
            for mode in 0..2 {
                let mut mesh = TriangleMesh::default();
                for t in 0..n {
                    let x = 368000.0 + (t % 256) as f64 * 2.0;
                    let y = 3955000.0 + (t / 256) as f64 * 2.0;
                    mesh.vertices
                        .extend([[x, y, 40.0], [x + 1.0, y, 40.0], [x, y + 1.0, 40.0]]);
                    mesh.triangles
                        .push([(3 * t) as u32, (3 * t + 1) as u32, (3 * t + 2) as u32]);
                }
                if mode == 1 {
                    mesh.vertices[0] = [f64::from_bits(0x7ff8_0000_0000_1234), -0.0, f64::INFINITY];
                }
                let old = pool.install(|| MeshBvh::new(&mesh).unwrap());
                let new = pool.install(|| IndexedMeshBvh::new(&mesh).unwrap());
                assert_eq!(node_bits(&old.storage.nodes), node_bits(&new.storage.nodes));
                assert_eq!(old.storage.order, new.storage.order);
                assert_eq!(old.storage.nodes.capacity(), new.storage.nodes.capacity());
                assert_eq!(old.storage.order.capacity(), new.storage.order.capacity());
                assert_eq!(old.storage.corners.capacity(), n);
                assert_eq!(new.storage.corners.capacity(), 0);
                assert_eq!(
                    old.storage.storage_bytes() - new.storage.storage_bytes(),
                    72 * n
                );
                for (slot, &t) in new.storage.order.iter().enumerate() {
                    assert_eq!(
                        old.storage.corners[slot].map(|p| p.map(f64::to_bits)),
                        mesh.corners(t as usize).map(|p| p.map(f64::to_bits))
                    );
                }
                let (mut old_stack, mut new_stack, mut guess) = (Vec::new(), Vec::new(), None);
                for t in [0, 1, 255, 256, n / 2, n - 2, n - 1] {
                    let p = mesh.vertices[t * 3];
                    for query in [
                        [p[0] + 0.25, p[1] + 0.25, p[2] + 0.5],
                        [p[0] + 0.25, p[1] + 0.25, p[2] - 0.5],
                        [f64::NAN; 3],
                        [1e308; 3],
                    ] {
                        let a = old.nearest_with_cached_bounds(query, guess, &mut old_stack);
                        let b = new.nearest(query, guess, &mut new_stack);
                        assert_eq!(
                            (
                                a.triangle,
                                a.point.map(f64::to_bits),
                                a.distance_sq.to_bits()
                            ),
                            (
                                b.triangle,
                                b.point.map(f64::to_bits),
                                b.distance_sq.to_bits()
                            )
                        );
                        guess = (a.triangle != usize::MAX).then_some(a.triangle);
                    }
                }
            }
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn eager_pruning_preserves_hit_bits_signs_and_missing_queries() {
    let bits = |h: MeshHit| {
        (
            h.triangle,
            h.point.map(f64::to_bits),
            h.distance_sq.to_bits(),
        )
    };
    for seed in 1..=8 {
        for (scale, origin) in [
            (1.0, [0.0; 3]),
            (0.01, [368000.0, 3955000.0, 40.0]),
            (1e-160, [0.0; 3]),
            (1e60, [0.0; 3]),
            (1e140, [0.0; 3]),
            (1e200, [0.0; 3]),
        ] {
            let mut vertices: Vec<_> = pseudo_random(291, seed)
                .into_iter()
                .map(|p| std::array::from_fn(|a| p[a] * scale + origin[a]))
                .collect();
            if seed % 4 == 0 {
                vertices[0] = [f64::NAN, f64::INFINITY, -0.0];
            }
            let mut triangles: Vec<_> = (0..97u32).map(|t| [3 * t, 3 * t + 1, 3 * t + 2]).collect();
            triangles.extend((0..17u32).map(|t| [3 * t + 2, 3 * t + 1, 3 * t]));
            triangles.extend([[0, 0, 0], [0, 1, 0]]);
            let mesh = TriangleMesh {
                vertices,
                triangles,
            };
            let bvh = MeshBvh::new(&mesh).unwrap();
            let mut queries: Vec<_> = pseudo_random(103, seed + 100)
                .into_iter()
                .map(|p| std::array::from_fn(|a| p[a] * scale + origin[a]))
                .collect();
            queries.extend(mesh.vertices.iter().take(7));
            queries.extend([
                [f64::from_bits(0x7ff8_0000_0000_1234), 0.0, 0.0],
                [f64::INFINITY; 3],
                [f64::NEG_INFINITY; 3],
                [1e308; 3],
                [0.0, -0.0, 0.0],
                [-0.0, 0.0, -0.0],
            ]);
            let (mut old_stack, mut new_stack) = (Vec::new(), Vec::new());
            for policy in 0..4 {
                let mut previous = None;
                for (i, &p) in queries.iter().enumerate() {
                    let guess = match policy {
                        0 => None,
                        1 => previous,
                        2 => Some(i % mesh.triangles.len()),
                        _ => Some(0),
                    };
                    let old = bvh.nearest_with_cached_bounds(p, guess, &mut old_stack);
                    let new = bvh.nearest_with_cached_bounds_pruned(p, guess, &mut new_stack);
                    assert_eq!(bits(old), bits(new));
                    if old.triangle != usize::MAX {
                        let signed = |h: MeshHit| {
                            let d = h.distance_sq.sqrt();
                            if dot(sub(p, h.point), mesh.normal(h.triangle)) < 0.0 {
                                -d
                            } else {
                                d
                            }
                        };
                        assert_eq!(signed(old).to_bits(), signed(new).to_bits());
                        previous = Some(old.triangle);
                    } else {
                        previous = None;
                    }
                    assert!(new_stack.capacity() <= old_stack.capacity());
                }
            }
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn paired_projection_preserves_hit_bits_signs_and_missing_queries() {
    let bits = |h: MeshHit| {
        (
            h.triangle,
            h.point.map(f64::to_bits),
            h.distance_sq.to_bits(),
        )
    };
    for seed in 1..=8 {
        for (scale, origin) in [
            (1.0, [0.0; 3]),
            (0.01, [368000.0, 3955000.0, 40.0]),
            (1e-160, [0.0; 3]),
            (1e60, [0.0; 3]),
            (1e140, [0.0; 3]),
            (1e200, [0.0; 3]),
        ] {
            let mut vertices: Vec<_> = pseudo_random(291, seed)
                .into_iter()
                .map(|p| std::array::from_fn(|a| p[a] * scale + origin[a]))
                .collect();
            if seed % 4 == 0 {
                vertices[0] = [f64::NAN, f64::INFINITY, -0.0];
            }
            let mut triangles: Vec<_> = (0..97u32).map(|t| [3 * t, 3 * t + 1, 3 * t + 2]).collect();
            triangles.extend((0..17u32).map(|t| [3 * t + 2, 3 * t + 1, 3 * t]));
            triangles.extend([[0, 0, 0], [0, 1, 0]]);
            let mesh = TriangleMesh {
                vertices,
                triangles,
            };
            let bvh = MeshBvh::new(&mesh).unwrap();
            let mut queries: Vec<_> = pseudo_random(103, seed + 100)
                .into_iter()
                .map(|p| std::array::from_fn(|a| p[a] * scale + origin[a]))
                .collect();
            queries.extend(mesh.vertices.iter().take(7));
            queries.extend([
                [f64::from_bits(0x7ff8_0000_0000_1234), 0.0, 0.0],
                [f64::INFINITY; 3],
                [f64::NEG_INFINITY; 3],
                [1e308; 3],
                [0.0, -0.0, 0.0],
                [-0.0, 0.0, -0.0],
            ]);
            let (mut old_stack, mut new_stack) = (Vec::new(), Vec::new());
            for policy in 0..4 {
                let mut previous = None;
                for (i, &p) in queries.iter().enumerate() {
                    let guess = match policy {
                        0 => None,
                        1 => previous,
                        2 => Some(i % mesh.triangles.len()),
                        _ => Some(0),
                    };
                    let old = bvh.nearest_with_cached_bounds_clamped(p, guess, &mut old_stack);
                    let new = bvh.nearest_with_paired_projection(p, guess, &mut new_stack);
                    assert_eq!(bits(old), bits(new));
                    if old.triangle != usize::MAX {
                        let signed = |h: MeshHit| {
                            let d = h.distance_sq.sqrt();
                            if dot(sub(p, h.point), mesh.normal(h.triangle)) < 0.0 {
                                -d
                            } else {
                                d
                            }
                        };
                        assert_eq!(signed(old).to_bits(), signed(new).to_bits());
                        previous = Some(old.triangle);
                    } else {
                        previous = None;
                    }
                    assert_eq!(new_stack.capacity(), old_stack.capacity());
                }
            }
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn paired_bounds_preserve_hit_bits_signs_and_missing_queries() {
    let bits = |h: MeshHit| {
        (
            h.triangle,
            h.point.map(f64::to_bits),
            h.distance_sq.to_bits(),
        )
    };
    for seed in 1..=8 {
        for (scale, origin) in [
            (1.0, [0.0; 3]),
            (0.01, [368000.0, 3955000.0, 40.0]),
            (1e-160, [0.0; 3]),
            (1e60, [0.0; 3]),
            (1e140, [0.0; 3]),
            (1e200, [0.0; 3]),
        ] {
            let mut vertices: Vec<_> = pseudo_random(291, seed)
                .into_iter()
                .map(|p| std::array::from_fn(|a| p[a] * scale + origin[a]))
                .collect();
            if seed % 4 == 0 {
                vertices[0] = [f64::NAN, f64::INFINITY, -0.0];
            }
            let mut triangles: Vec<_> = (0..97u32).map(|t| [3 * t, 3 * t + 1, 3 * t + 2]).collect();
            triangles.extend((0..17u32).map(|t| [3 * t + 2, 3 * t + 1, 3 * t]));
            triangles.extend([[0, 0, 0], [0, 1, 0]]);
            let mesh = TriangleMesh {
                vertices,
                triangles,
            };
            let bvh = MeshBvh::new(&mesh).unwrap();
            let mut queries: Vec<_> = pseudo_random(103, seed + 100)
                .into_iter()
                .map(|p| std::array::from_fn(|a| p[a] * scale + origin[a]))
                .collect();
            queries.extend(mesh.vertices.iter().take(7));
            queries.extend([
                [f64::from_bits(0x7ff8_0000_0000_1234), 0.0, 0.0],
                [f64::INFINITY; 3],
                [f64::NEG_INFINITY; 3],
                [1e308; 3],
                [0.0, -0.0, 0.0],
                [-0.0, 0.0, -0.0],
            ]);
            let (mut old_stack, mut new_stack, mut clamped_stack) =
                (Vec::new(), Vec::new(), Vec::new());
            for policy in 0..4 {
                let mut previous = None;
                for (i, &p) in queries.iter().enumerate() {
                    let guess = match policy {
                        0 => None,
                        1 => previous,
                        2 => Some(i % mesh.triangles.len()),
                        _ => Some(0),
                    };
                    let old = bvh.nearest_with_cached_bounds(p, guess, &mut old_stack);
                    let new = bvh.nearest_with_cached_bounds_paired(p, guess, &mut new_stack);
                    let clamped =
                        bvh.nearest_with_cached_bounds_clamped(p, guess, &mut clamped_stack);
                    assert_eq!(bits(old), bits(new));
                    assert_eq!(bits(old), bits(clamped));
                    if old.triangle != usize::MAX {
                        let signed = |h: MeshHit| {
                            let d = h.distance_sq.sqrt();
                            if dot(sub(p, h.point), mesh.normal(h.triangle)) < 0.0 {
                                -d
                            } else {
                                d
                            }
                        };
                        assert_eq!(signed(old).to_bits(), signed(new).to_bits());
                        assert_eq!(signed(old).to_bits(), signed(clamped).to_bits());
                        previous = Some(old.triangle);
                    } else {
                        previous = None;
                    }
                    assert_eq!(new_stack.capacity(), old_stack.capacity());
                    assert_eq!(clamped_stack.capacity(), old_stack.capacity());
                }
            }
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn triangle_seed_preserves_projection_hit_bits_and_missing_reset() {
    let bits = |h: MeshHit| {
        (
            h.triangle,
            h.point.map(f64::to_bits),
            h.distance_sq.to_bits(),
        )
    };
    for seed in 1..=8 {
        for (scale, origin) in [
            (1.0, [0.0; 3]),
            (0.01, [368000.0, 3955000.0, 40.0]),
            (1e-160, [0.0; 3]),
            (1e60, [0.0; 3]),
            (1e140, [0.0; 3]),
            (1e200, [0.0; 3]),
        ] {
            let mut vertices: Vec<_> = pseudo_random(291, seed)
                .into_iter()
                .map(|p| std::array::from_fn(|a| p[a] * scale + origin[a]))
                .collect();
            if seed % 4 == 0 {
                vertices[0] = [f64::NAN, f64::INFINITY, -0.0];
            }
            let mut triangles: Vec<_> = (0..97u32).map(|t| [3 * t, 3 * t + 1, 3 * t + 2]).collect();
            triangles.extend((0..17u32).map(|t| [3 * t + 2, 3 * t + 1, 3 * t]));
            triangles.extend([[0, 0, 0], [0, 1, 0]]);
            let mesh = TriangleMesh {
                vertices,
                triangles,
            };
            let bvh = MeshBvh::new(&mesh).unwrap();
            let mut queries: Vec<_> = pseudo_random(103, seed + 100)
                .into_iter()
                .map(|p| std::array::from_fn(|a| p[a] * scale + origin[a]))
                .collect();
            queries.extend(mesh.vertices.iter().take(7));
            queries.extend([
                [f64::from_bits(0x7ff8_0000_0000_1234), 0.0, 0.0],
                [f64::INFINITY; 3],
                [f64::NEG_INFINITY; 3],
                [1e308; 3],
                [0.0, -0.0, 0.0],
                [-0.0, 0.0, -0.0],
            ]);
            let (mut old_stack, mut new_stack) = (Vec::new(), Vec::new());
            for policy in 0..4 {
                let mut previous = None;
                let mut cached: Option<MeshTriangleSeed> = None;
                for (i, &p) in queries.iter().enumerate() {
                    let guess = match policy {
                        0 => None,
                        1 => previous,
                        2 => Some(i % mesh.triangles.len()),
                        _ => Some(0),
                    };
                    let old = bvh.nearest_with_cached_bounds(p, guess, &mut old_stack);
                    if cached.as_ref().map(|s| s.triangle) != guess {
                        cached = guess.map(|t| MeshTriangleSeed::new(&mesh, t));
                    }
                    if let Some(seed) = cached.as_ref() {
                        let [a, b, c] = mesh.corners(seed.triangle);
                        assert_eq!(
                            seed.corners.map(|p| p.map(f64::to_bits)),
                            [a, b, c].map(|p| p.map(f64::to_bits))
                        );
                        assert_eq!(
                            seed.edges.map(|p| p.map(f64::to_bits)),
                            [sub(b, a), sub(c, a)].map(|p| p.map(f64::to_bits))
                        );
                        assert_eq!(
                            closest_point_on_triangle(p, a, b, c).map(f64::to_bits),
                            closest_point_with_triangle_seed(p, seed).map(f64::to_bits)
                        );
                    }
                    let new = bvh.nearest_with_triangle_seed(p, &mut cached, &mut new_stack);
                    assert_eq!(
                        cached.as_ref().map(|s| s.triangle),
                        (old.triangle != usize::MAX).then_some(old.triangle)
                    );
                    assert_eq!(bits(old), bits(new));
                    if old.triangle != usize::MAX {
                        let signed = |h: MeshHit| {
                            let d = h.distance_sq.sqrt();
                            if dot(sub(p, h.point), mesh.normal(h.triangle)) < 0.0 {
                                -d
                            } else {
                                d
                            }
                        };
                        assert_eq!(signed(old).to_bits(), signed(new).to_bits());
                        previous = Some(old.triangle);
                    } else {
                        previous = None;
                    }
                    assert_eq!(new_stack.capacity(), old_stack.capacity());
                }
            }
        }
    }
}

#[test]
fn untrusted_mesh_candidates_preserve_bits_and_signed_ties() {
    for seed in 1..=12 {
        for (scale, origin) in [
            (1.0, [0.0; 3]),
            (0.01, [368000.0, 3955000.0, 40.0]),
            (1e-60, [0.0; 3]),
            (1e60, [0.0; 3]),
        ] {
            let mut vertices = pseudo_random(291, seed);
            for p in &mut vertices {
                for a in 0..3 {
                    p[a] = p[a] * scale + origin[a];
                }
            }
            // Repeated/reversed and almost collinear faces exercise ties
            // and unstable GPU projections; CPU retains the original order.
            if seed % 3 == 0 {
                for t in (0..291).step_by(3) {
                    vertices[t + 2] = std::array::from_fn(|a| {
                        vertices[t][a] + (vertices[t + 1][a] - vertices[t][a]) * (1.0 + 1e-12)
                    });
                }
            }
            let mut triangles: Vec<_> = (0..97u32).map(|t| [3 * t, 3 * t + 1, 3 * t + 2]).collect();
            triangles.extend((0..17u32).map(|t| [3 * t + 2, 3 * t + 1, 3 * t]));
            let mut prepared = PreparedMesh::new(TriangleMesh {
                vertices,
                triangles,
            })
            .unwrap();
            let mut queries: Vec<_> = pseudo_random(127, seed + 37)
                .iter()
                .map(|p| std::array::from_fn(|a| p[a] * scale + origin[a]))
                .collect();
            queries.extend_from_slice(&prepared.mesh.vertices[..31]);
            let exact_slots: Vec<_> = queries
                .iter()
                .map(|&p| {
                    prepared
                        .storage
                        .corners
                        .iter()
                        .enumerate()
                        .map(|(slot, &[a, b, c])| {
                            let d = sub(p, closest_point_on_triangle(p, a, b, c));
                            (slot, dot(d, d))
                        })
                        .min_by(|a, b| a.1.total_cmp(&b.1))
                        .unwrap()
                        .0 as u32
                })
                .collect();
            for exported in [false, true] {
                if exported {
                    let (shift, nodes, triangles) = prepared.gpu_buffers();
                    assert_eq!(shift, prepared.mesh.vertices[0]);
                    assert_eq!(nodes.len(), prepared.storage.nodes.len() * 8);
                    assert_eq!(triangles.len(), prepared.storage.corners.len() * 12);
                }
                for signed in [false, true] {
                    let expected: Vec<_> = prepared
                        .distances(&queries, signed)
                        .iter()
                        .map(|d| d.to_bits())
                        .collect();
                    for candidates in [
                        vec![u32::MAX; queries.len()],
                        vec![0; queries.len()],
                        exact_slots.clone(),
                        (0..queries.len())
                            .map(|i| (i % prepared.storage.corners.len()) as u32)
                            .collect(),
                    ] {
                        let actual = prepared
                            .distances_hinted(&queries, signed, &candidates)
                            .unwrap();
                        assert_eq!(
                            expected,
                            actual.iter().map(|d| d.to_bits()).collect::<Vec<_>>(),
                            "seed={seed} scale={scale} signed={signed} exported={exported}"
                        );
                    }
                }
            }
            assert!(prepared.distances_hinted(&queries, false, &[]).is_none());
            assert_eq!(prepared.distances_hinted(&[], true, &[]), Some(vec![]));
        }
    }
}

/// Distance to a triangle by dense sampling of its surface.
fn sampled_distance_sq(p: [f64; 3], a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
    let n = 200;
    let mut best = f64::INFINITY;
    for i in 0..=n {
        for j in 0..=n - i {
            let (u, v) = (i as f64 / n as f64, j as f64 / n as f64);
            let q: [f64; 3] = std::array::from_fn(|k| a[k] + u * (b[k] - a[k]) + v * (c[k] - a[k]));
            let d = sub(p, q);
            best = best.min(dot(d, d));
        }
    }
    best
}

#[test]
fn closest_point_covers_all_regions() {
    let (a, b, c) = ([0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 3.0, 0.0]);
    let cases = [
        ([1.0, 1.0, 2.0], [1.0, 1.0, 0.0]),  // face
        ([-1.0, -1.0, 0.0], a),              // vertex a
        ([6.0, -1.0, 1.0], b),               // vertex b
        ([-1.0, 5.0, 0.0], c),               // vertex c
        ([2.0, -2.0, 0.0], [2.0, 0.0, 0.0]), // edge ab
        ([-2.0, 1.5, 0.0], [0.0, 1.5, 0.0]), // edge ac
    ];
    for (p, expected) in cases {
        let q = closest_point_on_triangle(p, a, b, c);
        #[cfg(feature = "parallel")]
        assert_eq!(
            q.map(f64::to_bits),
            closest_point_with_paired_dots(p, a, b, c).map(f64::to_bits),
        );
        for k in 0..3 {
            assert!((q[k] - expected[k]).abs() < 1e-12, "{p:?} -> {q:?}");
        }
    }
    // Edge bc: the foot of the perpendicular from (4, 3, 0).
    let q = closest_point_on_triangle([4.0, 3.0, 0.0], a, b, c);
    #[cfg(feature = "parallel")]
    assert_eq!(
        q.map(f64::to_bits),
        closest_point_with_paired_dots([4.0, 3.0, 0.0], a, b, c).map(f64::to_bits),
    );
    let expected = [4.0 - 4.0 * 0.36, 3.0 * 0.36, 0.0]; // t = (0,3)·(-4,3) / 25
    for k in 0..3 {
        assert!((q[k] - expected[k]).abs() < 1e-12, "{q:?}");
    }
}

#[test]
fn closest_point_matches_sampling() {
    let pts = pseudo_random(60, 3);
    for chunk in pts.chunks(4) {
        let [a, b, c, p] = [chunk[0], chunk[1], chunk[2], chunk[3]];
        let q = closest_point_on_triangle(p, a, b, c);
        let d = sub(p, q);
        let sampled = sampled_distance_sq(p, a, b, c);
        assert!(dot(d, d) <= sampled + 1e-9, "closer than any sample");
        assert!(
            dot(d, d).sqrt() >= sampled.sqrt() - 0.05,
            "not much farther than sampling"
        );
    }
}

#[test]
fn bvh_matches_brute_force() {
    let pts = pseudo_random(900, 7);
    let mesh = TriangleMesh {
        vertices: pts.clone(),
        triangles: (0..300).map(|t| [3 * t, 3 * t + 1, 3 * t + 2]).collect(),
    };
    let bvh = MeshBvh::new(&mesh).unwrap();
    let mut guess = None;
    for q in pseudo_random(200, 11) {
        let hit = bvh.nearest(q, guess);
        guess = Some(hit.triangle);
        let brute = (0..mesh.triangles.len())
            .map(|t| {
                let [a, b, c] = mesh.corners(t);
                let d = sub(q, closest_point_on_triangle(q, a, b, c));
                dot(d, d)
            })
            .fold(f64::INFINITY, f64::min);
        assert_eq!(hit.distance_sq, brute);
    }
}

#[test]
fn normal_follows_winding() {
    let mesh = TriangleMesh {
        vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        triangles: vec![[0, 1, 2]],
    };
    assert_eq!(mesh.normal(0), [0.0, 0.0, 1.0]);
}
