use ca_core::{
    TriangleMesh, cloud_to_mesh,
    mesh::{MeshQueryOrder, PreparedMesh},
};

#[cfg(feature = "parallel")]
fn parallel_fixture() -> (TriangleMesh, Vec<[f64; 3]>) {
    let vertices: Vec<_> = (0..49)
        .map(|i| {
            [
                368000.0 + (i % 7) as f64,
                3955000.0 + (i / 7) as f64,
                40.0 + (i % 3) as f64 * 0.02,
            ]
        })
        .collect();
    let mut triangles = Vec::new();
    for y in 0..6u32 {
        for x in 0..6u32 {
            let a = y * 7 + x;
            triangles.extend([[a, a + 1, a + 7], [a + 7, a + 1, a + 8]]);
        }
    }
    triangles.extend(
        triangles[..17]
            .iter()
            .map(|&[a, b, c]| [c, b, a])
            .collect::<Vec<_>>(),
    );
    triangles.extend([[0, 0, 0], [0, 1, 0]]);
    let queries = (0..8193)
        .map(|i| {
            let mut p = vertices[i % vertices.len()];
            p[2] += (i % 5) as f64 * 0.1 - 0.2;
            p
        })
        .collect();
    (
        TriangleMesh {
            vertices,
            triangles,
        },
        queries,
    )
}

#[cfg(feature = "parallel")]
#[test]
fn retained_parallel_mesh_owns_inputs_and_resets_each_batch() {
    let (mesh, queries) = parallel_fixture();
    let mut source = mesh.clone();
    source.triangles.push([u32::MAX, 0, 1]);
    let prepared = PreparedMesh::new(source.clone()).unwrap();
    let bytes = prepared.retained_bytes();
    source.vertices.fill([99.0; 3]);
    source.triangles.clear();
    drop(source);
    let changed: Vec<_> = queries
        .iter()
        .rev()
        .map(|p| [p[0] + 0.25, p[1] - 0.5, p[2] + 1.0])
        .collect();
    for threads in [1, 2, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        for input in [&queries, &changed] {
            for count in [0, 1, 4095, 4096, 4097, 8193] {
                for signed in [false, true] {
                    let expected = pool.install(|| {
                        ca_core::distance::cloud_to_mesh_par(&input[..count], &mesh, signed)
                            .unwrap()
                    });
                    let actual = pool.install(|| prepared.distances_par(&input[..count], signed));
                    assert_eq!(
                        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                    assert_eq!(prepared.retained_bytes(), bytes);
                }
            }
        }
    }
}

#[cfg(feature = "parallel")]
#[test]
fn retained_parallel_mesh_supports_concurrent_readers() {
    fn assert_send_sync<T: Send + Sync>() {}
    assert_send_sync::<PreparedMesh>();
    let (mesh, queries) = parallel_fixture();
    let prepared = std::sync::Arc::new(PreparedMesh::new(mesh.clone()).unwrap());
    let bytes = prepared.retained_bytes();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(4)
        .build()
        .unwrap();
    let expected: Vec<_> = [false, true]
        .map(|signed| {
            pool.install(|| ca_core::distance::cloud_to_mesh_par(&queries, &mesh, signed).unwrap())
                .into_iter()
                .map(f64::to_bits)
                .collect::<Vec<_>>()
        })
        .into();
    for _ in 0..2 {
        std::thread::scope(|scope| {
            let handles: Vec<_> = (0..4)
                .map(|i| {
                    let retained = &prepared;
                    let pool = &pool;
                    let points = &queries;
                    scope.spawn(move || pool.install(|| retained.distances_par(points, i % 2 == 1)))
                })
                .collect();
            for (i, handle) in handles.into_iter().enumerate() {
                assert_eq!(
                    handle
                        .join()
                        .unwrap()
                        .into_iter()
                        .map(f64::to_bits)
                        .collect::<Vec<_>>(),
                    expected[i % 2]
                );
            }
        });
        assert_eq!(prepared.retained_bytes(), bytes);
    }
}

#[test]
fn owned_mesh_bvh_preserves_signed_bits_ties_batches_and_source_independence() {
    for n in [1, 9, 33, 257, 2049] {
        for mode in 0..4 {
            let vertices: Vec<_> = (0..n * 3)
                .map(|i| {
                    let p = [
                        (i % 11) as f64 * 0.1,
                        ((i / 11) % 7) as f64 * 0.1,
                        (i % 5) as f64 * 0.02,
                    ];
                    match mode {
                        1 => [368000.0 + p[0], 3955000.0 + p[1], 40.0 + p[2]],
                        2 => [if i % 2 == 0 { -0.0 } else { 0.0 }; 3],
                        3 => [(i % 3 == 1) as u8 as f64, (i % 3 == 2) as u8 as f64, 0.0],
                        _ => p,
                    }
                })
                .collect();
            let mesh = TriangleMesh {
                vertices,
                triangles: (0..n as u32)
                    .map(|i| {
                        if i % 2 == 0 {
                            [i * 3, i * 3 + 1, i * 3 + 2]
                        } else {
                            [i * 3 + 2, i * 3 + 1, i * 3]
                        }
                    })
                    .collect(),
            };
            let mut original = mesh.clone();
            let prepared = PreparedMesh::new(original.clone()).unwrap();
            let bytes = prepared.retained_bytes();
            assert!(bytes <= mesh.vertices.len() * 24 + mesh.triangles.len() * 144 + 4096);
            original.vertices.fill([99.0; 3]);
            original.triangles.clear();
            drop(original);
            let mut queries = mesh.vertices.clone();
            queries.extend(
                mesh.vertices
                    .iter()
                    .map(|p| [p[0] + 0.025, p[1] - 0.03, p[2] + 0.5]),
            );
            queries.extend(mesh.vertices.iter().map(|p| [p[0], p[1], p[2] - 0.5]));
            for signed in [false, true, false, true] {
                for input in [
                    &queries[..],
                    &queries[..queries.len() / 2],
                    &[][..],
                    &queries[queries.len() / 3..],
                ] {
                    let expected = cloud_to_mesh(input, &mesh, signed).unwrap();
                    assert_eq!(
                        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        prepared
                            .distances(input, signed)
                            .iter()
                            .map(|v| v.to_bits())
                            .collect::<Vec<_>>()
                    );
                }
                assert_eq!(prepared.retained_bytes(), bytes);
            }
        }
    }
    assert!(PreparedMesh::new(TriangleMesh::default()).is_none());
    let invalid = TriangleMesh {
        vertices: vec![[0.0; 3]],
        triangles: vec![[0, 1, 2]],
    };
    assert!(PreparedMesh::new(invalid).is_none());
}

#[test]
fn missing_mesh_hits_do_not_poison_later_queries_or_signed_distances() {
    let mesh = TriangleMesh {
        vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
        triangles: vec![[0, 1, 2]],
    };
    let prepared = PreparedMesh::new(mesh.clone()).unwrap();
    let bytes = prepared.retained_bytes();
    let invalid = [
        f64::from_bits(0x7ff8_0000_0000_0023),
        f64::INFINITY,
        f64::NEG_INFINITY,
    ];
    let mut queries = vec![[0.25, 0.25, -0.5]];
    for value in invalid {
        for axis in 0..3 {
            let mut p = [0.25, 0.25, 0.5];
            p[axis] = value;
            queries.extend([p, [0.25, 0.25, 0.5], [0.25, 0.25, -0.5]]);
        }
    }
    // Finite coordinates can also produce a missing hit when squared distances overflow.
    queries.extend([[1e308, 0.25, 0.5], [0.25, 0.25, 0.5]]);
    // Exercise several Rayon chunks as well as different Morton tie orders.
    queries = queries.repeat(300);
    let bits = |values: Vec<f64>| values.into_iter().map(f64::to_bits).collect::<Vec<_>>();
    for reversed in [false, true] {
        if reversed {
            queries.reverse();
        }
        for signed in [false, true] {
            let expected = queries
                .iter()
                .map(|p| {
                    if p.iter().any(|v| !v.is_finite()) || p[0] == 1e308 {
                        f64::INFINITY.to_bits()
                    } else {
                        (if signed { p[2] } else { p[2].abs() }).to_bits()
                    }
                })
                .collect::<Vec<_>>();
            assert_eq!(
                bits(cloud_to_mesh(&queries, &mesh, signed).unwrap()),
                expected
            );
            assert_eq!(bits(prepared.distances(&queries, signed)), expected);
            for candidate in [0, u32::MAX] {
                assert_eq!(
                    bits(
                        prepared
                            .distances_hinted(&queries, signed, &vec![candidate; queries.len()])
                            .unwrap()
                    ),
                    expected
                );
            }
            let mut cache = MeshQueryOrder::new(8 * 1024 * 1024);
            assert_eq!(
                bits(cache.distances(&prepared, queries.clone(), signed)),
                expected
            );
            assert!(!cache.hit());
            assert_eq!(
                bits(cache.distances(&prepared, queries.clone(), signed)),
                expected
            );
            assert!(cache.hit());
            #[cfg(feature = "parallel")]
            {
                assert_eq!(
                    bits(ca_core::distance::cloud_to_mesh_par(&queries, &mesh, signed).unwrap()),
                    expected
                );
                assert_eq!(bits(prepared.distances_par(&queries, signed)), expected);
            }
            assert_eq!(prepared.distances(&[[0.25, 0.25, -0.5]], true), vec![-0.5]);
            assert_eq!(prepared.retained_bytes(), bytes);
        }
    }
}

#[test]
fn mesh_without_a_computable_hit_returns_positive_infinity() {
    let mesh = TriangleMesh {
        vertices: vec![[f64::NAN; 3]; 3],
        triangles: vec![[0, 1, 2]],
    };
    let prepared = PreparedMesh::new(mesh.clone()).unwrap();
    let queries = [[0.0; 3], [1.0; 3], [f64::NAN; 3]];
    for signed in [false, true] {
        let expected = vec![f64::INFINITY; queries.len()];
        assert_eq!(cloud_to_mesh(&queries, &mesh, signed).unwrap(), expected);
        assert_eq!(prepared.distances(&queries, signed), expected);
        assert_eq!(
            prepared
                .distances_hinted(&queries, signed, &[0, 0, u32::MAX])
                .unwrap(),
            expected
        );
        #[cfg(feature = "parallel")]
        {
            assert_eq!(
                ca_core::distance::cloud_to_mesh_par(&queries, &mesh, signed).unwrap(),
                expected
            );
            assert_eq!(prepared.distances_par(&queries, signed), expected);
        }
    }
}
