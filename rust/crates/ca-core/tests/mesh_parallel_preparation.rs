#![cfg(feature = "parallel")]

use ca_core::{TriangleMesh, mesh::PreparedMesh};

#[test]
fn parallel_preparation_preserves_threshold_cases_signed_results_and_exact_capacities() {
    for n in [65_535, 65_536, 65_537] {
        for origin in [[0.0; 3], [368000.0, 3955000.0, 40.0]] {
            let mut mesh = TriangleMesh {
                vertices: Vec::with_capacity(n * 3),
                triangles: Vec::with_capacity(n),
            };
            for t in 0..n {
                let x = origin[0] + (t % 256) as f64 * 2.0;
                let y = origin[1] + (t / 256) as f64 * 2.0;
                mesh.vertices.extend([
                    [x, y, origin[2]],
                    [x + 1.0, y, origin[2]],
                    [x, y + 1.0, origin[2]],
                ]);
                mesh.triangles
                    .push([(t * 3) as u32, (t * 3 + 1) as u32, (t * 3 + 2) as u32]);
            }
            let selected = [0, 1, 255, 256, n / 2, n - 2, n - 1];
            let mut queries = Vec::new();
            for t in selected {
                let p = mesh.vertices[t * 3];
                queries.extend([
                    [p[0] + 0.25, p[1] + 0.25, p[2] + 0.5],
                    [p[0] + 0.25, p[1] + 0.25, p[2] - 0.5],
                ]);
            }
            queries.extend([[f64::NAN; 3], [f64::INFINITY; 3], [1e308; 3]]);
            // Original exact vector capacities: vertices, triangles, order,
            // corners and the unchanged 56-byte BVH node reservation.
            let expected_bytes = n * (3 * 24 + 12 + 4 + 72) + (2 * n / 4 + 1) * 56;
            for threads in [1, 2, 4] {
                let pool = rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap();
                let prepared = pool.install(|| PreparedMesh::new(mesh.clone()).unwrap());
                assert_eq!(
                    prepared.retained_bytes(),
                    expected_bytes,
                    "n={n} threads={threads}"
                );
                for signed in [false, true] {
                    let expected: Vec<_> = (0..queries.len())
                        .map(|i| {
                            if i >= selected.len() * 2 {
                                f64::INFINITY
                            } else if signed && i % 2 == 1 {
                                -0.5
                            } else {
                                0.5
                            }
                        })
                        .map(f64::to_bits)
                        .collect();
                    let bits =
                        |values: Vec<f64>| values.into_iter().map(f64::to_bits).collect::<Vec<_>>();
                    assert_eq!(bits(prepared.distances(&queries, signed)), expected);
                    assert_eq!(
                        bits(
                            prepared
                                .distances_hinted(&queries, signed, &vec![u32::MAX; queries.len()])
                                .unwrap()
                        ),
                        expected
                    );
                    assert_eq!(
                        bits(pool.install(|| {
                            ca_core::distance::cloud_to_mesh_par(&queries, &mesh, signed).unwrap()
                        })),
                        expected
                    );
                    assert_eq!(prepared.retained_bytes(), expected_bytes);
                }
            }
        }
    }
}
