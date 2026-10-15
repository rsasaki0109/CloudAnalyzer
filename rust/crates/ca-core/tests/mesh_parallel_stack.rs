#![cfg(feature = "parallel")]

use ca_core::{TriangleMesh, mesh::MeshBvh};

// Independent preceding batch policy: exact Morton keys, original unstable
// key-only sort, 4096-point chunks, and a fresh nearest() stack per query.
fn preceding(points: &[[f64; 3]], mesh: &TriangleMesh, signed: bool) -> Vec<f64> {
    let bvh = MeshBvh::new(mesh).unwrap();
    let mut out = vec![0.0; points.len()];
    let Some(first) = points.first() else {
        return out;
    };
    let (mut lo, mut hi) = (*first, *first);
    for p in points {
        for a in 0..3 {
            lo[a] = lo[a].min(p[a]);
            hi[a] = hi[a].max(p[a]);
        }
    }
    let scale: [f64; 3] = std::array::from_fn(|a| {
        let extent = hi[a] - lo[a];
        if extent > 0.0 {
            ((1u32 << 21) - 1) as f64 / extent
        } else {
            0.0
        }
    });
    let spread = |v: u64| (0..21).fold(0, |key, b| key | ((v >> b) & 1) << (3 * b));
    let mut keyed: Vec<(u64, usize)> = points
        .iter()
        .enumerate()
        .map(|(i, p)| {
            let cell = |a: usize| ((p[a] - lo[a]) * scale[a]) as u64;
            (
                spread(cell(0)) | spread(cell(1)) << 1 | spread(cell(2)) << 2,
                i,
            )
        })
        .collect();
    keyed.sort_unstable_by_key(|&(key, _)| key);
    for chunk in keyed.chunks(4096) {
        let mut guess = None;
        for &(_, i) in chunk {
            let p = points[i];
            let hit = bvh.nearest(p, guess);
            if hit.triangle == usize::MAX {
                guess = None;
                out[i] = f64::INFINITY;
                continue;
            }
            guess = Some(hit.triangle);
            let mut d = hit.distance_sq.sqrt();
            if signed {
                let n = mesh.normal(hit.triangle);
                let delta: [f64; 3] = std::array::from_fn(|a| p[a] - hit.point[a]);
                if delta[0] * n[0] + delta[1] * n[1] + delta[2] * n[2] < 0.0 {
                    d = -d;
                }
            }
            out[i] = d;
        }
    }
    out
}

#[test]
fn parallel_stack_reuse_preserves_preceding_chunks_signed_ties_and_missing_hits() {
    for (scale, origin) in [
        (1.0, [0.0; 3]),
        (0.01, [368000.0, 3955000.0, 40.0]),
        (1e-160, [0.0; 3]),
        (1e60, [0.0; 3]),
    ] {
        let vertices: Vec<_> = (0..81)
            .map(|i| {
                let p = [
                    (i % 9) as f64 * 0.1,
                    (i / 9) as f64 * 0.1,
                    (i % 3) as f64 * 0.007,
                ];
                std::array::from_fn(|a| p[a] * scale + origin[a])
            })
            .collect();
        let mut triangles = Vec::new();
        for i in 0..70u32 {
            if i % 9 < 8 {
                triangles.extend([[i, i + 1, i + 9], [i + 9, i + 1, i + 10]]);
            }
        }
        triangles.extend(
            triangles[..17]
                .iter()
                .map(|&[a, b, c]| [c, b, a])
                .collect::<Vec<_>>(),
        );
        triangles.extend([[0, 0, 0], [0, 1, 0]]);
        let mesh = TriangleMesh {
            vertices,
            triangles,
        };
        let prepared = ca_core::mesh::PreparedMesh::new(mesh.clone()).unwrap();
        let retained_bytes = prepared.retained_bytes();
        let all: Vec<_> = (0..65_537)
            .map(|i| {
                let mut p = mesh.vertices[i % 81];
                p[2] += (i % 5) as f64 * 0.013 * scale - 0.026 * scale;
                p
            })
            .collect();
        for count in [0, 1, 4095, 4096, 4097, 8193, 65_535, 65_536, 65_537] {
            let queries = &all[..count];
            for signed in [false, true] {
                let expected = preceding(queries, &mesh, signed)
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>();
                for threads in [1, 2, 4] {
                    let pool = rayon::ThreadPoolBuilder::new()
                        .num_threads(threads)
                        .build()
                        .unwrap();
                    let actual = pool.install(|| {
                        ca_core::distance::cloud_to_mesh_par(queries, &mesh, signed).unwrap()
                    });
                    assert_eq!(
                        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        expected,
                        "scale={scale} count={count} signed={signed} threads={threads}"
                    );
                    let retained = pool.install(|| prepared.distances_par(queries, signed));
                    assert_eq!(
                        retained.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        expected,
                        "retained scale={scale} count={count} signed={signed} threads={threads}"
                    );
                    assert_eq!(prepared.retained_bytes(), retained_bytes);
                }
            }
        }
        let mut missing = all;
        missing[4095] = [f64::NAN, 0.0, 0.0];
        missing[4096] = [f64::INFINITY; 3];
        missing[8192] = [1e308; 3];
        missing[65_535] = [f64::NEG_INFINITY; 3];
        missing[65_536] = [f64::from_bits(0x7ff8_0000_0000_1234), 0.0, 0.0];
        for signed in [false, true] {
            assert_eq!(
                ca_core::distance::cloud_to_mesh_par(&missing, &mesh, signed)
                    .unwrap()
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                preceding(&missing, &mesh, signed)
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(
                prepared
                    .distances_par(&missing, signed)
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>(),
                preceding(&missing, &mesh, signed)
                    .iter()
                    .map(|v| v.to_bits())
                    .collect::<Vec<_>>()
            );
            assert_eq!(prepared.retained_bytes(), retained_bytes);
        }
    }
}
