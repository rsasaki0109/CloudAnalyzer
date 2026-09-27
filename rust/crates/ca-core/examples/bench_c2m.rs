//! Synthetic C2M benchmark: `cargo run --release -p ca-core --example bench_c2m -- [points] [grid]`.

use std::time::Instant;

use ca_core::TriangleMesh;

fn height(x: f64, y: f64) -> f64 {
    2.0 * (x / 6.0).sin() * (y / 8.0).cos()
}

fn main() {
    let mut args = std::env::args().skip(1).map(|a| a.parse().unwrap());
    let n: usize = args.next().unwrap_or(1_000_000);
    let grid: usize = args.next().unwrap_or(400);

    // A height-field mesh over [0, 100]^2 with grid x grid cells.
    let step = 100.0 / grid as f64;
    let mut mesh = TriangleMesh::default();
    for j in 0..=grid {
        for i in 0..=grid {
            let (x, y) = (i as f64 * step, j as f64 * step);
            mesh.vertices.push([x, y, height(x, y)]);
        }
    }
    let w = grid as u32 + 1;
    for j in 0..grid as u32 {
        for i in 0..grid as u32 {
            let a = j * w + i;
            mesh.triangles.push([a, a + 1, a + w + 1]);
            mesh.triangles.push([a, a + w + 1, a + w]);
        }
    }

    let mut s = 0x9e37_79b9_7f4a_7c15u64;
    let mut next = move || {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        (s >> 11) as f64 / (1u64 << 53) as f64
    };
    let points: Vec<[f64; 3]> = (0..n)
        .map(|_| {
            let (x, y) = (next() * 100.0, next() * 100.0);
            [x, y, height(x, y) + (next() - 0.5) * 0.2]
        })
        .collect();

    let start = Instant::now();
    let d = ca_core::cloud_to_mesh(&points, &mesh, true).unwrap();
    let ms = start.elapsed().as_secs_f64() * 1e3;
    let mean_abs = d.iter().map(|v| v.abs()).sum::<f64>() / d.len() as f64;
    println!(
        "points={n} triangles={} c2m={ms:.0} ms mean|d|={mean_abs:.4}",
        mesh.triangles.len()
    );
}
