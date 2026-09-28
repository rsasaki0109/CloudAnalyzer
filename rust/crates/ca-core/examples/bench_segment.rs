//! Synthetic shape detection and clustering benchmark:
//! `cargo run --release -p ca-core --example bench_segment -- [points]`.
//!
//! A 20 m x 20 m room (floor and four walls) with columns and a little
//! noise, sampled at random.

use std::time::Instant;

use ca_core::cluster::euclidean_clusters;
use ca_core::normals::{Orientation, estimate_normals};
use ca_core::shapes::{Primitive, RansacParams, detect_shapes};

struct Rng(u64);

impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

fn main() {
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(1_000_000);
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let mut r = move || rng.next_f64();
    let points: Vec<[f64; 3]> = (0..n)
        .map(|_| {
            let (u, v, jitter) = (r() * 20.0, r() * 4.0, 0.004 * (r() - 0.5));
            match (r() * 10.0) as u32 {
                0..=3 => [u, r() * 20.0, jitter],
                4 => [u, jitter, v],
                5 => [u, 20.0 + jitter, v],
                6 => [jitter, u, v],
                7 => [20.0 + jitter, u, v],
                8 => {
                    // Four columns of radius 0.3.
                    let c =
                        [[5.0, 5.0], [15.0, 5.0], [5.0, 15.0], [15.0, 15.0]][(r() * 4.0) as usize];
                    let a = r() * std::f64::consts::TAU;
                    [c[0] + 0.3 * a.cos(), c[1] + 0.3 * a.sin(), v]
                }
                _ => [r() * 20.0, r() * 20.0, r() * 4.0],
            }
        })
        .collect();

    let t = Instant::now();
    let normals = estimate_normals(&points, 12, Orientation::Up);
    println!("normals: {:.0} ms", t.elapsed().as_secs_f64() * 1e3);

    for primitive in [Primitive::Plane, Primitive::Cylinder] {
        let t = Instant::now();
        let params = RansacParams {
            primitive,
            threshold: 0.02,
            min_support: n / 200,
            ..RansacParams::default()
        };
        let found = detect_shapes(&points, &normals, &params);
        println!(
            "{primitive:?}: {} shapes in {:.0} ms",
            found.len(),
            t.elapsed().as_secs_f64() * 1e3
        );
        for d in &found {
            println!(
                "  {} points, rms {:.4}: {:?}",
                d.indices.len(),
                d.rms,
                d.shape
            );
        }
    }

    let t = Instant::now();
    let clusters = euclidean_clusters(&points, 0.1, 100);
    println!(
        "clusters: {} (largest {:?}) in {:.0} ms",
        clusters.sizes.len(),
        &clusters.sizes[..clusters.sizes.len().min(5)],
        t.elapsed().as_secs_f64() * 1e3
    );
}
