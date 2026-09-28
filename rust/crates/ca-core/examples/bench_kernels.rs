//! Time the hot kernels on a synthetic surface, natively or under WASI:
//! `cargo run --release -p ca-core --example bench_kernels -- [points] [rounds] [kernels]`
//! where `kernels` is a comma-separated subset of
//! `kd,c2c,octree,sor,normals,voxel,transform` (default: all).
//! Each kernel runs `rounds` times; the first and the best time are printed
//! (under V8 the first call may still run baseline code), then a checksum of
//! the results to compare builds.

use std::time::Instant;

use ca_core::PointCloud;
use ca_core::icp::Rigid;
use ca_core::kdtree::KdTree;
use ca_core::normals::Orientation;
use ca_core::octree::{Octree, OctreeParams};

/// Deterministic xorshift generator so native and WASM runs see the same data.
struct Rng(u64);

impl Rng {
    fn next_f64(&mut self) -> f64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        (self.0 >> 11) as f64 / (1u64 << 53) as f64
    }
}

/// A wavy surface in random point order, offset like UTM data; the extent
/// grows with `n` so the density stays ~100 points per square metre.
fn surface(rng: &mut Rng, n: usize, bump: f64) -> PointCloud {
    let half = (n as f64 / 100.0).sqrt() / 2.0;
    let positions = (0..n)
        .map(|_| {
            let x = (rng.next_f64() * 2.0 - 1.0) * half;
            let y = (rng.next_f64() * 2.0 - 1.0) * half;
            let z = 2.0 * (x / 6.0).sin() * (y / 8.0).cos()
                + bump * (-((x - 15.0).powi(2) + (y + 10.0).powi(2)) / 60.0).exp();
            [x + 368_000.0, y + 3_955_000.0, z + 40.0]
        })
        .collect();
    PointCloud {
        positions,
        colors: None,
        attributes: Vec::new(),
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let n: usize = args
        .next()
        .and_then(|a| a.parse().ok())
        .unwrap_or(1_000_000);
    let rounds: usize = args.next().and_then(|a| a.parse().ok()).unwrap_or(3);
    let only = args.next();
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let reference = surface(&mut rng, n, 0.0);
    let compared = surface(&mut rng, n, 0.3);
    println!("n={n}");

    // Runs `run` if selected and returns a checksum of its last result.
    let time = |name: &str, run: &mut dyn FnMut() -> f64| {
        if only
            .as_ref()
            .is_some_and(|o| !o.split(',').any(|k| k == name))
        {
            return;
        }
        let (mut first, mut best, mut sum) = (0.0, f64::INFINITY, 0.0);
        for r in 0..rounds {
            let start = Instant::now();
            sum = run();
            let ms = start.elapsed().as_secs_f64() * 1e3;
            if r == 0 {
                first = ms;
            }
            best = best.min(ms);
        }
        println!("{name:<10} first {first:>7.0} ms  best {best:>7.0} ms  sum {sum:.12e}");
    };

    time("kd", &mut || {
        KdTree::new(&reference.positions).map_or(0.0, |t| t.len() as f64)
    });
    time("c2c", &mut || {
        ca_core::cloud_to_cloud(&compared, &reference)
            .unwrap()
            .iter()
            .sum()
    });
    time("octree", &mut || {
        let mut c = reference.clone();
        let tree = Octree::build_for_cloud(&mut c, OctreeParams::default()).unwrap();
        let order: f64 = (tree.order.iter().enumerate())
            .map(|(i, &o)| (o as f64) * (i % 1000) as f64)
            .sum();
        tree.nodes.len() as f64 + order
    });
    time("sor", &mut || {
        ca_core::filter::knn_mean_distances(&reference.positions, 8)
            .iter()
            .sum()
    });
    time("normals", &mut || {
        ca_core::normals::estimate_normals(&reference.positions, 10, Orientation::Up)
            .iter()
            .map(|v| (v[0] as f64) + 2.0 * (v[1] as f64) + 3.0 * (v[2] as f64))
            .sum()
    });
    time("voxel", &mut || {
        ca_core::filter::voxel_subsample(&reference, 0.1).len() as f64
    });
    let rigid = Rigid::from_matrix(&[
        0.8, -0.6, 0.0, 1.0, 0.6, 0.8, 0.0, 2.0, 0.0, 0.0, 1.0, 3.0, 0.0, 0.0, 0.0, 1.0,
    ]);
    time("transform", &mut || {
        let mut c = reference.clone();
        for p in &mut c.positions {
            *p = rigid.apply(p);
        }
        c.positions[n / 2][0]
    });
}
