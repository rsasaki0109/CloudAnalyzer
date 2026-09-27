//! Synthetic C2C benchmark that runs the same natively and under WASI:
//! `cargo run --release -p ca-core --example bench_c2c -- [points]`.

use std::time::Instant;

use ca_core::PointCloud;

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

/// A wavy 100 m x 100 m surface in random point order, offset like UTM data.
fn surface(rng: &mut Rng, n: usize, bump: f64) -> PointCloud {
    let positions = (0..n)
        .map(|_| {
            let x = rng.next_f64() * 100.0 - 50.0;
            let y = rng.next_f64() * 100.0 - 50.0;
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
    let n: usize = std::env::args()
        .nth(1)
        .and_then(|a| a.parse().ok())
        .unwrap_or(1_000_000);
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    let reference = surface(&mut rng, n, 0.0);
    let compared = surface(&mut rng, n, 0.3);

    let start = Instant::now();
    let distances = ca_core::cloud_to_cloud(&compared, &reference).unwrap();
    let elapsed = start.elapsed().as_secs_f64() * 1e3;
    let stats = ca_core::DistanceStats::from_distances(&distances).unwrap();
    println!(
        "n={n} c2c={elapsed:.0} ms mean={:.6} max={:.6}",
        stats.mean, stats.max
    );

    // Partitioned run: plan once, then time each part as a parallel worker would.
    for parts in [4, 8] {
        let start = Instant::now();
        let plan = ca_core::partition_c2c(&compared, &reference, parts).unwrap();
        let plan_ms = start.elapsed().as_secs_f64() * 1e3;
        let pick = |cloud: &PointCloud, idx: &[u32]| PointCloud {
            positions: idx.iter().map(|&i| cloud.positions[i as usize]).collect(),
            colors: None,
            attributes: Vec::new(),
        };
        let mut slowest = 0.0f64;
        let mut copies = 0;
        let mut merged = vec![0.0; compared.len()];
        for part in &plan {
            copies += part.reference.len();
            let (q, r) = (
                pick(&compared, &part.queries),
                pick(&reference, &part.reference),
            );
            let start = Instant::now();
            let d = ca_core::cloud_to_cloud(&q, &r).unwrap();
            slowest = slowest.max(start.elapsed().as_secs_f64() * 1e3);
            for (&i, d) in part.queries.iter().zip(d) {
                merged[i as usize] = d;
            }
        }
        assert_eq!(merged, distances);
        println!(
            "parts={parts} plan={plan_ms:.0} ms slowest part={slowest:.0} ms reference copies={:.2}x",
            copies as f64 / reference.len() as f64
        );
    }
}
