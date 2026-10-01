//! Write the small COPC file the web tests use:
//! `cargo run -p ca-core --example make_test_copc -- ../web/e2e/fixtures/small.copc.laz`.

use ca_core::io::copc::{VoxelKey, write_minimal_copc};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: make_test_copc <out>")?;
    // A 64 x 64 m site: level l has 4^l nodes of 2,000 points inside their
    // square, on a gentle slope.
    let (size, per_node) = (64.0, 2_000usize);
    let mut nodes = Vec::new();
    let mut seed = 7u64;
    let mut next = move || {
        seed ^= seed << 13;
        seed ^= seed >> 7;
        seed ^= seed << 17;
        (seed >> 11) as f64 / (1u64 << 53) as f64
    };
    for level in 0..3 {
        let n = 1 << level;
        let cell = size / n as f64;
        for j in 0..n {
            for i in 0..n {
                let points = (0..per_node)
                    .map(|_| {
                        let (x, y) = ((i as f64 + next()) * cell, (j as f64 + next()) * cell);
                        [x, y, 0.05 * x + 0.02 * y]
                    })
                    .collect();
                let key = VoxelKey {
                    level,
                    x: i,
                    y: j,
                    // The root Z range is [-32, 32]; positive slope points
                    // belong to its upper child, then [0, 16] at level 2.
                    z: if level == 0 { 0 } else { 1 << (level - 1) },
                };
                nodes.push((key, points));
            }
        }
    }
    std::fs::write(&path, write_minimal_copc(&nodes, [32.0, 32.0, 0.0], 32.0))?;
    println!("wrote {path}");
    Ok(())
}
