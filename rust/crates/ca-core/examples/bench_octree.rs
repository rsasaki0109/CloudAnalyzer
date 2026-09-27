//! Time octree construction: `cargo run --release -p ca-core --example bench_octree -- file`.

use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: bench_octree <file>")?;
    let mut cloud = ca_core::read(&path, &std::fs::read(&path)?)?;
    let start = Instant::now();
    let tree = ca_core::octree::Octree::build_in_place(&mut cloud.positions, Default::default())
        .ok_or("empty cloud")?;
    let depth = tree.nodes.iter().map(|n| n.level).max().unwrap_or(0);
    println!(
        "points={} nodes={} depth={depth} root={} build={:.0} ms",
        cloud.len(),
        tree.nodes.len(),
        tree.nodes[0].count,
        start.elapsed().as_secs_f64() * 1e3
    );
    Ok(())
}
