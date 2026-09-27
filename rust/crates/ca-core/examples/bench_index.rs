//! Compare the plain and bucketed octree builds:
//! `cargo run --release -p ca-core --example bench_index -- file`.

use ca_core::octree::{Octree, OctreeParams};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("usage: bench_index <file>")?;
    let cloud = ca_core::read(&path, &std::fs::read(&path)?)?;
    let params = OctreeParams::default();
    let mut copy = cloud.clone();
    let t = Instant::now();
    Octree::build_for_cloud(&mut copy, params);
    println!("plain build: {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
    let mut copy = cloud.clone();
    let t = Instant::now();
    Octree::build_bucketed(&mut copy, params);
    println!("bucketed build: {:.0} ms", t.elapsed().as_secs_f64() * 1e3);
    Ok(())
}
