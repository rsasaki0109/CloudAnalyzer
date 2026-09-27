//! Time the filters: `cargo run --release -p ca-core --example bench_filter -- file`.

use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: bench_filter <file>")?;
    let cloud = ca_core::read(&path, &std::fs::read(&path)?)?;
    let t = Instant::now();
    let kept = ca_core::filter::voxel_subsample(&cloud, 0.5).len();
    println!(
        "voxel 0.5: kept {kept} in {:.0} ms",
        t.elapsed().as_secs_f64() * 1e3
    );
    let t = Instant::now();
    let kept = ca_core::filter::statistical_outliers(&cloud, 8, 1.0).len();
    println!(
        "sor k=8: kept {kept} of {} in {:.0} ms",
        cloud.len(),
        t.elapsed().as_secs_f64() * 1e3
    );
    Ok(())
}
