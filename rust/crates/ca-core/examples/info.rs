//! Print a short summary of a point cloud file: `cargo run -p ca-core --example info -- file`.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).ok_or("usage: info <file>")?;
    let cloud = ca_core::read(&path, &std::fs::read(&path)?)?;
    let bounds = cloud.bounds().ok_or("empty cloud")?;
    let checksum: f64 = cloud.positions.iter().flatten().sum();
    let color_sum: u64 = cloud
        .colors
        .iter()
        .flatten()
        .flatten()
        .map(|&c| c as u64)
        .sum();
    println!(
        "points={} colors={} min={:?} max={:?} xyz_sum={checksum:.6} rgb_sum={color_sum}",
        cloud.len(),
        cloud.colors.is_some(),
        bounds.min,
        bounds.max
    );
    Ok(())
}
