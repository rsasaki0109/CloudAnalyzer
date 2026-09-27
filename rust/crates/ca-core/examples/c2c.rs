//! Print C2C distance statistics: `cargo run --example c2c -- compared reference`.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [compared, reference] = args.as_slice() else {
        return Err("usage: c2c <compared> <reference>".into());
    };
    let load = |path: &str| -> Result<ca_core::PointCloud, Box<dyn std::error::Error>> {
        Ok(ca_core::read(path, &std::fs::read(path)?)?)
    };
    let (compared, reference) = (load(compared)?, load(reference)?);
    let start = std::time::Instant::now();
    let distances = ca_core::cloud_to_cloud(&compared, &reference).ok_or("empty reference")?;
    println!("c2c: {:.0} ms", start.elapsed().as_secs_f64() * 1e3);
    let stats = ca_core::DistanceStats::from_distances(&distances).ok_or("empty compared")?;
    println!(
        "compared={} reference={} colors={}/{}",
        compared.len(),
        reference.len(),
        compared.colors.is_some(),
        reference.colors.is_some()
    );
    println!("{stats:?}");
    Ok(())
}
