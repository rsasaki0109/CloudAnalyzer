//! Time loop registration between two scans:
//! `cargo run --release -p ca-core --example bench_loop -- a.bin b.bin [voxel]`.

use ca_core::icp::{IcpParams, Rigid};
use ca_core::pose_graph::{overlap_fitness, register_loop, register_with_yaw_search};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let voxel: f64 = args.get(3).map_or(Ok(0.4), |v| v.parse())?;
    let load = |path: &str| -> Result<ca_core::PointCloud, Box<dyn std::error::Error>> {
        let cloud = ca_core::read(path, &std::fs::read(path)?)?;
        Ok(cloud.select(&ca_core::filter::voxel_subsample(&cloud, voxel)))
    };
    let (a, b) = (load(&args[1])?, load(&args[2])?);
    println!("{} and {} points", a.len(), b.len());
    let sample: usize = std::env::var("SAMPLE").map_or(50_000, |v| v.parse().unwrap());
    let params = IcpParams {
        overlap: 0.8,
        sample,
        ..IcpParams::default()
    };
    let t = Instant::now();
    let (m, r) = register_loop(&a, &b, &Rigid::IDENTITY, params).unwrap();
    let f = overlap_fitness(&a, &b, &m, 0.5);
    println!(
        "one ICP: {:.0} ms, {} iterations, fitness {f:.2}",
        t.elapsed().as_secs_f64() * 1e3,
        r.iterations
    );
    let t = Instant::now();
    let (m, f, _) = register_with_yaw_search(&a, &b, &Rigid::IDENTITY, 8, params, 0.5).unwrap();
    println!(
        "yaw search (8): {:.0} ms, fitness {f:.2}, t = {:.2?}",
        t.elapsed().as_secs_f64() * 1e3,
        m.translation
    );
    Ok(())
}
