//! LiDAR odometry over a folder of scans (KITTI `.bin`, or anything `ca_core::read` opens, in
//! name order, each with a `.times` file of f32 per point when the scan's timing is known),
//! printing the poses as KITTI lines:
//! `cargo run --release -p ca-core --example bench_odometry -- <scans dir> [max frames] > poses.txt`.

use ca_core::odometry::{Odometry, OdometryParams};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let dir = &args[1];
    let limit: usize = args.get(2).map_or(Ok(usize::MAX), |v| v.parse())?;
    let mut files: Vec<_> = std::fs::read_dir(dir)?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| {
            p.extension().is_some_and(|e| {
                ["bin", "pcd", "ply"].contains(&e.to_string_lossy().to_lowercase().as_str())
            })
        })
        .collect();
    files.sort();
    files.truncate(limit);
    let env = |k: &str, d: f64| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    let defaults = OdometryParams::default();
    let params = OdometryParams {
        map_voxel: env("ODOM_MAP_VOXEL", defaults.map_voxel),
        min_range: env("ODOM_MIN_RANGE", defaults.min_range),
        max_range: env("ODOM_MAX_RANGE", defaults.max_range),
        map_points: env("ODOM_MAP_POINTS", defaults.map_points as f64) as usize,
        deskew: env("ODOM_DESKEW", 0.0) != 0.0,
        sweep: env("ODOM_SWEEP", defaults.sweep),
        ..defaults
    };
    let mut odometry = Odometry::new(params);
    let start = Instant::now();
    for path in &files {
        let name = path.to_string_lossy();
        let mut cloud = ca_core::read(&name, &std::fs::read(path)?)?;
        // A sibling `.times` file (f32 per point) gives when in the scan each point was taken.
        if let Ok(bytes) = std::fs::read(path.with_extension("times")) {
            let times: Vec<f32> = bytes
                .as_chunks::<4>()
                .0
                .iter()
                .map(|b| f32::from_le_bytes(*b))
                .collect();
            if times.len() == cloud.len() {
                cloud.attributes.push(ca_core::Attribute {
                    name: ca_core::odometry::TIME.to_string(),
                    values: ca_core::AttributeValues::F32(times),
                });
            }
        }
        let pose = odometry.register(&cloud);
        let m = pose.to_matrix();
        println!(
            "{}",
            m[..12]
                .iter()
                .map(|v| format!("{v:.9}"))
                .collect::<Vec<_>>()
                .join(" ")
        );
    }
    eprintln!("{} scans in {:.1?}", files.len(), start.elapsed());
    Ok(())
}
