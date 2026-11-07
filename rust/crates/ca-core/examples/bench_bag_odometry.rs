//! Replay every bag scan with poses, iteration counts and registration timing.
//! Usage: bench_bag_odometry input.mcap output.json; RAYON_NUM_THREADS=1.
use ca_core::PointCloud;
use ca_core::bag::{Bag, POINT_CLOUD, decode_point_cloud2};
use ca_core::odometry::{Odometry, OdometryParams};
use serde_json::json;
use std::fs::OpenOptions;
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[2])?;
    let mut bag = Bag::open(&args[1])?;
    let topics: Vec<_> = bag
        .topics()
        .iter()
        .filter(|t| t.kind == POINT_CLOUD)
        .map(|t| t.name.clone())
        .collect();
    let names: Vec<_> = topics.iter().map(String::as_str).collect();
    let mut odometry = Odometry::new(OdometryParams {
        deskew: false,
        ..Default::default()
    });
    let mut rows = Vec::new();
    for message in bag.messages(&names) {
        let message = message?;
        let scan = decode_point_cloud2(&message.data, message.encoding)?;
        let cloud = PointCloud {
            positions: scan
                .positions
                .into_iter()
                .filter(|p| {
                    let d = p.iter().map(|v| v * v).sum::<f64>().sqrt();
                    d > 1.5 && d < 80.
                })
                .collect(),
            ..Default::default()
        };
        let tick = Instant::now();
        let pose = odometry.register(&cloud);
        let seconds = tick.elapsed().as_secs_f64();
        let report = odometry.last_report().unwrap();
        rows.push(
            json!({"stamp_ns":scan.stamp_ns, "pose":pose.to_matrix(), "seconds":seconds,
            "iterations":report.iterations, "termination":report.termination.as_str(),
            "registration_points":report.registration_points}),
        );
    }
    serde_json::to_writer(output, &json!({"frames":rows}))?;
    Ok(())
}
