//! Experimental fixed-gain ICP; production parameters and solver are unchanged.
//! Usage: bench_bag_relaxation input.mcap output.json gain; RAYON_NUM_THREADS=1.
//! Compare gain 1 with 1.5, retaining all scans and nonconverged results.
use ca_core::PointCloud;
use ca_core::bag::{Bag, POINT_CLOUD, decode_point_cloud2};
use ca_core::odometry::{Odometry, OdometryParams};
use serde_json::json;
use std::fs::OpenOptions;
use std::time::Instant;
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    let gain: f64 = args.get(3).ok_or("gain is required")?.parse()?;
    if ![1., 1.5].contains(&gain) {
        return Err("experiment only supports gain 1 or 1.5".into());
    }
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
        let mut registration = odometry.begin(&cloud);
        let mut updates = Vec::new();
        let mut align = || {
            if !registration.needs_alignment() {
                return;
            }
            loop {
                let (h, mut b) = odometry.equations(&registration);
                for value in &mut b {
                    *value *= gain;
                }
                let before = *registration.total();
                let finished = odometry.step(&mut registration, (h, b));
                let delta = registration
                    .total()
                    .compose(&ca_core::pose_graph::inverse(&before));
                let center = registration.center();
                let moved = delta.apply(&center);
                let shift = (0..3)
                    .map(|i| (moved[i] - center[i]).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let trace: f64 = (0..3).map(|i| delta.rotation[i][i]).sum();
                let turn = ((trace - 1.) / 2.).clamp(-1., 1.).acos();
                updates.push([shift, turn]);
                if finished {
                    break;
                }
            }
        };
        #[cfg(feature = "parallel")]
        rayon::scope(|_| align());
        #[cfg(not(feature = "parallel"))]
        align();
        let pose = odometry.finish(registration);
        let seconds = tick.elapsed().as_secs_f64();
        let report = odometry.last_report().unwrap();
        rows.push(
            json!({"stamp_ns":scan.stamp_ns, "pose":pose.to_matrix(), "seconds":seconds,
            "iterations":report.iterations, "termination":report.termination.as_str(),
            "registration_points":report.registration_points, "updates":updates}),
        );
    }
    serde_json::to_writer(output, &json!({"gain":gain,"frames":rows}))?;
    Ok(())
}
