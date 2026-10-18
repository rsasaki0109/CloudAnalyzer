//! Export the exact f64 bits used for validation after native JSON deserialization.
//! Expected poses are diagnostics only; they never initialize the estimator.

use serde::Deserialize;
use serde_json::json;
use std::fs::{self, OpenOptions};

#[derive(Deserialize)]
struct Expected {
    pose: [f64; 16],
}

#[derive(Deserialize)]
struct Frame {
    scan_index: usize,
    baseline: Expected,
    balanced: Expected,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    frames: Vec<Frame>,
}

#[derive(Deserialize)]
struct Input {
    cases: Vec<Case>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(
        args.len(),
        3,
        "audit_odometry_pose_inputs input.json fresh-output.json"
    );
    let input: Input = serde_json::from_slice(&fs::read(&args[1])?)?;
    let cases: Vec<_> = input
        .cases
        .into_iter()
        .map(|case| {
            let frames: Vec<_> = case
                .frames
                .into_iter()
                .map(|frame| {
                    json!({"scan_index": frame.scan_index,
                "baseline_bits": frame.baseline.pose.map(f64::to_bits),
                "balanced_bits": frame.balanced.pose.map(f64::to_bits)})
                })
                .collect();
            json!({"case": case.name, "frames": frames})
        })
        .collect();
    let output = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(&args[2])?;
    serde_json::to_writer(
        output,
        &json!({"cases": cases, "estimator_uses_expected_poses": false}),
    )?;
    Ok(())
}
