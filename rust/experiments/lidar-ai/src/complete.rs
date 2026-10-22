//! CPU-only opt-in experiment over frozen native debug tensors. No model runs.
use ca_lidar_ai::{
    completion,
    projection::{self, CLASS_NAMES, CLASSES, MAX_POINTS, PIXELS},
};
use serde_json::{Value, json};
use sha2::{Digest, Sha256};
use std::{
    env, fs,
    io::Read,
    path::{Path, PathBuf},
};

fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn read(path: &Path, limit: usize) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    let file = fs::File::open(path)?;
    if !file.metadata()?.is_file() || file.metadata()?.len() > limit as u64 {
        return Err(format!("file exceeds the completion budget: {}", path.display()).into());
    }
    let mut bytes = Vec::new();
    file.take(limit as u64 + 1).read_to_end(&mut bytes)?;
    if bytes.len() > limit {
        return Err("file grew beyond completion budget".into());
    }
    Ok(bytes)
}
fn require(condition: bool, message: &str) -> Result<(), Box<dyn std::error::Error>> {
    if condition {
        Ok(())
    } else {
        Err(message.into())
    }
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let (mut source, mut base, mut output) = (None, None, None);
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--scan" => source = Some(PathBuf::from(args.next().ok_or("missing scan path")?)),
            "--base" => {
                base = Some(PathBuf::from(
                    args.next().ok_or("missing baseline debug directory")?,
                ))
            }
            "--out" => {
                output = Some(PathBuf::from(
                    args.next().ok_or("missing output directory")?,
                ))
            }
            "--help" => {
                println!(
                    "ca-lidar-complete --scan ORIGINAL_SCAN.bin --base FROZEN_NATIVE_DEBUG_DIR --out NEW_DIR\nFixed experimental near-surface policy; no model, training, automatic threshold changes or browser-native result import."
                );
                return Ok(());
            }
            _ => return Err(format!("unknown argument {arg}").into()),
        }
    }
    let source = source.ok_or("--scan required")?;
    let base = base.ok_or("--base required (native result with --debug-tensors)")?;
    let output = output.ok_or("--out required")?;
    require(
        !output.exists(),
        "output must be a new directory; no existing results overwritten",
    )?;
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    require(parent.is_dir(), "output parent missing")?;
    let raw = read(&source, MAX_POINTS * 16)?;
    let points = projection::read_scan(&raw)?;
    let projected = projection::project(&points)?;
    require(projected.report.projected > 0, "no supported source pixels")?;
    let report_bytes = read(&base.join("report.json"), 65536)?;
    let report: Value = serde_json::from_slice(&report_bytes)?;
    require(
        report["experimental"] == true
            && report["model"] == "SalsaNext SemanticKITTI20"
            && report["backend"] == "wgpu"
            && report["device_selection"] == "discrete_gpu_0"
            && report["source_modified"] == false
            && report["source_weight_sha256"]
                == "1757b8917e81b39058feb78dd17d690d74a878cab272a2c7b47bd8e5869ee811"
            && report["onnx_sha256"]
                == "8d61577f0d9582adfa28baea4246f915c29d4c5d5e3b1d27adac83ffef4815f5"
            && report["class_names"] == json!(CLASS_NAMES),
        "unsupported native baseline",
    )?;
    require(
        report["input_sha256"] == sha(&raw) && report["mask_bytes"] == points.len(),
        "baseline/source identity differs",
    )?;
    require(
        report["projection"] == serde_json::to_value(&projected.report)?,
        "baseline projection differs",
    )?;
    require(
        report["sensor_profile"]
            == json!({"height":64,"width":2048,"fov_up_deg":3,"fov_down_deg":-25,"max_range_m":100,"remission":"unchanged normalized0..1"}),
        "unsupported sensor profile",
    )?;
    let threshold = report["selection"]["threshold"]
        .as_f64()
        .ok_or("missing baseline threshold")? as f32;
    let targets: Vec<usize> = report["selection"]["target_classes"]
        .as_array()
        .ok_or("missing target classes")?
        .iter()
        .map(|c| {
            CLASS_NAMES
                .iter()
                .position(|&name| c.as_str() == Some(name))
                .ok_or("unknown target class")
        })
        .collect::<Result<_, _>>()?;
    let input = read(&base.join("input.f32"), 5 * PIXELS * 4)?;
    require(
        input.len() == 5 * PIXELS * 4
            && input
                .as_chunks::<4>()
                .0
                .iter()
                .zip(&projected.input)
                .all(|(b, v)| *b == v.to_le_bytes()),
        "baseline debug input does not match this raw scan",
    )?;
    let probability_bytes = read(&base.join("probabilities.f32"), CLASSES * PIXELS * 4)?;
    require(
        probability_bytes.len() == CLASSES * PIXELS * 4,
        "unexpected probability tensor shape",
    )?;
    let probabilities: Vec<f32> = probability_bytes
        .as_chunks::<4>()
        .0
        .iter()
        .map(|b| f32::from_le_bytes(*b))
        .collect();
    let (baseline, selection) =
        projection::removal_mask(&projected, &probabilities, &targets, threshold)?;
    require(
        report["selection"] == serde_json::to_value(&selection)?,
        "baseline selection differs from frozen probabilities",
    )?;
    let baseline_bytes: Vec<u8> = baseline.iter().map(|&v| u8::from(v)).collect();
    require(
        read(&base.join("candidate-mask.u8"), MAX_POINTS)? == baseline_bytes,
        "baseline mask differs from frozen probabilities",
    )?;
    let (kept, removed) = projection::split_scan(&raw, &baseline)?;
    require(
        read(&base.join("retained.bin"), MAX_POINTS * 16)? == kept
            && report["retained_sha256"] == sha(&kept)
            && read(&base.join("candidates.bin"), MAX_POINTS * 16)? == removed
            && report["candidate_sha256"] == sha(&removed),
        "baseline source records or hashes differ",
    )?;
    let (mask, counts) =
        completion::complete(&points, &projected, &probabilities, &targets, threshold)?;
    let (kept, removed) = projection::split_scan(&raw, &mask)?;
    let added: Vec<u8> = mask
        .iter()
        .zip(&baseline)
        .map(|(&after, &before)| u8::from(after && !before))
        .collect();
    let mask_bytes: Vec<u8> = mask.into_iter().map(u8::from).collect();
    let metadata = json!({"app":"CloudAnalyzer LiDAR surface completion experiment","version":1,
        "policy":{"id":completion::POLICY,"radius_m":completion::RADIUS_M,"min_seed_separation_m":completion::MIN_SEED_SEPARATION_M,
            "seed_score":completion::SEED_SCORE,"conflict_score":completion::CONFLICT_SCORE,"max_neighborhood_owners":completion::MAX_NEIGHBORS},
        "base":{"report_sha256":sha(&report_bytes),"input_tensor_sha256":sha(&input),"probabilities_sha256":sha(&probability_bytes),
            "candidate_mask_sha256":sha(&baseline_bytes),"onnx_sha256":report["onnx_sha256"],"weights_sha256":report["weights_sha256"],
            "target_classes":report["selection"]["target_classes"],"threshold":threshold},
        "input_sha256":sha(&raw),"completed_mask_sha256":sha(&mask_bytes),"added_mask_sha256":sha(&added),
        "retained_sha256":sha(&kept),"candidate_sha256":sha(&removed),"counts":counts,"source_modified":false,
        "limitations":["Experimental completion of eligible collisions only; no automatic adoption",
            "Original owner decisions stay; added points never become seeds; different confident classes veto completion",
            "No labels, model execution, training or fused-map support","This is not a nearest-only native report; review separately"]});
    require(
        read(&source, MAX_POINTS * 16)? == raw,
        "source changed during completion",
    )?;
    let staging = parent.join(format!(
        ".{}-pending-{}",
        output
            .file_name()
            .ok_or("invalid output name")?
            .to_string_lossy(),
        std::process::id()
    ));
    require(!staging.exists(), "staging directory exists")?;
    fs::create_dir(&staging)?;
    fs::write(staging.join("completed-mask.u8"), mask_bytes)?;
    fs::write(staging.join("added-mask.u8"), added)?;
    fs::write(staging.join("retained.bin"), kept)?;
    fs::write(staging.join("candidates.bin"), removed)?;
    fs::write(
        staging.join("completion-report.json"),
        serde_json::to_string_pretty(&metadata)? + "\n",
    )?;
    require(
        !output.exists(),
        "output created by another process; result left in staging",
    )?;
    fs::rename(&staging, &output)?;
    println!("{}", serde_json::to_string(&metadata)?);
    Ok(())
}
