//! Native-only, explicitly selected SalsaNext/HDL-64 experimental adapter.
use burn::tensor::{Device, DeviceKind, Tensor, TensorData};
use ca_lidar_ai::input::read;
use ca_lidar_ai::projection::{self, CLASS_NAMES, CLASSES, HEIGHT, MAX_POINTS, PIXELS, WIDTH};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    env, fs,
    path::{Path, PathBuf},
    time::Instant,
};

#[allow(dead_code)] // The importer also emits an unused byte-loading constructor.
mod model {
    include!(concat!(env!("OUT_DIR"), "/model/salsanext.rs"));
}
fn sha(bytes: &[u8]) -> String {
    format!("{:x}", Sha256::digest(bytes))
}
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut scan = None;
    let mut output = None;
    let mut weights = None;
    let mut threshold = 0.9_f32;
    let mut targets = vec![1];
    let mut sensor_frame = false;
    let mut tensors = false;
    let mut args = env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--scan" => scan = Some(PathBuf::from(args.next().ok_or("missing scan path")?)),
            "--out" => output = Some(PathBuf::from(args.next().ok_or("missing output path")?)),
            "--weights" => {
                weights = Some(PathBuf::from(args.next().ok_or("missing Burnpack path")?))
            }
            "--threshold" => threshold = args.next().ok_or("missing threshold")?.parse()?,
            "--classes" => {
                targets = args
                    .next()
                    .ok_or("missing classes")?
                    .split(',')
                    .map(|s| {
                        CLASS_NAMES
                            .iter()
                            .position(|&c| c == s)
                            .ok_or_else(|| format!("unknown class {s}"))
                    })
                    .collect::<Result<Vec<_>, _>>()?;
            }
            "--sensor-frame" => sensor_frame = true,
            "--debug-tensors" => tensors = true,
            "--help" => {
                println!(
                    "ca-lidar-ai --scan SINGLE_SENSOR_SCAN.bin --sensor-frame --weights salsanext.bpk --out NEW_DIR [--classes car,truck,other-vehicle] [--threshold 0.9] [--debug-tensors]"
                );
                return Ok(());
            }
            _ => return Err(format!("unknown argument {arg}").into()),
        }
    }
    if !sensor_frame {
        return Err("declare --sensor-frame: one KITTI-format HDL-64 scan in metres (x forward, y left, z up), not a fused map".into());
    }
    if !threshold.is_finite()
        || !(0.0..=1.0).contains(&threshold)
        || targets.is_empty()
        || targets.contains(&0)
    {
        return Err("invalid threshold/targets; unlabeled cannot be removed".into());
    }
    let scan = scan.ok_or("--scan required")?;
    let output = output.ok_or("--out required")?;
    let weights = weights.ok_or("--weights required")?;
    if output.exists() {
        return Err("output must be a new directory; no files overwritten".into());
    }
    let raw = read(&scan, 16 * MAX_POINTS)?;
    let original_hash = sha(&raw);
    let points = projection::read_scan(&raw)?;
    let projected = projection::project(&points)?;
    if projected.report.projected == 0 {
        return Err("no supported single-scan input pixels; no inference or removal".into());
    }
    let weights_hash = sha(&read(&weights, 128 * 1024 * 1024)?);
    if weights_hash != env!("CA_LIDAR_PACK_SHA") {
        return Err(
            "weights differ from this binary's imported model; no inference or removal".into(),
        );
    }
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    if !parent.is_dir() {
        return Err("output parent directory missing".into());
    }
    let output_name = output
        .file_name()
        .ok_or("invalid output directory")?
        .to_string_lossy();
    let staging = parent.join(format!(".{output_name}-pending-{}", std::process::id()));
    if staging.exists() {
        return Err("staging directory already exists".into());
    }
    // A discrete adapter is required. No CPU or software-adapter fallback.
    let device = Device::wgpu(DeviceKind::DiscreteGpu(0));
    let started = Instant::now();
    let model = model::Model::from_file(weights.to_str().ok_or("non-UTF8 weights path")?, &device);
    let input = Tensor::<4>::from_data(
        TensorData::new(projected.input.clone(), [1, 5, HEIGHT, WIDTH]),
        &device,
    );
    let result = model.forward(input);
    if result.dims() != [1, CLASSES, HEIGHT, WIDTH] {
        return Err("unexpected model output shape".into());
    }
    let data = result.into_data();
    let probabilities = data.try_to_vec::<f32>()?;
    let inference_millis = started.elapsed().as_secs_f64() * 1000.;
    let (mask, report) = projection::removal_mask(&projected, &probabilities, &targets, threshold)?;
    let (retained, candidates) = projection::split_scan(&raw, &mask)?;
    if sha(&read(&scan, 16 * MAX_POINTS)?) != original_hash {
        return Err("source scan changed while inference was running".into());
    }
    let metadata = json!({
        "experimental":true,"model":"SalsaNext SemanticKITTI20", "backend":"wgpu", "device_selection":"discrete_gpu_0",
        "inference_millis_including_model_load_and_first_kernel_setup":inference_millis,
        "input_sha256":original_hash,"weights_sha256":weights_hash,
        "onnx_sha256":env!("CA_LIDAR_ONNX_SHA"),
        "source_weight_sha256":"1757b8917e81b39058feb78dd17d690d74a878cab272a2c7b47bd8e5869ee811",
        "sensor_profile":{"height":HEIGHT,"width":WIDTH,"fov_up_deg":3,"fov_down_deg":-25,"max_range_m":100,"remission":"unchanged normalized0..1"},
        "projection":projected.report,"selection":report,"class_names":CLASS_NAMES,
        "retained_sha256":sha(&retained),"candidate_sha256":sha(&candidates),
        "source_modified":false,"mask_bytes":mask.len(),
        "limitations":["Single-scan KITTI/HDL-64 profile only; no fused-map or cross-sensor validation","Class prediction does not establish motion; parked and moving cars share the car class","Only nearest projected returns can be removed; all unclassified returns retained","No k-NN label propagation; no geometry-based enlargement or source overwrite","No whole-dataset accuracy or performance claim from this experiment"]
    });
    // Do not publish a result directory until every model/shape/point check passes.
    fs::create_dir(&staging)?;
    fs::write(staging.join("retained.bin"), retained)?;
    fs::write(staging.join("candidates.bin"), candidates)?;
    fs::write(
        staging.join("candidate-mask.u8"),
        mask.into_iter().map(u8::from).collect::<Vec<_>>(),
    )?;
    if tensors {
        fs::write(
            staging.join("input.f32"),
            projected
                .input
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
        fs::write(
            staging.join("probabilities.f32"),
            probabilities
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )?;
    }
    fs::write(
        staging.join("report.json"),
        serde_json::to_string_pretty(&metadata)? + "\n",
    )?;
    if output.exists() {
        return Err("output was created by another process; result remains in staging".into());
    }
    fs::rename(&staging, &output)?;
    println!("{}", serde_json::to_string(&metadata)?);
    assert_eq!(probabilities.len(), CLASSES * PIXELS);
    Ok(())
}
