#![cfg(feature = "completion")]
//! Fabricated inference records exercise I/O, not learned semantic accuracy.
use ca_lidar_ai::projection::{self, CLASS_NAMES, CLASSES, PIXELS};
use serde_json::json;
use sha2::{Digest, Sha256};
use std::{
    fs,
    path::PathBuf,
    process::{Command, Output},
    time::{SystemTime, UNIX_EPOCH},
};

fn sha(b: &[u8]) -> String {
    format!("{:x}", Sha256::digest(b))
}
struct Fixture {
    root: PathBuf,
    source: PathBuf,
    base: PathBuf,
}
impl Fixture {
    fn new() -> Self {
        let name = format!(
            "ca-lidar-complete-{}-{}",
            std::process::id(),
            SystemTime::now()
                .duration_since(UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let root = std::env::temp_dir().join(name);
        fs::create_dir(&root).unwrap();
        let source = root.join("scan.bin");
        let base = root.join("base");
        fs::create_dir(&base).unwrap();
        let points = [
            [10_f32, 0., 0., 0.5],
            [10.02, 0., 0., 0.5],
            [10., 0.05, 0., 0.5],
        ];
        let raw: Vec<_> = points
            .into_iter()
            .flatten()
            .flat_map(f32::to_le_bytes)
            .collect();
        fs::write(&source, &raw).unwrap();
        let p = projection::project(&points).unwrap();
        let mut probabilities = vec![0_f32; CLASSES * PIXELS];
        probabilities[..PIXELS].fill(1.);
        for (pixel, owner) in p.owners.iter().enumerate() {
            if owner.is_some() {
                probabilities[pixel] = 1. - 0.999;
                probabilities[PIXELS + pixel] = 0.999;
            }
        }
        let (mask, selection) = projection::removal_mask(&p, &probabilities, &[1], 0.9).unwrap();
        let (kept, removed) = projection::split_scan(&raw, &mask).unwrap();
        fs::write(
            base.join("input.f32"),
            p.input
                .iter()
                .flat_map(|v| v.to_le_bytes())
                .collect::<Vec<_>>(),
        )
        .unwrap();
        fs::write(
            base.join("probabilities.f32"),
            probabilities
                .into_iter()
                .flat_map(f32::to_le_bytes)
                .collect::<Vec<_>>(),
        )
        .unwrap();
        fs::write(
            base.join("candidate-mask.u8"),
            mask.into_iter().map(u8::from).collect::<Vec<_>>(),
        )
        .unwrap();
        fs::write(base.join("retained.bin"), &kept).unwrap();
        fs::write(base.join("candidates.bin"), &removed).unwrap();
        let report = json!({"experimental":true,"model":"SalsaNext SemanticKITTI20","backend":"wgpu","device_selection":"discrete_gpu_0",
            "source_modified":false,"source_weight_sha256":"1757b8917e81b39058feb78dd17d690d74a878cab272a2c7b47bd8e5869ee811",
            "onnx_sha256":"8d61577f0d9582adfa28baea4246f915c29d4c5d5e3b1d27adac83ffef4815f5","weights_sha256":"1".repeat(64),
            "class_names":CLASS_NAMES,"input_sha256":sha(&raw),"mask_bytes":points.len(),"projection":p.report,"selection":selection,
            "retained_sha256":sha(&kept),"candidate_sha256":sha(&removed),
            "sensor_profile":{"height":64,"width":2048,"fov_up_deg":3,"fov_down_deg":-25,"max_range_m":100,"remission":"unchanged normalized0..1"}});
        fs::write(
            base.join("report.json"),
            serde_json::to_vec_pretty(&report).unwrap(),
        )
        .unwrap();
        Self { root, source, base }
    }
    fn run(&self) -> Output {
        Command::new(env!("CARGO_BIN_EXE_ca-lidar-complete"))
            .arg("--scan")
            .arg(&self.source)
            .arg("--base")
            .arg(&self.base)
            .arg("--out")
            .arg(self.root.join("out"))
            .output()
            .unwrap()
    }
}
impl Drop for Fixture {
    fn drop(&mut self) {
        // Resolve the exact owned test directory before recursive cleanup.
        let target = self.root.canonicalize().unwrap();
        let parent = std::env::temp_dir().canonicalize().unwrap();
        assert_eq!(target.parent(), Some(parent.as_path()));
        assert!(
            target
                .file_name()
                .unwrap()
                .to_string_lossy()
                .starts_with("ca-lidar-complete-")
        );
        fs::remove_dir_all(target).unwrap();
    }
}
#[test]
fn cpu_cli_writes_separate_exact_records_and_preserves_the_baseline() {
    let f = Fixture::new();
    let raw = fs::read(&f.source).unwrap();
    let before: Vec<_> = fs::read_dir(&f.base)
        .unwrap()
        .map(|p| {
            let p = p.unwrap().path();
            let bytes = fs::read(&p).unwrap();
            (p, bytes)
        })
        .collect();
    let output = f.run();
    assert!(
        output.status.success(),
        "{}",
        String::from_utf8_lossy(&output.stderr)
    );
    let out = f.root.join("out");
    assert_eq!(fs::read(out.join("completed-mask.u8")).unwrap(), [1, 1, 1]);
    assert_eq!(fs::read(out.join("added-mask.u8")).unwrap(), [0, 1, 0]);
    assert!(fs::read(out.join("retained.bin")).unwrap().is_empty());
    assert_eq!(fs::read(out.join("candidates.bin")).unwrap(), raw);
    assert!(!out.join("report.json").exists());
    assert!(out.join("completion-report.json").is_file());
    assert_eq!(fs::read(&f.source).unwrap(), raw);
    for (p, b) in before {
        assert_eq!(fs::read(p).unwrap(), b);
    }
}
#[test]
fn rejected_source_debug_tensor_mask_and_output_do_not_publish_or_clobber() {
    let f = Fixture::new();
    let original_source = fs::read(&f.source).unwrap();
    for (file, edit) in [
        ("input.f32", 0_u8),
        ("probabilities.f32", 1),
        ("candidate-mask.u8", 2),
        ("retained.bin", 3),
    ] {
        let path = f.base.join(file);
        let original = fs::read(&path).unwrap();
        let mut changed = original.clone();
        match edit {
            0 => changed[0] ^= 1,
            1 => changed[..4].copy_from_slice(&f32::NAN.to_le_bytes()),
            2 => changed[1] = 1,
            _ => changed[0] ^= 1,
        }
        fs::write(&path, changed).unwrap();
        assert!(!f.run().status.success());
        assert!(!f.root.join("out").exists());
        fs::write(path, original).unwrap();
    }
    let mut changed = original_source.clone();
    changed[0] ^= 1;
    fs::write(&f.source, changed).unwrap();
    assert!(!f.run().status.success());
    assert!(!f.root.join("out").exists());
    fs::write(&f.source, &original_source).unwrap();
    fs::create_dir(f.root.join("out")).unwrap();
    fs::write(f.root.join("out/sentinel"), b"keep").unwrap();
    assert!(!f.run().status.success());
    assert_eq!(fs::read(f.root.join("out/sentinel")).unwrap(), b"keep");
    assert_eq!(fs::read(&f.source).unwrap(), original_source);
}
