//! Score dynamic point detection against SemanticKITTI moving labels:
//! `cargo run --release -p ca-core --example bench_dynamic -- <velodyne dir> <labels dir> <poses.txt> [voxel] [window] [margin] [votes] [resolution] [margin ratio] [object link] [min object] [elevation tolerance deg]`.
//!
//! Scans are voxel-thinned as the web app does, and the labels follow the
//! points kept. Moving classes are 252 to 259.

use ca_core::PointCloud;
use ca_core::dynamic::{VisibilityParams, dynamic_points};
use ca_core::icp::Rigid;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let arg = |i: usize, default: f64| args.get(i).map_or(Ok(default), |v| v.parse::<f64>());
    let (velodyne, labels, poses_file) = (&args[1], &args[2], &args[3]);
    let voxel = arg(4, 0.4)?;
    let params = VisibilityParams {
        window: arg(5, 10.0)? as usize,
        margin: arg(6, 0.5)?,
        min_see_through: arg(7, 3.0)? as usize,
        resolution_deg: arg(8, 3.0)?,
        margin_ratio: arg(9, 0.02)?,
        object_link: arg(10, 0.7)?,
        min_object: arg(11, 15.0)? as usize,
        elevation_tolerance_deg: arg(12, 0.5)?,
        ..VisibilityParams::default()
    };
    let poses: Vec<Rigid> = std::fs::read_to_string(poses_file)?
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| {
            let v: Vec<f64> = l.split_whitespace().map(|x| x.parse().unwrap()).collect();
            let mut m = [0.0; 16];
            m[..12].copy_from_slice(&v[..12]);
            m[15] = 1.0;
            Rigid::from_matrix(&m)
        })
        .collect();
    let t = Instant::now();
    let mut scans: Vec<PointCloud> = Vec::new();
    let mut moving: Vec<Vec<bool>> = Vec::new();
    for k in 0..poses.len() {
        let name = format!("{velodyne}/{k:010}.bin");
        let path = if std::path::Path::new(&name).exists() {
            name
        } else {
            format!("{velodyne}/{k:06}.bin")
        };
        let cloud = ca_core::read(&path, &std::fs::read(&path)?)?;
        let label_bytes = std::fs::read(format!("{labels}/{k:06}.label"))?;
        let label: Vec<bool> = label_bytes
            .as_chunks::<4>()
            .0
            .iter()
            .map(|c| (252..=259).contains(&(u32::from_le_bytes(*c) & 0xFFFF)))
            .collect();
        let keep = ca_core::filter::voxel_subsample(&cloud, voxel);
        moving.push(keep.iter().map(|&i| label[i]).collect());
        scans.push(cloud.select(&keep));
    }
    println!("loaded {} scans in {:.1?}", scans.len(), t.elapsed());
    let t = Instant::now();
    let refs: Vec<Option<&PointCloud>> = scans.iter().map(Some).collect();
    let flags = dynamic_points(&poses, &refs, &params);
    let took = t.elapsed();
    let (mut tp, mut fp, mut fn_, mut tn) = (0u64, 0u64, 0u64, 0u64);
    for (f, m) in flags.iter().zip(&moving) {
        for (&d, &truth) in f.iter().zip(m) {
            match (d, truth) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fn_ += 1,
                (false, false) => tn += 1,
            }
        }
    }
    let precision = tp as f64 / (tp + fp).max(1) as f64;
    let recall = tp as f64 / (tp + fn_).max(1) as f64;
    let f1 = 2.0 * precision * recall / (precision + recall).max(f64::MIN_POSITIVE);
    println!(
        "{params:?}\n{:.1?}: moving {} of {} points; precision {precision:.3}, recall {recall:.3}, F1 {f1:.3}, static kept {:.4}",
        took,
        tp + fn_,
        tp + fp + fn_ + tn,
        tn as f64 / (tn + fp).max(1) as f64
    );
    Ok(())
}
