//! Compare CSF ground extraction with a file's existing classification:
//! `cargo run --release -p ca-core --example ground -- file.las [resolution] [threshold]`.

use ca_core::ground::{CsfParams, Rigidness, csf};
use ca_core::{AttributeValues, CLASSIFICATION};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut args = std::env::args().skip(1);
    let path = args
        .next()
        .ok_or("usage: ground <file> [resolution] [threshold]")?;
    let resolution: f64 = args.next().map_or(Ok(1.0), |a| a.parse())?;
    let threshold: f64 = args.next().map_or(Ok(0.5), |a| a.parse())?;
    let cloud = ca_core::read(&path, &std::fs::read(&path)?)?;
    let start = std::time::Instant::now();
    let params = CsfParams {
        cloth_resolution: resolution,
        class_threshold: threshold,
        rigidness: Rigidness::Relief,
        ..CsfParams::default()
    };
    let ground = csf(&cloud, params).ok_or("csf failed")?;
    println!(
        "{} points, {} ground in {:.0} ms",
        cloud.len(),
        ground.iter().filter(|g| **g).count(),
        start.elapsed().as_secs_f64() * 1e3
    );
    if let Some(AttributeValues::U8(classes)) = cloud.attribute(CLASSIFICATION).map(|a| &a.values) {
        let mut agree = 0;
        let (mut tp, mut fp, mut fneg) = (0, 0, 0);
        for (g, &c) in ground.iter().zip(classes) {
            let truth = c == 2;
            agree += usize::from(*g == truth);
            match (*g, truth) {
                (true, true) => tp += 1,
                (true, false) => fp += 1,
                (false, true) => fneg += 1,
                _ => {}
            }
        }
        println!(
            "vs file classes: agreement {:.1} %, ground found {tp}, missed {fneg}, false ground {fp}",
            100.0 * agree as f64 / cloud.len() as f64
        );
    }
    Ok(())
}
