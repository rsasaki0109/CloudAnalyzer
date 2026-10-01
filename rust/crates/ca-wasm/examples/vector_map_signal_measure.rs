//! Reproduce a selected-box measurement on a real survey without supplying reference geometry.
use ca_core::vector_map::signals::{SignalOptions, add};
use std::{env, fs};
use vectormap_io::{json, lanelet2};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().collect();
    if args.len() != 4 {
        return Err("usage: vector_map_signal_measure CLOUD MAP OPTIONS.json".into());
    }
    let data = fs::read(&args[1])?;
    let cloud = ca_core::read(&args[1], &data)?;
    let text = fs::read_to_string(&args[2])?;
    let loaded = if args[2].ends_with(".json") {
        json::from_str(&text)?
    } else {
        lanelet2::read_str(&text, &Default::default())?
    };
    if loaded
        .issues
        .iter()
        .any(|i| i.severity == vectormap_core::Severity::Error)
    {
        return Err("input map has import errors".into());
    }
    let mut map = loaded.map;
    let options: SignalOptions = serde_json::from_str(&fs::read_to_string(&args[3])?)?;
    let report = add(&mut map, &cloud, &options)?;
    let (osm, issues) = lanelet2::write_string(&map, &lanelet2::SaveOptions::autoware());
    let restored = lanelet2::read_str(&osm, &Default::default())?;
    let id = report.added.or(report.reused).ok_or("missing signal ID")?;
    let signal = restored
        .map
        .traffic_signal(id)
        .ok_or("signal missing after round trip")?;
    if signal
        .attributes
        .get_prefixed("lanelet2", "cloudanalyzer_geometry_source")
        != Some("point_cloud_box_fit")
    {
        return Err("measurement source lost".into());
    }
    let maximum_roundtrip_error = signal
        .geometry
        .points
        .iter()
        .zip(&report.geometry)
        .map(|(a, b)| (a.x - b[0]).hypot(a.y - b[1]).hypot(a.z - b[2]))
        .fold(0.0_f64, f64::max);
    if maximum_roundtrip_error > 0.005 {
        return Err("roundtrip geometry moved more than 5 mm".into());
    }
    println!(
        "{}",
        serde_json::to_string_pretty(
            &serde_json::json!({"signal":report,"import_issues":loaded.issues,"export_issues":issues,"maximum_roundtrip_error":maximum_roundtrip_error,"roundtrip_geometry":signal.geometry,"roundtrip_height":signal.height})
        )?
    );
    Ok(())
}
