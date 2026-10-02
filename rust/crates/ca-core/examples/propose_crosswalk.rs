//! `cargo run -p ca-core --example propose_crosswalk -- cloud.pcd options.json`
//! Reads the complete compatibility cloud; does not add or classify a feature.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("expected cloud path and box options JSON path".into());
    }
    let cloud = ca_core::read(&args[1], &std::fs::read(&args[1])?)?;
    let options = serde_json::from_str(&std::fs::read_to_string(&args[2])?)?;
    let report =
        ca_core::vector_map::crosswalks::propose(&vectormap_core::Map::new(), &cloud, &options)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
