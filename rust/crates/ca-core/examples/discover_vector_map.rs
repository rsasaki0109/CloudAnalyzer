//! Whole-cloud development evaluation: point cloud + recorded trajectory only.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 4 {
        return Err("expected cloud, trajectory CSV, build options JSON".into());
    }
    let cloud = ca_core::read(&args[1], &std::fs::read(&args[1])?)?;
    let mut map = vectormap_core::Map::new();
    let roads = if args[2] != "-" {
        let trajectory = ca_core::trajectory::parse(
            &std::fs::read_to_string(&args[2])?,
            ca_core::trajectory::Format::Csv,
        )?;
        let options = serde_json::from_str(&std::fs::read_to_string(&args[3])?)?;
        Some(ca_core::vector_map::build(
            &mut map,
            &cloud,
            &trajectory.positions,
            &options,
        )?)
    } else {
        None
    };
    let mut discovery_options = ca_core::vector_map::discovery::DiscoveryOptions::default();
    if args[2] == "-" {
        discovery_options.scope = ca_core::vector_map::discovery::SearchScope::GroundSurface;
    }
    let discovery = ca_core::vector_map::discovery::propose(&map, &cloud, &discovery_options)?;
    println!(
        "{}",
        serde_json::to_string_pretty(&serde_json::json!({"roads":roads,"discovery":discovery}))?
    );
    Ok(())
}
