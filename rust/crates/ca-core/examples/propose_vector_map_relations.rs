//! Read-only geometric association preview for a recorded map JSON.
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    if args.len() != 3 {
        return Err("expected map JSON and rule ID".into());
    }
    let map: vectormap_core::Map = serde_json::from_str(&std::fs::read_to_string(&args[1])?)?;
    let report = ca_core::vector_map::relation_proposals::propose(&map, args[2].parse()?)?;
    println!("{}", serde_json::to_string_pretty(&report)?);
    Ok(())
}
