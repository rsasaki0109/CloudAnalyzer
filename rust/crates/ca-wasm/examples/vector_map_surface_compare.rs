//! Compare source-footprint and width-prior roads without any reference inputs.
use ca_core::{
    trajectory,
    vector_map::{self, BuildOptions},
};
use serde_json::json;
use std::{env, fs, path::Path};
use vectormap_core::Map;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len() != 4 {
        return Err("expected cloud trajectory.csv options.json NEW-output-directory".into());
    }
    let output = Path::new(&args[3]);
    if output.exists() {
        return Err("choose a new output directory".into());
    }
    let cloud = ca_core::read(&args[0], &fs::read(&args[0])?)?;
    let poses = trajectory::parse(&fs::read_to_string(&args[1])?, trajectory::Format::Csv)?;
    let mut options: BuildOptions = serde_json::from_str(&fs::read_to_string(&args[2])?)?;
    fs::create_dir_all(output)?;
    let mut summaries = Vec::new();
    for mode in [false, true] {
        options.fit_source_surface = mode;
        let mut map = Map::new();
        let name = if mode {
            "source-footprint"
        } else {
            "width-prior"
        };
        match vector_map::build(&mut map, &cloud, &poses.positions, &options) {
            Ok(report) => {
                let quality = vector_map::quality::audit(&map, &cloud)?;
                let json = vectormap_io::json::to_string(&map);
                fs::write(output.join(format!("{name}.json")), json)?;
                fs::write(
                    output.join(format!("{name}-report.json")),
                    serde_json::to_string_pretty(
                        &json!({"build":report,"quality":quality,"reference_inputs":[]}),
                    )?,
                )?;
                summaries.push(json!({"mode":name,"generated_length":report.generated_length,"lanes":quality.lanes.len(),"needs_review":quality.low_support_lanes.len(),"surface_fit":report.surface_fit}));
            }
            Err(error) => summaries
                .push(json!({"mode":name,"deferred_entire_path":true,"error":error.to_string()})),
        }
    }
    fs::write(
        output.join("comparison.json"),
        serde_json::to_string_pretty(&summaries)?,
    )?;
    println!("{}", serde_json::to_string(&summaries)?);
    Ok(())
}
