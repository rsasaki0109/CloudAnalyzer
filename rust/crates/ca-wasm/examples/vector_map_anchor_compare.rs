//! Source-only anchor comparison. No survey/reference map is read here.
use ca_core::{
    trajectory,
    vector_map::{self, BuildOptions},
};
use serde_json::{Value, json};
use std::{env, fs, path::Path};
use vectormap_core::Map;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    if args.len() != 3 {
        return Err("expected cloud cases.json NEW-output-directory".into());
    }
    let output = Path::new(&args[2]);
    if output.exists() {
        return Err("choose a new output directory".into());
    }
    let config: Value = serde_json::from_str(&fs::read_to_string(&args[1])?)?;
    if config["reference_inputs"] != json!([]) {
        return Err("reference inputs must be empty".into());
    }
    let cloud = ca_core::read(&args[0], &fs::read(&args[0])?)?;
    fs::create_dir_all(output)?;
    let mut cases = vec![];
    let mut maps = [Map::new(), Map::new()];
    for (i, case) in config["cases"]
        .as_array()
        .ok_or("missing cases")?
        .iter()
        .enumerate()
    {
        let csv = if let Some(csv) = case["trajectory_csv"].as_str() {
            csv.to_owned()
        } else {
            fs::read_to_string(case["trajectory"].as_str().ok_or("missing trajectory")?)?
        };
        fs::write(output.join(format!("trajectory-{i}.csv")), &csv)?;
        let poses = trajectory::parse(&csv, trajectory::Format::Csv)?;
        let mut summaries = vec![];
        for (mode, map) in maps.iter_mut().enumerate() {
            let mut options: BuildOptions = serde_json::from_value(case["options"].clone())?;
            options.fit_source_surface = true;
            let align = config["comparison"].as_str() == Some("curb_trace_alignment");
            let paint = config["comparison"].as_str() == Some("paint_corridor");
            let divider = config["comparison"].as_str() == Some("paint_divider");
            options.fit_paint_divider = divider && mode == 1;
            options.physical_anchors_only = align || paint || divider || mode == 1;
            options.align_trace_to_curbs = paint || divider || (align && mode == 1);
            options.fit_paint_corridor = divider || (paint && mode == 1);
            let (roads, extracted) = vector_map::extract(&cloud, &poses.positions, &options)?;
            let shift = extracted
                .trace_alignment
                .as_ref()
                .map(|r| r.shift_xy)
                .unwrap_or([0.0; 2]);
            let geometry: Vec<_> = roads.iter().map(|r| {
                let operator_reference: Vec<_> = r.reference.iter().map(|p| [p[0]-shift[0], p[1]-shift[1], p[2]]).collect();
                if align || paint || divider { json!({"reference":r.reference,"operator_reference":operator_reference,"boundaries":r.boundaries,"evidence":r.evidence}) }
                else { json!({"reference":r.reference,"boundaries":r.boundaries,"evidence":r.evidence}) }
            }).collect();
            let report = vector_map::build(map, &cloud, &poses.positions, &options)?;
            let quality = vector_map::quality::audit(map, &cloud)?;
            if report.reused_intervals != 0 {
                return Err("paired profiles require all extracted intervals to be added".into());
            }
            let (osm, issues) = vectormap_io::lanelet2::write_string(
                map,
                &vectormap_io::lanelet2::SaveOptions::autoware(),
            );
            if issues
                .iter()
                .any(|i| i.severity == vectormap_core::Severity::Error)
            {
                return Err("Lanelet2 export failed".into());
            }
            let name = if mode == 0 { "before" } else { "after" };
            fs::write(
                output.join(format!("{name}-{i}.json")),
                vectormap_io::json::to_string(map),
            )?;
            fs::write(output.join(format!("{name}-{i}.osm")), osm)?;
            let value = json!({"name":case["name"],"extraction":report,"quality":quality});
            fs::write(
                output.join(format!("{name}-{i}-audit.json")),
                serde_json::to_string_pretty(&value)?,
            )?;
            fs::write(
                output.join(format!("{name}-{i}-profiles.json")),
                serde_json::to_string_pretty(
                    &json!({"roads":geometry,"extraction":extracted,"reference_inputs":[]}),
                )?,
            )?;
            summaries.push(json!({"mode":name,"generated_m":report.generated_length,"deferred_m":report.surface_fit.as_ref().map(|f|f.deferred_length_m),"ignored":report.coverage_edge_anchor_candidates_ignored,"flags":quality.low_support_lanes,"lane_fragments":quality.lanes.len()}));
        }
        cases.push(json!({"name":case["name"],"modes":summaries}));
    }
    fs::write(
        output.join("comparison.json"),
        serde_json::to_string_pretty(&cases)?,
    )?;
    println!("{}", serde_json::to_string(&cases)?);
    Ok(())
}
