//! Controlled junction ablation on an external reference map and real cloud.
//! Remove labelled connector lanes, keep surveyed road legs, generate connections
//! without reference topology, and only then compare pairs against that topology.
//! Usage: cloud.pcd reference.osm output-directory [options.json]

use std::{collections::BTreeSet, env, fs, path::PathBuf};

use ca_core::vector_map::junctions::{self, JunctionOptions};
use serde_json::json;
use vectormap_core::{LaneId, LaneKind, Map};
use vectormap_io::{autoware, json as irjson, lanelet2};

fn truth_pairs(reference: &Map, legs: &BTreeSet<LaneId>) -> BTreeSet<[u64; 2]> {
    let mut pairs = BTreeSet::new();
    for &from in legs {
        let mut queue: Vec<_> = reference.successors(from).to_vec();
        let mut visited = BTreeSet::new();
        while let Some(id) = queue.pop() {
            if !visited.insert(id) {
                continue;
            }
            if legs.contains(&id) {
                // Direct leg-to-leg links are retained in the ablated input.
                if !reference.successors(from).contains(&id) {
                    pairs.insert([from.0, id.0]);
                }
            } else if reference
                .lane(id)
                .is_some_and(|l| l.turn_direction.is_some())
            {
                queue.extend(reference.successors(id));
            }
        }
    }
    pairs
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    if !(3..=4).contains(&args.len()) {
        return Err("expected cloud.pcd reference.osm output-directory [options.json]".into());
    }
    let out = PathBuf::from(&args[2]);
    if out.exists() {
        return Err("output directory already exists".into());
    }
    let cloud = ca_core::read(&args[0], &fs::read(&args[0])?)?;
    let loaded = lanelet2::read_str(&fs::read_to_string(&args[1])?, &Default::default())?;
    let reference = loaded.map;
    let legs: BTreeSet<_> = reference
        .lanes()
        .filter(|l| l.kind == LaneKind::Driving && l.turn_direction.is_none())
        .map(|l| l.id)
        .collect();
    let expected = truth_pairs(&reference, &legs);
    let removed: Vec<_> = reference
        .lanes()
        .filter(|l| l.turn_direction.is_some())
        .map(|l| l.id)
        .collect();
    let mut map = reference.clone();
    for &id in &removed {
        map.remove_lane(id)?;
    }
    let input = map.clone();
    let options: JunctionOptions = if let Some(path) = args.get(3) {
        serde_json::from_str(&fs::read_to_string(path)?)?
    } else {
        Default::default()
    };
    // The reference is not passed to either proposal or connection generation.
    let preview = junctions::propose(&map, &cloud, &options)?;
    assert_eq!(map, input, "preview changed input");
    let result = junctions::connect(&mut map, &cloud, &options, None)?;
    let generated: BTreeSet<_> = result
        .added
        .iter()
        .map(|&id| {
            assert_eq!(map.predecessors(id).len(), 1);
            assert_eq!(map.successors(id).len(), 1);
            [map.predecessors(id)[0].0, map.successors(id)[0].0]
        })
        .collect();
    let proposed: BTreeSet<_> = preview
        .candidates
        .iter()
        .map(|c| [c.from.0, c.to.0])
        .collect();
    assert_eq!(
        generated, proposed,
        "applied geometry differs from preview pairs"
    );
    let old_geometry_retained = input.lanes().all(|l| map.lane(l.id) == Some(l))
        && input.boundaries().all(|b| map.boundary(b.id) == Some(b))
        && input
            .regulatory_elements()
            .all(|r| map.regulatory_element(r.id) == Some(r))
        && input.metadata() == map.metadata();
    assert!(old_geometry_retained);
    let connected = map.clone();
    let replay = junctions::connect(&mut map, &cloud, &options, None)?;
    assert_eq!(map, connected, "replay changed existing connections");
    let tp = generated.intersection(&expected).count();
    let precision = tp as f64 / generated.len().max(1) as f64;
    let recall = tp as f64 / expected.len().max(1) as f64;
    let (osm, export_issues) = lanelet2::write_string(&map, &lanelet2::SaveOptions::autoware());
    let (_, input_export_issues) =
        lanelet2::write_string(&input, &lanelet2::SaveOptions::autoware());
    let roundtrip = lanelet2::read_str(&osm, &Default::default())?;
    for &id in &result.added {
        let lane = roundtrip.map.lane(id).ok_or("export lost connector lane")?;
        assert_eq!(
            lane.attributes
                .get_prefixed("lanelet2", "cloudanalyzer_review_required"),
            Some("yes")
        );
        assert_eq!(roundtrip.map.predecessors(id), map.predecessors(id));
        assert_eq!(roundtrip.map.successors(id), map.successors(id));
    }
    let report = json!({
        "experiment":"labelled reference connectors removed; surveyed legs retained",
        "removed_connector_lanes":removed.len(), "reference_leg_lanes":legs.len(),
        "input_lanes":input.lane_count(), "cloud_points":cloud.len(),
        "truth_pairs":expected, "generated_pairs":generated, "true_positive_pairs":tp,
        "false_positive_pairs":generated.difference(&expected).collect::<Vec<_>>(),
        "missed_pairs":expected.difference(&generated).collect::<Vec<_>>(),
        "pair_precision":precision, "pair_recall":recall,
        "existing_geometry_ids_rules_metadata_retained":old_geometry_retained,
        "replay_added":replay.added.len(), "junctions":result, "options":options,
        "import_issues":loaded.issues, "export_issues":export_issues,
        "autoware_issues":autoware::check(&map),
        "input_autoware_issues":autoware::check(&input),
        "input_export_issues":input_export_issues,
        "input_validation":vectormap_validation::validate(&input,&Default::default()),
        "roundtrip_added_lanes":result.added.len(),
        "validation":vectormap_validation::validate(&map,&Default::default()),
        "limitations":"Controlled ablation with surveyed input legs, not an end-to-end lane extraction score or held-out accuracy. Ground support does not establish permitted turns."
    });
    fs::create_dir_all(&out)?;
    fs::write(out.join("input.json"), irjson::to_string(&input))?;
    fs::write(out.join("vector_map.json"), irjson::to_string(&map))?;
    fs::write(out.join("lanelet2_map.osm"), osm)?;
    fs::write(
        out.join("map_projector_info.yaml"),
        autoware::projector_info_yaml(map.metadata().georeference),
    )?;
    fs::write(
        out.join("report.json"),
        serde_json::to_string_pretty(&report)?,
    )?;
    println!(
        "{} generated pairs / {} reference pairs, precision {:.4}, recall {:.4}; {} old lanes retained, replay added {}",
        generated.len(),
        expected.len(),
        precision,
        recall,
        input.lane_count(),
        replay.added.len()
    );
    Ok(())
}
