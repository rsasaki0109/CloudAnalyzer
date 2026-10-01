//! Reproducible XY-boundary evaluation on external (not redistributed) maps.
//! Usage: cargo run -p ca-wasm --release --example vector_map_evaluate --
//!   cloud.pcd trajectory.csv reference.osm output-directory

use ca_core::{
    kdtree::KdTree,
    trajectory,
    vector_map::{self, BuildOptions},
};
use serde_json::json;
use std::{collections::BTreeSet, env, fs, path::PathBuf};
use vectormap_core::{BoundaryKind, Map, Point2, Polyline3};
use vectormap_io::{autoware, lanelet2};

fn sample(line: &Polyline3, spacing: f64) -> Vec<[f64; 3]> {
    let mut out = Vec::new();
    for p in line.points.windows(2) {
        let n = ((p[1].x - p[0].x).hypot(p[1].y - p[0].y) / spacing)
            .ceil()
            .max(1.0) as usize;
        for i in 0..n {
            let t = (i as f64 + 0.5) / n as f64;
            out.push([
                p[0].x + (p[1].x - p[0].x) * t,
                p[0].y + (p[1].y - p[0].y) * t,
                0.0,
            ]);
        }
    }
    out
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = env::args().skip(1).collect();
    if !(4..=5).contains(&args.len()) {
        return Err(
            "expected cloud.pcd trajectory.csv reference.osm output-directory [options.json]"
                .into(),
        );
    }
    let cloud = ca_core::read(&args[0], &fs::read(&args[0])?)?;
    let poses = trajectory::parse(&fs::read_to_string(&args[1])?, trajectory::Format::Csv)?;
    let truth = if args[2] == "-" {
        Map::new()
    } else {
        lanelet2::read_str(&fs::read_to_string(&args[2])?, &Default::default())?.map
    };
    let defaults = BuildOptions {
        left_hand_traffic: args[2] != "-",
        ..Default::default()
    };
    let options: BuildOptions = if let Some(path) = args.get(4) {
        serde_json::from_str(&fs::read_to_string(path)?)?
    } else {
        defaults
    };
    let mut map = Map::new();
    // Coordinate metadata only; reference lane geometry is used only for scoring.
    map.metadata_mut().georeference = truth.metadata().georeference;
    let report = vector_map::build(&mut map, &cloud, &poses.positions, &options)?;
    let (extracted, _) = vector_map::extract(&cloud, &poses.positions, &options)?;
    let observed: Vec<_> = extracted
        .iter()
        .flat_map(|road| {
            road.source_boundaries
                .as_ref()
                .unwrap_or(&road.boundaries)
                .iter()
                .zip(&road.evidence)
        })
        .flat_map(|(points, labels)| points.iter().zip(labels))
        .filter(|(_, e)| **e != vector_map::Evidence::WidthPrior)
        .map(|(p, _)| [p[0], p[1], 0.0])
        .collect();
    // Score only vehicle lanes visited by the recorded drive and their neighbors.
    let mut lane_ids = BTreeSet::new();
    for p in &poses.positions {
        if let Some(near) = truth.find_nearest_lane(Point2::new(p[0], p[1]))
            && near.distance < 5.0
        {
            lane_ids.insert(near.lane);
            for side in [vectormap_core::Side::Left, vectormap_core::Side::Right] {
                if let Some(n) = truth.neighbor(near.lane, side) {
                    lane_ids.insert(n.lane);
                }
            }
        }
    }
    let boundary_ids: BTreeSet<_> = lane_ids
        .iter()
        .filter_map(|id| truth.lane(*id))
        .flat_map(|l| [l.left.boundary, l.right.boundary])
        .collect();
    let ground_truth: Vec<_> = boundary_ids
        .iter()
        .filter_map(|id| truth.boundary(*id))
        .filter(|b| b.kind != BoundaryKind::Virtual)
        .flat_map(|b| sample(&b.geometry, 0.1))
        .collect();
    let generated: Vec<_> = map
        .boundaries()
        .flat_map(|b| sample(&b.geometry, 0.1))
        .collect();
    let out = PathBuf::from(&args[3]);
    fs::create_dir_all(&out)?;
    let comparison = json!({
        "options":options,"trajectory":poses.positions,
        "cloud":cloud.positions.iter().step_by((cloud.len()/20_000).max(1)).collect::<Vec<_>>(),
        "reference":truth.boundaries().map(|b| json!({"id":b.id,"kind":b.kind,"points":b.geometry,
            "scored":boundary_ids.contains(&b.id)})).collect::<Vec<_>>(),
        "reference_centers":truth.lanes().filter_map(|l|truth.centerline(l.id).map(|line|json!({"id":l.id,"points":line,"left":l.left.boundary,"right":l.right.boundary}))).collect::<Vec<_>>(),
        "generated":extracted.iter().map(|r|json!({"reference":r.reference,"boundaries":r.boundaries,"source_boundaries":r.source_boundaries,"evidence":r.evidence})).collect::<Vec<_>>()
    });
    fs::write(
        out.join("comparison.json"),
        serde_json::to_string(&comparison)?,
    )?;
    let issues = autoware::save(&map, &out)?;
    if args[2] == "-" {
        let result =
            json!({"extraction":report,"autoware_issues":issues,"xy_boundary_metrics":null});
        fs::write(
            out.join("report.json"),
            serde_json::to_string_pretty(&result)?,
        )?;
        println!("{}", serde_json::to_string_pretty(&result)?);
        return Ok(());
    }
    let truth_index =
        KdTree::new(&ground_truth).ok_or("no reference boundaries near the trajectory")?;
    let generated_index = KdTree::new(&generated).ok_or("no generated boundaries")?;
    let fraction = |points: &[[f64; 3]], tree: &KdTree| {
        points
            .iter()
            .filter(|p| tree.nearest(p, None).distance_sq <= 0.3 * 0.3)
            .count() as f64
            / points.len().max(1) as f64
    };
    let precision = fraction(&generated, &truth_index);
    let recall = fraction(&ground_truth, &generated_index);
    let f1 = if precision + recall > 0.0 {
        2.0 * precision * recall / (precision + recall)
    } else {
        0.0
    };
    let result = json!({"extraction":report,"xy_boundary_metrics":{"tolerance_m":0.3,"sample_spacing_m":0.1,
        "precision":precision,"recall":recall,"f1":f1,"observed_vertex_precision":fraction(&observed,&truth_index),
        "reference_lanes":lane_ids,"reference_samples":ground_truth.len(),"generated_samples":generated.len()},"autoware_issues":issues});
    fs::write(
        out.join("report.json"),
        serde_json::to_string_pretty(&result)?,
    )?;
    println!("{}", serde_json::to_string_pretty(&result)?);
    Ok(())
}
