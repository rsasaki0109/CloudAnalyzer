//! Parity with CloudCompare: its results on two synthetic surveys, recorded
//! by `scripts/make_cloudcompare_fixtures.py` (see `tests/cloudcompare/`).

use ca_core::ground::{CsfParams, Rigidness, csf};
use ca_core::m3c2::{M3c2Params, m3c2};
use ca_core::volume::{CellHeight, Surface, VolumeParams, volume};
use ca_core::{PointCloud, cloud_to_cloud, filter};

fn path(name: &str) -> String {
    format!("{}/tests/cloudcompare/{name}", env!("CARGO_MANIFEST_DIR"))
}

fn cloud(name: &str) -> PointCloud {
    ca_core::read(name, &std::fs::read(path(name)).unwrap()).unwrap()
}

fn numbers(name: &str) -> Vec<f64> {
    std::fs::read_to_string(path(name))
        .unwrap()
        .lines()
        .filter(|l| !l.trim().is_empty())
        .map(|l| l.trim().parse().unwrap())
        .collect()
}

fn indices(name: &str) -> Vec<usize> {
    numbers(name).into_iter().map(|v| v as usize).collect()
}

#[test]
fn c2c_distances_match() {
    let (before, after) = (cloud("before.xyz"), cloud("after.xyz"));
    let ours = cloud_to_cloud(&after, &before).unwrap();
    let theirs = numbers("c2c.txt");
    assert_eq!(ours.len(), theirs.len());
    let worst = ours
        .iter()
        .zip(&theirs)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    // CloudCompare stores float32 coordinates and prints 6 decimals.
    assert!(worst < 2e-5, "largest difference {worst}");
}

#[test]
fn sor_removes_the_same_points() {
    // CloudCompare's k = 8 counts the point itself.
    let before = cloud("before.xyz");
    let kept = filter::statistical_outliers(&before, 7, 1.0);
    let mut is_kept = vec![false; before.len()];
    for i in kept {
        is_kept[i] = true;
    }
    let removed: Vec<usize> = (0..before.len()).filter(|&i| !is_kept[i]).collect();
    assert_eq!(removed, indices("sor_removed.txt"));
}

#[test]
fn csf_classifies_the_same_points_as_ground() {
    let before = cloud("before.xyz");
    let params = CsfParams {
        cloth_resolution: 1.0,
        class_threshold: 0.3,
        rigidness: Rigidness::Relief,
        ..CsfParams::default()
    };
    let ground = csf(&before, params).unwrap();
    let ours: Vec<usize> = (0..before.len()).filter(|&i| ground[i]).collect();
    assert_eq!(ours, indices("csf_ground.txt"));
}

#[test]
fn m3c2_distances_match() {
    let (before, after) = (cloud("before.xyz"), cloud("after.xyz"));
    // CloudCompare's scales are diameters and its depth the half-length; it
    // measures with any number of points.
    let params = M3c2Params {
        normal_radius: 1.0,
        projection_radius: 0.5,
        max_depth: 2.0,
        min_points: 1,
        ..M3c2Params::default()
    };
    let ours = m3c2(
        &before.positions,
        &before.positions,
        &after.positions,
        params,
    )
    .unwrap()
    .distance;
    let theirs = numbers("m3c2.txt");
    let mut delta: Vec<f64> = ours
        .iter()
        .zip(&theirs)
        .filter(|(a, b)| a.is_finite() && b.is_finite())
        .map(|(a, b)| (a - b).abs())
        .collect();
    // The same core points are measured...
    let measured = |v: &[f64]| v.iter().filter(|x| x.is_finite()).count();
    assert_eq!(measured(&ours), measured(&theirs));
    assert_eq!(delta.len(), measured(&theirs));
    delta.sort_by(f64::total_cmp);
    let median = delta[delta.len() / 2];
    let p95 = delta[delta.len() * 95 / 100];
    // ...with the same distances, up to float32 and a few normals.
    assert!(
        median < 1e-4 && p95 < 5e-4,
        "median {median}, 95th percentile {p95}"
    );
}

#[test]
fn volumes_match() {
    let (before, after) = (cloud("before.xyz"), cloud("after.xyz"));
    let params = VolumeParams {
        cell: 1.0,
        height: CellHeight::Mean,
        fill_empty: false,
    };
    let ours = volume(
        Surface::Points(&before.positions),
        Surface::Points(&after.positions),
        params,
    )
    .unwrap();
    let theirs = numbers("volume.txt");
    for (ours, theirs) in [(ours.added, theirs[0]), (ours.removed, theirs[1])] {
        assert!(
            (ours / theirs - 1.0).abs() < 5e-4,
            "ours {ours}, CloudCompare {theirs}"
        );
    }
}
