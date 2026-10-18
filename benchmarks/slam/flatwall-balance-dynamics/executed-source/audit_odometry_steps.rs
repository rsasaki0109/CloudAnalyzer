//! Replay prepared real scans, observe selected ICP steps, or time native registration.
//! Build with `--release --features parallel`; input/output paths are explicit.

use ca_core::PointCloud;
use ca_core::icp::Rigid;
use ca_core::odometry::{Odometry, OdometryParams, Registration};
use serde::Deserialize;
use serde_json::{Value, json};
use std::collections::{BTreeMap, HashMap};
use std::fs::{self, OpenOptions};
use std::io::Write;
use std::path::Path;
use std::time::Instant;

#[derive(Deserialize)]
struct Recording {
    name: String,
    raw: String,
    gyro: String,
    scan_indices: Vec<usize>,
}

#[derive(Deserialize)]
struct Expected {
    pose: [f64; 16],
    iterations: usize,
    termination: String,
    threshold_m: f64,
}

#[derive(Deserialize)]
struct Frame {
    scan_index: usize,
    baseline: Expected,
    balanced: Expected,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    recording: String,
    compensation: String,
    policy: String,
    registration_voxel: Option<f64>,
    trace_indices: Vec<usize>,
    frames: Vec<Frame>,
}

#[derive(Deserialize)]
struct Input {
    schema: String,
    recordings: Vec<Recording>,
    cases: Vec<Case>,
    points_per_scan: usize,
    warmup_pairs: usize,
    measured_pairs: usize,
    tail_states: usize,
}

fn save(path: &Path, value: &Value) -> Result<(), Box<dyn std::error::Error>> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    serde_json::to_writer(&mut file, value)?;
    file.write_all(b"\n")?;
    Ok(())
}

fn load_scans(path: &str, count: usize, points: usize) -> Vec<PointCloud> {
    let bytes = fs::read(path).expect("prepared scan file");
    assert_eq!(bytes.len(), count * points * 24);
    bytes
        .chunks_exact(points * 24)
        .map(|scan| PointCloud {
            positions: scan
                .chunks_exact(24)
                .map(|p| {
                    std::array::from_fn(|i| {
                        f64::from_le_bytes(p[i * 8..i * 8 + 8].try_into().unwrap())
                    })
                })
                .collect(),
            ..PointCloud::default()
        })
        .collect()
}

fn key(point: &[f64; 3]) -> [i64; 3] {
    point.map(|v| v.floor() as i64)
}

fn coordinate(point: &[f64; 3]) -> [u64; 3] {
    point.map(|v| if v == 0.0 { 0 } else { v.to_bits() })
}

#[derive(Default)]
struct Mirror(BTreeMap<[i64; 3], Vec<[f64; 3]>>);

impl Mirror {
    fn insert(&mut self, points: &[[f64; 3]], center: [f64; 3]) {
        for point in points {
            let cell = self.0.entry(key(point)).or_default();
            if cell.len() < 20 {
                cell.push(*point);
            }
        }
        self.0.retain(|_, cell| {
            let d: f64 = cell[0]
                .iter()
                .zip(center)
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            d.sqrt() <= 80.0
        });
    }
}

/// Separate exhaustive cell search, without the native cell-bound pruning.
struct Observer {
    points: Vec<[f64; 3]>,
    cells: BTreeMap<[i64; 3], Vec<usize>>,
}

impl Observer {
    fn new(mirror: &Mirror) -> Self {
        let mut points = Vec::new();
        let cells = mirror
            .0
            .iter()
            .map(|(key, cell)| {
                let start = points.len();
                points.extend(cell);
                (*key, (start..points.len()).collect())
            })
            .collect();
        Self { points, cells }
    }

    fn nearest(&self, q: &[f64; 3]) -> Option<(usize, f64)> {
        let cell = key(q);
        let mut best = (0, f64::INFINITY);
        let search = |cell: [i64; 3], best: &mut (usize, f64)| {
            for &index in self.cells.get(&cell).into_iter().flatten() {
                let p = self.points[index];
                let squared = (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2);
                if squared < best.1 {
                    *best = (index, squared);
                }
            }
        };
        search(cell, &mut best);
        for x in -1..=1 {
            for y in -1..=1 {
                for z in -1..=1 {
                    if [x, y, z] != [0, 0, 0] {
                        search([cell[0] + x, cell[1] + y, cell[2] + z], &mut best);
                    }
                }
            }
        }
        best.1.is_finite().then_some(best)
    }

    fn state(&self, r: &Registration) -> Value {
        let world: Vec<_> = r.source().iter().map(|p| r.total().apply(p)).collect();
        let pairs: Vec<_> = world
            .iter()
            .enumerate()
            .filter_map(|(i, q)| {
                self.nearest(q)
                    .filter(|(_, d)| *d <= r.threshold_sq())
                    .map(|(j, d)| (i, j, d))
            })
            .collect();
        let mut counts = HashMap::<[u64; 3], usize>::new();
        for &(_, target, _) in &pairs {
            *counts.entry(coordinate(&self.points[target])).or_default() += 1;
        }
        let multiplicity: Vec<_> = pairs
            .iter()
            .map(|p| counts[&coordinate(&self.points[p.1])])
            .collect();
        let robust: Vec<_> = pairs
            .iter()
            .map(|p| r.kernel().powi(2) / (r.kernel() + p.2).powi(2))
            .collect();
        let weights: Vec<_> = robust
            .iter()
            .zip(&multiplicity)
            .map(|(w, n)| w / *n as f64)
            .collect();
        let cost: f64 = weights.iter().zip(&pairs).map(|(w, p)| w * p.2).sum();
        json!({"source_indices": pairs.iter().map(|p| p.0).collect::<Vec<_>>(),
            "target_indices": pairs.iter().map(|p| p.1).collect::<Vec<_>>(),
            "multiplicity": multiplicity, "weights": weights, "weighted_cost": cost,
            "matches": pairs.len(), "unique_targets": counts.len()})
    }
}

fn begin(odometry: &Odometry, scan: &PointCloud, policy: &str) -> Registration {
    if policy == "last-pose" && !odometry.poses().is_empty() {
        odometry
            .begin_with_prediction(scan, &Rigid::IDENTITY)
            .unwrap()
    } else {
        assert!(policy == "last-pose" || policy == "repeat-motion");
        odometry.begin(scan)
    }
}

fn check(odometry: &Odometry, expected: &Expected) -> Value {
    let report = odometry.last_report().unwrap();
    let pose = odometry.poses().last().unwrap().to_matrix();
    let error = pose
        .iter()
        .zip(expected.pose)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f64::max);
    assert!(error <= 2e-8, "replay pose differs: {error}");
    assert_eq!(report.iterations, expected.iterations);
    assert_eq!(report.termination.as_str(), expected.termination);
    assert!((report.threshold_m - expected.threshold_m).abs() <= 2e-8);
    json!({"pose": pose, "pose_max_abs_error": error,
        "pose_bits_match": pose.iter().zip(expected.pose).all(|(a,b)| a.to_bits()==b.to_bits()),
        "iterations": report.iterations, "termination": report.termination.as_str(),
        "threshold_m": report.threshold_m})
}

fn trace(
    odometry: &Odometry,
    mut r: Registration,
    mirror: &Mirror,
    tail: usize,
) -> (Registration, Value) {
    let observer = Observer::new(mirror);
    assert_eq!(observer.points.len(), odometry.map().len());
    let mut states = Vec::new();
    while r.termination().is_none() {
        let before = r.total().to_matrix();
        let (h, b) = odometry.equations(&r);
        let mut state = observer.state(&r);
        state["before"] = json!(before);
        state["normal_matrix"] = json!(h);
        state["right_hand_side"] = json!(b);
        odometry.step(&mut r, (h, b));
        state["after"] = json!(r.total().to_matrix());
        states.push(state);
        // Keep complete pair rows only in the predeclared tail; all systems/poses remain.
        if states.len() > tail {
            let index = states.len() - tail - 1;
            for name in [
                "source_indices",
                "target_indices",
                "multiplicity",
                "weights",
            ] {
                states[index].as_object_mut().unwrap().remove(name);
            }
        }
    }
    let value = json!({"source_at_prediction": r.source(), "map_points_world": observer.points,
        "center_world": r.center(), "threshold_sq": r.threshold_sq(), "kernel": r.kernel(),
        "states": states, "termination": r.termination().unwrap().as_str()});
    (r, value)
}

fn run_case(
    case: &Case,
    scans: &[PointCloud],
    balance: bool,
    out: Option<&Path>,
    tail: usize,
) -> Value {
    let mut odometry = Odometry::new(OdometryParams {
        diagnostics: out.is_some(),
        ..OdometryParams::default()
    });
    odometry.set_target_balance(balance);
    odometry
        .set_registration_voxel(case.registration_voxel)
        .unwrap();
    let mut mirror = Mirror::default();
    let mut rows = Vec::new();
    let mut times = Vec::new();
    for (scan, frame) in scans.iter().zip(&case.frames) {
        let observe = balance && out.is_some() && case.trace_indices.contains(&frame.scan_index);
        odometry.set_capture_linearization(observe);
        if observe {
            let r = begin(&odometry, scan, &case.policy);
            let (r, mut captured) = trace(&odometry, r, &mirror, tail);
            odometry.finish(r);
            let s = odometry.last_linearization().unwrap();
            captured["final_source_world"] = json!(s.source_world);
            captured["final_source_indices"] = json!(s.matched_source_indices);
            captured["final_target_world"] = json!(s.target_world);
            captured["final_weights"] = json!(s.weights);
            captured["returned_pose"] = json!(s.pose.to_matrix());
            let path = out
                .unwrap()
                .join(&case.name)
                .join(format!("scan-{:03}.json", frame.scan_index));
            fs::create_dir_all(path.parent().unwrap()).unwrap();
            save(&path, &captured).unwrap();
        } else {
            let start = Instant::now();
            if case.policy == "last-pose" && !odometry.poses().is_empty() {
                odometry
                    .register_with_prediction(scan, &Rigid::IDENTITY)
                    .unwrap();
            } else {
                odometry.register(scan);
            }
            times.push(start.elapsed().as_nanos() as u64);
        }
        let expected = if balance {
            &frame.balanced
        } else {
            &frame.baseline
        };
        let mut checked = check(&odometry, expected);
        checked["scan_index"] = json!(frame.scan_index);
        rows.push(checked);
        if out.is_some() {
            mirror.insert(
                odometry.placed(),
                odometry.poses().last().unwrap().translation,
            );
        }
    }
    json!({"case": case.name, "target_balance": balance, "frames": rows,
        "registration_ns": times, "total_registration_ns": times.iter().sum::<u64>()})
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args().collect();
    assert_eq!(
        args.len(),
        4,
        "audit_odometry_steps trace|time input.json fresh-output-dir"
    );
    let mode = &args[1];
    assert!(mode == "trace" || mode == "time");
    assert_eq!(std::env::var("RAYON_NUM_THREADS")?.as_str(), "4");
    let input: Input = serde_json::from_slice(&fs::read(&args[2])?)?;
    assert_eq!(
        input.schema,
        "cloudanalyzer.flatwall_balance_dynamics_input.v1"
    );
    assert_eq!(input.cases.len(), 16);
    assert_eq!(
        (input.warmup_pairs, input.measured_pairs, input.tail_states),
        (1, 3, 24)
    );
    let out = Path::new(&args[3]);
    fs::create_dir(out)?;
    // Read/prepare all points before any registration timing window.
    let mut prepared = HashMap::new();
    for recording in &input.recordings {
        for (arm, path) in [("raw", &recording.raw), ("gyro", &recording.gyro)] {
            prepared.insert(
                (recording.name.clone(), arm.to_string()),
                load_scans(path, recording.scan_indices.len(), input.points_per_scan),
            );
        }
    }
    let mut results = Vec::new();
    let rounds = if mode == "trace" {
        1
    } else {
        input.warmup_pairs + input.measured_pairs
    };
    for round in 0..rounds {
        for (case_number, case) in input.cases.iter().enumerate() {
            let scans = &prepared[&(case.recording.clone(), case.compensation.clone())];
            assert_eq!(scans.len(), case.frames.len());
            let order = if (round + case_number) % 2 == 0 {
                [false, true]
            } else {
                [true, false]
            };
            for balance in order {
                let mut row = run_case(
                    case,
                    scans,
                    balance,
                    (mode == "trace").then_some(out),
                    input.tail_states,
                );
                row["round"] = json!(round);
                row["warmup"] = json!(mode == "time" && round < input.warmup_pairs);
                results.push(row);
            }
            eprintln!(
                "{}",
                json!({"mode": mode, "round": round, "case": case.name, "complete": true})
            );
        }
    }
    save(
        &out.join("runs.json"),
        &json!({"mode": mode, "runs": results,
        "diagnostic_capture_enabled": mode == "trace", "requested_rayon_threads": 4,
        "metric": "sum of native register calls on preloaded real scans; includes filtering, ICP, threshold learning, map insertion/pruning; excludes bag I/O, deskew, input preparation, validation and output serialization"}),
    )?;
    Ok(())
}
