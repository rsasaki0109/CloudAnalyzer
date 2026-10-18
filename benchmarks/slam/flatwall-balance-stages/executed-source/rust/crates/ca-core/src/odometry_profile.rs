//! Ignored release-mode benchmark of fixed real odometry states.
//! No clocks or profiling helpers are compiled into library builds.

use super::{LocalMap, Mat6, Rigid, Vec6, target_balance};
use rayon::prelude::*;
use serde::Deserialize;
use serde_json::json;
use std::hint::black_box;
use std::path::Path;
use std::time::Instant;

#[derive(Deserialize)]
struct Definition {
    name: String,
    file: String,
    source_points: usize,
    map_points: usize,
}

#[derive(Deserialize)]
struct Input {
    schema: String,
    repetitions: usize,
    warmup_rounds: usize,
    measured_rounds: usize,
    states: Vec<Definition>,
}

struct State {
    name: String,
    source: Vec<[f64; 3]>,
    map: LocalMap,
    pose: Rigid,
    center: [f64; 3],
    threshold_sq: f64,
    kernel: f64,
}

struct Pair {
    world: [f64; 3],
    target: [f64; 3],
    distance_sq: f64,
}

fn load(base: &Path, definition: &Definition) -> State {
    let bytes = std::fs::read(base.join(&definition.file)).unwrap();
    assert_eq!(
        bytes.len(),
        (23 + 3 * (definition.source_points + definition.map_points)) * 8
    );
    let values: Vec<_> = bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|b| f64::from_le_bytes(*b))
        .collect();
    assert!(values.iter().all(|v| v.is_finite()));
    assert!(values[19] > 0.0 && values[20] > 0.0 && values[21] > 0.0);
    assert_eq!(values[22], 20.0);
    let source_end = 23 + 3 * definition.source_points;
    let points = |data: &[f64]| data.as_chunks::<3>().0.to_vec();
    let mut map = LocalMap::new(values[21], values[22] as usize);
    map.insert(&points(&values[source_end..]));
    assert_eq!(map.len(), definition.map_points, "map capacity changed");
    State {
        name: definition.name.clone(),
        source: points(&values[23..source_end]),
        map,
        pose: Rigid {
            rotation: std::array::from_fn(|i| std::array::from_fn(|j| values[4 * i + j])),
            translation: [values[3], values[7], values[11]],
        },
        center: values[16..19].try_into().unwrap(),
        threshold_sq: values[19],
        kernel: values[20],
    }
}

fn search(state: &State) -> Vec<Pair> {
    // Same transform, native nearest search, acceptance and ordered Rayon
    // collection as the production target-balanced equations.
    state
        .source
        .par_iter()
        .filter_map(|p| {
            let world = state.pose.apply(p);
            let (target, distance_sq) = state.map.nearest(&world)?;
            (distance_sq <= state.threshold_sq).then_some(Pair {
                world,
                target,
                distance_sq,
            })
        })
        .collect()
}

fn counts(pairs: &[Pair]) -> Vec<usize> {
    // Include the production temporary target-coordinate vector and its drop.
    target_balance::multiplicities(&pairs.iter().map(|p| p.target).collect::<Vec<_>>())
}

fn accumulate(pairs: &[Pair], counts: &[usize], state: &State) -> (Mat6, Vec6) {
    // Keep production arithmetic and 256-pair reduction chunks. Independent
    // verification compares this decomposition with both native equations and
    // published real pair rows; it never drives registration.
    let chunk = |(offset, pairs): (usize, &[Pair])| {
        let mut h = [[0.0; 6]; 6];
        let mut b = [0.0; 6];
        for (i, pair) in pairs.iter().enumerate() {
            let q: [f64; 3] = std::array::from_fn(|k| pair.world[k] - state.center[k]);
            let residual: [f64; 3] = std::array::from_fn(|k| pair.world[k] - pair.target[k]);
            let weight = state.kernel * state.kernel
                / (state.kernel + pair.distance_sq).powi(2)
                / counts[offset * 256 + i] as f64;
            let rows = [
                [1.0, 0.0, 0.0, 0.0, q[2], -q[1]],
                [0.0, 1.0, 0.0, -q[2], 0.0, q[0]],
                [0.0, 0.0, 1.0, q[1], -q[0], 0.0],
            ];
            for (j, r) in rows.iter().zip(residual) {
                for a in 0..6 {
                    b[a] -= weight * j[a] * r;
                    for k in 0..6 {
                        h[a][k] += weight * j[a] * j[k];
                    }
                }
            }
        }
        (h, b)
    };
    let add = |(mut h, mut b): (Mat6, Vec6), (other_h, other_b): (Mat6, Vec6)| {
        for i in 0..6 {
            b[i] += other_b[i];
            for j in 0..6 {
                h[i][j] += other_h[i][j];
            }
        }
        (h, b)
    };
    pairs
        .par_chunks(256)
        .enumerate()
        .map(chunk)
        .reduce(|| ([[0.0; 6]; 6], [0.0; 6]), add)
}

fn whole(state: &State, balance: bool) -> (Mat6, Vec6) {
    if balance {
        target_balance::equations(
            &state.map,
            &state.source,
            &state.pose,
            state.center,
            state.threshold_sq,
            state.kernel,
        )
    } else {
        state.map.normal_equations(
            &state.source,
            &state.pose,
            state.center,
            state.threshold_sq,
            state.kernel,
        )
    }
}

fn error(actual: &(Mat6, Vec6), expected: &(Mat6, Vec6)) -> f64 {
    actual
        .0
        .iter()
        .flatten()
        .chain(&actual.1)
        .zip(expected.0.iter().flatten().chain(&expected.1))
        .map(|(a, b)| {
            assert!(
                (a - b).abs() <= 5e-9 + 5e-12 * b.abs(),
                "decomposed/native H/b differs: {a} {b}"
            );
            (a - b).abs()
        })
        .fold(0.0, f64::max)
}

fn timed_whole(state: &State, balance: bool, repetitions: usize) -> (u64, (Mat6, Vec6)) {
    let mut result = ([[0.0; 6]; 6], [0.0; 6]);
    let start = Instant::now();
    for _ in 0..repetitions {
        result = black_box(whole(black_box(state), balance));
    }
    (start.elapsed().as_nanos().try_into().unwrap(), result)
}

fn timed_stages(state: &State, repetitions: usize) -> ([u64; 5], (Mat6, Vec6)) {
    let mut times = [0_u64; 5];
    let mut result = ([[0.0; 6]; 6], [0.0; 6]);
    let outer = Instant::now();
    for _ in 0..repetitions {
        let start = Instant::now();
        let pairs = search(black_box(state));
        times[0] += start.elapsed().as_nanos() as u64;
        let start = Instant::now();
        let multiplicity = counts(black_box(&pairs));
        times[1] += start.elapsed().as_nanos() as u64;
        let start = Instant::now();
        result = black_box(accumulate(
            black_box(&pairs),
            black_box(&multiplicity),
            state,
        ));
        times[2] += start.elapsed().as_nanos() as u64;
        let start = Instant::now();
        drop(multiplicity);
        drop(pairs);
        times[3] += start.elapsed().as_nanos() as u64;
    }
    times[4] = outer.elapsed().as_nanos().try_into().unwrap();
    assert!(times[..4].iter().sum::<u64>() <= times[4]);
    (times, result)
}

#[test]
#[ignore = "explicit real-state input/output paths; release-mode native profiling only"]
fn record_fixed_states() {
    use std::io::Write;
    let input_path = std::env::var("CA_ODOMETRY_PROFILE_INPUT").expect("input JSON path");
    let output_path = std::env::var("CA_ODOMETRY_PROFILE_OUTPUT").expect("fresh output path");
    let threads: usize = std::env::var("CA_ODOMETRY_PROFILE_THREADS")
        .unwrap()
        .parse()
        .unwrap();
    assert!([1, 4].contains(&threads));
    let input_path = Path::new(&input_path);
    let input: Input = serde_json::from_slice(&std::fs::read(input_path).unwrap()).unwrap();
    assert_eq!(input.schema, "cloudanalyzer.odometry_fixed_states.v1");
    assert_eq!(
        (
            input.repetitions,
            input.warmup_rounds,
            input.measured_rounds
        ),
        (32, 1, 5)
    );
    assert_eq!(input.states.len(), 64);
    let states: Vec<_> = input
        .states
        .iter()
        .map(|d| load(input_path.parent().unwrap(), d))
        .collect();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    let (validation, samples) = pool.install(|| {
        let validation: Vec<_> = states.iter().map(|state| {
            let pairs = search(state);
            let multiplicity = counts(&pairs);
            let source_indices: Vec<_> = state.source.iter().enumerate().filter_map(|(i,p)| {
                state.map.nearest(&state.pose.apply(p))
                    .filter(|(_,d)| *d <= state.threshold_sq).map(|_| i)
            }).collect();
            assert_eq!(pairs.len(), source_indices.len());
            let balanced = whole(state,true);
            let decomposed = accumulate(&pairs,&multiplicity,state);
            let baseline = whole(state,false);
            let maximum_error = error(&decomposed,&balanced);
            json!({"name":state.name, "source_points":state.source.len(), "map_points":state.map.len(),
                "source_indices":source_indices, "targets":pairs.iter().map(|p|p.target).collect::<Vec<_>>(),
                "distance_sq":pairs.iter().map(|p|p.distance_sq).collect::<Vec<_>>(), "multiplicity":multiplicity,
                "baseline_h":baseline.0, "baseline_b":baseline.1,
                "balanced_h":balanced.0, "balanced_b":balanced.1,
                "decomposed_h":decomposed.0, "decomposed_b":decomposed.1,
                "maximum_decomposition_abs_error":maximum_error})
        }).collect();
        let mut samples = Vec::new();
        for round in 0..input.warmup_rounds+input.measured_rounds {
            for (state_index,state) in states.iter().enumerate() {
                let order: Vec<_> = (0..3).map(|i|(round+state_index+i)%3).collect();
                let mut baseline = (0, ([[0.0;6];6],[0.0;6]));
                let mut balanced = baseline;
                let mut stages = ([0;5],baseline.1);
                for &method in &order {
                    match method {
                        0 => baseline = timed_whole(state,false,input.repetitions),
                        1 => balanced = timed_whole(state,true,input.repetitions),
                        2 => stages = timed_stages(state,input.repetitions),
                        _ => unreachable!(),
                    }
                }
                // Validation, JSON construction and serialization stay outside clocks.
                let reference_baseline = whole(state,false);
                let reference_balanced = whole(state,true);
                let baseline_error = error(&baseline.1,&reference_baseline);
                let balanced_error = error(&balanced.1,&reference_balanced);
                let stage_error = error(&stages.1,&reference_balanced);
                samples.push(json!({"round":round,"warmup":round<input.warmup_rounds,
                    "state_index":state_index,"order":order,"repetitions":input.repetitions,
                    "baseline_ns":baseline.0,"balanced_ns":balanced.0,
                    "search_ns":stages.0[0],"count_ns":stages.0[1],"accumulate_ns":stages.0[2],
                    "destroy_ns":stages.0[3],"staged_ns":stages.0[4],
                    "maximum_baseline_abs_error":baseline_error,
                    "maximum_balanced_abs_error":balanced_error,
                    "maximum_staged_abs_error":stage_error}));
            }
        }
        (validation,samples)
    });
    let output = json!({"schema":"cloudanalyzer.odometry_stage_profile.v1",
        "threads":pool.current_num_threads(), "repetitions":input.repetitions,
        "validation":validation,"samples":samples});
    let mut file = std::fs::OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(output_path)
        .unwrap();
    serde_json::to_writer(&mut file, &output).unwrap();
    file.write_all(b"\n").unwrap();
}
