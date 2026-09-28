//! Parity with the Python trajectory evaluation (`cloudanalyzer/ca/trajectory.py`).
//!
//! The fixture in `tests/trajectory/` and the expected numbers come from
//! `PYTHONPATH=cloudanalyzer python scripts/make_trajectory_fixtures.py`,
//! which writes the two TUM files and prints `evaluate_trajectory(estimate,
//! reference, max_time_delta=0.05, rpe_distances_m=[0.5])` with no alignment,
//! `align_origin` and `align_rigid`.

use ca_core::trajectory::{
    Alignment, EvalParams, Evaluation, Format, RpeDelta, Stats, Trajectory, evaluate, parse,
};

fn load(name: &str) -> Trajectory {
    let path = format!("{}/tests/trajectory/{name}", env!("CARGO_MANIFEST_DIR"));
    parse(&std::fs::read_to_string(path).unwrap(), Format::Tum).unwrap()
}

fn run(alignment: Alignment, rpe_delta: RpeDelta) -> Evaluation {
    let params = EvalParams {
        max_time_delta: 0.05,
        alignment,
        rpe_delta,
    };
    evaluate(&load("estimate.tum"), &load("reference.tum"), &params).unwrap()
}

/// `values` summarised as Python's `[rmse, mean, median, std, min, max]`.
fn check(what: &str, values: &[f64], python: [f64; 6]) {
    let s = Stats::of(values).unwrap();
    let ours = [s.rmse, s.mean, s.median, s.std, s.min, s.max];
    for (a, b) in ours.iter().zip(python) {
        assert!(
            (a - b).abs() <= 1e-8 * b.abs().max(1.0),
            "{what}: {ours:?} != {python:?}"
        );
    }
}

#[test]
fn unaligned_matches_python() {
    let e = run(Alignment::None, RpeDelta::Frames(1));
    assert_eq!(e.timestamps.len(), 20);
    #[rustfmt::skip]
    check("ATE", &e.ate, [1.8240224940383387, 1.8027989048887025, 1.7686402209099072, 0.27744218729302306, 1.439664038290323, 2.3104704579911854]);
    #[rustfmt::skip]
    check("ATE rotation", e.ate_rotation.as_ref().unwrap(), [20.140126162980984, 20.138316005053944, 20.0965819491828, 0.2700191500233196, 19.654170875554613, 20.927502427467097]);
    #[rustfmt::skip]
    check("RPE", &e.rpe_translation, [0.06211259862495902, 0.05991138302749126, 0.056342860628122264, 0.0163890540202445, 0.029982847996145968, 0.08887032807883624]);
    #[rustfmt::skip]
    check("RPE rotation", e.rpe_rotation.as_ref().unwrap(), [0.4841950372340446, 0.4238029551059646, 0.38061925171218236, 0.23417064146798935, 0.121323135280711, 0.9587121014764052]);
    assert!((e.endpoint_drift - 0.9560685368053901).abs() < 1e-12);
}

#[test]
fn origin_alignment_matches_python() {
    let e = run(Alignment::Origin, RpeDelta::Frames(1));
    #[rustfmt::skip]
    check("ATE", &e.ate, [0.6064173380240556, 0.5292895225149308, 0.5544405009098695, 0.29596383091874295, 0.0, 0.9560685368053901]);
}

#[test]
fn rigid_alignment_matches_python() {
    let e = run(Alignment::Se3, RpeDelta::Frames(1));
    #[rustfmt::skip]
    check("ATE", &e.ate, [0.038873695325804564, 0.03586036309405184, 0.034268693697727726, 0.015006616775484208, 0.004067945936428971, 0.06953207392838842]);
    #[rustfmt::skip]
    check("ATE rotation", e.ate_rotation.as_ref().unwrap(), [1.3190611255619833, 1.3096115808558506, 1.2951963081136921, 0.1576063458655331, 0.9970430981449889, 1.5557239902737838]);
    #[rustfmt::skip]
    check("RPE", &e.rpe_translation, [0.03305913988160749, 0.03144895392642099, 0.0323502424101531, 0.010191664566965505, 0.011568280588513462, 0.06131300148901618]);
    assert!((e.endpoint_drift - 0.11455253071290834).abs() < 1e-9);
    let t = e.alignment.translation;
    for (a, b) in t
        .iter()
        .zip([-0.19171022529250026, 2.187922141428276, -0.7027065258475748])
    {
        assert!((a - b).abs() < 1e-9, "{t:?}");
    }
    // Sim(3) (not in Python) fits the fixture's 0.97 scale and does better.
    let sim3 = run(Alignment::Sim3, RpeDelta::Frames(1));
    assert!((sim3.alignment.scale - 1.0 / 0.97).abs() < 0.01);
    assert!(Stats::of(&sim3.ate).unwrap().rmse < 0.038873695325804564);
}

#[test]
fn rpe_by_distance_matches_python() {
    let e = run(Alignment::Se3, RpeDelta::Meters(0.5));
    assert_eq!(e.rpe_pairs.len(), 16);
    #[rustfmt::skip]
    check("RPE 0.5 m", &e.rpe_translation, [0.03958926653664348, 0.03765609822877862, 0.03762303089366596, 0.012219995543942484, 0.011348077005251863, 0.05999126757148725]);
    let rotation = Stats::of(e.rpe_rotation.as_ref().unwrap()).unwrap();
    assert!((rotation.rmse - 0.5928344693675993).abs() < 1e-8);
}
