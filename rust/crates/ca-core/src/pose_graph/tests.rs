use super::*;

fn pose(t: [f64; 3], w: [f64; 3]) -> Rigid {
    Rigid {
        rotation: exp_so3(&w),
        translation: t,
    }
}

fn close(a: &Rigid, b: &Rigid, tolerance: f64) -> bool {
    let d = inverse(a).compose(b);
    let w = log_so3(&d.rotation);
    d.translation.iter().chain(&w).all(|x| x.abs() < tolerance)
}

#[test]
fn so3_log_inverts_exp() {
    for w in [
        [0.0, 0.0, 0.0],
        [1e-9, -2e-9, 0.0],
        [0.3, -0.2, 0.9],
        [0.0, 0.0, std::f64::consts::PI - 1e-7],
        [2.0, -1.0, 0.5],
    ] {
        let back = log_so3(&exp_so3(&w));
        for i in 0..3 {
            assert!((back[i] - w[i]).abs() < 1e-6, "{w:?} -> {back:?}");
        }
    }
}

#[test]
fn block_cholesky_solves_a_dense_system() {
    // Three variables all connected: the factor fills in completely.
    let adjacency = vec![
        BTreeSet::from([1, 2]),
        BTreeSet::from([0, 2]),
        BTreeSet::from([0, 1]),
    ];
    let (position, columns) = symbolic(3, &adjacency);
    // A = M M^T + n I for a fixed M, 18x18.
    let n = 18;
    let entry = |i: usize, j: usize| ((i * 7 + j * 3) % 11) as f64 - 5.0;
    let a: Vec<Vec<f64>> = (0..n)
        .map(|i| {
            (0..n)
                .map(|j| {
                    (0..n).map(|k| entry(i, k) * entry(j, k)).sum::<f64>()
                        + if i == j { n as f64 } else { 0.0 }
                })
                .collect()
        })
        .collect();
    let mut m = BlockMatrix {
        diagonal: vec![ZERO6; 3],
        below: columns
            .iter()
            .map(|rows| rows.iter().map(|&r| (r, ZERO6)).collect())
            .collect(),
    };
    for vi in 0..3 {
        for vj in 0..3 {
            if position[vj] > position[vi] {
                continue;
            }
            let block = m.block(position[vi], position[vj]);
            for i in 0..6 {
                for j in 0..6 {
                    block[i][j] = a[vi * 6 + i][vj * 6 + j];
                }
            }
        }
    }
    let b: Vec<f64> = (0..n).map(|i| i as f64 - 4.0).collect();
    let mut x = vec![[0.0; 6]; 3];
    for v in 0..3 {
        for i in 0..6 {
            x[position[v]][i] = b[v * 6 + i];
        }
    }
    assert!(factor(&mut m));
    substitute(&m, &mut x);
    for i in 0..n {
        let ax: f64 = (0..n).map(|j| a[i][j] * x[position[j / 6]][j % 6]).sum();
        assert!((ax - b[i]).abs() < 1e-8, "row {i}: {ax} != {}", b[i]);
    }
}

/// A square loop of `n` poses per side with noisy odometry.
fn loop_graph(drift: f64) -> (PoseGraph, Vec<Rigid>) {
    let mut truth = Vec::new();
    for side in 0..4 {
        let yaw = side as f64 * std::f64::consts::FRAC_PI_2;
        for k in 0..10 {
            let s = k as f64;
            let corner = [[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]][side];
            let (c, sn) = (yaw.cos(), yaw.sin());
            truth.push(pose(
                [corner[0] + c * s, corner[1] + sn * s, 0.1 * s],
                [0.0, 0.0, yaw],
            ));
        }
    }
    let info = isotropic_information(0.05, 0.01);
    let mut graph = PoseGraph::from_poses(&truth, info);
    // Corrupt the odometry with a steady yaw and forward bias, then dead-reckon.
    for (k, e) in graph.edges.iter_mut().enumerate() {
        let bias = pose([drift, 0.0, 0.0], [0.0, 0.01 * drift, drift * 0.3]);
        e.measurement = e
            .measurement
            .compose(&bias)
            .compose(&pose([0.0, 0.001 * (k % 3) as f64, 0.0], [0.0; 3]));
    }
    for i in 1..graph.nodes.len() {
        let prev = graph.nodes[i - 1].pose;
        graph.nodes[i].pose = prev.compose(&graph.edges[i - 1].measurement);
    }
    let last = truth.len() - 1;
    graph.edges.push(Edge {
        from: last,
        to: 0,
        measurement: inverse(&truth[last]).compose(&truth[0]),
        information: info,
        kind: EdgeKind::Loop,
    });
    (graph, truth)
}

#[test]
fn a_loop_closure_pulls_the_drifted_end_back() {
    let (mut graph, truth) = loop_graph(0.05);
    let end = graph.nodes.len() - 1;
    let before = inverse(&graph.nodes[end].pose).compose(&truth[end]);
    assert!(
        before.translation.iter().map(|x| x.abs()).sum::<f64>() > 2.0,
        "{before:?}"
    );

    let loop_error_before = *graph.edge_errors().last().unwrap();
    let report = optimize(&mut graph, &OptimizeParams::default()).unwrap();
    assert!(report.converged, "{report:?}");
    // The biased odometry cannot all be met, but the result is a minimum:
    // nudging any free pose in any direction raises the cost.
    assert!(report.final_cost < report.initial_cost * 0.05, "{report:?}");
    let params = OptimizeParams::default();
    let cost = graph.cost(&params);
    for i in (1..graph.nodes.len()).step_by(7) {
        for c in 0..6 {
            for h in [-1e-3, 1e-3] {
                let mut nudged = graph.clone();
                let mut d = [0.0; 6];
                d[c] = h;
                nudged.nodes[i].pose = retract(&nudged.nodes[i].pose, &d);
                assert!(nudged.cost(&params) > cost - 1e-9, "node {i} axis {c}");
            }
        }
    }
    assert!(close(&graph.nodes[0].pose, &truth[0], 1e-12));
    let after = inverse(&graph.nodes[end].pose).compose(&truth[end]);
    let err: f64 = after.translation.iter().map(|x| x * x).sum::<f64>().sqrt();
    assert!(err < 0.5, "end still {err} m off");
    // The loop's error is spread evenly over all edges.
    let errors = graph.edge_errors();
    let loop_error = errors[errors.len() - 1];
    assert!(
        loop_error < loop_error_before / 100.0,
        "{loop_error_before} -> {errors:?}"
    );
    let max = errors.iter().copied().fold(0.0, f64::max);
    assert!(loop_error > max / 2.0, "{errors:?}");
}

#[test]
fn exact_measurements_are_already_optimal() {
    let (_, truth) = loop_graph(0.0);
    let mut graph = PoseGraph::from_poses(&truth, isotropic_information(0.1, 0.1));
    let report = optimize(&mut graph, &OptimizeParams::default()).unwrap();
    assert!(report.initial_cost < 1e-18, "{report:?}");
    for (n, t) in graph.nodes.iter().zip(&truth) {
        assert!(close(&n.pose, t, 1e-9));
    }
}

#[test]
fn huber_kernel_limits_a_wrong_loop() {
    let (clean, truth) = loop_graph(0.0);
    let mut plain = clean.clone();
    // A bogus loop claiming poses 5 and 25 coincide.
    let bogus = Edge {
        from: 5,
        to: 25,
        measurement: Rigid::IDENTITY,
        information: isotropic_information(0.05, 0.01),
        kind: EdgeKind::Loop,
    };
    plain.edges.push(bogus.clone());
    let mut robust = plain.clone();
    optimize(&mut plain, &OptimizeParams::default()).unwrap();
    optimize(
        &mut robust,
        &OptimizeParams {
            loop_kernel: Some(1.0),
            ..OptimizeParams::default()
        },
    )
    .unwrap();
    let error = |g: &PoseGraph| -> f64 {
        g.nodes
            .iter()
            .zip(&truth)
            .map(|(n, t)| {
                let d = inverse(&n.pose).compose(t).translation;
                (d[0] * d[0] + d[1] * d[1] + d[2] * d[2]).sqrt()
            })
            .fold(0.0, f64::max)
    };
    assert!(
        error(&robust) < error(&plain) / 3.0,
        "{} vs {}",
        error(&robust),
        error(&plain)
    );
}

#[test]
fn g2o_round_trips() {
    let (graph, _) = loop_graph(0.02);
    let mut graph = graph;
    graph.nodes[3].fixed = true;
    let text = graph.to_g2o();
    let back = PoseGraph::from_g2o(&text).unwrap();
    assert_eq!(back.nodes.len(), graph.nodes.len());
    assert_eq!(back.edges.len(), graph.edges.len());
    assert!(back.nodes[0].fixed && back.nodes[3].fixed && !back.nodes[1].fixed);
    for (a, b) in back.nodes.iter().zip(&graph.nodes) {
        assert!(close(&a.pose, &b.pose, 1e-12));
    }
    for (a, b) in back.edges.iter().zip(&graph.edges) {
        assert_eq!((a.from, a.to, a.kind), (b.from, b.to, b.kind));
        assert!(close(&a.measurement, &b.measurement, 1e-12));
        assert_eq!(a.information, b.information);
    }
}

#[test]
fn g2o_reads_the_reference_example() {
    let text = "\
# comment
VERTEX_SE3:QUAT 0 0 0 0 0 0 0 1
VERTEX_SE3:QUAT 1 1 0 0 0 0 0.7071067811865476 0.7071067811865476
VERTEX_SE3:QUAT 7 1 1 0 0 0 1 0
EDGE_SE3:QUAT 0 1 1 0 0 0 0 0.7071067811865476 0.7071067811865476 1 0 0 0 0 0 1 0 0 0 0 1 0 0 0 4 0 0 4 0 4
EDGE_SE3:QUAT 1 7 1 0 0 0 0 0.7071067811865476 0.7071067811865476 1 0 0 0 0 0 1 0 0 0 0 1 0 0 0 4 0 0 4 0 4
VERTEX_SE2 9 0 0 0
FIX 1
";
    let graph = PoseGraph::from_g2o(text).unwrap();
    assert_eq!(graph.nodes.len(), 3);
    assert_eq!(graph.nodes[2].id, 7);
    assert!(graph.nodes[1].fixed);
    assert_eq!(graph.edges[0].kind, EdgeKind::Odometry);
    assert_eq!(graph.edges[1].kind, EdgeKind::Loop);
    assert_eq!(graph.edges[1].information[3][3], 4.0);
    // Both edges agree with the vertices.
    for e in graph.edge_errors() {
        assert!(e < 1e-20, "{e}");
    }
    assert!(PoseGraph::from_g2o("EDGE_SE3:QUAT 0 1").is_err());
    assert!(PoseGraph::from_g2o("VERTEX_SE3:QUAT 0 0 0 0 0 0 0 1\nEDGE_SE3:QUAT 0 5 0 0 0 0 0 0 1 1 0 0 0 0 0 1 0 0 0 0 1 0 0 0 1 0 0 1 0 1").is_err());
}

#[test]
fn an_imu_mounted_turned_and_upside_down_is_calibrated_from_the_drive() {
    // A drive that turns and tilts a little: its keyframes' rotations.
    let mut seed = 11u64;
    let mut noise = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((seed >> 11) as f64 / (1u64 << 53) as f64 - 0.5) * 0.1
    };
    let mul = |a: &Mat3, b: &Mat3| -> Mat3 {
        std::array::from_fn(|i| std::array::from_fn(|j| (0..3).map(|k| a[i][k] * b[k][j]).sum()))
    };
    let rotations: Vec<Mat3> = (0..80)
        .map(|k| {
            let yaw = exp_so3(&[0.0, 0.0, k as f64 * 0.075]);
            mul(&yaw, &exp_so3(&[noise(), noise(), 0.0]))
        })
        .collect();
    // The IMU is turned a quarter about z, tilted 5 degrees and upside down against the LiDAR.
    let mount = mul(
        &mul(
            &exp_so3(&[0.0, 0.0, std::f64::consts::FRAC_PI_2]),
            &exp_so3(&[5f64.to_radians(), 0.0, 0.0]),
        ),
        &exp_so3(&[std::f64::consts::PI, 0.0, 0.0]),
    );
    let transpose =
        |m: &Mat3| -> Mat3 { std::array::from_fn(|i| std::array::from_fn(|j| m[j][i])) };
    let ups: Vec<[f64; 3]> = rotations
        .iter()
        .map(|r| apply3(&transpose(&mount), &apply3(&transpose(r), &[0.0, 0.0, 1.0])))
        .collect();
    let identity = exp_so3(&[0.0; 3]);
    assert!(median(up_spread(&rotations, &ups, &identity)) > 4.0);
    let found = calibrate_ups(&rotations, &ups);
    assert!(median(up_spread(&rotations, &ups, &found)) < 0.01);

    // Tied through a graph: the estimate is used, and the spread left sets the sigma.
    let poses: Vec<Rigid> = rotations
        .iter()
        .enumerate()
        .map(|(k, r)| Rigid {
            rotation: *r,
            translation: [k as f64, 0.0, 0.0],
        })
        .collect();
    let mut graph = PoseGraph::from_poses(&poses, isotropic_information(0.1, 0.01));
    // (80 keyframes: enough to estimate the mount from.)
    let measured: Vec<(usize, [f64; 3])> = ups.iter().copied().enumerate().collect();
    let tie = set_gravity_calibrated(&mut graph, &measured, 0.1, true).unwrap();
    assert_eq!(tie.tied, 80);
    assert!(tie.spread > 4.0 && tie.mount.as_ref().unwrap().1 < 0.01);
    assert_eq!(tie.sigma_deg, 0.1);
    let raw = set_gravity_calibrated(&mut graph, &measured, 0.1, false).unwrap();
    assert!(raw.mount.is_none());

    // Drift alone (up directions exact, poses pitching 0.6 degrees a keyframe) is not noise.
    let drifted: Vec<Mat3> = (0..40)
        .map(|k| exp_so3(&[0.0, (0.6 * k as f64).to_radians(), 0.0]))
        .collect();
    let exact: Vec<[f64; 3]> = vec![[0.0, 0.0, 1.0]; 40];
    assert!(up_noise(&drifted, &exact, &identity) < 1e-6);
    // Noise alone is: 1 degree either way about x, alternately.
    let level: Vec<Mat3> = (0..40).map(|_| identity).collect();
    let noisy: Vec<[f64; 3]> = (0..40)
        .map(|k| {
            apply3(
                &exp_so3(&[if k % 2 == 0 { 1f64 } else { -1.0 }.to_radians(), 0.0, 0.0]),
                &[0.0, 0.0, 1.0],
            )
        })
        .collect();
    assert!(up_noise(&level, &noisy, &identity) > 0.5);
}

/// Points on three walls and a floor of a room, in the room's frame.
fn room() -> PointCloud {
    let mut positions = Vec::new();
    for a in 0..40 {
        for b in 0..40 {
            let (u, v) = (a as f64 * 0.25, b as f64 * 0.1);
            positions.push([u, 0.0, v]);
            positions.push([0.0, u * 0.8, v]);
            positions.push([u, 8.0, v * 1.2]);
            positions.push([u, u * 0.2 + v, 0.0]);
        }
    }
    PointCloud {
        positions,
        ..PointCloud::default()
    }
}

#[test]
fn register_loop_refines_a_rough_guess() {
    let truth = pose([0.8, -0.4, 0.1], [0.02, -0.01, 0.2]);
    let from_scan = room();
    // Seen from `to`, the room is the `from` scan moved by truth^-1.
    let to_scan = transformed(&from_scan, &inverse(&truth));
    let guess = truth.compose(&pose([0.15, 0.1, 0.0], [0.0, 0.0, 0.03]));
    let (measured, result) =
        register_loop(&from_scan, &to_scan, &guess, IcpParams::default()).unwrap();
    assert!(result.rms_final < 1e-3, "{result:?}");
    assert!(close(&measured, &truth, 1e-3));
}

#[test]
fn assemble_places_scans_at_their_poses() {
    let poses = [
        pose([0.0; 3], [0.0; 3]),
        pose([10.0, 0.0, 0.0], [0.0, 0.0, 1.0]),
    ];
    let graph = PoseGraph::from_poses(&poses, isotropic_information(1.0, 1.0));
    let scan = |i: f32| PointCloud {
        positions: vec![[1.0, 0.0, 0.0]],
        colors: None,
        attributes: vec![Attribute {
            name: INTENSITY.into(),
            values: AttributeValues::F32(vec![i]),
        }],
    };
    let map = assemble(&graph, &[Some(scan(1.0)), Some(scan(2.0))]);
    assert_eq!(map.len(), 2);
    let p = map.positions[1];
    assert!((p[0] - (10.0 + 1f64.cos())).abs() < 1e-12 && (p[1] - 1f64.sin()).abs() < 1e-12);
    assert_eq!(
        map.attribute(INTENSITY).unwrap().values,
        AttributeValues::F32(vec![1.0, 2.0])
    );
    // A scan without intensity drops the attribute; a missing scan adds nothing.
    let bare = PointCloud {
        positions: vec![[0.0; 3]],
        ..PointCloud::default()
    };
    let map = assemble(&graph, &[Some(bare), None]);
    assert_eq!(map.len(), 1);
    assert!(map.attribute(INTENSITY).is_none());
}

#[test]
fn kitti_and_tum_text() {
    let poses = [
        pose([1.0, 2.0, 3.0], [0.0; 3]),
        pose([0.0; 3], [0.0, 0.0, 0.5]),
    ];
    let graph = PoseGraph::from_poses(&poses, isotropic_information(1.0, 1.0));
    let kitti = graph.to_kitti();
    assert_eq!(kitti.lines().next().unwrap(), "1 0 0 1 0 1 0 2 0 0 1 3");
    let tum = graph.to_tum(&[10.5, 11.5]);
    assert!(
        tum.starts_with(
            "10.5 1 2 3 0 0 0 1
"
        ),
        "{tum}"
    );
    assert!(
        graph
            .to_tum(&[])
            .lines()
            .nth(1)
            .unwrap()
            .starts_with("1 0 0 0 0 0 ")
    );
    use crate::trajectory::{Format, parse};
    let back = parse(&kitti, Format::Kitti).unwrap();
    assert_eq!(back.positions[0], [1.0, 2.0, 3.0]);
}

#[test]
fn loop_candidates_pair_the_ends_of_a_loop() {
    let (_, truth) = loop_graph(0.0);
    let mut graph = PoseGraph::from_poses(&truth, isotropic_information(0.1, 0.1));
    let candidates = loop_candidates(&graph, 3.0, 0.0, 20.0, 5.0);
    assert!(!candidates.is_empty());
    for c in &candidates {
        assert!(
            c.from < 3 && c.to >= 36 && c.travel > 30.0,
            "{candidates:?}"
        );
    }
    // Joined pairs are not proposed again; a short path is never a loop.
    let c = candidates[0];
    graph.add_loop(
        c.from,
        c.to,
        graph.relative(c.from, c.to),
        isotropic_information(0.1, 0.1),
    );
    let again = loop_candidates(&graph, 3.0, 0.0, 20.0, 5.0);
    assert!(!again.iter().any(|d| (d.from, d.to) == (c.from, c.to)));
    assert!(loop_candidates(&graph, 3.0, 0.0, 100.0, 5.0).is_empty());
}

#[test]
fn the_search_widens_with_the_path_travelled() {
    // The end of a drifted loop is a few metres from the start after about 40 m.
    let (graph, _) = loop_graph(0.05);
    let end = graph.nodes.len() - 1;
    let apart = (0..3)
        .map(|j| {
            let d = sub3(
                &graph.nodes[end].pose.translation,
                &graph.nodes[j].pose.translation,
            );
            dot3(&d, &d).sqrt()
        })
        .fold(f64::INFINITY, f64::min);
    assert!(apart > 1.0, "{apart}");
    let reaches_start = |c: &[LoopCandidate]| c.iter().any(|c| c.from < 8 && c.to == end);
    assert!(!reaches_start(&loop_candidates(
        &graph, 1.0, 0.0, 20.0, 0.0
    )));
    // Drift of 20 % over the ~39 m path allows 1 + 7.8 m.
    assert!(apart < drift_budget(1.0, 0.2, 38.0), "{apart}");
    assert!(reaches_start(&loop_candidates(&graph, 1.0, 0.2, 20.0, 0.0)));
    assert!((drift_budget(3.0, 0.2, 40.0) - 11.0).abs() < 1e-12);
}

#[test]
fn overlap_fitness_tells_a_good_registration_from_a_bad_one() {
    let truth = pose([0.8, -0.4, 0.1], [0.02, -0.01, 0.2]);
    let from_scan = room();
    let to_scan = transformed(&from_scan, &inverse(&truth));
    assert!(overlap_fitness(&from_scan, &to_scan, &truth, 0.05) > 0.99);
    let wrong = pose([0.0, 3.0, 0.0], [0.0; 3]).compose(&truth);
    assert!(overlap_fitness(&from_scan, &to_scan, &wrong, 0.05) < 0.5);
}

#[test]
fn yaw_search_registers_without_a_guess() {
    let truth = pose([1.5, -0.8, 0.0], [0.0, 0.0, 1.9]);
    let from_scan = room();
    let to_scan = transformed(&from_scan, &inverse(&truth));
    // From the identity, plain ICP cannot turn 109 degrees.
    let (plain, _) =
        register_loop(&from_scan, &to_scan, &Rigid::IDENTITY, IcpParams::default()).unwrap();
    assert!(!close(&plain, &truth, 1e-2));
    let (measured, fitness, _) = register_with_yaw_search(
        &from_scan,
        &to_scan,
        &Rigid::IDENTITY,
        8,
        IcpParams::default(),
        0.05,
    )
    .unwrap();
    assert!(fitness > 0.99, "{fitness}");
    assert!(close(&measured, &truth, 1e-3));
}

#[test]
fn append_moves_and_renumbers_the_other_graph() {
    let info = isotropic_information(1.0, 1.0);
    let mut a = PoseGraph::from_poses(&[Rigid::IDENTITY, pose([1.0, 0.0, 0.0], [0.0; 3])], info);
    let mut b = PoseGraph::from_poses(&[Rigid::IDENTITY, pose([0.0, 2.0, 0.0], [0.0; 3])], info);
    b.nodes[0].fixed = true;
    let t = pose([10.0, 0.0, 0.0], [0.0, 0.0, std::f64::consts::FRAC_PI_2]);
    let offset = a.append(&b, &t);
    assert_eq!(offset, 2);
    assert_eq!(
        a.nodes.iter().map(|n| n.id).collect::<Vec<_>>(),
        [0, 1, 2, 3]
    );
    assert!(a.nodes[0].fixed && !a.nodes[2].fixed);
    assert!(close(&a.nodes[3].pose, &t.compose(&b.nodes[1].pose), 1e-12));
    assert_eq!((a.edges[1].from, a.edges[1].to), (2, 3));
    // Moving both ends by the same transform keeps the edge satisfied.
    assert!(a.edge_errors()[1] < 1e-20);
    // Ids that do not clash are kept.
    let mut c = PoseGraph::from_poses(&[Rigid::IDENTITY], info);
    c.nodes[0].id = 100;
    a.append(&c, &Rigid::IDENTITY);
    assert_eq!(a.nodes[4].id, 100);
}

#[test]
fn planes_move_between_frames() {
    let x = pose([1.0, -2.0, 0.5], [0.2, -0.1, 0.7]);
    let world = [0.0, 0.6, 0.8, -3.0];
    let local = plane_in_frame(&world, &x);
    let back = plane_in_world(&local, &x);
    for i in 0..4 {
        assert!((back[i] - world[i]).abs() < 1e-12);
    }
    // A point on the plane stays on it in the pose's frame.
    let on_world = [2.0, 0.0, 3.75];
    let on_local = inverse(&x).apply(&on_world);
    assert!(
        (local[0] * on_local[0] + local[1] * on_local[1] + local[2] * on_local[2] + local[3]).abs()
            < 1e-12
    );
}

#[test]
fn detect_floor_finds_the_ground_under_the_sensor() {
    // A slightly tilted floor 1.8 m below the sensor, a wall and a ceiling.
    let mut positions = Vec::new();
    for a in -30..=30 {
        for b in -30..=30 {
            let (x, y) = (a as f64 * 0.3, b as f64 * 0.3);
            positions.push([x, y, -1.8 + 0.02 * x]);
            positions.push([x, 6.0, 1.0 + b as f64 * 0.05]);
            positions.push([x, y, 1.2]);
        }
    }
    let scan = PointCloud {
        positions,
        ..PointCloud::default()
    };
    let floor = detect_floor(&scan, [0.0, 0.0, 1.0], 0.5, 0.05, 100).unwrap();
    let n = unit3([-0.02, 0.0, 1.0]);
    for i in 0..3 {
        assert!((floor[i] - n[i]).abs() < 1e-6, "{floor:?}");
    }
    assert!((floor[3] - 1.8 * n[2]).abs() < 1e-6, "{floor:?}");
    // Looking for a floor along another axis finds none below the sensor.
    assert!(detect_floor(&scan, [1.0, 0.0, 0.0], 0.3, 0.05, 100).is_none());
}

#[test]
fn a_shared_floor_removes_height_drift() {
    // A straight 30-pose drive on flat ground; odometry pitches down a little each step.
    let truth: Vec<Rigid> = (0..30)
        .map(|k| pose([k as f64, 0.0, 0.0], [0.0; 3]))
        .collect();
    let info = isotropic_information(0.05, 0.01);
    let mut graph = PoseGraph::from_poses(&truth, info);
    for e in &mut graph.edges {
        e.measurement = e.measurement.compose(&pose([0.0; 3], [0.0, 0.004, 0.0]));
    }
    for i in 1..graph.nodes.len() {
        graph.nodes[i].pose = graph.nodes[i - 1]
            .pose
            .compose(&graph.edges[i - 1].measurement);
    }
    let height = |g: &PoseGraph| {
        g.nodes
            .iter()
            .map(|n| n.pose.translation[2].abs())
            .fold(0.0, f64::max)
    };
    assert!(height(&graph) > 1.0, "{}", height(&graph));
    // Every pose sees the floor 1.8 m below it.
    let floor = [0.0, 0.0, 1.0, 1.8];
    let seen: Vec<[f64; 4]> = truth.iter().map(|x| plane_in_frame(&floor, x)).collect();
    let plane = graph.add_plane(plane_in_world(&seen[0], &graph.nodes[0].pose));
    for (node, measurement) in seen.iter().enumerate() {
        graph.plane_edges.push(PlaneEdge {
            node,
            plane,
            measurement: *measurement,
            information: plane_information(0.01, 0.02),
        });
    }
    let report = optimize(&mut graph, &OptimizeParams::default()).unwrap();
    assert!(report.converged, "{report:?}");
    // What is left is the compromise with the biased odometry.
    assert!(height(&graph) < 0.1, "{}", height(&graph));
    let c = graph.planes[plane].coefficients;
    assert!(
        (c[2] - 1.0).abs() < 1e-4 && (c[3] - 1.8).abs() < 0.05,
        "{c:?}"
    );
}

#[test]
fn planes_round_trip_through_g2o_and_append() {
    let info = isotropic_information(1.0, 1.0);
    let mut graph =
        PoseGraph::from_poses(&[Rigid::IDENTITY, pose([1.0, 0.0, 0.0], [0.0; 3])], info);
    let plane = graph.add_plane([0.0, 0.0, 1.0, 1.5]);
    assert_eq!(graph.planes[plane].id, 2);
    graph.plane_edges.push(PlaneEdge {
        node: 1,
        plane,
        measurement: [0.0, 0.0, 1.0, 1.5],
        information: plane_information(0.01, 0.02),
    });
    let text = graph.to_g2o();
    assert!(text.contains("VERTEX_PLANE 2 0 0 1 1.5"), "{text}");
    let back = PoseGraph::from_g2o(&text).unwrap();
    assert_eq!(back.planes, graph.planes);
    assert_eq!(back.plane_edges.len(), 1);
    assert_eq!(back.plane_edges[0].node, 1);
    assert!((back.plane_edges[0].information[2][2] - 2500.0).abs() < 1e-9);
    assert!(back.plane_edge_errors()[0] < 1e-20);

    // Appended somewhere else, the plane moves with its graph.
    let mut other = PoseGraph::from_poses(&[Rigid::IDENTITY], info);
    let t = pose([5.0, 0.0, 2.0], [0.0; 3]);
    other.append(&back, &t);
    assert_eq!(other.planes.len(), 1);
    assert!((other.planes[0].coefficients[3] - (1.5 - 2.0)).abs() < 1e-12);
    assert_eq!(other.plane_edges[0].node, 2);
    assert!(other.plane_edge_errors()[0] < 1e-20);
}

#[test]
fn one_tilted_plane_fits_a_constant_grade() {
    let grade = 0.1f64.atan();
    let truth: Vec<Rigid> = (0..40)
        .map(|k| {
            let s = k as f64;
            pose([s * grade.cos(), 0.0, s * grade.sin()], [0.0, -grade, 0.0])
        })
        .collect();
    let seen: Vec<(usize, [f64; 4])> = (0..truth.len())
        .map(|i| (i, [0.0, 0.0, 1.0, 1.8]))
        .collect();
    let mut graph = PoseGraph::from_poses(&truth, isotropic_information(0.05, 0.005));
    let before = graph.cost(&OptimizeParams::default());
    tie_to_floor(&mut graph, &seen, plane_information(0.005, 0.02)).unwrap();
    let errors = graph.plane_edge_errors();
    let worst = errors.iter().copied().fold(0.0, f64::max);
    assert!(
        before < 1e-12 && worst < 1e-12,
        "{before} {worst} {:?}",
        graph.planes
    );
    optimize(&mut graph, &OptimizeParams::default()).unwrap();
    for (n, t) in graph.nodes.iter().zip(&truth) {
        assert!((n.pose.translation[2] - t.translation[2]).abs() < 1e-6);
    }
}

#[test]
fn one_floor_flattens_hills() {
    // Height 5 sin(s / 60) over 300 m; the vehicle pitches with the road.
    let truth: Vec<Rigid> = (0..300)
        .map(|k| {
            let s = k as f64;
            let slope = (5.0 / 60.0 * (s / 60.0).cos()).atan();
            pose([s, 0.0, 5.0 * (s / 60.0).sin()], [0.0, -slope, 0.0])
        })
        .collect();
    let seen: Vec<(usize, [f64; 4])> = (0..truth.len())
        .map(|i| (i, [0.0, 0.0, 1.0, 1.8]))
        .collect();
    let mut graph = PoseGraph::from_poses(&truth, isotropic_information(0.1, 0.0175));
    let plane = tie_to_floor(&mut graph, &seen, plane_information(0.0087, 0.05)).unwrap();
    optimize(&mut graph, &OptimizeParams::default()).unwrap();
    // What the guide warns about: every pose ends up 1.8 m above one
    // plane, so the 10 m of hills are gone.
    let c = graph.planes[plane].coefficients;
    let off = graph
        .nodes
        .iter()
        .map(|n| (dot3(&[c[0], c[1], c[2]], &n.pose.translation) + c[3] - 1.8).abs())
        .fold(0.0, f64::max);
    assert!(off < 0.5, "{off}");
    assert!(tie_to_floor(&mut graph, &[], plane_information(0.01, 0.01)).is_none());
}

#[test]
fn gravity_takes_out_tilt_drift_and_keeps_hills() {
    // Height 5 sin(s / 60) over 300 m; the vehicle pitches with the road.
    let truth: Vec<Rigid> = (0..300)
        .map(|k| {
            let s = k as f64;
            let slope = (5.0 / 60.0 * (s / 60.0).cos()).atan();
            pose([s, 0.0, 5.0 * (s / 60.0).sin()], [0.0, -slope, 0.0])
        })
        .collect();
    let mut graph = PoseGraph::from_poses(&truth, isotropic_information(0.05, 0.004));
    // Odometry that pitches up 0.05 degrees too much every metre.
    for e in &mut graph.edges {
        e.measurement = e
            .measurement
            .compose(&pose([0.0; 3], [0.0, -0.000_873, 0.0]));
    }
    for i in 1..graph.nodes.len() {
        graph.nodes[i].pose = graph.nodes[i - 1]
            .pose
            .compose(&graph.edges[i - 1].measurement);
    }
    let worst = |g: &PoseGraph| {
        g.nodes
            .iter()
            .zip(&truth)
            .map(|(n, t)| (n.pose.translation[2] - t.translation[2]).abs())
            .fold(0.0, f64::max)
    };
    let drifted = worst(&graph);
    assert!(drifted > 20.0, "{drifted}");
    // An IMU sees true up in each body frame.
    let ups: Vec<(usize, [f64; 3])> = truth
        .iter()
        .enumerate()
        .map(|(i, t)| (i, std::array::from_fn(|k| t.rotation[2][k])))
        .collect();
    assert!(tie_to_gravity(
        &mut graph,
        &ups,
        gravity_information(0.002),
        5
    ));
    assert!((graph.gravity[2] - 1.0).abs() < 1e-4, "{:?}", graph.gravity);
    let report = optimize(&mut graph, &OptimizeParams::default()).unwrap();
    assert!(report.converged, "{report:?}");
    let corrected = worst(&graph);
    assert!(corrected < 1.0, "{drifted} -> {corrected}");
    // The hills are kept: the top is still about 5 m up.
    let top = graph
        .nodes
        .iter()
        .map(|n| n.pose.translation[2])
        .fold(f64::MIN, f64::max);
    assert!((top - 5.0).abs() < 1.0, "{top}");
    assert!(!tie_to_gravity(
        &mut graph,
        &[],
        gravity_information(0.01),
        5
    ));
}

#[test]
fn correction_is_each_points_shift() {
    let info = isotropic_information(1.0, 1.0);
    let mut graph =
        PoseGraph::from_poses(&[Rigid::IDENTITY, pose([5.0, 0.0, 0.0], [0.0; 3])], info);
    let before: Vec<Rigid> = graph.nodes.iter().map(|n| n.pose).collect();
    // Node 1 turns a quarter about its own origin.
    graph.nodes[1].pose = pose([5.0, 0.0, 0.0], [0.0, 0.0, std::f64::consts::FRAC_PI_2]);
    let scans = [
        Some(scan_of(&[[0.0; 3]])),
        Some(scan_of(&[[0.0; 3], [2.0, 0.0, 0.0]])),
    ];
    let moved = correction(&graph, &scans, &before);
    // Node 0's point stays; node 1's origin stays, its point 2 m out swings by 2 sqrt(2).
    assert_eq!(moved.len(), 3);
    assert!(moved[0] == 0.0 && moved[1] == 0.0);
    assert!(
        (moved[2] - 2.0 * std::f32::consts::SQRT_2).abs() < 1e-5,
        "{moved:?}"
    );
    assert_eq!(moved.len(), assemble(&graph, &scans).len());
}

fn scan_of(points: &[[f64; 3]]) -> PointCloud {
    PointCloud {
        positions: points.to_vec(),
        ..PointCloud::default()
    }
}
