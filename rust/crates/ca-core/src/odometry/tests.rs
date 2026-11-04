use super::*;

#[test]
fn sparse_equations_match_dense_jacobian_reference() {
    let points: Vec<_> = (0..80)
        .map(|i| {
            [
                (i % 7) as f64 * 0.3 - 1.,
                (i % 11) as f64 * 0.2 - 1.,
                (i % 13) as f64 * 0.1 - 0.5,
            ]
        })
        .collect();
    let mut map = LocalMap::new(1., 20);
    map.insert(&points);
    let source: Vec<_> = points
        .iter()
        .map(|p| [p[0] + 0.02, p[1] - 0.03, p[2] + 0.01])
        .collect();
    for center in [[0.; 3], [0.5, -2., 3.]] {
        for threshold in [0.0001, 0.04, 9.] {
            let kernel = 0.7;
            let (actual_h, actual_b) =
                map.normal_equations(&source, &Rigid::IDENTITY, center, threshold, kernel);
            let mut h = [[0.; 6]; 6];
            let mut b = [0.; 6];
            for world in &source {
                let Some((target, distance)) = map.nearest(world) else {
                    continue;
                };
                if distance > threshold {
                    continue;
                }
                let q: [f64; 3] = std::array::from_fn(|i| world[i] - center[i]);
                let residual: [f64; 3] = std::array::from_fn(|i| world[i] - target[i]);
                let w = kernel * kernel / (kernel + distance).powi(2);
                let jacobian = [
                    [1., 0., 0., 0., q[2], -q[1]],
                    [0., 1., 0., -q[2], 0., q[0]],
                    [0., 0., 1., q[1], -q[0], 0.],
                ];
                for (row, residual) in jacobian.iter().zip(residual) {
                    for i in 0..6 {
                        b[i] -= w * row[i] * residual;
                        for j in 0..6 {
                            h[i][j] += w * row[i] * row[j];
                        }
                    }
                }
            }
            assert_eq!(actual_h, h);
            assert_eq!(actual_b, b);
        }
    }
}

// Exhaust all 27 cells in the same center-first order, without bounds.
// This checks target choice and distance bits independently of pruning.
fn ordered_neighborhood(map: &LocalMap, q: &[f64; 3]) -> Option<([f64; 3], f64)> {
    let key = q.map(|x| (x / map.voxel).floor() as i64);
    let mut cells = vec![key];
    for dx in -1..=1 {
        for dy in -1..=1 {
            for dz in -1..=1 {
                if [dx, dy, dz] != [0; 3] {
                    cells.push([key[0] + dx, key[1] + dy, key[2] + dz]);
                }
            }
        }
    }
    let mut best = ([0.0; 3], f64::INFINITY);
    for cell in cells {
        for m in map.map.get(&cell).into_iter().flatten() {
            let distance = (m[0] - q[0]).powi(2) + (m[1] - q[1]).powi(2) + (m[2] - q[2]).powi(2);
            if distance < best.1 {
                best = (*m, distance);
            }
        }
    }
    best.1.is_finite().then_some(best)
}

fn assert_nearest_bits(map: &LocalMap, q: [f64; 3]) {
    let bits = |hit: Option<([f64; 3], f64)>| {
        hit.map(|(point, distance)| (point.map(f64::to_bits), distance.to_bits()))
    };
    assert_eq!(bits(map.nearest(&q)), bits(ordered_neighborhood(map, &q)));
}

#[test]
fn nearest_face_bounds_match_ordered_sparse_neighborhoods() {
    for voxel in [0.1875, 0.75, 1.0, 4.0] {
        let mut map = LocalMap::new(voxel, 20);
        for x in -3..=3 {
            for y in -3..=3 {
                for z in -3..=3 {
                    if (x + 2 * y + 3 * z) % 3 == 0 {
                        for fraction in [0.125, 0.5, 0.875] {
                            map.insert(&[[
                                (f64::from(x) + fraction) * voxel,
                                (f64::from(y) + fraction) * voxel,
                                (f64::from(z) + fraction) * voxel,
                            ]]);
                        }
                    }
                }
            }
        }
        let mut state = 1_u64;
        for _ in 0..4096 {
            let query = std::array::from_fn(|_| {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((state >> 32) as f64 / u32::MAX as f64 * 10.0 - 5.0) * voxel
            });
            assert_nearest_bits(&map, query);
        }
        assert_nearest_bits(&map, [30.0 * voxel; 3]);
        assert_nearest_bits(&LocalMap::new(voxel, 20), [0.0; 3]);
    }
}

#[test]
fn nearest_face_bounds_preserve_boundaries_ties_and_signed_zero() {
    for voxel in [0.1875, 0.75, 1.0, 4.0] {
        let mut map = LocalMap::new(voxel, 20);
        map.insert(&[[-0.0, 0.0, -0.0], [0.0; 3]]);
        for x in -2..=2 {
            for y in -2..=2 {
                for z in -2..=2 {
                    map.insert(&[[
                        f64::from(x) * voxel,
                        f64::from(y) * voxel,
                        f64::from(z) * voxel,
                    ]]);
                }
            }
        }
        for cell in -3..=3 {
            let boundary = f64::from(cell) * voxel;
            for value in [
                boundary.next_down(),
                boundary,
                boundary.next_up(),
                boundary + 0.5 * voxel,
            ] {
                for axis in 0..3 {
                    let mut query = [-0.0, 0.0, -0.0];
                    query[axis] = value;
                    assert_nearest_bits(&map, query);
                }
            }
        }
        assert_eq!(
            map.nearest(&[0.0; 3]).unwrap().0.map(f64::to_bits),
            [-0.0, 0.0, -0.0].map(f64::to_bits)
        );
        assert_nearest_bits(&map, [f64::NAN, 0.0, 0.0]);
    }
}

#[test]
fn nearest_prefix_bounds_preserve_overflowing_face_squares() {
    for voxel in [1e155, 1e200] {
        let mut map = LocalMap::new(voxel, 20);
        map.insert(&[
            [-voxel; 3],
            [0.0; 3],
            [voxel; 3],
            [2.0 * voxel; 3],
            [voxel, 0.0, 0.0],
        ]);
        for query in [
            [0.5 * voxel; 3],
            [voxel; 3],
            [voxel.next_down(), 0.0, 0.0],
            [voxel.next_up(), 0.0, 0.0],
            [30.0 * voxel; 3],
        ] {
            assert_nearest_bits(&map, query);
        }
    }
}

fn sampling_cloud() -> PointCloud {
    PointCloud {
        positions: (0..12)
            .flat_map(|i| {
                (0..4).map(move |j| {
                    [
                        3.0 * (i % 4) as f64 - 6.0 + 0.03125 + 0.09375 * j as f64,
                        3.0 * (i / 4) as f64 + 2.03125,
                        (i % 3) as f64 + 0.03125,
                    ]
                })
            })
            .collect(),
        ..PointCloud::default()
    }
}

#[cfg(feature = "parallel")]
#[test]
fn alignment_worker_gate_uses_captured_policy_and_existing_chunk_boundary() {
    let mut odometry = Odometry::new(OdometryParams {
        min_range: 0.0,
        ..OdometryParams::default()
    });
    let scan = sampling_cloud();
    odometry.set_target_balance(true);
    assert!(!odometry.begin(&scan).needs_alignment_worker()); // Empty map.
    odometry.register(&scan);
    let mut balanced = odometry.begin(&scan);
    odometry.set_target_balance(false);
    assert!(!balanced.needs_alignment_worker()); // First iteration stays here.
    balanced.iterations = 1;
    assert!(balanced.needs_alignment_worker()); // Choice belongs to begin.
    balanced.termination = Some(RegistrationTermination::Converged);
    assert!(!balanced.needs_alignment_worker());
    balanced.termination = None;
    balanced.source.clear();
    assert!(!balanced.needs_alignment_worker());
    let mut default = odometry.begin(&scan);
    default.iterations = 1;
    odometry.set_target_balance(true);
    for count in [0, 10, 255, 256, 257] {
        default.source.resize(count, [1.0, 2.0, 3.0]);
        assert_eq!(default.needs_alignment_worker(), count > 256);
    }
    assert!(
        !odometry
            .begin(&PointCloud::default())
            .needs_alignment_worker()
    );
}

#[cfg(feature = "parallel")]
#[test]
fn alignment_worker_gate_preserves_nested_local_pools() {
    let mut odometry = Odometry::new(OdometryParams::default());
    let scan = sampling_cloud();
    odometry.register(&scan);
    let mut registration = odometry.begin(&scan);
    registration.source.resize(257, [1.0, 2.0, 3.0]);
    registration.iterations = 1;
    assert!(registration.needs_alignment_worker());
    for workers in [1, 4] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(workers)
            .build()
            .unwrap();
        pool.install(|| {
            assert_eq!(rayon::current_num_threads(), workers);
            assert!(!registration.needs_alignment_worker());
            rayon::scope(|scope| {
                scope.spawn(|_| {
                    assert_eq!(rayon::current_num_threads(), workers);
                    assert!(!registration.needs_alignment_worker());
                })
            });
        });
    }
}

#[test]
fn automatic_alignment_preserves_stepped_state_for_both_policies_and_sizes() {
    for dense in [false, true] {
        for balanced in [false, true] {
            for predicted in [false, true] {
                let params = OdometryParams {
                    min_range: 0.0,
                    max_iterations: 4,
                    diagnostics: true,
                    ..OdometryParams::default()
                };
                let mut automatic = Odometry::new(params);
                let mut stepped = Odometry::new(params);
                for odometry in [&mut automatic, &mut stepped] {
                    odometry.set_target_balance(balanced);
                    odometry.set_registration_voxel(Some(0.125)).unwrap();
                    odometry.set_capture_linearization(true);
                }
                let mut scan = if dense {
                    PointCloud {
                        positions: (0..12)
                            .flat_map(|i| {
                                (0..12).flat_map(move |j| {
                                    (0..3).map(move |k| {
                                        [
                                            2.0 * i as f64 + 2.0,
                                            2.0 * j as f64 - 10.0,
                                            2.0 * k as f64 - 2.0 + 0.125 * (i % 3) as f64,
                                        ]
                                    })
                                })
                            })
                            .collect(),
                        ..PointCloud::default()
                    }
                } else {
                    sampling_cloud()
                };
                for frame in 0..4 {
                    let mut r = if predicted && frame > 0 {
                        stepped
                            .begin_with_prediction(&scan, &Rigid::IDENTITY)
                            .unwrap()
                    } else {
                        stepped.begin(&scan)
                    };
                    if frame > 0 {
                        assert_eq!(r.source().len() > 256, dense);
                    }
                    if r.needs_alignment() {
                        loop {
                            let equations = stepped.equations(&r);
                            if stepped.step(&mut r, equations) {
                                break;
                            }
                        }
                    }
                    let expected = stepped.finish(r);
                    let actual = if predicted && frame > 0 {
                        automatic
                            .register_with_prediction(&scan, &Rigid::IDENTITY)
                            .unwrap()
                    } else {
                        automatic.register(&scan)
                    };
                    assert_eq!(
                        actual.to_matrix().map(f64::to_bits),
                        expected.to_matrix().map(f64::to_bits)
                    );
                    assert_eq!(automatic.last_report(), stepped.last_report());
                    assert_eq!(automatic.map.map, stepped.map.map);
                    assert_eq!(automatic.placed(), stepped.placed());
                    assert_eq!(automatic.error_sum.to_bits(), stepped.error_sum.to_bits());
                    assert_eq!(automatic.error_count, stepped.error_count);
                    let a = automatic.last_linearization().unwrap();
                    let b = stepped.last_linearization().unwrap();
                    assert_eq!(
                        a.normal_matrix.map(|row| row.map(f64::to_bits)),
                        b.normal_matrix.map(|row| row.map(f64::to_bits))
                    );
                    assert_eq!(
                        a.right_hand_side.map(f64::to_bits),
                        b.right_hand_side.map(f64::to_bits)
                    );
                    assert_eq!(a.matched_source_indices, b.matched_source_indices);
                    assert_eq!(
                        a.weights.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        b.weights.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                    assert_eq!(a.target_multiplicity, b.target_multiplicity);
                    for point in &mut scan.positions {
                        point[0] -= 0.03125;
                        point[1] += 0.015625;
                    }
                }
                let empty = PointCloud::default();
                assert_eq!(automatic.register(&empty), stepped.register(&empty));
                assert_eq!(automatic.last_report(), stepped.last_report());
            }
        }
    }
}

#[test]
fn registration_sampling_is_independent_of_map_insertion() {
    let params = OdometryParams {
        min_range: 0.0,
        diagnostics: true,
        ..OdometryParams::default()
    };
    let mut legacy = Odometry::new(params);
    let mut dense = Odometry::new(params);
    dense.set_registration_voxel(Some(0.0625)).unwrap();
    let cloud = sampling_cloud();
    assert_eq!(
        legacy.register(&cloud).to_matrix(),
        dense.register(&cloud).to_matrix()
    );
    assert_eq!(legacy.placed(), dense.placed());
    assert_eq!(legacy.placed().len(), 12);
    assert_eq!(legacy.map.map, dense.map.map);
    assert_eq!(legacy.map.voxel, dense.map.voxel);
    assert_eq!(legacy.map.max_points, dense.map.max_points);
    let sparse = legacy.begin(&cloud);
    let full = dense.begin(&cloud);
    assert_eq!(sparse.source().len(), 12);
    assert_eq!(full.source(), cloud.positions);
    assert_eq!(full.frame, sparse.frame);
    assert_eq!(dense.registration_voxel(), Some(0.0625));
}

#[test]
fn registration_sampling_preserves_first_source_order_and_range_filtering() {
    let mut odometry = Odometry::new(OdometryParams {
        min_range: 1.0,
        ..OdometryParams::default()
    });
    let cloud = sampling_cloud();
    odometry.register(&cloud);
    odometry.set_registration_voxel(Some(0.125)).unwrap();
    let expected: Vec<_> = cloud.positions.iter().step_by(4).copied().collect();
    let mut positions = vec![[0.0; 3], [100.0, 0.0, 0.0], [f64::NAN, 0.0, 0.0]];
    for point in &expected {
        positions.push(*point);
        positions.push([point[0] + 0.03125, point[1], point[2]]);
    }
    positions.extend(expected.iter().copied());
    let scan = PointCloud {
        positions,
        ..PointCloud::default()
    };
    assert_eq!(odometry.begin(&scan).source(), expected);
}

#[test]
fn registration_sampling_changes_are_atomic_and_only_affect_future_begins() {
    let mut odometry = Odometry::new(OdometryParams {
        min_range: 0.0,
        ..OdometryParams::default()
    });
    let cloud = sampling_cloud();
    odometry.register(&cloud);
    let pending = odometry.begin(&cloud);
    odometry.set_registration_voxel(Some(0.0625)).unwrap();
    assert_eq!(pending.source().len(), 12);
    assert_eq!(odometry.begin(&cloud).source().len(), 48);
    let report = odometry.last_report().cloned();
    let placed = odometry.placed().to_vec();
    let poses: Vec<_> = odometry.poses().iter().map(Rigid::to_matrix).collect();
    for value in [0.0, -0.0, -0.1, f64::NAN, f64::INFINITY, f64::NEG_INFINITY] {
        assert_eq!(
            odometry.set_registration_voxel(Some(value)),
            Err(RegistrationVoxelError)
        );
        assert_eq!(odometry.registration_voxel(), Some(0.0625));
        assert_eq!(odometry.last_report(), report.as_ref());
        assert_eq!(odometry.placed(), placed);
        assert_eq!(
            odometry
                .poses()
                .iter()
                .map(Rigid::to_matrix)
                .collect::<Vec<_>>(),
            poses
        );
    }
    odometry.set_registration_voxel(None).unwrap();
    assert_eq!(odometry.registration_voxel(), None);
    assert_eq!(odometry.begin(&cloud).source(), pending.source());
}

#[test]
fn registration_sampling_happens_after_constant_motion_deskew() {
    let mut odometry = Odometry::new(OdometryParams {
        min_range: 0.0,
        deskew: true,
        ..OdometryParams::default()
    });
    let mut cloud = sampling_cloud();
    odometry.register(&cloud);
    let motion = Rigid {
        translation: [0.25, 0.0, 0.0],
        ..Rigid::IDENTITY
    };
    odometry.poses.push(motion);
    odometry.set_registration_voxel(Some(0.03125)).unwrap();
    let times: Vec<f32> = (0..cloud.len()).map(|i| (i % 2) as f32).collect();
    cloud.attributes.push(crate::Attribute {
        name: TIME.into(),
        values: crate::AttributeValues::F32(times.clone()),
    });
    let predicted = Rigid {
        translation: [0.5, 0.0, 0.0],
        ..Rigid::IDENTITY
    };
    let expected: Vec<_> = cloud
        .positions
        .iter()
        .zip(times)
        .map(|(point, time)| {
            // Pure translation: undo time relative to mid-sweep, then apply the prediction.
            [
                point[0] + (f64::from(time) - 0.5) * 0.25 + 0.5,
                point[1],
                point[2],
            ]
        })
        .collect();
    assert_eq!(odometry.begin(&cloud).source(), expected);
    assert_eq!(
        odometry.begin(&cloud).prediction.to_matrix(),
        predicted.to_matrix()
    );
}

fn diagnostic_scene() -> PointCloud {
    PointCloud {
        positions: (0..12)
            .map(|i| [3.0 * (i % 4) as f64, 3.0 * (i / 4) as f64, (i % 3) as f64])
            .collect(),
        ..PointCloud::default()
    }
}

#[test]
fn linearization_capture_choice_belongs_to_begin_and_empty_results_are_owned() {
    let mut odometry = Odometry::new(OdometryParams {
        min_range: 0.0,
        ..OdometryParams::default()
    });
    assert!(!odometry.capture_linearization());
    let scan = diagnostic_scene();
    let pending = odometry.begin(&scan);
    odometry.set_capture_linearization(true);
    odometry.finish(pending);
    assert!(odometry.last_linearization().is_none());
    let pending = odometry.begin(&PointCloud::default());
    odometry.set_capture_linearization(false);
    odometry.finish(pending);
    let snapshot = odometry.last_linearization().unwrap().clone();
    assert!(!snapshot.alignment_eligible);
    assert!(snapshot.source_sensor.is_empty());
    assert!(snapshot.matched_source_indices.is_empty());
    assert_eq!(snapshot.normal_matrix, [[0.0; 6]; 6]);
    assert_eq!(snapshot.right_hand_side, [0.0; 6]);
    assert_eq!(snapshot.weighted_rms_radius_m, None);
    assert!(snapshot.finite);
    odometry.register(&scan);
    assert!(odometry.last_linearization().is_none());
    assert!(snapshot.inserted_world.is_empty());
}

#[test]
fn linearization_capture_preserves_solver_map_and_report_bits() {
    let params = OdometryParams {
        min_range: 0.0,
        diagnostics: true,
        ..OdometryParams::default()
    };
    let mut plain = Odometry::new(params);
    let mut measured = Odometry::new(params);
    measured.set_capture_linearization(true);
    let mut scan = diagnostic_scene();
    for _ in 0..5 {
        let a = plain.register(&scan);
        let b = measured.register(&scan);
        assert_eq!(
            a.to_matrix().map(f64::to_bits),
            b.to_matrix().map(f64::to_bits)
        );
        assert_eq!(plain.last_report(), measured.last_report());
        assert_eq!(plain.map.map, measured.map.map);
        assert_eq!(plain.error_sum.to_bits(), measured.error_sum.to_bits());
        assert_eq!(plain.error_count, measured.error_count);
        let snapshot = measured.last_linearization().unwrap();
        assert_eq!(snapshot.inserted_world, measured.placed());
        assert_eq!(snapshot.pose, b);
        for point in &mut scan.positions {
            point[0] -= 0.1;
        }
    }
}

#[test]
fn linearization_flags_nonfinite_weights_without_repair_or_pose_feedback() {
    let mut odometry = Odometry::new(OdometryParams {
        min_range: 0.0,
        ..OdometryParams::default()
    });
    odometry.set_capture_linearization(true);
    let scan = diagnostic_scene();
    odometry.register(&scan);
    let mut pending = odometry.begin(&scan);
    pending.kernel = 0.0; // Legacy formula has 0/0 on exact matches.
    assert_eq!(odometry.finish(pending), Rigid::IDENTITY);
    let snapshot = odometry.last_linearization().unwrap();
    assert!(!snapshot.finite);
    assert!(snapshot.weights.iter().all(|w| w.is_nan()));
    assert_eq!(snapshot.weighted_rms_radius_m, None);
    assert_eq!(snapshot.inserted_world, scan.positions);
}

#[test]
fn linearization_is_recomputed_after_the_last_step_at_returned_pose() {
    let mut odometry = Odometry::new(OdometryParams {
        min_range: 0.0,
        diagnostics: true,
        ..OdometryParams::default()
    });
    odometry.set_capture_linearization(true);
    let scan = diagnostic_scene();
    odometry.register(&scan);
    let mut pending = odometry.begin(&scan);
    let mut identity = [[0.0; 6]; 6];
    for (i, row) in identity.iter_mut().enumerate() {
        row[i] = 1.0;
    }
    odometry.step(
        &mut pending,
        (identity, [0.04, -0.02, 0.01, 0.01, 0.02, -0.01]),
    );
    let pose = odometry.finish(pending);
    let snapshot = odometry.last_linearization().unwrap();
    assert_eq!(
        snapshot.source_world,
        scan.positions
            .iter()
            .map(|p| pose.apply(p))
            .collect::<Vec<_>>()
    );
    assert_eq!(snapshot.matched_source_indices.len(), 12);
    assert_ne!(snapshot.normal_matrix, identity);
    assert!(snapshot.squared_distances.iter().any(|&d| d > 0.0));
    assert_eq!(snapshot.target_world, scan.positions);
    assert!(snapshot.finite);
}

#[test]
fn reports_distinguish_no_alignment_and_early_finish() {
    let mut odometry = Odometry::new(OdometryParams {
        min_range: 0.0,
        diagnostics: true,
        ..OdometryParams::default()
    });
    assert!(odometry.last_report().is_none());
    assert_eq!(odometry.last_iterations(), 0);
    odometry.register(&diagnostic_scene());
    let report = odometry.last_report().unwrap();
    assert_eq!(report.termination, RegistrationTermination::EmptyMap);
    assert_eq!(report.input_points, 12);
    assert_eq!(report.registration_points, 12);
    assert!(report.correspondences.is_none());
    let small = PointCloud {
        positions: vec![[2.0, 1.0, 0.0]],
        ..PointCloud::default()
    };
    odometry.register(&small);
    let report = odometry.last_report().unwrap();
    assert_eq!(
        report.termination,
        RegistrationTermination::InsufficientPoints
    );
    assert_eq!(report.registration_points, 1);
    assert_eq!(report.iterations, 0);
    assert!(report.correspondences.is_none());
    let registration = odometry.begin(&diagnostic_scene());
    assert_eq!(registration.termination(), None);
    odometry.finish(registration);
    assert_eq!(
        odometry.last_report().unwrap().termination,
        RegistrationTermination::Unfinished
    );
}

#[test]
fn zero_matches_and_collinear_matches_are_not_convergence() {
    for collinear in [false, true] {
        let mut odometry = Odometry::new(OdometryParams {
            min_range: 0.0,
            max_range: 200.0,
            diagnostics: true,
            ..OdometryParams::default()
        });
        let mut scan = diagnostic_scene();
        if collinear {
            scan.positions = (0..12).map(|i| [3.0 * i as f64, 0.0, 0.0]).collect();
        }
        odometry.register(&scan);
        if !collinear {
            for p in &mut scan.positions {
                p[0] += 50.0;
            }
        }
        assert_eq!(odometry.register(&scan), Rigid::IDENTITY);
        let report = odometry.last_report().unwrap();
        assert_eq!(report.termination, RegistrationTermination::SingularSystem);
        assert_eq!(report.iterations, 0);
        let stats = report.correspondences.as_ref().unwrap();
        assert_eq!(stats.matched_points, if collinear { 12 } else { 0 });
        assert_eq!(stats.rmse_m, collinear.then_some(0.0));
        assert_eq!(stats.max_residual_m, collinear.then_some(0.0));
        assert_eq!(stats.matched_fraction, if collinear { 1.0 } else { 0.0 });
        // Diagnostic only: the existing policy still inserts every scan.
        assert_eq!(odometry.placed().len(), 12);
    }
}

#[test]
fn reports_convergence_limits_and_repeated_steps_without_mutation() {
    for limited in [false, true] {
        let mut odometry = Odometry::new(OdometryParams {
            min_range: 0.0,
            max_iterations: 1,
            ..OdometryParams::default()
        });
        let scan = diagnostic_scene();
        odometry.register(&scan);
        let mut registration = odometry.begin(&scan);
        let mut h = [[0.0; 6]; 6];
        for (i, row) in h.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        let dx = if limited { [0.01; 6] } else { [0.0; 6] };
        assert!(odometry.step(&mut registration, (h, dx)));
        let total = *registration.total();
        assert!(odometry.step(&mut registration, (h, [1.0; 6])));
        assert_eq!(*registration.total(), total);
        odometry.finish(registration);
        let report = odometry.last_report().unwrap();
        assert_eq!(report.iterations, 1);
        assert_eq!(
            report.termination,
            if limited {
                RegistrationTermination::IterationLimit
            } else {
                RegistrationTermination::Converged
            }
        );
        assert!(report.correspondences.is_none());
    }
}

#[test]
fn nonfinite_systems_and_updates_preserve_the_last_finite_transform() {
    let mut odometry = Odometry::new(OdometryParams {
        min_range: 0.0,
        ..OdometryParams::default()
    });
    let scan = diagnostic_scene();
    odometry.register(&scan);
    for case in 0..4 {
        let mut registration = odometry.begin(&scan);
        let mut h = [[0.0; 6]; 6];
        for (i, row) in h.iter_mut().enumerate() {
            row[i] = 1.0;
        }
        let mut b = [0.0; 6];
        match case {
            0 => h[2][3] = f64::NAN,
            1 => b[1] = f64::INFINITY,
            2 => {
                h[0][0] = f64::MAX;
                h[0][1] = -f64::MAX;
                h[1][0] = f64::MAX / 2.0;
                h[1][1] = f64::MAX;
            }
            _ => b[3] = f64::MAX,
        }
        assert!(odometry.step(&mut registration, (h, b)));
        assert_eq!(*registration.total(), Rigid::IDENTITY);
        assert_eq!(registration.iterations, 0);
        assert_eq!(
            registration.termination(),
            Some(if case == 3 {
                RegistrationTermination::NonFiniteUpdate
            } else {
                RegistrationTermination::NonFiniteSystem
            })
        );
    }
}

#[test]
fn final_residuals_match_an_independent_search_before_insertion() {
    let mut odometry = Odometry::new(OdometryParams {
        min_range: 0.0,
        initial_sigma: 0.1,
        max_iterations: 1,
        diagnostics: true,
        ..OdometryParams::default()
    });
    let mut scan = diagnostic_scene();
    odometry.register(&scan);
    let old_map = odometry.map.map.clone();
    for (i, p) in scan.positions.iter_mut().enumerate() {
        p[0] += 0.10 + 0.003 * i as f64;
        p[1] -= 0.02 * (i % 3) as f64;
    }
    scan.positions[0][0] += 0.6; // In the stencil, outside the threshold.
    scan.positions.push([35.0, 0.0, 0.0]); // Outside the stencil entirely.
    let registration = odometry.begin(&scan);
    let source = registration.diagnostic_source.clone().unwrap();
    let pose = odometry.complete(registration);
    let report = odometry.last_report().unwrap();
    let stats = report.correspondences.as_ref().unwrap();
    let mut distances = Vec::new();
    for p in &source {
        let q = pose.apply(p);
        let key = q.map(|v| v.floor() as i64);
        let best = old_map
            .iter()
            .filter(|(k, _)| (0..3).all(|i| (k[i] - key[i]).abs() <= 1))
            .flat_map(|(_, points)| points)
            .map(|m| (0..3).map(|i| (q[i] - m[i]).powi(2)).sum::<f64>())
            .min_by(f64::total_cmp);
        if let Some(d) = best.filter(|d| *d <= report.threshold_m.powi(2)) {
            distances.push(d);
        }
    }
    assert_eq!(stats.matched_points, 11);
    assert_eq!(stats.matched_points, distances.len());
    assert_eq!(report.registration_points, 13);
    assert_eq!(stats.matched_fraction, 11.0 / 13.0);
    let rms = (distances.iter().sum::<f64>() / distances.len() as f64).sqrt();
    assert!((stats.rmse_m.unwrap() - rms).abs() < 1e-14);
    assert!(rms > 1e-3); // Would be zero after inserting the current points.
    let maximum = distances.iter().copied().reduce(f64::max).unwrap().sqrt();
    assert!((stats.max_residual_m.unwrap() - maximum).abs() < 1e-14);
}

#[test]
fn optional_diagnostics_preserve_pose_bits_and_threshold_history() {
    let params = OdometryParams {
        min_range: 0.0,
        ..OdometryParams::default()
    };
    let mut plain = Odometry::new(params);
    let mut measured = Odometry::new(OdometryParams {
        diagnostics: true,
        ..params
    });
    let mut scan = diagnostic_scene();
    for _ in 0..5 {
        let a = plain.register(&scan);
        let b = measured.register(&scan);
        assert_eq!(
            a.to_matrix().map(f64::to_bits),
            b.to_matrix().map(f64::to_bits)
        );
        assert_eq!(plain.error_sum.to_bits(), measured.error_sum.to_bits());
        assert_eq!(plain.error_count, measured.error_count);
        assert_eq!(plain.last_iterations(), measured.last_iterations());
        assert_eq!(
            plain.last_report().unwrap().termination,
            measured.last_report().unwrap().termination
        );
        for p in &mut scan.positions {
            p[0] -= 0.1;
        }
    }
}

#[test]
fn relative_prediction_composes_on_the_right_without_changing_deskew() {
    let mut odometry = Odometry::new(OdometryParams {
        deskew: true,
        ..OdometryParams::default()
    });
    let empty = PointCloud::default();
    odometry.register(&empty);
    let last = Rigid {
        rotation: crate::pose_graph::exp_so3(&[0.0, 0.0, 0.4]),
        translation: [4.0, -2.0, 1.0],
    };
    odometry.register_with_prediction(&empty, &last).unwrap();
    let motion = Rigid {
        rotation: crate::pose_graph::exp_so3(&[0.1, -0.2, 0.3]),
        translation: [0.5, -0.3, 0.2],
    };
    let mut scan = PointCloud {
        positions: (0..20).map(|k| [2.0, k as f64, 1.0]).collect(),
        ..PointCloud::default()
    };
    scan.attributes.push(crate::Attribute {
        name: TIME.into(),
        values: crate::AttributeValues::F32((0..20).map(|k| k as f32).collect()),
    });
    let original = odometry.begin(&scan);
    let predicted = odometry.begin_with_prediction(&scan, &motion).unwrap();
    assert_eq!(original.frame, predicted.frame);
    assert_eq!(predicted.prediction, last.compose(&motion));
    assert_ne!(predicted.prediction, motion.compose(&last));
    let found = odometry.register_with_prediction(&empty, &motion).unwrap();
    for (a, b) in found
        .to_matrix()
        .iter()
        .zip(last.compose(&motion).to_matrix())
    {
        assert!((a - b).abs() < 1e-14);
    }
    assert_eq!(odometry.last_iterations(), 0);
}

#[test]
fn invalid_predictions_cannot_change_poses_map_or_threshold_history() {
    let mut odometry = Odometry::new(OdometryParams::default());
    let empty = PointCloud::default();
    assert!(
        odometry
            .register_with_prediction(&empty, &Rigid::IDENTITY)
            .is_err()
    );
    assert!(odometry.poses().is_empty());
    odometry.register(&empty);
    for case in 0..4 {
        let mut bad = Rigid::IDENTITY;
        match case {
            0 => bad.rotation[0][0] = 2.0,
            1 => bad.rotation[0][0] = -1.0,
            2 => bad.translation[1] = f64::NAN,
            _ => bad.rotation[2][1] = f64::INFINITY,
        }
        assert!(odometry.register_with_prediction(&empty, &bad).is_err());
        assert_eq!(odometry.poses(), &[Rigid::IDENTITY]);
        assert!(odometry.map().is_empty());
        assert_eq!(odometry.error_count, 0);
        assert_eq!(odometry.last_iterations(), 0);
    }
}

/// Ground and boxes scattered about, in world coordinates.
fn town() -> Vec<[f64; 3]> {
    let mut out = Vec::new();
    let mut seed = 7u64;
    let mut next = || {
        seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        (seed >> 33) as f64 / (1u64 << 31) as f64
    };
    // Ground, at random places so that no shift matches it with itself.
    for _ in 0..100_000 {
        out.push([next() * 120.0 - 40.0, next() * 60.0 - 30.0, 0.0]);
    }
    // Boxes of 2..4 m at random places and heights.
    for _ in 0..200 {
        let (cx, cy) = (next() * 110.0 - 30.0, next() * 50.0 - 25.0);
        if cy.abs() < 4.0 {
            continue;
        }
        let (sx, sy, h) = (2.0 + 2.0 * next(), 2.0 + 2.0 * next(), 1.0 + 4.0 * next());
        let steps = |len: f64| (0..=(len / 0.2) as usize).map(move |k| k as f64 * 0.2);
        for z in steps(h) {
            for x in steps(sx) {
                out.push([cx + x, cy, z]);
                out.push([cx + x, cy + sy, z]);
            }
            for y in steps(sy) {
                out.push([cx, cy + y, z]);
                out.push([cx + sx, cy + y, z]);
            }
        }
    }
    out
}

/// Scans of `world` along `truth`, each taken over the move to the next
/// pose (the sensor spinning once, so the azimuth tells the time).
fn skewed_scans(world: &[[f64; 3]], truth: &[Rigid]) -> Vec<PointCloud> {
    truth
        .windows(2)
        .map(|w| {
            let (a, b) = (&w[0], &w[1]);
            let step = crate::pose_graph::inverse(a).compose(b);
            let spin = crate::pose_graph::log_so3(&step.rotation);
            // In a's frame, which azimuth each point is at (a full turn from -pi).
            let mut seen: Vec<([f64; 3], f64)> = world
                .iter()
                .map(|p| crate::pose_graph::inverse(a).apply(p))
                .filter(|p| norm(p) < 40.0)
                .map(|p| (p, p[1].atan2(p[0])))
                .collect();
            seen.sort_by(|x, y| x.1.total_cmp(&y.1));
            let positions = seen
                .iter()
                .map(|(p, azimuth)| {
                    // Taken this far into the move: seen from there.
                    let f = (azimuth + std::f64::consts::PI) / std::f64::consts::TAU;
                    let there = Rigid {
                        rotation: crate::pose_graph::exp_so3(&spin.map(|w| w * f)),
                        translation: step.translation.map(|t| t * f),
                    };
                    crate::pose_graph::inverse(&there).apply(p)
                })
                .collect();
            PointCloud {
                positions,
                ..PointCloud::default()
            }
        })
        .collect()
}

#[test]
fn a_turn_taken_while_scanning_is_undone() {
    let world = town();
    // Pulling away into a turn of 6 degrees a scan while moving 0.6 m:
    // the end of each sweep sees the world turned against its start.
    let mut u = 0.0;
    let truth: Vec<Rigid> = (0..30)
        .map(|k| {
            u += 0.1 * k.min(10) as f64;
            let yaw = 0.1 * u;
            let (s, c) = yaw.sin_cos();
            Rigid {
                rotation: [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
                translation: [6.0 * yaw.sin(), 6.0 * (1.0 - yaw.cos()), 1.5],
            }
        })
        .collect();
    let scans = skewed_scans(&world, &truth);
    let errors: Vec<f64> = [true, false]
        .map(|deskew| {
            let mut odometry = Odometry::new(OdometryParams {
                deskew,
                ..OdometryParams::default()
            });
            for scan in &scans {
                odometry.register(scan);
            }
            // Each pose stands for the middle of its scan: halfway between the two.
            let found = odometry.poses()[28];
            let first = crate::pose_graph::inverse(&truth[0]);
            let (a, b) = (first.compose(&truth[28]), first.compose(&truth[29]));
            let mid: [f64; 3] =
                std::array::from_fn(|k| 0.5 * (a.translation[k] + b.translation[k]));
            norm(&std::array::from_fn(|k| found.translation[k] - mid[k]))
        })
        .to_vec();
    assert!(
        errors[0] < 0.15,
        "deskewed {}, skewed {}",
        errors[0],
        errors[1]
    );
    assert!(
        errors[0] < 0.5 * errors[1],
        "deskewed {}, skewed {}",
        errors[0],
        errors[1]
    );
}

#[test]
fn a_drive_through_a_town_is_followed() {
    let world = town();
    let mut odometry = Odometry::new(OdometryParams::default());
    // Pulling away to 1 m a scan, turning slowly; the scans see 40 m around.
    let mut x = 0.0;
    let truth: Vec<Rigid> = (0..40)
        .map(|k| {
            x += 0.1 * k.min(10) as f64;
            let yaw: f64 = 0.004 * k as f64;
            let (s, c) = yaw.sin_cos();
            Rigid {
                rotation: [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
                translation: [x, 0.1 * x, 1.5],
            }
        })
        .collect();
    for pose in &truth {
        let back = crate::pose_graph::inverse(pose);
        let scan = PointCloud {
            positions: world
                .iter()
                .map(|p| back.apply(p))
                .filter(|p| norm(p) < 40.0)
                .collect(),
            ..PointCloud::default()
        };
        odometry.register(&scan);
    }
    // Relative to the first pose, as odometry starts at the identity.
    let last = crate::pose_graph::inverse(&truth[0]).compose(&truth[39]);
    let found = odometry.poses()[39];
    let error = norm(&std::array::from_fn(|k| {
        found.translation[k] - last.translation[k]
    }));
    assert!(error < 0.2, "{error}");
}
