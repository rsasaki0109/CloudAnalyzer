use ca_core::{PointCloud, filter, icp, kdtree::KdTree, m3c2, map_quality, normals};

fn points(n: usize) -> Vec<[f64; 3]> {
    (0..n)
        .map(|i| {
            let j = i % 997;
            let x = (j % 37) as f64 * 0.1;
            let y = (j / 37) as f64 * 0.1;
            [x, y, (x * 0.7).sin() * 0.1 + y * y * 0.02]
        })
        .collect()
}

#[test]
fn retained_knn_preserves_indices_distance_bits_ties_and_changing_lengths() {
    for n in [1, 16, 17, 257, 1500] {
        let pts = points(n);
        let tree = KdTree::new(&pts).unwrap();
        let mut scratch = vec![(usize::MAX, f64::NAN); 40];
        for k in [12, 0, 1, 24, 3, n + 3, 2] {
            for q in pts
                .iter()
                .take(23)
                .chain([[f64::NAN, 0.0, 0.0], [f64::INFINITY, -0.0, 0.0]].iter())
            {
                let expected = tree.nearest_k(q, k);
                tree.nearest_k_into(q, k, &mut scratch);
                let bits = |hits: &[(usize, f64)]| {
                    hits.iter()
                        .map(|&(i, d)| (i, d.to_bits()))
                        .collect::<Vec<_>>()
                };
                assert_eq!(bits(&scratch), bits(&expected));
            }
        }
    }
}

#[test]
fn normals_keep_f32_bits_with_reused_knn_and_parallel_chunks() {
    for n in [1, 17, 1500] {
        let pts = points(n);
        for k in [0, 3, 12, 24] {
            for orientation in [
                normals::Orientation::Up,
                normals::Orientation::Away([0.0; 3]),
            ] {
                let expected = normals::estimate_normals_with_scratch(&pts, k, orientation, false);
                let actual = normals::estimate_normals_with_scratch(&pts, k, orientation, true);
                let bits = |values: &[[f32; 3]]| {
                    values
                        .iter()
                        .map(|v| v.map(f32::to_bits))
                        .collect::<Vec<_>>()
                };
                assert_eq!(bits(&expected), bits(&actual));
                #[cfg(feature = "parallel")]
                assert_eq!(
                    bits(&expected),
                    bits(&normals::estimate_normals_par(&pts, k, orientation))
                );
            }
        }
    }
}

#[test]
fn sor_reused_knn_preserves_means_and_kept_points() {
    let pts = points(1500);
    for k in [1, 12, 24] {
        let a = filter::knn_mean_distances_with_scratch(&pts, k, false);
        let b = filter::knn_mean_distances_with_scratch(&pts, k, true);
        assert_eq!(
            a.iter().map(|x| x.to_bits()).collect::<Vec<_>>(),
            b.iter().map(|x| x.to_bits()).collect::<Vec<_>>()
        );
        for ratio in [0.0, 1.0, 2.0] {
            assert_eq!(filter::sor_keep(&a, ratio), filter::sor_keep(&b, ratio));
        }
    }
}

#[test]
fn m3c2_cached_cylinder_preserves_all_f64_bits() {
    let a = points(700);
    let b = a
        .iter()
        .map(|p| [p[0], p[1], p[2] + 0.02])
        .collect::<Vec<_>>();
    for normal_radius in [0.1, 0.4, 2.0] {
        for (projection_radius, max_depth) in
            [(0.25, 0.5), (f64::MIN_POSITIVE, 0.5), (f64::MAX, f64::MAX)]
        {
            for bounds in [false, true] {
                let params = m3c2::M3c2Params {
                    normal_radius,
                    projection_radius,
                    max_depth,
                    ..Default::default()
                };
                let cached = m3c2::M3c2::with_radius_bounds(&a, &b, params, bounds)
                    .unwrap()
                    .with_cached_geometry(true);
                let control = m3c2::M3c2::with_radius_bounds(&a, &b, params, bounds)
                    .unwrap()
                    .with_cached_geometry(false);
                for shared in [false, true] {
                    for core in a.iter().step_by(41) {
                        let x =
                            cached.at_with_search(core, &mut Vec::new(), &mut Vec::new(), shared);
                        let y =
                            control.at_with_search(core, &mut Vec::new(), &mut Vec::new(), shared);
                        assert_eq!(
                            (
                                x.0.to_bits(),
                                x.1.to_bits(),
                                x.2.map(|n| n.map(f64::to_bits))
                            ),
                            (
                                y.0.to_bits(),
                                y.1.to_bits(),
                                y.2.map(|n| n.map(f64::to_bits))
                            )
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn icp_scratch_keeps_transform_rms_iteration_and_convergence_bits() {
    let pts = points(1500);
    let reference = PointCloud {
        positions: pts.clone(),
        ..Default::default()
    };
    let moving = PointCloud {
        positions: pts
            .iter()
            .map(|p| [p[0] + 0.01, p[1] - 0.02, p[2] + 0.025])
            .collect(),
        ..Default::default()
    };
    for metric in [icp::IcpMetric::PointToPoint, icp::IcpMetric::PointToPlane] {
        for overlap in [0.7, 1.0] {
            let params = icp::IcpParams {
                metric,
                overlap,
                max_iterations: 8,
                ..Default::default()
            };
            let a = icp::icp_with_scratch(&moving, &reference, params, false).unwrap();
            let b = icp::icp_with_scratch(&moving, &reference, params, true).unwrap();
            assert_eq!(
                a.transform.to_matrix().map(f64::to_bits),
                b.transform.to_matrix().map(f64::to_bits)
            );
            assert_eq!(
                (
                    a.rms_initial.to_bits(),
                    a.rms_final.to_bits(),
                    a.iterations,
                    a.converged
                ),
                (
                    b.rms_initial.to_bits(),
                    b.rms_final.to_bits(),
                    b.iterations,
                    b.converged
                )
            );
        }
    }
}

#[test]
fn map_neighbour_scratch_preserves_voxel_counts_and_existing_reduction_precision() {
    let pts = points(1500);
    let shifted = pts
        .iter()
        .map(|p| [p[0] + 0.01, p[1], p[2] + 0.02])
        .collect::<Vec<_>>();
    for voxel in [0.1, 0.5, 2.0] {
        let a = map_quality::voxel_scores_with_scratch(&pts, &shifted, voxel, 5, false, false);
        let b = map_quality::voxel_scores_with_scratch(&pts, &shifted, voxel, 5, false, true);
        assert_eq!(a.voxels, b.voxels);
        for (x, y) in [(a.awd, b.awd), (a.scs, b.scs)] {
            assert!(
                (x.is_nan() && y.is_nan())
                    || (x - y).abs() <= 1e-12 * x.abs().max(y.abs()).max(1.0)
            );
        }
    }
}

#[test]
fn prepared_icp_owns_reference_and_preserves_parameter_changes_and_repeated_fits() {
    let original = PointCloud {
        positions: points(1500),
        ..Default::default()
    };
    let mut source = original.clone();
    let mut prepared = icp::IcpPrepared::new(&source).unwrap();
    source.positions.fill([0.0; 3]);
    assert!(prepared.retained_bytes() <= original.len() * 96 + 16_384);
    for metric in [
        icp::IcpMetric::PointToPlane,
        icp::IcpMetric::PointToPoint,
        icp::IcpMetric::PointToPlane,
    ] {
        for shift in [0.02, -0.01, 0.03] {
            let moving = PointCloud {
                positions: original
                    .positions
                    .iter()
                    .map(|p| [p[0], p[1], p[2] + shift])
                    .collect(),
                ..Default::default()
            };
            let params = icp::IcpParams {
                metric,
                max_iterations: 8,
                overlap: 0.8,
                ..Default::default()
            };
            let a = icp::icp(&moving, &original, params).unwrap();
            let b = prepared.compute(&moving, params).unwrap();
            assert_eq!(
                a.transform.to_matrix().map(f64::to_bits),
                b.transform.to_matrix().map(f64::to_bits)
            );
            assert_eq!(
                (
                    a.rms_initial.to_bits(),
                    a.rms_final.to_bits(),
                    a.iterations,
                    a.converged
                ),
                (
                    b.rms_initial.to_bits(),
                    b.rms_final.to_bits(),
                    b.iterations,
                    b.converged
                )
            );
        }
    }
}
