use ca_core::{
    filter::{KnnPart, Regions},
    normals::{Orientation, estimate_normals},
};

fn f64_bits(a: &[f64], b: &[f64]) {
    assert_eq!(
        a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn prepared_normals_match_fresh_for_parameters_ties_and_nonfinite_points() {
    for count in [0, 1, 17, 257] {
        for mode in 0..5 {
            let points: Vec<_> = (0..count)
                .map(|i| {
                    let j = i % 37;
                    let p = [
                        (j % 7) as f64 * 0.1,
                        (j / 7) as f64 * 0.1,
                        (j as f64).sin() * 0.02,
                    ];
                    match mode {
                        1 => [p[0] + 368000.0, p[1] + 3955000.0, p[2] + 40.0],
                        2 => p.map(|v| v * 1e155),
                        3 => [1.0, -0.0, 0.0],
                        4 if i == 0 => [f64::NAN, f64::INFINITY, f64::NEG_INFINITY],
                        _ => p,
                    }
                })
                .collect();
            let part = KnnPart::prepared(points.clone());
            let bytes = part.retained_bytes();
            for k in [1, 3, 12, 24, count + 2] {
                for orientation in [Orientation::Up, Orientation::Away([0.0, 0.0, 0.0])] {
                    let a = part.normals(k, orientation).into_flattened();
                    let bounded = part
                        .normals_with_bounds(k, orientation, true)
                        .into_flattened();
                    let b = estimate_normals(&points, k, orientation).into_flattened();
                    assert_eq!(
                        a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        b.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                    assert_eq!(
                        a.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        bounded.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                }
            }
            assert_eq!(part.retained_bytes(), bytes);
        }
    }
}

#[test]
fn prepared_sor_preserves_open_order_reach_candidates_and_query_bits() {
    let points: Vec<_> = (0..257)
        .map(|i| {
            [
                (i % 13) as f64 * 0.1,
                (i / 13) as f64 * 0.1,
                (i as f64).sin() * 0.02,
            ]
        })
        .collect();
    let fresh = KnnPart::new(points.clone());
    let prepared = KnnPart::prepared(points.clone());
    let regions = Regions::from_flat(&[
        0.0, 0.0, 0.0, -1.0, 1.0, 2.0, 1.0, 1.0, 0.0, -1.0, 2.0, 2.0, 1.0,
    ]);
    // Use the documented bbox encoding rather than a grid's ownership order.
    let regions = regions.expect("valid bounds regions");
    for k in [1, 3, 12, 24, 300] {
        let a = fresh.local_with_scratch(k, 0, &regions, false);
        let b = prepared.local(k, 0, &regions);
        let bounded = prepared.local_with_options(k, 0, &regions, true, true);
        assert_eq!(a.open, b.open);
        assert_eq!(a.reach, b.reach);
        f64_bits(&a.means, &b.means);
        f64_bits(&a.candidates, &b.candidates);
        assert_eq!(a.open, bounded.open);
        assert_eq!(a.reach, bounded.reach);
        f64_bits(&a.means, &bounded.means);
        f64_bits(&a.candidates, &bounded.candidates);
        let queries = [
            [0.1, 0.0, 0.02],
            [f64::NAN, 0.0, 0.0],
            [f64::INFINITY, -0.0, 1.0],
        ];
        f64_bits(
            &fresh.within_with_scratch(&queries, k, false),
            &prepared.within(&queries, k),
        );
        f64_bits(
            &fresh.within(&queries, k),
            &prepared.within_with_bounds(&queries, k, true, false, true),
        );
    }
}

#[test]
fn prepared_points_are_owned_and_repeated_fits_do_not_retain_results() {
    let mut points: Vec<_> = (0..129)
        .map(|i| [(i % 11) as f64, (i / 11) as f64, 0.0])
        .collect();
    let part = KnnPart::prepared(points.clone());
    let expected = estimate_normals(&points, 12, Orientation::Up);
    let bytes = part.retained_bytes();
    assert!(bytes >= points.len() * std::mem::size_of::<[f64; 3]>());
    points.fill([100.0; 3]);
    for _ in 0..3 {
        assert_eq!(part.normals(12, Orientation::Up), expected);
        assert_eq!(part.retained_bytes(), bytes);
    }
}
