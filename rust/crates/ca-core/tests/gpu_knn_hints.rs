use ca_core::{
    filter::{KnnPart, Regions},
    normals::Orientation,
};

fn normal_bits(a: &[[f32; 3]], b: &[[f32; 3]]) {
    assert_eq!(
        a.iter().flatten().map(|v| v.to_bits()).collect::<Vec<_>>(),
        b.iter().flatten().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
}

#[test]
fn block_hints_preserve_normals_and_local_sor_ties_and_nonfinite_fallback() {
    for count in [1, 17, 65, 257] {
        for mode in 0..6 {
            let points: Vec<_> = (0..count)
                .map(|i| {
                    let j = i % 47;
                    let p = [
                        (j % 7) as f64 * 0.1,
                        (j / 7) as f64 * 0.1,
                        (j as f64).sin() * 0.02,
                    ];
                    match mode {
                        1 => [p[0] + 368000.0, p[1] + 3955000.0, p[2] + 40.0],
                        2 => [1.0, -0.0, 0.0],
                        3 => p.map(|v| v * 1e155),
                        4 if i == 0 => [f64::NAN, f64::INFINITY, f64::NEG_INFINITY],
                        5 => [if i % 2 == 0 { -0.0 } else { 0.0 }, (i % 5) as f64, 0.0],
                        _ => p,
                    }
                })
                .collect();
            let mut part = KnnPart::prepared(points);
            let bytes = part.retained_bytes();
            let (_, queries) = part.gpu_hint_buffers().unwrap();
            assert_eq!(queries.len(), count * 4);
            assert!(part.retained_bytes() > bytes);
            let regions = Regions::Boxes(vec![([-1e200; 3], [1e200; 3]), ([-1.0; 3], [1.0; 3])]);
            for k in [3, 12, 24, 31, 32, 64] {
                let normal_candidates = part.cpu_hint_candidates(k).unwrap();
                let local_candidates = part.cpu_hint_candidates(k + 1).unwrap();
                for orientation in [Orientation::Up, Orientation::Away([0.25, 0.25, 0.0])] {
                    let expected = part.normals(k, orientation);
                    for slots in [
                        (0..count * k)
                            .map(|i| ((i * 19 + 11) % count) as u32)
                            .collect::<Vec<_>>(),
                        vec![0; count * k],
                        vec![u32::MAX; count * k],
                    ] {
                        normal_bits(
                            &expected,
                            &part.normals_candidates(k, orientation, &slots).unwrap(),
                        );
                    }
                    normal_bits(
                        &expected,
                        &part
                            .normals_hinted(k, orientation, &normal_candidates)
                            .unwrap(),
                    );
                    normal_bits(
                        &expected,
                        &part
                            .normals_hinted(k, orientation, &vec![u32::MAX; count])
                            .unwrap(),
                    );
                    normal_bits(
                        &expected,
                        &part
                            .normals_hinted(k, orientation, &vec![0; count])
                            .unwrap(),
                    );
                }
                let a = part.local(k, 0, &regions);
                let slots: Vec<_> = (0..count * (k + 1))
                    .map(|i| ((i * 19 + 11) % count) as u32)
                    .collect();
                let b = part.local_candidates(k, 0, &regions, &slots).unwrap();
                assert_eq!(a.open, b.open);
                assert_eq!(a.reach, b.reach);
                assert_eq!(
                    a.means.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    b.means.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
                assert_eq!(
                    a.candidates.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                    b.candidates.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                );
                for candidates in [&local_candidates, &vec![u32::MAX; count], &vec![0; count]] {
                    let b = part.local_hinted(k, 0, &regions, candidates).unwrap();
                    assert_eq!(a.open, b.open);
                    assert_eq!(a.reach, b.reach);
                    assert_eq!(
                        a.means.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        b.means.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                    assert_eq!(
                        a.candidates.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
                        b.candidates.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
                    );
                }
            }
            assert!(part.normals_hinted(24, Orientation::Up, &[]).is_none());
            assert!(part.local_hinted(24, 0, &regions, &[]).is_none());
            assert!(part.normals_candidates(24, Orientation::Up, &[]).is_none());
            assert!(part.local_candidates(24, 0, &regions, &[]).is_none());
        }
    }
}
