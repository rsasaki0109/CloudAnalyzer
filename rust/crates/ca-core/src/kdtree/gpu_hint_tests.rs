use super::*;

#[test]
fn hinted_k_neighbours_preserve_exact_indices_distances_and_tie_order() {
    for mode in 0..5 {
        let points: Vec<_> = (0..257)
            .map(|i| {
                let p = [(i % 7) as f64, ((i / 7) % 7) as f64, (i % 3) as f64];
                match mode {
                    1 => p.map(|v| v + 3955000.0),
                    2 => [1.0, -0.0, 0.0],
                    3 => p.map(|v| v * 1e155),
                    4 if i == 0 => [f64::NAN, f64::INFINITY, 0.0],
                    _ => p,
                }
            })
            .collect();
        let tree = KdTree::new(&points).unwrap();
        let hints = KnnRadiusHints::new(&tree);
        let queries = points.iter().step_by(17).copied().chain([
            [0.5; 3],
            [-0.0; 3],
            [f64::INFINITY; 3],
            [f64::NAN; 3],
        ]);
        for q in queries {
            for k in [0, 1, 3, 12, 24, 25, 32, 64, 500] {
                let expected = tree
                    .nearest_k(&q, k)
                    .into_iter()
                    .map(|(i, d)| (i, d.to_bits()))
                    .collect::<Vec<_>>();
                let nearest: Vec<_> = expected
                    .iter()
                    .map(|(index, _)| {
                        tree.indices
                            .iter()
                            .position(|v| *v as usize == *index)
                            .unwrap() as u32
                    })
                    .collect();
                let arbitrary: Vec<_> = (0..k)
                    .map(|i| ((i * 19 + 113) % tree.len()) as u32)
                    .collect();
                let mut reversed = nearest.clone();
                reversed.reverse();
                for slots in [
                    nearest,
                    reversed,
                    arbitrary,
                    vec![0; k],
                    vec![u32::MAX; k],
                    Vec::new(),
                ] {
                    let mut found = Vec::new();
                    tree.nearest_k_into_candidates(&q, k, &slots, &hints, &mut found);
                    assert_eq!(
                        expected,
                        found
                            .into_iter()
                            .map(|(i, d)| (i, d.to_bits()))
                            .collect::<Vec<_>>()
                    );
                }
                for node in (0..tree.nodes.len() as u32).chain([u32::MAX]) {
                    let mut found = Vec::new();
                    tree.nearest_k_into_hinted(&q, k, node, &hints, &mut found);
                    assert_eq!(
                        expected,
                        found
                            .into_iter()
                            .map(|(i, d)| (i, d.to_bits()))
                            .collect::<Vec<_>>()
                    );
                }
            }
        }
    }
}
