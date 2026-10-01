//! Small synthetic, multi-level COPC for spatial signal reader regressions.
//! `cargo run -p ca-core --example make_test_signal_copc -- ../cloudanalyzer/tests/data/signal.copc.laz`
use ca_core::io::copc::{VoxelKey, write_minimal_copc};
use std::collections::BTreeMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args()
        .nth(1)
        .ok_or("usage: make_test_signal_copc <out>")?;
    let center = [49995., 50001., 7.25];
    let mut points = Vec::new();
    for x in 0..25 {
        for z in (0..51).step_by(5) {
            points.push([49994.4 + x as f64 * 0.05, 50001., 7. + z as f64 * 0.01]);
        }
    }
    // Outside the head box, but correctly inside the root and its children.
    points.extend((0..5000).map(|i| [49996.5 + (i % 100) as f64 * 0.001, 50002., 9.]));
    let mut root = Vec::new();
    let mut children = BTreeMap::<[i32; 3], Vec<[f64; 3]>>::new();
    for (i, point) in points.into_iter().enumerate() {
        if i % 3 == 0 {
            root.push(point);
        } else {
            let key = std::array::from_fn(|a| i32::from(point[a] >= center[a]));
            children.entry(key).or_default().push(point);
        }
    }
    let mut nodes = vec![(
        VoxelKey {
            level: 0,
            x: 0,
            y: 0,
            z: 0,
        },
        root,
    )];
    nodes.extend(
        children
            .into_iter()
            .map(|([x, y, z], points)| (VoxelKey { level: 1, x, y, z }, points)),
    );
    let bytes = write_minimal_copc(&nodes, center, 2.);
    std::fs::write(path, bytes)?;
    Ok(())
}
