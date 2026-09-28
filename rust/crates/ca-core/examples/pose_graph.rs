//! Optimise a g2o pose graph: `cargo run --release --example pose_graph -- in.g2o [out.g2o]`.

use ca_core::pose_graph::{OptimizeParams, PoseGraph, optimize};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let Some(input) = args.get(1) else {
        eprintln!("usage: pose_graph <in.g2o> [out.g2o]");
        std::process::exit(2);
    };
    let text = std::fs::read_to_string(input).expect("read input");
    let mut graph = PoseGraph::from_g2o(&text).expect("parse g2o");
    println!("{} nodes, {} edges", graph.nodes.len(), graph.edges.len());
    let start = std::time::Instant::now();
    let report = optimize(&mut graph, &OptimizeParams::default()).expect("no free nodes");
    println!("{report:?} in {:.2?}", start.elapsed());
    if let Some(output) = args.get(2) {
        std::fs::write(output, graph.to_g2o()).expect("write output");
    }
}
