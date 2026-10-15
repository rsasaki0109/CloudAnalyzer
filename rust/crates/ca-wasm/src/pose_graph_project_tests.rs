use super::{GraphSnapshot, PoseGraphSession};
use ca_core::icp::Rigid;
use ca_core::pose_graph::{EdgeKind, GravityEdge, PoseGraph};

#[test]
fn project_preserves_constraints_initial_poses_and_timestamps() {
    let mut graph = PoseGraph::from_poses(&[Rigid::IDENTITY, Rigid::IDENTITY], [[1.; 6]; 6]);
    graph.nodes[0].fixed = false;
    graph.nodes[1].fixed = true;
    graph.edges[0].kind = EdgeKind::Loop;
    graph.gravity = [0., 0., 1.];
    graph.gravity_edges.push(GravityEdge {
        node: 1,
        up: [0., 0., 1.],
        information: [[4., 0.], [0., 4.]],
    });
    graph.add_plane([0., 0., 1., -2.]);
    let mut session = PoseGraphSession::new(graph, vec![1.25, 2.5]);
    session.initial[1].translation[0] = 3.;
    let restored = PoseGraphSession::from_snapshot(&session.to_snapshot().unwrap()).unwrap();
    assert_eq!(restored.graph, session.graph);
    assert_eq!(restored.initial, session.initial);
    assert_eq!(restored.timestamps, session.timestamps);
}

#[test]
fn malformed_project_indices_are_rejected_before_use() {
    let mut snapshot = GraphSnapshot {
        version: 1,
        graph: PoseGraph::from_poses(&[Rigid::IDENTITY, Rigid::IDENTITY], [[1.; 6]; 6]),
        timestamps: vec![],
        initial: vec![Rigid::IDENTITY; 2],
        dynamic: vec![],
    };
    assert!(snapshot.validate().is_ok());
    snapshot.graph.edges[0].to = 2;
    assert!(snapshot.validate().is_err());
    snapshot.graph.edges[0].to = 1;
    snapshot.graph.nodes[1].id = snapshot.graph.nodes[0].id;
    assert!(snapshot.validate().is_err());
}
