//! WebAssembly bindings for the CloudAnalyzer Web viewer.

use ca_core::icp::{IcpMetric, IcpParams, Rigid};
use ca_core::octree::{BUCKETS, BucketLayout, NO_CHILD, Octree, OctreeNode, OctreeParams};
use ca_core::{
    AttributeValues, CLASSIFICATION, DistanceStats, INTENSITY, PointCloud, TriangleMesh,
};
use std::borrow::Cow;
use wasm_bindgen::prelude::*;

mod bag;
mod cloud;
mod copc_box;
mod distance;
mod gpu_c2c;
mod icp;
mod knn;
mod lod;
mod m3c2;
mod merge;
mod mesh;
mod odometry;
mod pose_graph;
mod raster;
mod readers;
mod segmentation;
mod stream;
mod trajectory;
mod vector_map;
mod volume;
pub use bag::{BagFile, BagMessage};
pub use cloud::*;
pub use copc_box::CopcBoxReader;
pub use distance::*;
pub use gpu_c2c::{GpuC2cJob, GpuC2cReference, GpuC2cVerifier};
pub use icp::*;
pub use knn::*;
pub use lod::*;
pub use m3c2::{M3c2Plan, M3c2Prepared, compute_m3c2, m3c2_cloud_distances, m3c2_distances};
pub use merge::*;
pub use mesh::*;
pub use odometry::{LidarOdometry, OdometryMap, Registration};
pub use pose_graph::{PoseGraphSession, register_scans};
pub use raster::*;
pub use readers::*;
pub use segmentation::*;
pub use stream::*;
pub use trajectory::*;
pub use vector_map::VectorMapSession;
pub use volume::*;

/// Numbers per node in [`Cloud::lod_nodes`].
const NODE_STRIDE: usize = 15;

// The imported flat WASM slice already owns this call's query snapshot.
// Borrow its triples until the synchronous search returns instead of copying
// them into another allocation. Retained query-order entries still need ownership.
// Default off: both real-source Autoware passes regress in whole C2M calls.
// Used in `mesh`; kept here because web/scripts/benchmark-mesh-query-borrow.mjs
// reads this file for the flag.
const USE_MESH_BORROWED_QUERIES: bool = false;

/// Run the heavy kernels once on a small synthetic cloud. Browsers first
/// run WebAssembly with a baseline compiler and optimize a function only
/// after it has been busy, and never in the middle of a call; one long call
/// on a large cloud would run unoptimized from start to end. Calling this
/// when the page is idle gets the optimized code compiled beforehand.
#[wasm_bindgen(js_name = warmUp)]
pub fn warm_up() {
    let n = 60_000;
    let positions: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let (x, y) = ((i % 250) as f64 * 0.1, (i / 250) as f64 * 0.1);
            [x, y, (x * 0.7).sin() + (y * 0.3).cos()]
        })
        .collect();
    let mut cloud = PointCloud {
        positions: positions.clone(),
        colors: Some(vec![[1, 2, 3]; n]),
        attributes: Vec::new(),
    };
    let params = OctreeParams {
        max_leaf: 2_000,
        ..OctreeParams::default()
    };
    let _ = Octree::build_bucketed(&mut cloud, params);
    let _ = Octree::build_for_cloud(&mut cloud.clone(), params);
    let _ = ca_core::io::write_las(&cloud, &[], true);
    let queries: Vec<[f64; 3]> = positions
        .iter()
        .step_by(3)
        .map(|p| [p[0] + 0.03, p[1], p[2] + 0.1])
        .collect();
    let _ = ca_core::cloud_to_cloud(
        &PointCloud {
            positions: queries.clone(),
            colors: None,
            attributes: Vec::new(),
        },
        &cloud,
    );
    let mesh = TriangleMesh {
        vertices: positions[..2_500].to_vec(),
        triangles: (0..9u32)
            .flat_map(|j| {
                (0..249u32).flat_map(move |i| {
                    let k = j * 250 + i;
                    [[k, k + 1, k + 250], [k + 1, k + 251, k + 250]]
                })
            })
            .collect(),
    };
    let _ = ca_core::cloud_to_mesh(&queries[..5_000], &mesh, true);
    let split = ca_core::filter::split_for_knn(&positions, 2);
    let part: Vec<[f64; 3]> = split.parts[0]
        .iter()
        .map(|&i| positions[i as usize])
        .collect();
    let part = ca_core::filter::KnnPart::new(part);
    let local = part.local(8, 0, &split.regions);
    let _ = part.within(&queries[..2_000], 8);
    let _ = ca_core::filter::sor_keep(&local.means, 1.0);
    let _ = ca_core::filter::spatial_subsample(&cloud, 0.05);
    let _ = ca_core::filter::octree_subsample(&cloud, 6);
    let _ = ca_core::raster::rasterize(
        &cloud,
        ca_core::raster::RasterParams {
            cell: 0.25,
            height: ca_core::raster::RasterHeight::Percentile(50.0),
            fill_empty: true,
            class: None,
        },
    );
    let _ = ca_core::delaunay::delaunay_25d(&positions, ca_core::delaunay::MaxEdge::Auto);
    let normals = ca_core::normals::estimate_normals(
        &positions[..20_000],
        12,
        ca_core::normals::Orientation::Up,
    );
    for primitive in [
        ca_core::shapes::Primitive::Plane,
        ca_core::shapes::Primitive::Cylinder,
    ] {
        let params = ca_core::shapes::RansacParams {
            primitive,
            threshold: 0.05,
            min_support: 2_000,
            max_shapes: 2,
            candidates: 100,
            ..Default::default()
        };
        let _ = ca_core::shapes::detect_shapes(&positions[..20_000], &normals, &params);
    }
    let _ = ca_core::cluster::euclidean_clusters(&positions[..20_000], 0.15, 10);
}
