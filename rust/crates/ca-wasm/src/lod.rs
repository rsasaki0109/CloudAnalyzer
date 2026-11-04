//! Level-of-detail octree layout and buckets.

use super::*;

pub(crate) fn nodes_to_flat(nodes: &[OctreeNode]) -> Vec<f64> {
    let mut out = Vec::with_capacity(nodes.len() * NODE_STRIDE);
    for n in nodes {
        out.extend([n.start as f64, n.count as f64]);
        out.extend(n.min);
        out.extend([n.size, n.level as f64]);
        out.extend(
            n.children
                .map(|c| if c == NO_CHILD { -1.0 } else { c as f64 }),
        );
    }
    out
}

pub(crate) fn nodes_from_flat(flat: &[f64]) -> Result<Vec<OctreeNode>, JsError> {
    if !flat.len().is_multiple_of(NODE_STRIDE) {
        return Err(JsError::new("node table has the wrong length"));
    }
    Ok(flat
        .as_chunks::<NODE_STRIDE>()
        .0
        .iter()
        .map(|n| OctreeNode {
            start: n[0] as u32,
            count: n[1] as u32,
            min: [n[2], n[3], n[4]],
            size: n[5],
            level: n[6] as u8,
            children: std::array::from_fn(|c| {
                if n[7 + c] < 0.0 {
                    NO_CHILD
                } else {
                    n[7 + c] as u32
                }
            }),
        })
        .collect())
}

pub(crate) fn cube_of(cube: &[f64]) -> Result<([f64; 3], f64), JsError> {
    match cube {
        &[x, y, z, size] if size > 0.0 => Ok(([x, y, z], size)),
        _ => Err(JsError::new("cube must be [minX, minY, minZ, size]")),
    }
}

fn colors_of(colors: Option<Vec<u8>>, points: usize) -> Result<Option<Vec<[u8; 3]>>, JsError> {
    match colors {
        Some(c) if c.len() != 3 * points => Err(JsError::new("colors do not match positions")),
        c => Ok(c.map(|c| c.as_chunks::<3>().0.to_vec())),
    }
}

/// Points reordered off the main worker, with what the reordering produced.
#[wasm_bindgen]
pub struct Reordered {
    positions: Vec<[f64; 3]>,
    colors: Option<Vec<[u8; 3]>>,
    order: Vec<u32>,
    counts: Vec<u32>,
    nodes: Vec<OctreeNode>,
}

#[wasm_bindgen]
impl Reordered {
    pub fn positions(&self) -> Vec<f64> {
        self.positions.as_flattened().to_vec()
    }

    pub fn colors(&self) -> Option<Vec<u8>> {
        self.colors.as_ref().map(|c| c.as_flattened().to_vec())
    }

    /// `order[i]` is the input index of the point now at `i`.
    pub fn order(&self) -> Vec<u32> {
        self.order.clone()
    }

    /// [`bucket_chunk`]: points per bucket. [`build_bucket`]: the root's and
    /// the level-1 node's share of the bucket.
    pub fn counts(&self) -> Vec<u32> {
        self.counts.clone()
    }

    /// [`build_bucket`]: the subtree's node table (see [`Cloud::lod_nodes`]),
    /// ranges relative to the subtree.
    pub fn nodes(&self) -> Vec<f64> {
        nodes_to_flat(&self.nodes)
    }
}

/// Step 1 of a parallel index build (see [`Cloud::index_cube`]): sort a
/// slice of the cloud by bucket, keeping the order within each bucket.
#[wasm_bindgen(js_name = bucketChunk)]
pub fn bucket_chunk(
    positions: &[f64],
    colors: Option<Vec<u8>>,
    cube: &[f64],
) -> Result<Reordered, JsError> {
    let (lo, size) = cube_of(cube)?;
    let points = positions.as_chunks::<3>().0;
    let colors = colors_of(colors, points.len())?;
    let (order, counts) = ca_core::octree::bucket_order(points, lo, size);
    Ok(Reordered {
        positions: order.iter().map(|&i| points[i as usize]).collect(),
        colors: colors.map(|c| order.iter().map(|&i| c[i as usize]).collect()),
        order,
        counts: counts.to_vec(),
        nodes: Vec::new(),
    })
}

/// Step 2 of a parallel index build: one bucket's points (concatenated from
/// every slice in order) reordered into `[root's | level-1 node's | subtree]`.
#[wasm_bindgen(js_name = buildBucket)]
pub fn build_bucket(
    positions: &[f64],
    colors: Option<Vec<u8>>,
    cube: &[f64],
    key: usize,
) -> Result<Reordered, JsError> {
    let (lo, size) = cube_of(cube)?;
    let mut points = positions.as_chunks::<3>().0.to_vec();
    let mut colors = colors_of(colors, points.len())?;
    let bucket = ca_core::octree::build_bucket(
        &mut points,
        colors.as_deref_mut(),
        lo,
        size,
        key,
        OctreeParams::default(),
    )
    .ok_or_else(|| JsError::new("invalid bucket"))?;
    Ok(Reordered {
        positions: points,
        colors,
        order: bucket.order,
        counts: vec![bucket.root_kept as u32, bucket.level1_kept as u32],
        nodes: bucket.nodes,
    })
}

/// Reorder `cloud` in place into octree order (colors follow) and return
/// the octree without the permutation, which the viewer does not need.
pub(crate) fn build_lod(cloud: &mut PointCloud) -> Result<Octree, JsError> {
    let mut lod = Octree::build_for_cloud(cloud, OctreeParams::default())
        .ok_or_else(|| JsError::new("cloud is empty or too large"))?;
    lod.order = Vec::new();
    Ok(lod)
}
