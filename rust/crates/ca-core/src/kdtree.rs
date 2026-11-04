//! A static 3D k-d tree specialised for nearest-neighbour queries.
//!
//! Points are copied into tree order so leaves are contiguous in memory.
//! Construction splits at the median of the widest axis, so it is
//! `O(n log n)` regardless of input order or duplicate coordinates.

const LEAF_SIZE: usize = 16;

#[cfg(test)]
mod gpu_hint_tests;

#[derive(Debug, Clone, Copy)]
struct Node {
    /// Range of `points` covered by this node.
    start: u32,
    end: u32,
    /// Split coordinate and axis; `axis == LEAF` marks a leaf.
    split: f64,
    axis: u8,
    /// Index of the right child; the left child is always `self + 1`.
    right: u32,
}

const LEAF: u8 = u8::MAX;

#[derive(Debug, Clone)]
pub struct KdTree {
    nodes: Vec<Node>,
    points: Vec<[f64; 3]>,
    /// Original index of each entry in `points`.
    indices: Vec<u32>,
}

/// Exact subtree boxes for optional GPU-selected k-neighbour block hints.
/// A hint only provides an upper radius; the original traversal still selects
/// and orders every final hit, including ties.
pub(crate) struct KnnRadiusHints {
    bounds: Vec<([f64; 3], [f64; 3])>,
    finite: bool,
}

impl KnnRadiusHints {
    pub(crate) fn new(tree: &KdTree) -> Self {
        let mut out = Self {
            bounds: vec![([0.0; 3], [0.0; 3]); tree.nodes.len()],
            finite: tree.points.iter().flatten().all(|v| v.is_finite()),
        };
        fn fill(tree: &KdTree, node: usize, bounds: &mut [([f64; 3], [f64; 3])]) {
            let n = tree.nodes[node];
            bounds[node] = if n.axis == LEAF {
                crate::distance::bounds(tree.points[n.start as usize..n.end as usize].iter())
            } else {
                fill(tree, node + 1, bounds);
                fill(tree, n.right as usize, bounds);
                (
                    std::array::from_fn(|a| {
                        bounds[node + 1].0[a].min(bounds[n.right as usize].0[a])
                    }),
                    std::array::from_fn(|a| {
                        bounds[node + 1].1[a].max(bounds[n.right as usize].1[a])
                    }),
                )
            };
        }
        fill(tree, 0, &mut out.bounds);
        out
    }

    pub(crate) fn storage_bytes(&self) -> usize {
        self.bounds.capacity() * std::mem::size_of::<([f64; 3], [f64; 3])>()
    }
}

/// A nearest-neighbour hit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Nearest {
    /// Index into the slice the tree was built from.
    pub index: usize,
    pub distance_sq: f64,
    /// Position of the hit in tree order, for use as a later `guess`.
    slot: u32,
}

/// Call-local state shared by all recursive nodes of one bounded query.
/// Keep the invariant radius beside the incumbent instead of passing another
/// f64 argument at every recursive call. No heap allocation is involved.
struct BoundedNearest {
    best: Nearest,
    upper: f64,
}

// Whole-RPC real-source comparisons found the shared-state prototype slower.
// Keep the original radius argument unless explicitly rebuilding for comparison.
const USE_SHARED_NEAREST_STATE: bool = false;

impl Nearest {
    pub(crate) fn tree_slot(self) -> u32 {
        self.slot
    }
}

/// Finiteness checked once for an immutable tree before bounded queries.
pub(crate) struct NearestRadiusHints {
    finite: bool,
}

impl NearestRadiusHints {
    pub(crate) fn new(tree: &KdTree) -> Self {
        Self {
            finite: tree.points.iter().flatten().all(|v| v.is_finite()),
        }
    }
}

/// Exact verification of untrusted tree-order candidates. A leaf certificate
/// can exclude every other leaf using original f64 sibling extrema; otherwise
/// verification searches either the complete tree or the leaf's ancestral
/// siblings with exact subtree boxes.
#[derive(Debug)]
pub struct CandidateKdTree {
    tree: KdTree,
    leaves: Vec<LeafCertificate>,
    leaf_for_slot: Vec<u32>,
    bounds: Vec<([f64; 3], [f64; 3])>,
    parents: Vec<u32>,
    finite: bool,
}

#[derive(Debug, Clone, Copy)]
struct LeafCertificate {
    node: u32,
    start: u32,
    end: u32,
    /// Every point outside this leaf belongs to an ancestral sibling. That
    /// sibling is separated along its split axis by one of these extrema.
    lower: [f64; 3],
    upper: [f64; 3],
    coincident: bool,
}

impl CandidateKdTree {
    pub fn new(points: &[[f64; 3]]) -> Option<Self> {
        let tree = KdTree::new(points)?;
        Some(Self::from_tree(tree))
    }

    fn from_tree(tree: KdTree) -> Self {
        Self::from_tree_with_bounds(tree, None)
    }

    fn from_tree_with_bounds(tree: KdTree, imported: Option<Vec<SnapshotBounds>>) -> Self {
        // With total-order extrema, any NaN or infinity occurs at an axis
        // minimum or maximum. Six root values certify finiteness without
        // another pass over every imported coordinate.
        let finite = imported.as_ref().map_or_else(
            || tree.points.iter().flatten().all(|v| v.is_finite()),
            |bounds| {
                bounds[0]
                    .0
                    .iter()
                    .chain(&bounds[0].1)
                    .all(|v| v.is_finite())
            },
        );
        let mut out = Self {
            finite,
            leaf_for_slot: vec![0; tree.len()],
            tree,
            leaves: Vec::new(),
            bounds: Vec::new(),
            parents: Vec::new(),
        };
        // Keep exact subtree boxes as well as the cheap axis certificates.
        // They can exclude diagonal/corner regions when one axis cannot.
        // Validation uses total ordering, whose finite extrema differ from
        // numeric min/max only in signed-zero bits. Comparisons and squared
        // gaps retain their values. Keep the old path for nonfinite data.
        let imported = imported.filter(|_| out.finite);
        let mut bounds = imported.unwrap_or_default();
        out.parents = vec![0; out.tree.nodes.len()];
        if bounds.is_empty() {
            bounds.resize(out.tree.nodes.len(), ([0.0; 3], [0.0; 3]));
            out.subtree_bounds(0, &mut bounds);
        }
        out.certificates(0, [f64::NEG_INFINITY; 3], [f64::INFINITY; 3], &bounds);
        out.bounds = bounds;
        out
    }

    /// Portable exact tree data; derived verification certificates are rebuilt on import.
    pub fn snapshot(&self) -> Vec<u8> {
        self.tree.snapshot()
    }

    pub fn from_snapshot(bytes: &[u8], max_points: usize) -> Result<Self, &'static str> {
        Self::from_snapshot_with_reused_bounds(bytes, max_points, true)
    }

    /// Retain validated finite subtree extrema instead of rescanning points.
    /// Snapshot bytes and every validation step remain unchanged.
    pub fn from_snapshot_with_reused_bounds(
        bytes: &[u8],
        max_points: usize,
        reuse_bounds: bool,
    ) -> Result<Self, &'static str> {
        if reuse_bounds {
            let (tree, bounds) = KdTree::read_snapshot::<true>(bytes, max_points)?;
            Ok(Self::from_tree_with_bounds(tree, Some(bounds)))
        } else {
            KdTree::from_snapshot(bytes, max_points).map(Self::from_tree)
        }
    }

    fn subtree_bounds(&self, node: usize, bounds: &mut [([f64; 3], [f64; 3])]) {
        let n = self.tree.nodes[node];
        bounds[node] = if n.axis == LEAF {
            crate::distance::bounds(self.tree.points[n.start as usize..n.end as usize].iter())
        } else {
            let left = node + 1;
            let right = n.right as usize;
            self.subtree_bounds(left, bounds);
            self.subtree_bounds(right, bounds);
            (
                std::array::from_fn(|a| bounds[left].0[a].min(bounds[right].0[a])),
                std::array::from_fn(|a| bounds[left].1[a].max(bounds[right].1[a])),
            )
        };
    }

    fn certificates(
        &mut self,
        node: usize,
        lower: [f64; 3],
        upper: [f64; 3],
        bounds: &[([f64; 3], [f64; 3])],
    ) {
        let n = self.tree.nodes[node];
        if n.axis == LEAF {
            self.leaf_for_slot[n.start as usize..n.end as usize].fill(self.leaves.len() as u32);
            self.leaves.push(LeafCertificate {
                node: node as u32,
                start: n.start,
                end: n.end,
                lower,
                upper,
                coincident: bounds[node].0 == bounds[node].1,
            });
        } else {
            self.parents[node + 1] = node as u32;
            self.parents[n.right as usize] = node as u32;
            let axis = n.axis as usize;
            let mut left_upper = upper;
            left_upper[axis] = left_upper[axis].min(bounds[n.right as usize].0[axis]);
            self.certificates(node + 1, lower, left_upper, bounds);
            let mut right_lower = lower;
            right_lower[axis] = right_lower[axis].max(bounds[node + 1].1[axis]);
            self.certificates(n.right as usize, right_lower, upper, bounds);
        }
    }

    /// Returns the exact hit and whether a leaf certificate sufficed. Invalid
    /// slots, nonfinite data and failed certificates use the complete search.
    pub fn nearest(&self, query: &[f64; 3], slot: u32) -> (Nearest, bool) {
        self.nearest_with_options(query, slot, false, false)
    }

    /// Scan the candidate leaf, then exclude or search its ancestral siblings
    /// with exact subtree boxes. Those disjoint subtrees cover every point
    /// outside the initial leaf, even for an incorrect candidate.
    pub fn nearest_bounded(&self, query: &[f64; 3], slot: u32) -> (Nearest, bool) {
        self.nearest_with_options(query, slot, true, true)
    }

    /// Select exact sibling bounds and pairwise WASM SIMD independently.
    /// SIMD keeps scalar arithmetic and hit order; other targets use scalar.
    pub fn nearest_with_options(
        &self,
        query: &[f64; 3],
        slot: u32,
        bounded: bool,
        simd: bool,
    ) -> (Nearest, bool) {
        if !self.finite || !query.iter().all(|v| v.is_finite()) || slot as usize >= self.tree.len()
        {
            return (self.tree.nearest_seeded(query, slot), false);
        }
        let leaf = self.leaves[self.leaf_for_slot[slot as usize] as usize];
        let mut best = Nearest {
            index: 0,
            distance_sq: distance_sq(&self.tree.points[slot as usize], query),
            slot,
        };
        if !leaf.coincident {
            scan_candidate_leaf(
                &self.tree.points,
                leaf.start,
                leaf.end,
                query,
                &mut best,
                simd,
            );
        }
        let mut certified = best.distance_sq.is_finite()
            && (0..3).all(|a| {
                let low = query[a] - leaf.lower[a];
                let high = leaf.upper[a] - query[a];
                low >= 0.0
                    && high >= 0.0
                    && low * low >= best.distance_sq
                    && high * high >= best.distance_sq
            });
        if !certified && bounded && best.distance_sq.is_finite() {
            let mut searched_leaf = false;
            let mut node = leaf.node as usize;
            while node != 0 {
                let parent = self.parents[node] as usize;
                let axis = self.tree.nodes[parent].axis as usize;
                let (sibling, gap) = if node == parent + 1 {
                    let sibling = self.tree.nodes[parent].right as usize;
                    (sibling, self.bounds[sibling].0[axis] - query[axis])
                } else {
                    let sibling = parent + 1;
                    (sibling, query[axis] - self.bounds[sibling].1[axis])
                };
                // Most ancestral siblings are far along their split axis.
                // Reject those with one exact component before loading and
                // projecting the full box.
                if !(gap >= 0.0 && gap * gap >= best.distance_sq) {
                    self.search_bounded(sibling, query, &mut best, &mut searched_leaf, simd);
                }
                node = parent;
            }
            certified = !searched_leaf;
        } else if !certified {
            self.tree.search::<false>(0, query, &mut best);
        }
        best.index = self.tree.indices[best.slot as usize] as usize;
        (best, certified)
    }

    fn search_bounded(
        &self,
        node: usize,
        q: &[f64; 3],
        best: &mut Nearest,
        searched_leaf: &mut bool,
        simd: bool,
    ) {
        let (lo, hi) = self.bounds[node];
        let delta: [f64; 3] = std::array::from_fn(|a| {
            if q[a] < lo[a] {
                lo[a] - q[a]
            } else if q[a] > hi[a] {
                hi[a] - q[a]
            } else {
                0.0
            }
        });
        // Same component order and arithmetic as point distance. Each
        // projected component is <= the corresponding component of every
        // point in the box; monotone nonnegative sums preserve the bound.
        let lower = delta[0] * delta[0] + delta[1] * delta[1] + delta[2] * delta[2];
        if lower >= best.distance_sq {
            return;
        }
        let n = self.tree.nodes[node];
        if n.axis == LEAF {
            *searched_leaf = true;
            let end = if lo == hi { n.start + 1 } else { n.end };
            scan_candidate_leaf(&self.tree.points, n.start, end, q, best, simd);
        } else {
            let diff = q[n.axis as usize] - n.split;
            let (near, far) = if diff < 0.0 {
                (node + 1, n.right as usize)
            } else {
                (n.right as usize, node + 1)
            };
            self.search_bounded(near, q, best, searched_leaf, simd);
            if diff * diff < best.distance_sq {
                self.search_bounded(far, q, best, searched_leaf, simd);
            }
        }
    }

    pub fn candidate_buffers(&self, origin: [f64; 3]) -> (Vec<f32>, Vec<u32>) {
        self.tree.candidate_buffers(origin)
    }

    /// Restore the input order so another WASM instance reconstructs identical
    /// tree slots. Never round these coordinates through the GPU buffers.
    pub fn original_points(&self) -> Vec<[f64; 3]> {
        let mut points = vec![[0.0; 3]; self.tree.len()];
        for (&point, &index) in self.tree.points.iter().zip(&self.tree.indices) {
            points[index as usize] = point;
        }
        points
    }

    pub fn original_point(&self, index: usize) -> Option<[f64; 3]> {
        self.tree
            .indices
            .iter()
            .position(|&i| i as usize == index)
            .map(|slot| self.tree.points[slot])
    }
}

fn read_u32(bytes: &[u8]) -> u32 {
    u32::from_le_bytes(bytes.try_into().unwrap())
}

type SnapshotBounds = ([f64; 3], [f64; 3]);

fn validate_snapshot_node<const KEEP_BOUNDS: bool>(
    tree: &KdTree,
    node: usize,
    start: u32,
    end: u32,
    bounds: &mut Vec<SnapshotBounds>,
) -> Option<(usize, SnapshotBounds)> {
    let n = *tree.nodes.get(node)?;
    if n.start != start || n.end != end || start >= end {
        return None;
    }
    if n.axis == LEAF {
        if n.right != 0 || n.split.to_bits() != 0 {
            return None;
        }
        let mut lo = tree.points[start as usize];
        let mut hi = lo;
        for p in &tree.points[start as usize + 1..end as usize] {
            for a in 0..3 {
                if p[a].total_cmp(&lo[a]).is_lt() {
                    lo[a] = p[a];
                }
                if p[a].total_cmp(&hi[a]).is_gt() {
                    hi[a] = p[a];
                }
            }
        }
        if KEEP_BOUNDS {
            bounds[node] = (lo, hi);
        }
        return Some((node + 1, (lo, hi)));
    }
    if end - start <= LEAF_SIZE as u32 {
        return None;
    }
    // Balanced splits limit recursion depth even for a malicious snapshot.
    let mid = start + (end - start) / 2;
    let (right, left_bounds) =
        validate_snapshot_node::<KEEP_BOUNDS>(tree, node + 1, start, mid, bounds)?;
    if n.right as usize != right {
        return None;
    }
    let (next, right_bounds) =
        validate_snapshot_node::<KEEP_BOUNDS>(tree, right, mid, end, bounds)?;
    let axis = n.axis as usize;
    if left_bounds.1[axis].total_cmp(&n.split).is_gt()
        || right_bounds.0[axis].total_cmp(&n.split).is_lt()
    {
        return None;
    }
    let lower = std::array::from_fn(|a| {
        if left_bounds.0[a].total_cmp(&right_bounds.0[a]).is_lt() {
            left_bounds.0[a]
        } else {
            right_bounds.0[a]
        }
    });
    let upper = std::array::from_fn(|a| {
        if left_bounds.1[a].total_cmp(&right_bounds.1[a]).is_gt() {
            left_bounds.1[a]
        } else {
            right_bounds.1[a]
        }
    });
    if KEEP_BOUNDS {
        bounds[node] = (lower, upper);
    }
    Some((next, (lower, upper)))
}

fn scan_candidate_leaf(
    points: &[[f64; 3]],
    start: u32,
    end: u32,
    q: &[f64; 3],
    best: &mut Nearest,
    simd: bool,
) {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    if simd {
        use core::arch::wasm32::*;
        let (qx, qy, qz) = (f64x2_splat(q[0]), f64x2_splat(q[1]), f64x2_splat(q[2]));
        let (pairs, remainder) = points[start as usize..end as usize].as_chunks::<2>();
        let mut minimum = f64x2_splat(best.distance_sq);
        for pair in pairs {
            // SAFETY: each pair contains six contiguous f64s. The three
            // unaligned 16-byte loads exactly cover those 48 valid bytes.
            let (xy, zx, yz) = unsafe {
                let ptr = pair.as_ptr().cast::<f64>();
                (
                    v128_load(ptr.cast()),
                    v128_load(ptr.add(2).cast()),
                    v128_load(ptr.add(4).cast()),
                )
            };
            let dx = f64x2_sub(i64x2_shuffle::<0, 3>(xy, zx), qx);
            let dy = f64x2_sub(i64x2_shuffle::<1, 2>(xy, yz), qy);
            let dz = f64x2_sub(i64x2_shuffle::<0, 3>(zx, yz), qz);
            // Preserve (dx*dx + dy*dy) + dz*dz, with separate multiply/add.
            // No fused/relaxed arithmetic or reassociated sums.
            let d = f64x2_add(
                f64x2_add(f64x2_mul(dx, dx), f64x2_mul(dy, dy)),
                f64x2_mul(dz, dz),
            );
            // Candidates usually already have the nearest distance. Reduce
            // distances without extracting lanes/updating indices per pair.
            // Finite inputs imply squared distances are nonnegative (possibly
            // infinite), so lane-wise min cannot introduce NaN or signed zero.
            minimum = f64x2_pmin(minimum, d);
        }
        if f64x2_extract_lane::<0>(minimum) >= best.distance_sq
            && f64x2_extract_lane::<1>(minimum) >= best.distance_sq
        {
            if let Some(point) = remainder.first() {
                let d = distance_sq(point, q);
                if d < best.distance_sq {
                    best.distance_sq = d;
                    best.slot = end - 1;
                }
            }
            return;
        }
        // A closer point requires its slot. The scalar scan preserves the
        // original strict comparisons and tie order, including odd tails.
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    let _ = simd;
    for i in start..end {
        let d = distance_sq(&points[i as usize], q);
        if d < best.distance_sq {
            best.distance_sq = d;
            best.slot = i;
        }
    }
}

/// Ordinary nearest searches often find a better hit in a leaf. Retain two
/// lane minima and slots, then select the first strict improvement in tree
/// order without scalar comparisons/branches for every point.
fn scan_nearest_leaf(points: &[[f64; 3]], start: u32, end: u32, q: &[f64; 3], best: &mut Nearest) {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        use core::arch::wasm32::*;
        let qx = f64x2_splat(q[0]);
        let qy = f64x2_splat(q[1]);
        let qz = f64x2_splat(q[2]);
        let mut minimum = f64x2_splat(best.distance_sq);
        let mut hits = i64x2_splat(i64::from(best.slot));
        let mut slots = i64x2(i64::from(start), i64::from(start) + 1);
        let step = i64x2_splat(2);
        let (pairs, tail) = points[start as usize..end as usize].as_chunks::<2>();
        for pair in pairs {
            let ptr = pair.as_ptr().cast::<f64>();
            // Exactly six f64 coordinates; each load is within this pair.
            let (xy, zx, yz) = unsafe {
                (
                    v128_load(ptr.cast()),
                    v128_load(ptr.add(2).cast()),
                    v128_load(ptr.add(4).cast()),
                )
            };
            let dx = f64x2_sub(i64x2_shuffle::<0, 3>(xy, zx), qx);
            let dy = f64x2_sub(i64x2_shuffle::<1, 2>(xy, yz), qy);
            let dz = f64x2_sub(i64x2_shuffle::<0, 3>(zx, yz), qz);
            let d = f64x2_add(
                f64x2_add(f64x2_mul(dx, dx), f64x2_mul(dy, dy)),
                f64x2_mul(dz, dz),
            );
            let closer = f64x2_lt(d, minimum);
            minimum = v128_bitselect(d, minimum, closer);
            hits = v128_bitselect(slots, hits, closer);
            slots = i64x2_add(slots, step);
        }
        select_leaf_minimum(
            best,
            [
                f64x2_extract_lane::<0>(minimum),
                f64x2_extract_lane::<1>(minimum),
            ],
            [
                i64x2_extract_lane::<0>(hits) as u32,
                i64x2_extract_lane::<1>(hits) as u32,
            ],
        );
        if let Some(point) = tail.first() {
            let d = distance_sq(point, q);
            if d < best.distance_sq {
                best.distance_sq = d;
                best.slot = end - 1;
            }
        }
    }
    #[cfg(not(all(target_arch = "wasm32", target_feature = "simd128")))]
    scan_candidate_leaf(points, start, end, q, best, false);
}

#[cfg(any(test, all(target_arch = "wasm32", target_feature = "simd128")))]
fn select_leaf_minimum(best: &mut Nearest, distances: [f64; 2], slots: [u32; 2]) {
    let seed = best.distance_sq;
    // A tied seed must survive. Among strict improvements from different
    // parity lanes, the smallest tree slot is the scalar scan's first hit.
    for k in 0..2 {
        if distances[k] < seed
            && (distances[k] < best.distance_sq
                || (distances[k] == best.distance_sq && slots[k] < best.slot))
        {
            best.distance_sq = distances[k];
            best.slot = slots[k];
        }
    }
}

impl KdTree {
    /// Build a tree over `points`. Returns `None` when `points` is empty or
    /// holds more than `u32::MAX` entries.
    pub fn new(points: &[[f64; 3]]) -> Option<Self> {
        if points.is_empty() || points.len() > u32::MAX as usize {
            return None;
        }
        // Partition (point, index) pairs in place: no indirection, so the
        // median selections stream through contiguous memory.
        let mut items: Vec<([f64; 3], u32)> =
            points.iter().zip(0..).map(|(&p, i)| (p, i)).collect();
        let mut nodes = Vec::with_capacity(2 * points.len() / LEAF_SIZE + 1);
        build(&mut items, 0, &mut nodes);
        Some(Self {
            nodes,
            points: items.iter().map(|&(p, _)| p).collect(),
            indices: items.iter().map(|&(_, i)| i).collect(),
        })
    }

    /// Portable, versioned exact tree data for independent verification workers.
    /// Coordinates and splits retain all f64 bits; no Rust struct layout or
    /// rounded GPU buffers are serialized. Derived certificates are rebuilt.
    pub fn snapshot(&self) -> Vec<u8> {
        let tree = self;
        let mut bytes = Vec::with_capacity(16 + tree.len() * 28 + tree.nodes.len() * 24);
        bytes.extend_from_slice(b"CAKD");
        bytes.extend_from_slice(&1_u32.to_le_bytes());
        bytes.extend_from_slice(&(tree.len() as u32).to_le_bytes());
        bytes.extend_from_slice(&(tree.nodes.len() as u32).to_le_bytes());
        for point in &tree.points {
            for v in point {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
        }
        for index in &tree.indices {
            bytes.extend_from_slice(&index.to_le_bytes());
        }
        for node in &tree.nodes {
            for v in [node.start, node.end, node.right, u32::from(node.axis)] {
                bytes.extend_from_slice(&v.to_le_bytes());
            }
            bytes.extend_from_slice(&node.split.to_le_bytes());
        }
        bytes
    }

    /// Import in O(n), validating sizes, index permutation, balanced preorder
    /// topology and split partitions before any traversal or certificate use.
    /// The explicit point budget bounds allocation even for untrusted headers.
    pub fn from_snapshot(bytes: &[u8], max_points: usize) -> Result<Self, &'static str> {
        Self::read_snapshot::<false>(bytes, max_points).map(|(tree, _)| tree)
    }

    fn read_snapshot<const KEEP_BOUNDS: bool>(
        bytes: &[u8],
        max_points: usize,
    ) -> Result<(Self, Vec<SnapshotBounds>), &'static str> {
        const INVALID: &str = "invalid exact k-d tree snapshot";
        if bytes.len() < 16 || &bytes[..4] != b"CAKD" || read_u32(&bytes[4..8]) != 1 {
            return Err(INVALID);
        }
        let count = read_u32(&bytes[8..12]) as usize;
        let nodes = read_u32(&bytes[12..16]) as usize;
        if count == 0 || count > max_points || nodes == 0 || nodes > count.saturating_mul(2) - 1 {
            return Err(INVALID);
        }
        let expected = count
            .checked_mul(28)
            .and_then(|n| nodes.checked_mul(24).and_then(|m| n.checked_add(m)))
            .and_then(|n| n.checked_add(16))
            .ok_or(INVALID)?;
        if bytes.len() != expected {
            return Err(INVALID);
        }
        // Balanced internal ranges contain > LEAF_SIZE points, so non-root
        // leaves contain at least LEAF_SIZE/2 points. Reject an impossible
        // node header before reserving the optional derived array.
        let max_valid_nodes = count.saturating_mul(2) / (LEAF_SIZE / 2) + 1;
        if KEEP_BOUNDS && nodes > max_valid_nodes {
            return Err(INVALID);
        }
        let point_end = 16 + count * 24;
        let index_end = point_end + count * 4;
        let points = bytes[16..point_end]
            .as_chunks::<24>()
            .0
            .iter()
            .map(|p| {
                std::array::from_fn(|a| f64::from_le_bytes(p[a * 8..a * 8 + 8].try_into().unwrap()))
            })
            .collect();
        let mut seen = vec![false; count];
        let mut indices = Vec::with_capacity(count);
        for index in bytes[point_end..index_end]
            .as_chunks::<4>()
            .0
            .iter()
            .map(|b| read_u32(b))
        {
            if index as usize >= count || std::mem::replace(&mut seen[index as usize], true) {
                return Err(INVALID);
            }
            indices.push(index);
        }
        let mut decoded = Vec::with_capacity(nodes);
        for node in bytes[index_end..].as_chunks::<24>().0 {
            let axis = read_u32(&node[12..16]);
            if axis > 2 && axis != u32::from(LEAF) {
                return Err(INVALID);
            }
            decoded.push(Node {
                start: read_u32(&node[..4]),
                end: read_u32(&node[4..8]),
                right: read_u32(&node[8..12]),
                axis: axis as u8,
                split: f64::from_le_bytes(node[16..24].try_into().unwrap()),
            });
        }
        let tree = KdTree {
            points,
            indices,
            nodes: decoded,
        };
        let mut bounds = if KEEP_BOUNDS {
            // One bulk initialization; validated nodes fill their own slot.
            vec![([0.0; 3], [0.0; 3]); nodes]
        } else {
            Vec::new()
        };
        let (next, _) =
            validate_snapshot_node::<KEEP_BOUNDS>(&tree, 0, 0, count as u32, &mut bounds)
                .ok_or(INVALID)?;
        if next != nodes {
            return Err(INVALID);
        }
        Ok((tree, bounds))
    }

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Coordinates in traversal order, retaining the original f64 bits.
    pub fn tree_points(&self) -> &[[f64; 3]] {
        &self.points
    }

    /// Nearest point to `query`. `guess`, typically the hit of a spatially
    /// close previous query, seeds the search so most branches are pruned.
    pub fn nearest(&self, query: &[f64; 3], guess: Option<Nearest>) -> Nearest {
        self.nearest_with_simd(query, guess, false)
    }

    /// Optional paired f64 WASM leaf arithmetic. Traversal, guesses and strict
    /// tie comparisons retain scalar order; other targets remain scalar.
    pub fn nearest_with_simd(
        &self,
        query: &[f64; 3],
        guess: Option<Nearest>,
        simd: bool,
    ) -> Nearest {
        let slot = guess.map_or(0, |g| g.slot);
        self.nearest_from_slot(query, slot, simd)
    }

    /// A candidate supplies only a recomputed f64 upper radius. The original
    /// seed and strict improvement/tie order are retained. `hints` must have
    /// been constructed for this immutable tree. The boolean records a tighter
    /// radius, not adoption of the candidate or a guaranteed branch saving.
    pub(crate) fn nearest_with_radius_hint(
        &self,
        query: &[f64; 3],
        guess: Option<Nearest>,
        candidate: u32,
        hints: &NearestRadiusHints,
        simd: bool,
    ) -> (Nearest, bool) {
        self.nearest_with_radius_layout::<USE_SHARED_NEAREST_STATE>(
            query, guess, candidate, hints, simd,
        )
    }

    fn nearest_with_radius_layout<const SHARED: bool>(
        &self,
        query: &[f64; 3],
        guess: Option<Nearest>,
        candidate: u32,
        hints: &NearestRadiusHints,
        simd: bool,
    ) -> (Nearest, bool) {
        let upper = self.points.get(candidate as usize).and_then(|p| {
            if !hints.finite || !query.iter().all(|v| v.is_finite()) {
                return None;
            }
            let d = distance_sq(p, query);
            d.is_finite().then_some(d)
        });
        let Some(upper) = upper else {
            return (self.nearest_with_simd(query, guess, simd), false);
        };
        let slot = guess.map_or(0, |g| g.slot);
        let mut best = Nearest {
            index: 0,
            distance_sq: distance_sq(&self.points[slot as usize], query),
            slot,
        };
        let tighter = upper < best.distance_sq;
        if SHARED && tighter {
            let mut state = BoundedNearest { best, upper };
            if simd {
                self.search_hinted_context::<true>(0, query, &mut state);
            } else {
                self.search_hinted_context::<false>(0, query, &mut state);
            }
            best = state.best;
        } else if tighter {
            if simd {
                self.search_hinted::<true>(0, query, &mut best, upper);
            } else {
                self.search_hinted::<false>(0, query, &mut best, upper);
            }
        } else if simd {
            self.search::<true>(0, query, &mut best);
        } else {
            self.search::<false>(0, query, &mut best);
        }
        best.index = self.indices[best.slot as usize] as usize;
        (best, tighter)
    }

    /// An untrusted candidate in tree order only seeds the upper bound. The
    /// full f64 search still runs, so a rounded or incorrect GPU candidate
    /// cannot change the distance. Out-of-range candidates use slot zero.
    pub fn nearest_seeded(&self, query: &[f64; 3], slot: u32) -> Nearest {
        let slot = if (slot as usize) < self.points.len() {
            slot
        } else {
            0
        };
        self.nearest_from_slot(query, slot, false)
    }

    fn nearest_from_slot(&self, query: &[f64; 3], slot: u32, simd: bool) -> Nearest {
        let mut best = Nearest {
            index: 0,
            distance_sq: distance_sq(&self.points[slot as usize], query),
            slot,
        };
        if simd {
            self.search::<true>(0, query, &mut best);
        } else {
            self.search::<false>(0, query, &mut best);
        }
        best.index = self.indices[best.slot as usize] as usize;
        best
    }

    /// Packed WebGPU candidate-search data relative to `origin`. Points are
    /// vec4<f32>; each node is five u32 words: start, end, right, axis, and
    /// the shifted split's f32 bits. Axis 255 denotes a leaf. These rounded
    /// copies must never be used as final distance measurements.
    pub fn candidate_buffers(&self, origin: [f64; 3]) -> (Vec<f32>, Vec<u32>) {
        let points = self
            .points
            .iter()
            .flat_map(|p| {
                [
                    (p[0] - origin[0]) as f32,
                    (p[1] - origin[1]) as f32,
                    (p[2] - origin[2]) as f32,
                    0.0,
                ]
            })
            .collect();
        (points, self.candidate_nodes(origin))
    }

    /// Rounded split data without the unused tree-order point buffer.
    pub fn candidate_nodes(&self, origin: [f64; 3]) -> Vec<u32> {
        self.nodes
            .iter()
            .flat_map(|n| {
                [
                    n.start,
                    n.end,
                    n.right,
                    u32::from(n.axis),
                    if n.axis == LEAF {
                        0
                    } else {
                        ((n.split - origin[n.axis as usize]) as f32).to_bits()
                    },
                ]
            })
            .collect()
    }

    /// Indices of all points within `radius` of `query` (unordered).
    pub fn within(&self, query: &[f64; 3], radius: f64, out: &mut Vec<usize>) {
        self.within_with_bounds(query, radius, out, false);
    }

    /// Combine ancestral split-plane lower bounds without changing hit order.
    /// No subtree boxes or additional per-tree storage are needed.
    pub fn within_with_bounds(
        &self,
        query: &[f64; 3],
        radius: f64,
        out: &mut Vec<usize>,
        bounds: bool,
    ) {
        out.clear();
        if radius >= 0.0 {
            if bounds {
                self.search_within::<true, false>(0, query, radius * radius, [0.0; 3], out);
            } else {
                self.search_within::<false, false>(0, query, radius * radius, [0.0; 3], out);
            }
        }
    }

    /// Same unsorted hit order as `within_with_bounds`, expressed as tree slots.
    /// Consumers can use `tree_points` without retaining the original input.
    pub fn within_slots_with_bounds(
        &self,
        query: &[f64; 3],
        radius: f64,
        out: &mut Vec<usize>,
        bounds: bool,
    ) {
        out.clear();
        if radius >= 0.0 {
            if bounds {
                self.search_within::<true, true>(0, query, radius * radius, [0.0; 3], out);
            } else {
                self.search_within::<false, true>(0, query, radius * radius, [0.0; 3], out);
            }
        }
    }

    fn search_within<const BOUNDS: bool, const SLOTS: bool>(
        &self,
        node: usize,
        q: &[f64; 3],
        r2: f64,
        lower: [f64; 3],
        out: &mut Vec<usize>,
    ) {
        let n = self.nodes[node];
        if n.axis == LEAF {
            for i in n.start as usize..n.end as usize {
                if distance_sq(&self.points[i], q) <= r2 {
                    out.push(if SLOTS { i } else { self.indices[i] as usize });
                }
            }
            return;
        }
        let diff = q[n.axis as usize] - n.split;
        let (near, far) = if diff < 0.0 {
            (node + 1, n.right as usize)
        } else {
            (n.right as usize, node + 1)
        };
        self.search_within::<BOUNDS, SLOTS>(near, q, r2, lower, out);
        if diff * diff <= r2 {
            let mut far_lower = lower;
            if BOUNDS {
                // Every point in the far child is at least this far along
                // the split axis. Retain stronger bounds from older splits.
                // Sum in the same x+y+z order as distance_sq; each nonnegative
                // component is a lower bound on the corresponding leaf term.
                far_lower[n.axis as usize] = far_lower[n.axis as usize].max(diff * diff);
                if far_lower[0] + far_lower[1] + far_lower[2] > r2 {
                    return;
                }
            }
            self.search_within::<BOUNDS, SLOTS>(far, q, r2, far_lower, out);
        }
    }

    /// Visit radius hits in the same order as `within_slots_with_bounds`,
    /// reusing leaf deltas/distance for a consumer without a temporary hit list.
    pub fn visit_within_with_bounds(
        &self,
        query: &[f64; 3],
        radius: f64,
        bounds: bool,
        visit: &mut impl FnMut(usize, [f64; 3], f64),
    ) {
        if radius >= 0.0 {
            if bounds {
                self.visit_within::<true>(0, query, radius * radius, [0.0; 3], visit);
            } else {
                self.visit_within::<false>(0, query, radius * radius, [0.0; 3], visit);
            }
        }
    }

    fn visit_within<const BOUNDS: bool>(
        &self,
        node: usize,
        q: &[f64; 3],
        r2: f64,
        lower: [f64; 3],
        visit: &mut impl FnMut(usize, [f64; 3], f64),
    ) {
        let n = self.nodes[node];
        if n.axis == LEAF {
            for i in n.start as usize..n.end as usize {
                let p = self.points[i];
                let v = [p[0] - q[0], p[1] - q[1], p[2] - q[2]];
                let d2 = v[0] * v[0] + v[1] * v[1] + v[2] * v[2];
                if d2 <= r2 {
                    visit(i, v, d2);
                }
            }
            return;
        }
        let diff = q[n.axis as usize] - n.split;
        let (near, far) = if diff < 0.0 {
            (node + 1, n.right as usize)
        } else {
            (n.right as usize, node + 1)
        };
        self.visit_within::<BOUNDS>(near, q, r2, lower, visit);
        if diff * diff <= r2 {
            let mut far_lower = lower;
            if BOUNDS {
                far_lower[n.axis as usize] = far_lower[n.axis as usize].max(diff * diff);
                if far_lower[0] + far_lower[1] + far_lower[2] > r2 {
                    return;
                }
            }
            self.visit_within::<BOUNDS>(far, q, r2, far_lower, visit);
        }
    }

    /// Two independent radius queries in one traversal. Each output retains
    /// exactly its own query's unsorted slot order, including boundary hits.
    pub fn within_two_slots_with_bounds(
        &self,
        query: &[f64; 3],
        radii: [f64; 2],
        out: [&mut Vec<usize>; 2],
        bounds: bool,
    ) {
        let [first, second] = out;
        first.clear();
        second.clear();
        // A negative squared radius disables a query in a pruned subtree.
        let squared = radii.map(|r| if r >= 0.0 { r * r } else { -1.0 });
        if squared.iter().all(|&r| r < 0.0) {
            return;
        }
        if bounds {
            self.search_within_two::<true>(0, query, squared, [0.0; 3], first, second);
        } else {
            self.search_within_two::<false>(0, query, squared, [0.0; 3], first, second);
        }
    }

    fn search_within_two<const BOUNDS: bool>(
        &self,
        node: usize,
        q: &[f64; 3],
        radii: [f64; 2],
        lower: [f64; 3],
        first: &mut Vec<usize>,
        second: &mut Vec<usize>,
    ) {
        // Once one radius has pruned this subtree, retain the single-query
        // traversal for the survivor instead of carrying a disabled query.
        if radii[0] < 0.0 {
            self.search_within::<BOUNDS, true>(node, q, radii[1], lower, second);
            return;
        }
        if radii[1] < 0.0 {
            self.search_within::<BOUNDS, true>(node, q, radii[0], lower, first);
            return;
        }
        let n = self.nodes[node];
        if n.axis == LEAF {
            for i in n.start as usize..n.end as usize {
                let distance = distance_sq(&self.points[i], q);
                if distance <= radii[0] {
                    first.push(i);
                }
                if distance <= radii[1] {
                    second.push(i);
                }
            }
            return;
        }
        let diff = q[n.axis as usize] - n.split;
        let (near, far) = if diff < 0.0 {
            (node + 1, n.right as usize)
        } else {
            (n.right as usize, node + 1)
        };
        self.search_within_two::<BOUNDS>(near, q, radii, lower, first, second);
        let mut far_lower = lower;
        if BOUNDS {
            far_lower[n.axis as usize] = far_lower[n.axis as usize].max(diff * diff);
        }
        let bound = far_lower[0] + far_lower[1] + far_lower[2];
        let far_radii = radii.map(|r| {
            if diff * diff <= r && !(BOUNDS && bound > r) {
                r
            } else {
                -1.0
            }
        });
        if far_radii.iter().any(|&r| r >= 0.0) {
            self.search_within_two::<BOUNDS>(far, q, far_radii, far_lower, first, second);
        }
    }

    /// The `k` nearest points to `query` as `(index, squared distance)`,
    /// closest first. Intended for small `k` (e.g. normal estimation).
    pub fn nearest_k(&self, query: &[f64; 3], k: usize) -> Vec<(usize, f64)> {
        let mut found: Vec<(usize, f64)> = Vec::with_capacity(k + 1);
        if k > 0 {
            self.search_k::<false, false>(0, query, k, [0.0; 3], &mut found);
        }
        found
            .into_iter()
            .map(|(slot, d)| (self.indices[slot] as usize, d))
            .collect()
    }

    /// Retained vector capacities, excluding allocator overhead and temporary queries.
    pub fn storage_bytes(&self) -> usize {
        self.points.capacity() * std::mem::size_of::<[f64; 3]>()
            + self.indices.capacity() * std::mem::size_of::<u32>()
            + self.nodes.capacity() * std::mem::size_of::<Node>()
    }

    /// Same hits and tie order, using one retained caller-owned query buffer.
    pub fn nearest_k_into(&self, query: &[f64; 3], k: usize, found: &mut Vec<(usize, f64)>) {
        self.nearest_k_into_with_simd(query, k, found, false);
    }

    /// Optional paired f64 leaf arithmetic, retaining scalar insertion/tie order.
    pub fn nearest_k_into_with_simd(
        &self,
        query: &[f64; 3],
        k: usize,
        found: &mut Vec<(usize, f64)>,
        simd: bool,
    ) {
        self.nearest_k_into_with_options(query, k, found, simd, false);
    }

    /// Combine ancestral split-plane bounds in x/y/z distance order.
    pub fn nearest_k_into_with_options(
        &self,
        query: &[f64; 3],
        k: usize,
        found: &mut Vec<(usize, f64)>,
        simd: bool,
        bounds: bool,
    ) {
        self.nearest_k_slots_into_with_options(query, k, found, simd, bounds);
        for (slot, _) in found {
            *slot = self.indices[*slot] as usize;
        }
    }

    /// Same traversal/distances, leaving internal tree slots in the scratch.
    /// Distance-only consumers need not gather original point indices.
    #[inline]
    pub(crate) fn nearest_k_slots_into_with_options(
        &self,
        query: &[f64; 3],
        k: usize,
        found: &mut Vec<(usize, f64)>,
        simd: bool,
        bounds: bool,
    ) {
        found.clear();
        found.reserve(k.min(self.len()));
        if k > 0 {
            match (simd, bounds) {
                (false, false) => self.search_k::<false, false>(0, query, k, [0.0; 3], found),
                (false, true) => self.search_k::<false, true>(0, query, k, [0.0; 3], found),
                (true, false) => self.search_k::<true, false>(0, query, k, [0.0; 3], found),
                (true, true) => self.search_k::<true, true>(0, query, k, [0.0; 3], found),
            }
        }
    }

    /// An arbitrary valid block containing k distinct points supplies a safe
    /// f64 upper radius. Invalid, oversized or nonfinite hints use the original
    /// scalar search. No candidate is inserted ahead of the original traversal.
    pub(crate) fn nearest_k_into_hinted(
        &self,
        query: &[f64; 3],
        k: usize,
        node: u32,
        hints: &KnnRadiusHints,
        found: &mut Vec<(usize, f64)>,
    ) {
        let upper = self.nodes.get(node as usize).and_then(|n| {
            let count = (n.end - n.start) as usize;
            if !hints.finite
                || !query.iter().all(|v| v.is_finite())
                || k == 0
                || k > 32
                || count < k
                || count > 64
            {
                return None;
            }
            let mut distances = [0.0; 64];
            for (d, p) in distances[..count]
                .iter_mut()
                .zip(&self.points[n.start as usize..n.end as usize])
            {
                *d = distance_sq(p, query);
                if !d.is_finite() {
                    return None;
                }
            }
            let (_, kth, _) = distances[..count].select_nth_unstable_by(k - 1, f64::total_cmp);
            Some(*kth)
        });
        let Some(upper) = upper else {
            self.nearest_k_into(query, k, found);
            return;
        };
        self.nearest_k_with_radius(query, k, upper, hints, found);
    }

    /// GPU slots are untrusted tree-order indices, not final neighbours. Any
    /// k distinct finite points bound the true kth distance from above. Keep
    /// candidate insertion out of the original traversal to preserve ties.
    pub(crate) fn nearest_k_into_candidates(
        &self,
        query: &[f64; 3],
        k: usize,
        slots: &[u32],
        hints: &KnnRadiusHints,
        found: &mut Vec<(usize, f64)>,
    ) {
        let upper = (|| {
            if k == 0
                || k > 32
                || slots.len() != k
                || !hints.finite
                || !query.iter().all(|v| v.is_finite())
            {
                return None;
            }
            let mut seen = 0u64;
            let mut upper = 0.0f64;
            for (i, &slot) in slots.iter().enumerate() {
                let bit = 1u64 << (slot % 64);
                if seen & bit != 0 && slots[..i].contains(&slot) {
                    return None;
                }
                seen |= bit;
                let d = distance_sq(self.points.get(slot as usize)?, query);
                if !d.is_finite() {
                    return None;
                }
                upper = upper.max(d);
            }
            Some(upper)
        })();
        if let Some(upper) = upper {
            found.clear();
            found.reserve(k);
            self.search_k_radius(0, query, k, upper, found);
            for (slot, _) in found {
                *slot = self.indices[*slot] as usize;
            }
        } else {
            self.nearest_k_into(query, k, found);
        }
    }

    /// The known upper radius can exclude a far subtree even before the
    /// original traversal has filled its hit list. No per-node box work.
    fn search_k_radius(
        &self,
        node: usize,
        q: &[f64; 3],
        k: usize,
        upper: f64,
        found: &mut Vec<(usize, f64)>,
    ) {
        let n = self.nodes[node];
        if n.axis == LEAF {
            for (i, p) in self.points[n.start as usize..n.end as usize]
                .iter()
                .enumerate()
            {
                insert_knn(found, k, n.start as usize + i, distance_sq(p, q));
            }
            return;
        }
        let diff = q[n.axis as usize] - n.split;
        let (near, far) = if diff < 0.0 {
            (node + 1, n.right as usize)
        } else {
            (n.right as usize, node + 1)
        };
        self.search_k_radius(near, q, k, upper, found);
        let plane = diff * diff;
        // Equality stays eligible while the original list is incomplete;
        // retain the original strict split test once the list is full.
        if plane <= upper && (found.len() < k || plane < found[found.len() - 1].1) {
            self.search_k_radius(far, q, k, upper, found);
        }
    }

    fn nearest_k_with_radius(
        &self,
        query: &[f64; 3],
        k: usize,
        upper: f64,
        hints: &KnnRadiusHints,
        found: &mut Vec<(usize, f64)>,
    ) {
        found.clear();
        found.reserve(k);
        self.search_k_hinted(0, query, k, upper, hints, found);
        for (slot, _) in found {
            *slot = self.indices[*slot] as usize;
        }
    }

    /// CPU comparison for the same rounded block walk as the GPU shader.
    pub(crate) fn rounded_knn_block(&self, query: &[f64; 3], origin: [f64; 3], k: usize) -> u32 {
        let q: [f32; 3] = std::array::from_fn(|a| (query[a] - origin[a]) as f32);
        let mut node = 0usize;
        for _ in 0..32 {
            let n = self.nodes[node];
            if n.axis == LEAF {
                break;
            }
            let left = self.nodes[node + 1];
            let right = self.nodes[n.right as usize];
            if ((left.end - left.start).min(right.end - right.start) as usize) < k {
                break;
            }
            let split = (n.split - origin[n.axis as usize]) as f32;
            node = if q[n.axis as usize] < split {
                node + 1
            } else {
                n.right as usize
            };
        }
        node as u32
    }

    fn search_k_hinted(
        &self,
        node: usize,
        q: &[f64; 3],
        k: usize,
        upper: f64,
        hints: &KnnRadiusHints,
        found: &mut Vec<(usize, f64)>,
    ) {
        let (lo, hi) = hints.bounds[node];
        let gaps: [f64; 3] = std::array::from_fn(|a| {
            if q[a] < lo[a] {
                lo[a] - q[a]
            } else if q[a] > hi[a] {
                q[a] - hi[a]
            } else {
                0.0
            }
        });
        let lower = gaps[0] * gaps[0] + gaps[1] * gaps[1] + gaps[2] * gaps[2];
        let limit = if found.len() == k {
            upper.min(found[k - 1].1)
        } else {
            upper
        };
        // Strict > retains equal-distance nodes. Original near/far order and
        // full-list split-plane comparisons below preserve historical ties.
        if lower > limit {
            return;
        }
        let n = self.nodes[node];
        if n.axis == LEAF {
            for (i, p) in self.points[n.start as usize..n.end as usize]
                .iter()
                .enumerate()
            {
                insert_knn(found, k, n.start as usize + i, distance_sq(p, q));
            }
            return;
        }
        let diff = q[n.axis as usize] - n.split;
        let (near, far) = if diff < 0.0 {
            (node + 1, n.right as usize)
        } else {
            (n.right as usize, node + 1)
        };
        self.search_k_hinted(near, q, k, upper, hints, found);
        if found.len() < k || diff * diff < found[found.len() - 1].1 {
            self.search_k_hinted(far, q, k, upper, hints, found);
        }
    }

    fn search_k<const SIMD: bool, const BOUNDS: bool>(
        &self,
        node: usize,
        q: &[f64; 3],
        k: usize,
        lower: [f64; 3],
        found: &mut Vec<(usize, f64)>,
    ) {
        let n = self.nodes[node];
        if n.axis == LEAF {
            let range = &self.points[n.start as usize..n.end as usize];
            if SIMD {
                let (pairs, tail) = range.as_chunks::<2>();
                for (i, pair) in pairs.iter().enumerate() {
                    let d = pair_distances(pair, q);
                    insert_knn(found, k, n.start as usize + i * 2, d[0]);
                    insert_knn(found, k, n.start as usize + i * 2 + 1, d[1]);
                }
                if let Some(p) = tail.first() {
                    insert_knn(found, k, n.end as usize - 1, distance_sq(p, q));
                }
            } else {
                for (i, p) in range.iter().enumerate() {
                    insert_knn(found, k, n.start as usize + i, distance_sq(p, q));
                }
            }
            return;
        }
        let diff = q[n.axis as usize] - n.split;
        let (near, far) = if diff < 0.0 {
            (node + 1, n.right as usize)
        } else {
            (n.right as usize, node + 1)
        };
        self.search_k::<SIMD, BOUNDS>(near, q, k, lower, found);
        if found.len() < k || diff * diff < found[found.len() - 1].1 {
            let mut far_lower = lower;
            if BOUNDS {
                let axis = n.axis as usize;
                far_lower[axis] = far_lower[axis].max(diff * diff);
                // Same positive-term addition order as distance_sq. Strict >
                // retains equality/NaN visits; an incomplete list never prunes.
                let bound = far_lower[0] + far_lower[1] + far_lower[2];
                if found.len() == k && bound > found[found.len() - 1].1 {
                    return;
                }
            }
            self.search_k::<SIMD, BOUNDS>(far, q, k, far_lower, found);
        }
    }

    fn search<const SIMD: bool>(&self, node: usize, q: &[f64; 3], best: &mut Nearest) {
        let n = self.nodes[node];
        if n.axis == LEAF {
            if SIMD {
                scan_nearest_leaf(&self.points, n.start, n.end, q, best);
            } else {
                scan_candidate_leaf(&self.points, n.start, n.end, q, best, false);
            }
            return;
        }
        let diff = q[n.axis as usize] - n.split;
        let (near, far) = if diff < 0.0 {
            (node + 1, n.right as usize)
        } else {
            (n.right as usize, node + 1)
        };
        self.search::<SIMD>(near, q, best);
        if diff * diff < best.distance_sq {
            self.search::<SIMD>(far, q, best);
        }
    }

    fn search_hinted_context<const SIMD: bool>(
        &self,
        node: usize,
        q: &[f64; 3],
        state: &mut BoundedNearest,
    ) {
        let n = self.nodes[node];
        if n.axis == LEAF {
            if SIMD {
                scan_nearest_leaf(&self.points, n.start, n.end, q, &mut state.best);
            } else {
                scan_candidate_leaf(&self.points, n.start, n.end, q, &mut state.best, false);
            }
            return;
        }
        let diff = q[n.axis as usize] - n.split;
        let (near, far) = if diff < 0.0 {
            (node + 1, n.right as usize)
        } else {
            (n.right as usize, node + 1)
        };
        self.search_hinted_context::<SIMD>(near, q, state);
        // Retain equality visits at the hint radius. It never becomes the
        // incumbent, so the original seed and traversal choose all ties.
        if diff * diff < state.best.distance_sq && diff * diff <= state.upper {
            self.search_hinted_context::<SIMD>(far, q, state);
        }
    }

    fn search_hinted<const SIMD: bool>(
        &self,
        node: usize,
        q: &[f64; 3],
        best: &mut Nearest,
        upper: f64,
    ) {
        let n = self.nodes[node];
        if n.axis == LEAF {
            if SIMD {
                scan_nearest_leaf(&self.points, n.start, n.end, q, best);
            } else {
                scan_candidate_leaf(&self.points, n.start, n.end, q, best, false);
            }
            return;
        }
        let diff = q[n.axis as usize] - n.split;
        let (near, far) = if diff < 0.0 {
            (node + 1, n.right as usize)
        } else {
            (n.right as usize, node + 1)
        };
        self.search_hinted::<SIMD>(near, q, best, upper);
        if diff * diff < best.distance_sq && diff * diff <= upper {
            self.search_hinted::<SIMD>(far, q, best, upper);
        }
    }
}

#[inline]
fn insert_knn(found: &mut Vec<(usize, f64)>, k: usize, slot: usize, d: f64) {
    if found.len() < k || d < found[found.len() - 1].1 {
        // Sorted insert after equal distances; a full list drops
        // its last entry. Copy each displaced hit once, then write
        // the new hit once at its final position. Comparisons and
        // insertion after equal distances keep the historical order.
        if found.len() < k {
            found.push((slot, d));
        }
        let mut j = found.len() - 1;
        while j > 0 && found[j - 1].1 > d {
            found[j] = found[j - 1];
            j -= 1;
        }
        found[j] = (slot, d);
    }
}

#[inline]
fn pair_distances(pair: &[[f64; 3]; 2], q: &[f64; 3]) -> [f64; 2] {
    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    {
        use core::arch::wasm32::*;
        let ptr = pair.as_ptr().cast::<f64>();
        // SAFETY: three unaligned loads cover exactly this pair's six f64 values.
        let (xy, zx, yz) = unsafe {
            (
                v128_load(ptr.cast()),
                v128_load(ptr.add(2).cast()),
                v128_load(ptr.add(4).cast()),
            )
        };
        let dx = f64x2_sub(i64x2_shuffle::<0, 3>(xy, zx), f64x2_splat(q[0]));
        let dy = f64x2_sub(i64x2_shuffle::<1, 2>(xy, yz), f64x2_splat(q[1]));
        let dz = f64x2_sub(i64x2_shuffle::<0, 3>(zx, yz), f64x2_splat(q[2]));
        let d = f64x2_add(
            f64x2_add(f64x2_mul(dx, dx), f64x2_mul(dy, dy)),
            f64x2_mul(dz, dz),
        );
        let result = [f64x2_extract_lane::<0>(d), f64x2_extract_lane::<1>(d)];
        // Keep scalar NaN propagation as well as neighbour ordering.
        if result.iter().all(|v| !v.is_nan()) {
            return result;
        }
    }
    [distance_sq(&pair[0], q), distance_sq(&pair[1], q)]
}

#[inline(always)]
fn distance_sq(p: &[f64; 3], q: &[f64; 3]) -> f64 {
    let (dx, dy, dz) = (p[0] - q[0], p[1] - q[1], p[2] - q[2]);
    dx * dx + dy * dy + dz * dz
}

/// Build the subtree for `items` (which starts at `offset` in the full
/// array) and return its node index.
fn build(items: &mut [([f64; 3], u32)], offset: usize, nodes: &mut Vec<Node>) -> usize {
    let id = nodes.len();
    nodes.push(Node {
        start: offset as u32,
        end: (offset + items.len()) as u32,
        split: 0.0,
        axis: LEAF,
        right: 0,
    });
    if items.len() <= LEAF_SIZE {
        return id;
    }
    let (lo, hi) = crate::distance::bounds(items.iter().map(|(p, _)| p));
    let axis = (0..3)
        .max_by(|&a, &b| (hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b])))
        .unwrap();
    if hi[axis] - lo[axis] <= 0.0 {
        return id; // all points coincide
    }
    let mid = items.len() / 2;
    items.select_nth_unstable_by(mid, |a, b| a.0[axis].total_cmp(&b.0[axis]));
    let split = items[mid].0[axis];
    let (left, right) = items.split_at_mut(mid);
    build(left, offset, nodes);
    let right_id = build(right, offset + mid, nodes);
    let node = &mut nodes[id];
    node.axis = axis as u8;
    node.split = split;
    node.right = right_id as u32;
    id
}

#[cfg(test)]
mod tests;
