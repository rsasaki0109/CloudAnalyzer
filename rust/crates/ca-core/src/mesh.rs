//! Triangle meshes and point-to-mesh distance queries.

/// An indexed triangle mesh.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TriangleMesh {
    pub vertices: Vec<[f64; 3]>,
    pub triangles: Vec<[u32; 3]>,
}

impl TriangleMesh {
    pub fn corners(&self, triangle: usize) -> [[f64; 3]; 3] {
        self.triangles[triangle].map(|v| self.vertices[v as usize])
    }

    /// Unit normal following the right-hand rule, or zero for a degenerate
    /// triangle.
    pub fn normal(&self, triangle: usize) -> [f64; 3] {
        let [a, b, c] = self.corners(triangle);
        let n = cross(sub(b, a), sub(c, a));
        let len = dot(n, n).sqrt();
        if len > 0.0 {
            n.map(|v| v / len)
        } else {
            [0.0; 3]
        }
    }

    /// Drop triangles that reference missing vertices.
    pub fn validate(&mut self) {
        let n = self.vertices.len() as u32;
        self.triangles.retain(|t| t.iter().all(|&v| v < n));
    }
}

/// Closest point to `p` on triangle `abc` (Ericson, *Real-Time Collision
/// Detection*, 5.1.5).
pub fn closest_point_on_triangle(p: [f64; 3], a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> [f64; 3] {
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = sub(p, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return add(a, scale(ab, v));
    }
    let cp = sub(p, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return add(a, scale(ac, w));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return add(b, scale(sub(c, b), w));
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    add(a, add(scale(ab, v), scale(ac, w)))
}

// Independent dot products keep their original left-to-right reductions.
fn closest_point_with_paired_dots(p: [f64; 3], a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> [f64; 3] {
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(p, a);
    let (d1, d2) = projection_dot_pair(ab, ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = sub(p, b);
    let (d3, d4) = projection_dot_pair(ab, ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return add(a, scale(ab, v));
    }
    let cp = sub(p, c);
    let (d5, d6) = projection_dot_pair(ab, ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return add(a, scale(ac, w));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return add(b, scale(sub(c, b), w));
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    add(a, add(scale(ab, v), scale(ac, w)))
}

// Keep every projection expression in the preceding evaluation order.
#[cfg(feature = "parallel")]
fn closest_point_with_triangle_seed(p: [f64; 3], seed: &MeshTriangleSeed) -> [f64; 3] {
    let [a, b, c] = seed.corners;
    let [ab, ac] = seed.edges;
    let ap = sub(p, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = sub(p, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return add(a, scale(ab, v));
    }
    let cp = sub(p, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return add(a, scale(ac, w));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return add(b, scale(sub(c, b), w));
    }
    let denom = 1.0 / (va + vb + vc);
    let v = vb * denom;
    let w = vc * denom;
    add(a, add(scale(ab, v), scale(ac, w)))
}

/// A hit from [`MeshBvh::nearest`]. If no triangle has a computable distance,
/// `triangle` is `usize::MAX` and `distance_sq` is positive infinity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshHit {
    pub triangle: usize,
    pub point: [f64; 3],
    pub distance_sq: f64,
}

/// One chunk's previous hit, using the original corner and edge bits.
#[cfg(feature = "parallel")]
#[derive(Debug, Clone, Copy)]
pub(crate) struct MeshTriangleSeed {
    triangle: usize,
    corners: [[f64; 3]; 3],
    edges: [[f64; 3]; 2],
}

#[cfg(feature = "parallel")]
impl MeshTriangleSeed {
    fn new(mesh: &TriangleMesh, triangle: usize) -> Self {
        let [a, b, c] = mesh.corners(triangle);
        Self {
            triangle,
            corners: [a, b, c],
            edges: [sub(b, a), sub(c, a)],
        }
    }
}

#[derive(Debug, Clone, Copy)]
struct BvhNode {
    min: [f64; 3],
    max: [f64; 3],
    /// Leaf: range of `order`; inner: `start` is the right child (left is next).
    start: u32,
    count: u32,
}

/// Bounding volume hierarchy over a mesh's triangles.
#[derive(Debug, Clone)]
pub struct MeshBvh<'a> {
    mesh: &'a TriangleMesh,
    storage: std::borrow::Cow<'a, BvhStorage>,
}

#[derive(Debug, Clone)]
struct BvhStorage {
    nodes: Vec<BvhNode>,
    order: Vec<u32>,
    /// Triangle corners in `order`, so leaf tests read contiguous memory.
    corners: Vec<[[f64; 3]; 3]>,
}

/// Native-only BVH whose leaf tests read the immutable original mesh.
#[cfg(feature = "parallel")]
pub(crate) struct IndexedMeshBvh<'a> {
    mesh: &'a TriangleMesh,
    storage: BvhStorage,
}

#[cfg(feature = "parallel")]
impl<'a> IndexedMeshBvh<'a> {
    pub(crate) fn new(mesh: &'a TriangleMesh) -> Option<Self> {
        Some(Self {
            mesh,
            storage: BvhStorage::new_with_corner_policy(mesh, USE_MESH_PAR_PREPARATION, false)?,
        })
    }

    pub(crate) fn nearest(
        &self,
        p: [f64; 3],
        guess: Option<usize>,
        stack: &mut Vec<(usize, f64)>,
    ) -> MeshHit {
        let mut best = MeshHit {
            triangle: usize::MAX,
            point: [0.0; 3],
            distance_sq: f64::INFINITY,
        };
        if let Some(t) = guess {
            let [a, b, c] = self.mesh.corners(t);
            test_triangle(t, a, b, c, p, &mut best);
        }
        self.storage
            .search_cached_bounds::<false, false, true, 0, false>(p, best, stack, self.mesh)
    }
}

const BVH_LEAF: usize = 4;

// Indexed collection preserves triangle order and exact vector capacities.
const USE_MESH_PAR_PREPARATION: bool = true;
// Nonzero finite extrema have identical bits regardless of min/max grouping.
const USE_MESH_BOUND_MERGE: bool = true;
#[cfg(feature = "parallel")]
const USE_MESH_PAR_BUILD: bool = true;
// Parallelize only large centroid scans, retaining exact split axes.
#[cfg(feature = "parallel")]
const USE_MESH_PAR_CENTROID_BOUNDS: bool = false;
#[cfg(feature = "parallel")]
const PAR_CENTROID_BOUNDS_MIN_ITEMS: usize = 262_144;
#[cfg(feature = "parallel")]
const PAR_CENTROID_BOUNDS_CHUNK_ITEMS: usize = 16_384;
#[cfg(feature = "parallel")]
const PAR_PREPARATION_MIN_TRIANGLES: usize = 65_536;
#[cfg(feature = "parallel")]
const PAR_BUILD_MIN_TRIANGLES: usize = 32_768;

impl<'a> MeshBvh<'a> {
    /// Returns `None` for a mesh without triangles.
    pub fn new(mesh: &'a TriangleMesh) -> Option<Self> {
        Some(Self {
            mesh,
            storage: std::borrow::Cow::Owned(BvhStorage::new(mesh)?),
        })
    }

    /// Closest point of the mesh to `p`. `guess`, typically the hit of a
    /// nearby previous query, seeds the search so most nodes are pruned.
    pub fn nearest(&self, p: [f64; 3], guess: Option<usize>) -> MeshHit {
        self.nearest_with(p, guess, &mut Vec::new())
    }

    /// [`MeshBvh::nearest`] reusing a caller-provided traversal stack.
    pub fn nearest_with(
        &self,
        p: [f64; 3],
        guess: Option<usize>,
        stack: &mut Vec<usize>,
    ) -> MeshHit {
        let mut best = MeshHit {
            triangle: usize::MAX,
            point: [0.0; 3],
            distance_sq: f64::INFINITY,
        };
        if let Some(t) = guess {
            self.test(t, p, &mut best);
        }
        self.storage.search(p, best, stack)
    }

    fn test(&self, t: usize, p: [f64; 3], best: &mut MeshHit) {
        let [a, b, c] = self.mesh.corners(t);
        test_triangle(t, a, b, c, p, best);
    }

    /// Reuse child bounds computed for ordering instead of recomputing on pop.
    pub(crate) fn nearest_with_cached_bounds(
        &self,
        p: [f64; 3],
        guess: Option<usize>,
        stack: &mut Vec<(usize, f64)>,
    ) -> MeshHit {
        self.nearest_with_cached_bounds_impl::<false, false, 0, false>(p, guess, stack)
    }

    /// The seeded triangle has already been tested with the same f64 corners.
    #[cfg(feature = "parallel")]
    pub(crate) fn nearest_with_cached_bounds_unique(
        &self,
        p: [f64; 3],
        guess: Option<usize>,
        stack: &mut Vec<(usize, f64)>,
    ) -> MeshHit {
        self.nearest_with_cached_bounds_impl::<true, false, 0, false>(p, guess, stack)
    }

    /// Reject already-pruned children before storing them, preserving NaNs.
    #[cfg(feature = "parallel")]
    pub(crate) fn nearest_with_cached_bounds_pruned(
        &self,
        p: [f64; 3],
        guess: Option<usize>,
        stack: &mut Vec<(usize, f64)>,
    ) -> MeshHit {
        self.nearest_with_cached_bounds_impl::<false, true, 0, false>(p, guess, stack)
    }

    /// Pair sibling bounds without changing each lane's axis reduction order.
    #[cfg(feature = "parallel")]
    pub(crate) fn nearest_with_cached_bounds_paired(
        &self,
        p: [f64; 3],
        guess: Option<usize>,
        stack: &mut Vec<(usize, f64)>,
    ) -> MeshHit {
        self.nearest_with_cached_bounds_impl::<false, false, 1, false>(p, guess, stack)
    }

    /// Fold NaN handling into each nonnegative difference before pairing.
    #[cfg(feature = "parallel")]
    pub(crate) fn nearest_with_cached_bounds_clamped(
        &self,
        p: [f64; 3],
        guess: Option<usize>,
        stack: &mut Vec<(usize, f64)>,
    ) -> MeshHit {
        self.nearest_with_cached_bounds_impl::<false, false, 2, false>(p, guess, stack)
    }

    /// Pair independent projection dots while keeping adopted sibling bounds.
    #[cfg(feature = "parallel")]
    pub(crate) fn nearest_with_paired_projection(
        &self,
        p: [f64; 3],
        guess: Option<usize>,
        stack: &mut Vec<(usize, f64)>,
    ) -> MeshHit {
        self.nearest_with_cached_bounds_impl::<false, false, 2, true>(p, guess, stack)
    }

    fn nearest_with_cached_bounds_impl<
        const SKIP_GUESS: bool,
        const EAGER_PRUNE: bool,
        const BOUND_PAIR: u8,
        const PAIRED_PROJECTION: bool,
    >(
        &self,
        p: [f64; 3],
        guess: Option<usize>,
        stack: &mut Vec<(usize, f64)>,
    ) -> MeshHit {
        let mut best = MeshHit {
            triangle: usize::MAX,
            point: [0.0; 3],
            distance_sq: f64::INFINITY,
        };
        if let Some(t) = guess {
            if PAIRED_PROJECTION {
                let [a, b, c] = self.mesh.corners(t);
                test_triangle_projected::<true>(t, a, b, c, p, &mut best);
            } else {
                self.test(t, p, &mut best);
            }
        }
        self.storage
            .search_cached_bounds::<SKIP_GUESS, EAGER_PRUNE, false, BOUND_PAIR, PAIRED_PROJECTION>(
                p, best, stack, self.mesh,
            )
    }

    /// Retain only the previous hit's corners/edges within a query chunk.
    #[cfg(feature = "parallel")]
    pub(crate) fn nearest_with_triangle_seed(
        &self,
        p: [f64; 3],
        seed: &mut Option<MeshTriangleSeed>,
        stack: &mut Vec<(usize, f64)>,
    ) -> MeshHit {
        let mut best = MeshHit {
            triangle: usize::MAX,
            point: [0.0; 3],
            distance_sq: f64::INFINITY,
        };
        if let Some(cached) = seed.as_ref() {
            let q = closest_point_with_triangle_seed(p, cached);
            let d = sub(p, q);
            let d2 = dot(d, d);
            if d2 < best.distance_sq {
                best = MeshHit {
                    triangle: cached.triangle,
                    point: q,
                    distance_sq: d2,
                };
            }
        }
        let hit = self
            .storage
            .search_cached_bounds::<false, false, false, 0, false>(p, best, stack, self.mesh);
        if hit.triangle == usize::MAX {
            *seed = None;
        } else if seed.as_ref().map(|s| s.triangle) != Some(hit.triangle) {
            *seed = Some(MeshTriangleSeed::new(self.mesh, hit.triangle));
        }
        hit
    }
}

impl BvhStorage {
    fn new(mesh: &TriangleMesh) -> Option<Self> {
        Self::new_with_parallel(mesh, USE_MESH_PAR_PREPARATION)
    }

    fn new_with_parallel(mesh: &TriangleMesh, _parallel: bool) -> Option<Self> {
        Self::new_with_corner_policy(mesh, _parallel, true)
    }

    fn new_with_corner_policy(
        mesh: &TriangleMesh,
        _parallel: bool,
        copy_corners: bool,
    ) -> Option<Self> {
        if mesh.triangles.is_empty() {
            return None;
        }
        #[cfg(feature = "parallel")]
        use rayon::prelude::*;
        #[cfg(feature = "parallel")]
        let parallel = _parallel && mesh.triangles.len() >= PAR_PREPARATION_MIN_TRIANGLES;
        let centroid = |t: u32| {
            let [a, b, c] = mesh.corners(t as usize);
            (t, std::array::from_fn(|i| (a[i] + b[i] + c[i]) / 3.0))
        };
        let serial_items = || (0..mesh.triangles.len() as u32).map(centroid).collect();
        #[cfg(feature = "parallel")]
        let mut items: Vec<(u32, [f64; 3])> = if parallel {
            (0..mesh.triangles.len() as u32)
                .into_par_iter()
                .map(centroid)
                .collect()
        } else {
            serial_items()
        };
        #[cfg(not(feature = "parallel"))]
        let mut items: Vec<(u32, [f64; 3])> = serial_items();
        let mut nodes = Vec::with_capacity(2 * items.len() / BVH_LEAF + 1);
        let merge = USE_MESH_BOUND_MERGE && can_merge_bounds(mesh);
        #[cfg(feature = "parallel")]
        if parallel && USE_MESH_PAR_BUILD {
            nodes.resize(bvh_node_count(items.len()), empty_node());
            build_parallel(
                mesh,
                &mut items,
                0,
                0,
                &mut nodes,
                merge,
                USE_MESH_PAR_CENTROID_BOUNDS,
            );
        } else {
            build(mesh, &mut items, 0, &mut nodes, merge);
        }
        #[cfg(not(feature = "parallel"))]
        build(mesh, &mut items, 0, &mut nodes, merge);
        // Allocate exact u32 capacity instead of retaining the much wider
        // temporary centroid allocation through in-place iterator collection.
        let order: Vec<u32> = items.iter().map(|&(t, _)| t).collect();
        drop(items);
        let corner = |&t: &u32| mesh.corners(t as usize);
        #[cfg(feature = "parallel")]
        let corners = if !copy_corners {
            Vec::new()
        } else if parallel {
            order.par_iter().map(corner).collect()
        } else {
            order.iter().map(corner).collect()
        };
        #[cfg(not(feature = "parallel"))]
        let corners = if copy_corners {
            order.iter().map(corner).collect()
        } else {
            Vec::new()
        };
        Some(Self {
            nodes,
            order,
            corners,
        })
    }

    fn search(&self, p: [f64; 3], best: MeshHit, stack: &mut Vec<usize>) -> MeshHit {
        self.search_limited::<false>(p, best, f64::INFINITY, stack)
    }

    fn search_limited<const HINT: bool>(
        &self,
        p: [f64; 3],
        mut best: MeshHit,
        upper: f64,
        stack: &mut Vec<usize>,
    ) -> MeshHit {
        stack.clear();
        stack.push(0);
        while let Some(index) = stack.pop() {
            let node = self.nodes[index];
            let lower = box_distance_sq(&node, p);
            if lower >= best.distance_sq || HINT && lower > upper {
                continue;
            }
            if node.count > 0 {
                for slot in node.start as usize..(node.start + node.count) as usize {
                    let [a, b, c] = self.corners[slot];
                    test_triangle(self.order[slot] as usize, a, b, c, p, &mut best);
                }
                continue;
            }
            // Visit the nearer child first (pushed last).
            let (left, right) = (index + 1, node.start as usize);
            let (dl, dr) = (
                box_distance_sq(&self.nodes[left], p),
                box_distance_sq(&self.nodes[right], p),
            );
            if dl <= dr {
                stack.extend([right, left]);
            } else {
                stack.extend([left, right]);
            }
        }
        best
    }

    fn storage_bytes(&self) -> usize {
        self.nodes.capacity() * std::mem::size_of::<BvhNode>()
            + self.order.capacity() * std::mem::size_of::<u32>()
            + self.corners.capacity() * std::mem::size_of::<[[f64; 3]; 3]>()
    }

    fn search_cached_bounds<
        const SKIP_GUESS: bool,
        const EAGER_PRUNE: bool,
        const INDEXED_CORNERS: bool,
        const BOUND_PAIR: u8,
        const PAIRED_PROJECTION: bool,
    >(
        &self,
        p: [f64; 3],
        mut best: MeshHit,
        stack: &mut Vec<(usize, f64)>,
        mesh: &TriangleMesh,
    ) -> MeshHit {
        let seeded_triangle = best.triangle;
        stack.clear();
        stack.push((0, box_distance_sq(&self.nodes[0], p)));
        while let Some((index, lower)) = stack.pop() {
            if lower >= best.distance_sq {
                continue;
            }
            let node = self.nodes[index];
            if node.count > 0 {
                for slot in node.start as usize..(node.start + node.count) as usize {
                    let t = self.order[slot] as usize;
                    if SKIP_GUESS && t == seeded_triangle {
                        continue;
                    }
                    let [a, b, c] = if INDEXED_CORNERS {
                        mesh.corners(t)
                    } else {
                        self.corners[slot]
                    };
                    test_triangle_projected::<PAIRED_PROJECTION>(t, a, b, c, p, &mut best);
                }
                continue;
            }
            let (left, right) = (index + 1, node.start as usize);
            let (dl, dr) = if BOUND_PAIR == 2 {
                box_distance_sq_pair_clamped(&self.nodes[left], &self.nodes[right], p)
            } else if BOUND_PAIR == 1 {
                box_distance_sq_pair(&self.nodes[left], &self.nodes[right], p)
            } else {
                (
                    box_distance_sq(&self.nodes[left], p),
                    box_distance_sq(&self.nodes[right], p),
                )
            };
            let children = if dl <= dr {
                [(right, dr), (left, dl)]
            } else {
                [(left, dl), (right, dr)]
            };
            if EAGER_PRUNE {
                for child in children {
                    // Best distances only decrease. A NaN lower bound keeps
                    // the preceding comparison's false result and is queued.
                    if child.1 >= best.distance_sq {
                        continue;
                    }
                    stack.push(child);
                }
            } else {
                stack.extend(children);
            }
        }
        best
    }
}

/// Owned immutable mesh/BVH, independent of the original source's lifetime.
pub struct PreparedMesh {
    mesh: TriangleMesh,
    storage: BvhStorage,
    hint_finite: std::sync::OnceLock<bool>,
}

impl PreparedMesh {
    pub fn new(mut mesh: TriangleMesh) -> Option<Self> {
        mesh.validate();
        let storage = BvhStorage::new(&mesh)?;
        Some(Self {
            mesh,
            storage,
            hint_finite: std::sync::OnceLock::new(),
        })
    }

    pub fn retained_bytes(&self) -> usize {
        self.mesh.vertices.capacity() * std::mem::size_of::<[f64; 3]>()
            + self.mesh.triangles.capacity() * std::mem::size_of::<[u32; 3]>()
            + self.storage.storage_bytes()
    }

    pub fn distances(&self, points: &[[f64; 3]], signed: bool) -> Vec<f64> {
        let bvh = MeshBvh {
            mesh: &self.mesh,
            storage: std::borrow::Cow::Borrowed(&self.storage),
        };
        crate::distance::distances_with_mesh_bvh(points, &self.mesh, &bvh, signed)
    }

    /// Parallel C2M using the retained immutable BVH (feature `parallel`).
    /// Query order, chunk-local guesses and scratch are rebuilt for every call;
    /// the geometry/BVH may be shared by concurrent readers without rebuilding.
    /// Uses the same copied-corner query policy as native `cloud_to_mesh_par`.
    #[cfg(feature = "parallel")]
    pub fn distances_par(&self, points: &[[f64; 3]], signed: bool) -> Vec<f64> {
        let bvh = MeshBvh {
            mesh: &self.mesh,
            storage: std::borrow::Cow::Borrowed(&self.storage),
        };
        crate::distance::distances_with_mesh_bvh_par(points, &self.mesh, &bvh, signed)
    }

    fn distances_ordered(&self, points: &[[f64; 3]], order: &[usize], signed: bool) -> Vec<f64> {
        let bvh = MeshBvh {
            mesh: &self.mesh,
            storage: std::borrow::Cow::Borrowed(&self.storage),
        };
        crate::distance::distances_with_mesh_bvh_ordered(points, &self.mesh, &bvh, order, signed)
    }

    /// Rounded, shifted geometry is only for untrusted GPU candidate selection.
    /// Nodes are min/max f32 bits, start/right, count; triangles are three vec4s.
    pub fn gpu_buffers(&mut self) -> ([f64; 3], Vec<u32>, Vec<f32>) {
        let origin = self.mesh.vertices[0];
        self.hint_finite
            .get_or_init(|| self.mesh.vertices.iter().flatten().all(|v| v.is_finite()));
        let nodes = self
            .storage
            .nodes
            .iter()
            .flat_map(|n| {
                [
                    ((n.min[0] - origin[0]) as f32).to_bits(),
                    ((n.min[1] - origin[1]) as f32).to_bits(),
                    ((n.min[2] - origin[2]) as f32).to_bits(),
                    ((n.max[0] - origin[0]) as f32).to_bits(),
                    ((n.max[1] - origin[1]) as f32).to_bits(),
                    ((n.max[2] - origin[2]) as f32).to_bits(),
                    n.start,
                    n.count,
                ]
            })
            .collect();
        let triangles = self
            .storage
            .corners
            .iter()
            .flat_map(|t| {
                t.iter().flat_map(|p| {
                    [
                        (p[0] - origin[0]) as f32,
                        (p[1] - origin[1]) as f32,
                        (p[2] - origin[2]) as f32,
                        0.0,
                    ]
                })
            })
            .collect();
        (origin, nodes, triangles)
    }

    /// Candidates only bound traversal; original guesses, visitation and strict
    /// triangle insertion still select every final hit, including signed ties.
    pub fn distances_hinted(
        &self,
        points: &[[f64; 3]],
        signed: bool,
        candidates: &[u32],
    ) -> Option<Vec<f64>> {
        if candidates.len() != points.len() {
            return None;
        }
        let bvh = MeshBvh {
            mesh: &self.mesh,
            storage: std::borrow::Cow::Borrowed(&self.storage),
        };
        let mut out = vec![0.0; points.len()];
        let finite = *self
            .hint_finite
            .get_or_init(|| self.mesh.vertices.iter().flatten().all(|v| v.is_finite()));
        let mut guess = None;
        let mut stack = Vec::new();
        for i in crate::distance::morton_order(points) {
            let p = points[i];
            let upper = self
                .storage
                .corners
                .get(candidates[i] as usize)
                .and_then(|&[a, b, c]| {
                    if !finite || !p.iter().all(|v| v.is_finite()) {
                        return None;
                    }
                    let delta = sub(p, closest_point_on_triangle(p, a, b, c));
                    let d = dot(delta, delta);
                    d.is_finite().then_some(d)
                });
            let hit = if let Some(upper) = upper {
                let mut seed = MeshHit {
                    triangle: usize::MAX,
                    point: [0.0; 3],
                    distance_sq: f64::INFINITY,
                };
                if let Some(t) = guess {
                    bvh.test(t, p, &mut seed);
                }
                let hit = self
                    .storage
                    .search_limited::<true>(p, seed, upper, &mut stack);
                // A rounded/projection edge case may exclude even its candidate.
                // Never publish an incomplete bound search.
                if hit.distance_sq > upper {
                    bvh.nearest_with(p, guess, &mut stack)
                } else {
                    hit
                }
            } else {
                bvh.nearest_with(p, guess, &mut stack)
            };
            if hit.triangle == usize::MAX {
                guess = None;
                out[i] = f64::INFINITY;
                continue;
            }
            guess = Some(hit.triangle);
            let mut d = hit.distance_sq.sqrt();
            if signed && dot(sub(p, hit.point), self.mesh.normal(hit.triangle)) < 0.0 {
                d = -d;
            }
            out[i] = d;
        }
        Some(out)
    }
}

struct QueryOrder {
    points: Box<[[f64; 3]]>,
    order: Box<[usize]>,
}

impl QueryOrder {
    fn matches(&self, points: &[[f64; 3]]) -> bool {
        self.points.len() == points.len()
            && self
                .points
                .iter()
                .zip(points)
                .all(|(a, b)| a.map(f64::to_bits) == b.map(f64::to_bits))
    }

    fn bytes(&self) -> usize {
        self.points.len() * std::mem::size_of::<[f64; 3]>()
            + self.order.len() * std::mem::size_of::<usize>()
    }
}

/// One worker's bounded content-matched query orders, independent of mesh identity.
pub struct MeshQueryOrder {
    entries: Vec<QueryOrder>,
    max_bytes: usize,
    hit: bool,
}

impl MeshQueryOrder {
    pub fn new(max_bytes: usize) -> Self {
        Self {
            entries: Vec::new(),
            max_bytes: max_bytes.min(8 * 1024 * 1024),
            hit: false,
        }
    }

    pub fn clear(&mut self) {
        self.entries.clear();
        self.hit = false;
    }

    pub fn hit(&self) -> bool {
        self.hit
    }

    pub fn entry_count(&self) -> usize {
        self.entries.len()
    }

    /// Exact boxed payload; at most eight entry headers are additional.
    pub fn retained_bytes(&self) -> usize {
        self.entries.iter().map(QueryOrder::bytes).sum()
    }

    pub fn distances(
        &mut self,
        mesh: &PreparedMesh,
        points: Vec<[f64; 3]>,
        signed: bool,
    ) -> Vec<f64> {
        self.hit = false;
        if let Some(slot) = self.entries.iter().position(|entry| entry.matches(&points)) {
            self.hit = true;
            let entry = self.entries.remove(slot);
            let distances = mesh.distances_ordered(&points, &entry.order, signed);
            self.entries.push(entry);
            return distances;
        }
        let needed = points
            .len()
            .checked_mul(std::mem::size_of::<[f64; 3]>() + std::mem::size_of::<usize>());
        let Some(needed) = needed.filter(|&bytes| bytes > 0 && bytes <= self.max_bytes) else {
            self.clear();
            return mesh.distances(&points, signed);
        };
        // Release old payload before forming a replacement. Each export owns
        // its incoming points already, just like the uncached WASM calculation.
        while self.entries.len() >= 8 || self.retained_bytes() > self.max_bytes - needed {
            self.entries.remove(0);
        }
        let order = crate::distance::morton_order(&points).into_boxed_slice();
        let entry = QueryOrder {
            points: points.into_boxed_slice(),
            order,
        };
        let distances = mesh.distances_ordered(&entry.points, &entry.order, signed);
        self.entries.push(entry);
        distances
    }
}

fn test_triangle(t: usize, a: [f64; 3], b: [f64; 3], c: [f64; 3], p: [f64; 3], best: &mut MeshHit) {
    test_triangle_projected::<false>(t, a, b, c, p, best);
}

fn test_triangle_projected<const PAIRED: bool>(
    t: usize,
    a: [f64; 3],
    b: [f64; 3],
    c: [f64; 3],
    p: [f64; 3],
    best: &mut MeshHit,
) {
    let q = if PAIRED {
        closest_point_with_paired_dots(p, a, b, c)
    } else {
        closest_point_on_triangle(p, a, b, c)
    };
    let d = sub(p, q);
    let d2 = dot(d, d);
    if d2 < best.distance_sq {
        *best = MeshHit {
            triangle: t,
            point: q,
            distance_sq: d2,
        };
    }
}

fn can_merge_bounds(mesh: &TriangleMesh) -> bool {
    mesh.vertices
        .iter()
        .flatten()
        .all(|v| v.is_finite() && *v != 0.0)
}

#[cfg(feature = "parallel")]
fn empty_node() -> BvhNode {
    BvhNode {
        min: [f64::INFINITY; 3],
        max: [f64::NEG_INFINITY; 3],
        start: 0,
        count: 0,
    }
}

/// Midpoint splits depend only on length, so preorder ranges are known before
/// partitioning. At the last complete level, sizes are between 4 and 7;
/// subtrees longer than 4 each add two leaves.
#[cfg(feature = "parallel")]
fn bvh_node_count(n: usize) -> usize {
    if n <= BVH_LEAF {
        return 1;
    }
    let width = 1usize << (usize::BITS - 1 - (n / BVH_LEAF).leading_zeros());
    let extra = if n / width == BVH_LEAF {
        n % width
    } else {
        width
    };
    2 * (width + extra) - 1
}

/// Original serial extrema scan, including target-dependent zero/NaN ties.
#[cfg(feature = "parallel")]
fn centroid_bounds_serial(items: &[(u32, [f64; 3])]) -> ([f64; 3], [f64; 3]) {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for (_, c) in items {
        for a in 0..3 {
            lo[a] = lo[a].min(c[a]);
            hi[a] = hi[a].max(c[a]);
        }
    }
    (lo, hi)
}

#[cfg(feature = "parallel")]
fn centroid_bounds_parallel(items: &[(u32, [f64; 3])]) -> ([f64; 3], [f64; 3]) {
    use rayon::prelude::*;
    let (lo, hi, exact) = items
        .par_chunks(PAR_CENTROID_BOUNDS_CHUNK_ITEMS)
        .map(|chunk| {
            let mut lo = [f64::INFINITY; 3];
            let mut hi = [f64::NEG_INFINITY; 3];
            let mut exact = true;
            for (_, c) in chunk {
                for a in 0..3 {
                    exact &= c[a].is_finite() && c[a] != 0.0;
                    lo[a] = lo[a].min(c[a]);
                    hi[a] = hi[a].max(c[a]);
                }
            }
            (lo, hi, exact)
        })
        .reduce(
            || ([f64::INFINITY; 3], [f64::NEG_INFINITY; 3], true),
            |(a, b, exact_a), (c, d, exact_b)| {
                (
                    std::array::from_fn(|i| a[i].min(c[i])),
                    std::array::from_fn(|i| b[i].max(d[i])),
                    exact_a && exact_b,
                )
            },
        );
    if exact {
        (lo, hi)
    } else {
        // Preserve original signed-zero/NaN tie bits before choosing the axis.
        centroid_bounds_serial(items)
    }
}

/// Build in disjoint preorder ranges; each unstable partition stays serial.
#[cfg(feature = "parallel")]
fn build_parallel(
    mesh: &TriangleMesh,
    items: &mut [(u32, [f64; 3])],
    offset: usize,
    node_offset: usize,
    nodes: &mut [BvhNode],
    merge: bool,
    parallel_bounds: bool,
) {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    if !merge || items.len() <= BVH_LEAF {
        for &(t, _) in items.iter() {
            for v in mesh.corners(t as usize) {
                for a in 0..3 {
                    min[a] = min[a].min(v[a]);
                    max[a] = max[a].max(v[a]);
                }
            }
        }
    }
    nodes[0] = BvhNode {
        min,
        max,
        start: offset as u32,
        count: items.len() as u32,
    };
    if items.len() <= BVH_LEAF {
        return;
    }
    let (lo, hi) = if parallel_bounds && items.len() >= PAR_CENTROID_BOUNDS_MIN_ITEMS {
        centroid_bounds_parallel(items)
    } else {
        centroid_bounds_serial(items)
    };
    let axis = (0..3)
        .max_by(|&a, &b| (hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b])))
        .unwrap();
    let mid = items.len() / 2;
    let parallel = items.len() >= PAR_BUILD_MIN_TRIANGLES;
    items.select_nth_unstable_by(mid, |x, y| x.1[axis].total_cmp(&y.1[axis]));
    let (left, right) = items.split_at_mut(mid);
    let left_count = bvh_node_count(left.len());
    let right_id = node_offset + 1 + left_count;
    let (parent, children) = nodes.split_at_mut(1);
    let (left_nodes, right_nodes) = children.split_at_mut(left_count);
    if parallel {
        rayon::join(
            || {
                build_parallel(
                    mesh,
                    left,
                    offset,
                    node_offset + 1,
                    left_nodes,
                    merge,
                    parallel_bounds,
                )
            },
            || {
                build_parallel(
                    mesh,
                    right,
                    offset + mid,
                    right_id,
                    right_nodes,
                    merge,
                    parallel_bounds,
                )
            },
        );
    } else {
        build_parallel(
            mesh,
            left,
            offset,
            node_offset + 1,
            left_nodes,
            merge,
            parallel_bounds,
        );
        build_parallel(
            mesh,
            right,
            offset + mid,
            right_id,
            right_nodes,
            merge,
            parallel_bounds,
        );
    }
    if merge {
        for a in 0..3 {
            parent[0].min[a] = left_nodes[0].min[a].min(right_nodes[0].min[a]);
            parent[0].max[a] = left_nodes[0].max[a].max(right_nodes[0].max[a]);
        }
    }
    parent[0].start = right_id as u32;
    parent[0].count = 0;
}

fn build(
    mesh: &TriangleMesh,
    items: &mut [(u32, [f64; 3])],
    offset: usize,
    nodes: &mut Vec<BvhNode>,
    merge: bool,
) -> usize {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    if !merge || items.len() <= BVH_LEAF {
        for &(t, _) in items.iter() {
            for v in mesh.corners(t as usize) {
                for a in 0..3 {
                    min[a] = min[a].min(v[a]);
                    max[a] = max[a].max(v[a]);
                }
            }
        }
    }
    let id = nodes.len();
    nodes.push(BvhNode {
        min,
        max,
        start: offset as u32,
        count: items.len() as u32,
    });
    if items.len() <= BVH_LEAF {
        return id;
    }
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for (_, c) in items.iter() {
        for a in 0..3 {
            lo[a] = lo[a].min(c[a]);
            hi[a] = hi[a].max(c[a]);
        }
    }
    let axis = (0..3)
        .max_by(|&a, &b| (hi[a] - lo[a]).total_cmp(&(hi[b] - lo[b])))
        .unwrap();
    let mid = items.len() / 2;
    items.select_nth_unstable_by(mid, |x, y| x.1[axis].total_cmp(&y.1[axis]));
    let (left, right) = items.split_at_mut(mid);
    let left_id = build(mesh, left, offset, nodes, merge);
    let right_id = build(mesh, right, offset + mid, nodes, merge);
    if merge {
        for a in 0..3 {
            nodes[id].min[a] = nodes[left_id].min[a].min(nodes[right_id].min[a]);
            nodes[id].max[a] = nodes[left_id].max[a].max(nodes[right_id].max[a]);
        }
    }
    nodes[id].start = right_id as u32;
    nodes[id].count = 0;
    id
}

fn box_distance_sq(node: &BvhNode, p: [f64; 3]) -> f64 {
    (0..3)
        .map(|a| {
            let d = (node.min[a] - p[a]).max(p[a] - node.max[a]).max(0.0);
            d * d
        })
        .sum()
}

#[inline]
fn box_distance_sq_pair(left: &BvhNode, right: &BvhNode, p: [f64; 3]) -> (f64, f64) {
    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    {
        use std::arch::x86_64::*;
        // SAFETY: SSE2 is part of the x86_64 baseline. These intrinsics operate
        // only on register values; no pointers or unchecked memory accesses.
        unsafe {
            let zero = _mm_setzero_pd();
            let mut sum = zero;
            for (axis, point) in p.into_iter().enumerate() {
                let point = _mm_set1_pd(point);
                let below = _mm_sub_pd(_mm_set_pd(right.min[axis], left.min[axis]), point);
                let above = _mm_sub_pd(point, _mm_set_pd(right.max[axis], left.max[axis]));
                // MAXPD selects the second input when it is NaN. f64::max
                // keeps the numeric input instead. Select `below` in those
                // lanes, then max with zero also handles two NaNs. Any choice
                // of signed zero is erased by the following square and sum.
                let nan = _mm_cmpunord_pd(above, above);
                let distance = _mm_or_pd(
                    _mm_and_pd(nan, below),
                    _mm_andnot_pd(nan, _mm_max_pd(below, above)),
                );
                let distance = _mm_max_pd(distance, zero);
                // Retain 0 + dx*dx + dy*dy + dz*dz separately in each lane.
                sum = _mm_add_pd(sum, _mm_mul_pd(distance, distance));
            }
            (_mm_cvtsd_f64(sum), _mm_cvtsd_f64(_mm_unpackhi_pd(sum, sum)))
        }
    }
    #[cfg(not(all(feature = "parallel", target_arch = "x86_64")))]
    {
        (box_distance_sq(left, p), box_distance_sq(right, p))
    }
}

#[inline]
fn box_distance_sq_pair_clamped(left: &BvhNode, right: &BvhNode, p: [f64; 3]) -> (f64, f64) {
    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    {
        use std::arch::x86_64::*;
        // SAFETY: x86_64 guarantees SSE2. All operands are register values;
        // there are no pointers or unchecked memory accesses.
        unsafe {
            let zero = _mm_setzero_pd();
            let mut sum = zero;
            for (axis, point) in p.into_iter().enumerate() {
                let point = _mm_set1_pd(point);
                let below = _mm_sub_pd(_mm_set_pd(right.min[axis], left.min[axis]), point);
                let above = _mm_sub_pd(point, _mm_set_pd(right.max[axis], left.max[axis]));
                // MAXPD(x, 0) maps NaN/negative/zero to +0. Pairing these
                // numeric nonnegative values gives maxnum(below, above, 0),
                // up to zero signs erased by the square and initial +0 sum.
                let distance = _mm_max_pd(_mm_max_pd(below, zero), _mm_max_pd(above, zero));
                sum = _mm_add_pd(sum, _mm_mul_pd(distance, distance));
            }
            (_mm_cvtsd_f64(sum), _mm_cvtsd_f64(_mm_unpackhi_pd(sum, sum)))
        }
    }
    #[cfg(not(all(feature = "parallel", target_arch = "x86_64")))]
    {
        (box_distance_sq(left, p), box_distance_sq(right, p))
    }
}

#[inline]
fn projection_dot_pair(a: [f64; 3], b: [f64; 3], v: [f64; 3]) -> (f64, f64) {
    #[cfg(all(feature = "parallel", target_arch = "x86_64"))]
    {
        use std::arch::x86_64::*;
        // SAFETY: SSE2 is guaranteed on x86_64. All intrinsics operate only
        // on register operands, without pointers or unchecked memory access.
        unsafe {
            let x = _mm_mul_pd(_mm_set_pd(b[0], a[0]), _mm_set1_pd(v[0]));
            let y = _mm_mul_pd(_mm_set_pd(b[1], a[1]), _mm_set1_pd(v[1]));
            let z = _mm_mul_pd(_mm_set_pd(b[2], a[2]), _mm_set1_pd(v[2]));
            let sum = _mm_add_pd(_mm_add_pd(x, y), z);
            (_mm_cvtsd_f64(sum), _mm_cvtsd_f64(_mm_unpackhi_pd(sum, sum)))
        }
    }
    #[cfg(not(all(feature = "parallel", target_arch = "x86_64")))]
    {
        (dot(a, v), dot(b, v))
    }
}

fn sub(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn add(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [a[0] + b[0], a[1] + b[1], a[2] + b[2]]
}

fn scale(a: [f64; 3], s: f64) -> [f64; 3] {
    [a[0] * s, a[1] * s, a[2] * s]
}

pub(crate) fn dot(a: [f64; 3], b: [f64; 3]) -> f64 {
    a[0] * b[0] + a[1] * b[1] + a[2] * b[2]
}

fn cross(a: [f64; 3], b: [f64; 3]) -> [f64; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

#[cfg(test)]
// Explicit, because examples/profile_c2m_stages.rs includes this file with
// #[path], and an included file would look for src/tests.rs instead.
#[path = "mesh/tests.rs"]
mod tests;
