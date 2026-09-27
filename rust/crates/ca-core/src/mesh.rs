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

/// A hit from [`MeshBvh::nearest`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct MeshHit {
    pub triangle: usize,
    pub point: [f64; 3],
    pub distance_sq: f64,
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
    nodes: Vec<BvhNode>,
    order: Vec<u32>,
    /// Triangle corners in `order`, so leaf tests read contiguous memory.
    corners: Vec<[[f64; 3]; 3]>,
}

const BVH_LEAF: usize = 4;

impl<'a> MeshBvh<'a> {
    /// Returns `None` for a mesh without triangles.
    pub fn new(mesh: &'a TriangleMesh) -> Option<Self> {
        if mesh.triangles.is_empty() {
            return None;
        }
        let mut items: Vec<(u32, [f64; 3])> = (0..mesh.triangles.len() as u32)
            .map(|t| {
                let [a, b, c] = mesh.corners(t as usize);
                (t, std::array::from_fn(|i| (a[i] + b[i] + c[i]) / 3.0))
            })
            .collect();
        let mut nodes = Vec::with_capacity(2 * items.len() / BVH_LEAF + 1);
        build(mesh, &mut items, 0, &mut nodes);
        let order: Vec<u32> = items.into_iter().map(|(t, _)| t).collect();
        let corners = order.iter().map(|&t| mesh.corners(t as usize)).collect();
        Some(Self {
            mesh,
            nodes,
            order,
            corners,
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
        stack.clear();
        stack.push(0);
        while let Some(index) = stack.pop() {
            let node = self.nodes[index];
            if box_distance_sq(&node, p) >= best.distance_sq {
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

    fn test(&self, t: usize, p: [f64; 3], best: &mut MeshHit) {
        let [a, b, c] = self.mesh.corners(t);
        test_triangle(t, a, b, c, p, best);
    }
}

fn test_triangle(t: usize, a: [f64; 3], b: [f64; 3], c: [f64; 3], p: [f64; 3], best: &mut MeshHit) {
    let q = closest_point_on_triangle(p, a, b, c);
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

fn build(
    mesh: &TriangleMesh,
    items: &mut [(u32, [f64; 3])],
    offset: usize,
    nodes: &mut Vec<BvhNode>,
) -> usize {
    let mut min = [f64::INFINITY; 3];
    let mut max = [f64::NEG_INFINITY; 3];
    for &(t, _) in items.iter() {
        for v in mesh.corners(t as usize) {
            for a in 0..3 {
                min[a] = min[a].min(v[a]);
                max[a] = max[a].max(v[a]);
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
    build(mesh, left, offset, nodes);
    let right_id = build(mesh, right, offset + mid, nodes);
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
mod tests {
    use super::*;

    fn pseudo_random(n: usize, seed: u64) -> Vec<[f64; 3]> {
        let mut s = seed;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..n)
            .map(|_| {
                [
                    next() * 10.0 - 5.0,
                    next() * 10.0 - 5.0,
                    next() * 10.0 - 5.0,
                ]
            })
            .collect()
    }

    /// Distance to a triangle by dense sampling of its surface.
    fn sampled_distance_sq(p: [f64; 3], a: [f64; 3], b: [f64; 3], c: [f64; 3]) -> f64 {
        let n = 200;
        let mut best = f64::INFINITY;
        for i in 0..=n {
            for j in 0..=n - i {
                let (u, v) = (i as f64 / n as f64, j as f64 / n as f64);
                let q: [f64; 3] =
                    std::array::from_fn(|k| a[k] + u * (b[k] - a[k]) + v * (c[k] - a[k]));
                let d = sub(p, q);
                best = best.min(dot(d, d));
            }
        }
        best
    }

    #[test]
    fn closest_point_covers_all_regions() {
        let (a, b, c) = ([0.0, 0.0, 0.0], [4.0, 0.0, 0.0], [0.0, 3.0, 0.0]);
        let cases = [
            ([1.0, 1.0, 2.0], [1.0, 1.0, 0.0]),  // face
            ([-1.0, -1.0, 0.0], a),              // vertex a
            ([6.0, -1.0, 1.0], b),               // vertex b
            ([-1.0, 5.0, 0.0], c),               // vertex c
            ([2.0, -2.0, 0.0], [2.0, 0.0, 0.0]), // edge ab
            ([-2.0, 1.5, 0.0], [0.0, 1.5, 0.0]), // edge ac
        ];
        for (p, expected) in cases {
            let q = closest_point_on_triangle(p, a, b, c);
            for k in 0..3 {
                assert!((q[k] - expected[k]).abs() < 1e-12, "{p:?} -> {q:?}");
            }
        }
        // Edge bc: the foot of the perpendicular from (4, 3, 0).
        let q = closest_point_on_triangle([4.0, 3.0, 0.0], a, b, c);
        let expected = [4.0 - 4.0 * 0.36, 3.0 * 0.36, 0.0]; // t = (0,3)·(-4,3) / 25
        for k in 0..3 {
            assert!((q[k] - expected[k]).abs() < 1e-12, "{q:?}");
        }
    }

    #[test]
    fn closest_point_matches_sampling() {
        let pts = pseudo_random(60, 3);
        for chunk in pts.chunks(4) {
            let [a, b, c, p] = [chunk[0], chunk[1], chunk[2], chunk[3]];
            let q = closest_point_on_triangle(p, a, b, c);
            let d = sub(p, q);
            let sampled = sampled_distance_sq(p, a, b, c);
            assert!(dot(d, d) <= sampled + 1e-9, "closer than any sample");
            assert!(
                dot(d, d).sqrt() >= sampled.sqrt() - 0.05,
                "not much farther than sampling"
            );
        }
    }

    #[test]
    fn bvh_matches_brute_force() {
        let pts = pseudo_random(900, 7);
        let mesh = TriangleMesh {
            vertices: pts.clone(),
            triangles: (0..300).map(|t| [3 * t, 3 * t + 1, 3 * t + 2]).collect(),
        };
        let bvh = MeshBvh::new(&mesh).unwrap();
        let mut guess = None;
        for q in pseudo_random(200, 11) {
            let hit = bvh.nearest(q, guess);
            guess = Some(hit.triangle);
            let brute = (0..mesh.triangles.len())
                .map(|t| {
                    let [a, b, c] = mesh.corners(t);
                    let d = sub(q, closest_point_on_triangle(q, a, b, c));
                    dot(d, d)
                })
                .fold(f64::INFINITY, f64::min);
            assert_eq!(hit.distance_sq, brute);
        }
    }

    #[test]
    fn normal_follows_winding() {
        let mesh = TriangleMesh {
            vertices: vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]],
            triangles: vec![[0, 1, 2]],
        };
        assert_eq!(mesh.normal(0), [0.0, 0.0, 1.0]);
    }
}
