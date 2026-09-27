//! A static 3D k-d tree specialised for nearest-neighbour queries.
//!
//! Points are copied into tree order so leaves are contiguous in memory.
//! Construction splits at the median of the widest axis, so it is
//! `O(n log n)` regardless of input order or duplicate coordinates.

const LEAF_SIZE: usize = 16;

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

/// A nearest-neighbour hit.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Nearest {
    /// Index into the slice the tree was built from.
    pub index: usize,
    pub distance_sq: f64,
    /// Position of the hit in tree order, for use as a later `guess`.
    slot: u32,
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

    pub fn len(&self) -> usize {
        self.points.len()
    }

    pub fn is_empty(&self) -> bool {
        self.points.is_empty()
    }

    /// Nearest point to `query`. `guess`, typically the hit of a spatially
    /// close previous query, seeds the search so most branches are pruned.
    pub fn nearest(&self, query: &[f64; 3], guess: Option<Nearest>) -> Nearest {
        let slot = guess.map_or(0, |g| g.slot);
        let mut best = Nearest {
            index: 0,
            distance_sq: distance_sq(&self.points[slot as usize], query),
            slot,
        };
        self.search(0, query, &mut best);
        best.index = self.indices[best.slot as usize] as usize;
        best
    }

    /// The `k` nearest points to `query` as `(index, squared distance)`,
    /// closest first. Intended for small `k` (e.g. normal estimation).
    pub fn nearest_k(&self, query: &[f64; 3], k: usize) -> Vec<(usize, f64)> {
        let mut found: Vec<(u32, f64)> = Vec::with_capacity(k + 1);
        if k > 0 {
            self.search_k(0, query, k, &mut found);
        }
        found
            .into_iter()
            .map(|(slot, d)| (self.indices[slot as usize] as usize, d))
            .collect()
    }

    fn search_k(&self, node: usize, q: &[f64; 3], k: usize, found: &mut Vec<(u32, f64)>) {
        let n = self.nodes[node];
        if n.axis == LEAF {
            for i in n.start as usize..n.end as usize {
                let d = distance_sq(&self.points[i], q);
                if found.len() < k || d < found[found.len() - 1].1 {
                    // Sorted insert; `found` stays at most `k` long.
                    let at = found.partition_point(|&(_, e)| e <= d);
                    found.insert(at, (i as u32, d));
                    found.truncate(k);
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
        self.search_k(near, q, k, found);
        if found.len() < k || diff * diff < found[found.len() - 1].1 {
            self.search_k(far, q, k, found);
        }
    }

    fn search(&self, node: usize, q: &[f64; 3], best: &mut Nearest) {
        let n = self.nodes[node];
        if n.axis == LEAF {
            for i in n.start as usize..n.end as usize {
                let d = distance_sq(&self.points[i], q);
                if d < best.distance_sq {
                    best.distance_sq = d;
                    best.slot = i as u32;
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
        self.search(near, q, best);
        if diff * diff < best.distance_sq {
            self.search(far, q, best);
        }
    }
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
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for (p, _) in items.iter() {
        for a in 0..3 {
            lo[a] = lo[a].min(p[a]);
            hi[a] = hi[a].max(p[a]);
        }
    }
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
mod tests {
    use super::*;

    fn brute(points: &[[f64; 3]], q: &[f64; 3]) -> f64 {
        points
            .iter()
            .map(|p| (0..3).map(|a| (p[a] - q[a]).powi(2)).sum::<f64>())
            .fold(f64::INFINITY, f64::min)
    }

    fn pseudo_random(n: usize, seed: u64) -> Vec<[f64; 3]> {
        let mut s = seed;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        (0..n)
            .map(|_| [next() * 10.0, next() * 10.0, (next() * 4.0).floor()])
            .collect()
    }

    #[test]
    fn matches_brute_force() {
        let points = pseudo_random(2000, 7);
        let tree = KdTree::new(&points).unwrap();
        for q in pseudo_random(300, 11) {
            let hit = tree.nearest(&q, None);
            assert_eq!(hit.distance_sq, brute(&points, &q));
            let p = points[hit.index];
            let d: f64 = (0..3).map(|a| (p[a] - q[a]).powi(2)).sum();
            assert_eq!(d, hit.distance_sq);
        }
    }

    #[test]
    fn any_guess_yields_the_nearest() {
        let points = pseudo_random(500, 3);
        let tree = KdTree::new(&points).unwrap();
        let far = tree.nearest(&[100.0, 100.0, 100.0], None);
        for q in pseudo_random(50, 5) {
            assert_eq!(tree.nearest(&q, Some(far)).distance_sq, brute(&points, &q));
        }
    }

    #[test]
    fn nearest_k_matches_brute_force() {
        let points = pseudo_random(1500, 13);
        let tree = KdTree::new(&points).unwrap();
        for q in pseudo_random(40, 17) {
            let got = tree.nearest_k(&q, 7);
            let mut all: Vec<f64> = points
                .iter()
                .map(|p| (0..3).map(|a| (p[a] - q[a]).powi(2)).sum())
                .collect();
            all.sort_by(f64::total_cmp);
            assert_eq!(
                got.iter().map(|g| g.1).collect::<Vec<_>>(),
                all[..7].to_vec()
            );
        }
        assert_eq!(tree.nearest_k(&[0.0; 3], 5000).len(), 1500);
    }

    #[test]
    fn handles_duplicates_and_tiny_inputs() {
        let points = vec![[1.0, 2.0, 3.0]; 100];
        let tree = KdTree::new(&points).unwrap();
        assert_eq!(tree.nearest(&[1.0, 2.0, 4.0], None).distance_sq, 1.0);
        let single = KdTree::new(&[[0.0; 3]]).unwrap();
        assert_eq!(single.nearest(&[3.0, 4.0, 0.0], None).distance_sq, 25.0);
        assert!(KdTree::new(&[]).is_none());
    }
}
