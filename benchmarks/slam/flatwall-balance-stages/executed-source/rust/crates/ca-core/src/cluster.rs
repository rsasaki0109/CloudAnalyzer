//! Euclidean clustering (connected components): points closer than `epsilon`
//! belong to the same cluster, like CloudCompare's "Label Connected
//! Components" but on exact distances instead of an octree grid.

use crate::kdtree::KdTree;

/// Label of points in no cluster of at least the minimum size.
pub const NOISE: u32 = u32::MAX;

#[derive(Debug, Clone, PartialEq)]
pub struct Clusters {
    /// Cluster of each point (0 = the largest), or [`NOISE`].
    pub labels: Vec<u32>,
    /// Point count of each cluster, descending.
    pub sizes: Vec<usize>,
}

/// Connected components of the graph linking points within `epsilon` of
/// each other; components smaller than `min_size` are noise.
///
/// Union-find over a grid of cells (see [`link_by_grid`]), or over k-d
/// tree radius queries when the grid would be too fine; never `O(n²)`
/// for an `epsilon` of a few point spacings.
pub fn euclidean_clusters(points: &[[f64; 3]], epsilon: f64, min_size: usize) -> Clusters {
    let n = points.len();
    let mut parent: Vec<u32> = (0..n as u32).collect();
    if !link_by_grid(points, epsilon, &mut parent) {
        link_by_tree(points, epsilon, &mut parent);
    }
    // Root of each point, then components by size (ties: first point).
    let roots: Vec<u32> = (0..n as u32).map(|i| find(&mut parent, i)).collect();
    let mut sizes = vec![0usize; n];
    for &r in &roots {
        sizes[r as usize] += 1;
    }
    let mut kept: Vec<u32> = (0..n as u32)
        .filter(|&r| roots[r as usize] == r && sizes[r as usize] >= min_size.max(1))
        .collect();
    kept.sort_by_key(|&r| (std::cmp::Reverse(sizes[r as usize]), r));
    let mut label_of_root = vec![NOISE; n];
    for (label, &r) in kept.iter().enumerate() {
        label_of_root[r as usize] = label as u32;
    }
    Clusters {
        labels: roots.iter().map(|&r| label_of_root[r as usize]).collect(),
        sizes: kept.iter().map(|&r| sizes[r as usize]).collect(),
    }
}

/// Link each point to its neighbours within `epsilon`, one k-d tree radius
/// query per point.
fn link_by_tree(points: &[[f64; 3]], epsilon: f64, parent: &mut [u32]) {
    let Some(tree) = KdTree::new(points) else {
        return;
    };
    let mut hits = Vec::new();
    // Morton order keeps consecutive queries in the same tree leaves.
    for i in crate::distance::morton_order(points) {
        tree.within(&points[i], epsilon, &mut hits);
        for &j in &hits {
            // The relation is symmetric: each pair is linked from one side.
            if j > i {
                union(parent, i as u32, j as u32);
            }
        }
    }
}

/// Grid cells per axis above which [`link_by_grid`] gives up.
const MAX_CELLS: u64 = 1 << 20;

/// Link points within `epsilon` through cells `epsilon / √3` wide. The
/// points of a cell are all within `epsilon` of each other, so they are
/// joined without a distance check; two cells up to two apart are joined
/// by the first close enough pair, and not checked at all once they are
/// connected. In dense clouds that skips nearly every distance a radius
/// query would compute. Returns false (doing nothing) when the grid would
/// be too fine.
fn link_by_grid(points: &[[f64; 3]], epsilon: f64, parent: &mut [u32]) -> bool {
    if points.is_empty() {
        return true;
    }
    // A hair under epsilon / √3, so rounding cannot put a cell's opposite
    // corners farther apart than epsilon.
    let size = epsilon / 3f64.sqrt() * (1.0 - 1e-9);
    let (lo, hi) = crate::distance::bounds(points.iter());
    let dims: [u64; 3] = std::array::from_fn(|a| ((hi[a] - lo[a]) / size) as u64 + 1);
    if size.is_nan() || size <= 0.0 || dims.iter().any(|&d| d >= MAX_CELLS) {
        return false;
    }
    let coords = |p: &[f64; 3]| -> [u64; 3] {
        std::array::from_fn(|a| (((p[a] - lo[a]) / size) as u64).min(dims[a] - 1))
    };
    let key = |c: [u64; 3]| c[0] | c[1] << 20 | c[2] << 40;
    // Points by cell, and each cell's range of them.
    let mut sorted: Vec<(u64, u32)> = points
        .iter()
        .zip(0..)
        .map(|(p, i)| (key(coords(p)), i))
        .collect();
    sorted.sort_unstable();
    let mut cells: Vec<(u64, usize, usize)> = Vec::new();
    for (at, &(k, i)) in sorted.iter().enumerate() {
        match cells.last_mut() {
            Some(last) if last.0 == k => {
                last.2 = at + 1;
                union(parent, sorted[last.1].1, i);
            }
            _ => cells.push((k, at, at + 1)),
        }
    }
    // Positions in cell order, for locality in the pair checks.
    let at: Vec<[f64; 3]> = sorted.iter().map(|&(_, i)| points[i as usize]).collect();
    let eps2 = epsilon * epsilon;
    // Cells are sorted by (z, y, x), so the neighbours of a cell in each row
    // (y + dy, z + dz) are a run of cells, and where that run starts only
    // moves forward from one cell to the next: one cursor per row. Each pair
    // of cells is checked once, from the earlier one.
    let rows: Vec<[i64; 2]> = (-2..=2)
        .flat_map(|dz| (-2..=2).map(move |dy| [dy, dz]))
        .collect();
    let mut cursors = vec![0usize; rows.len()];
    for &(k, start, end) in &cells {
        let [x, y, z] = [k & 0xf_ffff, k >> 20 & 0xf_ffff, k >> 40];
        for (row, cursor) in rows.iter().zip(&mut cursors) {
            let (ny, nz) = (y as i64 + row[0], z as i64 + row[1]);
            if ny < 0 || nz < 0 || ny >= dims[1] as i64 || nz >= dims[2] as i64 {
                continue;
            }
            let first = key([x.saturating_sub(2), ny as u64, nz as u64]);
            let last = key([(x + 2).min(dims[0] - 1), ny as u64, nz as u64]);
            while *cursor < cells.len() && cells[*cursor].0 < first {
                *cursor += 1;
            }
            for &(nk, os, oe) in cells[*cursor..].iter().take_while(|n| n.0 <= last) {
                if nk <= k || find(parent, sorted[start].1) == find(parent, sorted[os].1) {
                    continue;
                }
                'pairs: for i in start..end {
                    for j in os..oe {
                        let (p, q) = (at[i], at[j]);
                        let d2 =
                            (p[0] - q[0]).powi(2) + (p[1] - q[1]).powi(2) + (p[2] - q[2]).powi(2);
                        if d2 <= eps2 {
                            union(parent, sorted[i].1, sorted[j].1);
                            break 'pairs;
                        }
                    }
                }
            }
        }
    }
    true
}

/// Root of `i`, halving the path on the way.
fn find(parent: &mut [u32], mut i: u32) -> u32 {
    while parent[i as usize] != i {
        let up = parent[parent[i as usize] as usize];
        parent[i as usize] = up;
        i = up;
    }
    i
}

/// Join the sets of `a` and `b` (the smaller root index becomes the root).
fn union(parent: &mut [u32], a: u32, b: u32) {
    let (ra, rb) = (find(parent, a), find(parent, b));
    if ra != rb {
        let (lo, hi) = if ra < rb { (ra, rb) } else { (rb, ra) };
        parent[hi as usize] = lo;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A `side` x `side` x `side` lattice of points `step` apart at `origin`.
    fn blob(origin: [f64; 3], side: usize, step: f64) -> Vec<[f64; 3]> {
        let mut out = Vec::new();
        for i in 0..side {
            for j in 0..side {
                for k in 0..side {
                    out.push([
                        origin[0] + i as f64 * step,
                        origin[1] + j as f64 * step,
                        origin[2] + k as f64 * step,
                    ]);
                }
            }
        }
        out
    }

    #[test]
    fn two_separated_blobs_are_two_clusters() {
        let mut points = blob([0.0; 3], 10, 0.1); // 1000 points
        points.extend(blob([5.0, 0.0, 0.0], 8, 0.1)); // 512 points
        points.push([2.5, 2.5, 2.5]); // isolated
        points.extend([[-3.0, 0.0, 0.0], [-3.05, 0.0, 0.0]]); // a pair
        let c = euclidean_clusters(&points, 0.15, 3);
        assert_eq!(c.sizes, vec![1000, 512]);
        assert!(c.labels[..1000].iter().all(|&l| l == 0));
        assert!(c.labels[1000..1512].iter().all(|&l| l == 1));
        assert!(c.labels[1512..].iter().all(|&l| l == NOISE));
        // With a minimum of 2 the pair is a cluster too.
        assert_eq!(
            euclidean_clusters(&points, 0.15, 2).sizes,
            vec![1000, 512, 2]
        );
    }

    #[test]
    fn grid_and_tree_find_the_same_components() {
        // Sparse random points: many small components, some chains.
        let mut s = 0x2545_f491_4f6c_dd1du64;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut points: Vec<[f64; 3]> = (0..5000)
            .map(|_| [next() * 10.0, next() * 10.0, next() * 2.0])
            .collect();
        // And a dense slab, with many points per cell.
        points.extend((0..5000).map(|_| [next(), next(), 3.0 + 0.1 * next()]));
        for epsilon in [0.02, 0.2, 0.35, 0.6] {
            let mut grid: Vec<u32> = (0..points.len() as u32).collect();
            let mut tree = grid.clone();
            assert!(link_by_grid(&points, epsilon, &mut grid));
            link_by_tree(&points, epsilon, &mut tree);
            for i in 0..points.len() as u32 {
                assert_eq!(find(&mut grid, i), find(&mut tree, i), "epsilon {epsilon}");
            }
        }
        // Too fine a grid falls back to the tree.
        let far = [[0.0; 3], [1e9, 0.0, 0.0], [1e9 + 0.5, 0.0, 0.0]];
        assert!(!link_by_grid(&far, 1.0, &mut [0, 1, 2]));
        assert_eq!(euclidean_clusters(&far, 1.0, 2).sizes, vec![2]);
    }

    #[test]
    fn chains_link_through_neighbours() {
        // Points 0.1 apart along a line are one cluster at epsilon 0.1,
        // and all singletons just below it.
        let line: Vec<[f64; 3]> = (0..100).map(|i| [i as f64 * 0.1, 0.0, 0.0]).collect();
        assert_eq!(euclidean_clusters(&line, 0.1001, 1).sizes, vec![100]);
        let apart = euclidean_clusters(&line, 0.0999, 1);
        assert_eq!(apart.sizes.len(), 100);
        assert!(
            euclidean_clusters(&line, 0.0999, 2)
                .labels
                .iter()
                .all(|&l| l == NOISE)
        );
        assert!(euclidean_clusters(&[], 1.0, 1).labels.is_empty());
    }
}
