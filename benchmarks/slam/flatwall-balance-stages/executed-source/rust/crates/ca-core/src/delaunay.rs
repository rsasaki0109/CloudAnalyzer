//! 2.5D Delaunay meshing (like CloudCompare's "Delaunay 2.5D (XY plane)"):
//! triangulate the points' XY positions and keep their Z.

use delaunator::{EMPTY, Point, next_halfedge};

use crate::PointCloud;
use crate::filter::voxel_subsample;
use crate::mesh::TriangleMesh;

/// Which triangles to drop for having a long (horizontal) edge.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MaxEdge {
    /// Keep every triangle, up to the convex hull.
    Unlimited,
    /// [`AUTO_EDGE_FACTOR`] times the median edge length.
    Auto,
    Length(f64),
}

/// The automatic max edge length as a multiple of the median edge: well
/// above the spacing of evenly sampled points, well below a gap or a hull
/// edge across a concave outline.
pub const AUTO_EDGE_FACTOR: f64 = 4.0;

#[derive(Debug, Clone, PartialEq)]
pub struct Meshed {
    /// Only the points some kept triangle uses, in input order. Triangles
    /// wind counter-clockwise seen from above, so their normals face +Z.
    pub mesh: TriangleMesh,
    /// The longest horizontal edge allowed (infinite for
    /// [`MaxEdge::Unlimited`]).
    pub max_edge: f64,
    /// Triangles dropped for a longer edge.
    pub removed: usize,
}

/// Triangulate `points` in the XY plane. Points sharing an XY position are
/// used once. Returns `None` when fewer than three points are not collinear
/// in XY.
pub fn delaunay_25d(points: &[[f64; 3]], max_edge: MaxEdge) -> Option<Meshed> {
    // Relative to the first point, so georeferenced coordinates keep their
    // precision in the in-circle tests.
    let origin = *points.first()?;
    let xy: Vec<Point> = points
        .iter()
        .map(|p| Point {
            x: p[0] - origin[0],
            y: p[1] - origin[1],
        })
        .collect();
    let t = delaunator::triangulate(&xy);
    drop(xy);
    if t.triangles.is_empty() {
        return None;
    }
    let length = |a: usize, b: usize| {
        let (p, q) = (points[a], points[b]);
        (p[0] - q[0]).hypot(p[1] - q[1])
    };
    let max_edge = match max_edge {
        MaxEdge::Unlimited => f64::INFINITY,
        MaxEdge::Length(l) => l,
        MaxEdge::Auto => {
            // Every edge once (hull edges have no twin), sampled so the
            // median of a huge mesh stays cheap.
            let step = (t.triangles.len() / 1_000_000).max(1);
            let mut edges: Vec<f64> = (0..t.triangles.len())
                .step_by(step)
                .filter(|&e| t.halfedges[e] == EMPTY || e < t.halfedges[e])
                .map(|e| length(t.triangles[e], t.triangles[next_halfedge(e)]))
                .collect();
            let mid = edges.len() / 2;
            AUTO_EDGE_FACTOR * *edges.select_nth_unstable_by(mid, f64::total_cmp).1
        }
    };
    let mut index = vec![u32::MAX; points.len()];
    let mut mesh = TriangleMesh::default();
    let mut removed = 0;
    for &[a, b, c] in t.triangles.as_chunks::<3>().0 {
        if length(a, b).max(length(b, c)).max(length(c, a)) > max_edge {
            removed += 1;
            continue;
        }
        // delaunator winds clockwise in a y-up frame; reversed, normals face up.
        mesh.triangles.push([a, c, b].map(|v| {
            if index[v] == u32::MAX {
                index[v] = mesh.vertices.len() as u32;
                mesh.vertices.push(points[v]);
            }
            index[v]
        }));
    }
    Some(Meshed {
        mesh,
        max_edge,
        removed,
    })
}

/// Points to mesh when `cloud` has more than `limit`: one per voxel, the
/// voxel growing until at most `limit` are left. Returns the kept indices
/// and the voxel size, or `None` when the cloud is small enough.
pub fn thin_for_meshing(cloud: &PointCloud, limit: usize) -> Option<(Vec<usize>, f64)> {
    if cloud.len() <= limit {
        return None;
    }
    let b = cloud.bounds()?;
    let extent = |a: usize| (b.max[a] - b.min[a]).max(f64::MIN_POSITIVE);
    // A surface seen from above holds about area / voxel² points.
    let mut voxel = (extent(0) * extent(1) / limit.max(1) as f64).sqrt();
    loop {
        let keep = voxel_subsample(cloud, voxel);
        if keep.len() <= limit {
            return Some((keep, voxel));
        }
        voxel *= 1.25;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid(n: usize, x0: f64) -> Vec<[f64; 3]> {
        (0..n * n)
            .map(|k| {
                let (i, j) = ((k % n) as f64, (k / n) as f64);
                [x0 + i * 0.1, j * 0.1, (i * 0.3).sin()]
            })
            .collect()
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
            .map(|_| [next() * 10.0, next() * 10.0, next()])
            .collect()
    }

    #[test]
    fn regular_grid_gives_two_triangles_per_cell() {
        let n = 30;
        for max_edge in [MaxEdge::Unlimited, MaxEdge::Auto] {
            let out = delaunay_25d(&grid(n, 0.0), max_edge).unwrap();
            assert_eq!(out.mesh.triangles.len(), 2 * (n - 1) * (n - 1));
            assert_eq!(out.mesh.vertices.len(), n * n);
            assert_eq!(out.removed, 0);
        }
    }

    #[test]
    fn no_point_inside_any_circumcircle() {
        let points = pseudo_random(300, 9);
        let mesh = delaunay_25d(&points, MaxEdge::Unlimited).unwrap().mesh;
        // Every point is used, so vertices can be checked instead of points.
        assert_eq!(mesh.vertices.len(), points.len());
        for t in 0..mesh.triangles.len() {
            let [a, b, c] = mesh.corners(t);
            // Circumcenter of abc in XY.
            let (bx, by, cx, cy) = (b[0] - a[0], b[1] - a[1], c[0] - a[0], c[1] - a[1]);
            let d = 2.0 * (bx * cy - by * cx);
            let (b2, c2) = (bx * bx + by * by, cx * cx + cy * cy);
            let ux = (cy * b2 - by * c2) / d;
            let uy = (bx * c2 - cx * b2) / d;
            let r2 = ux * ux + uy * uy;
            for p in &mesh.vertices {
                let (dx, dy) = (p[0] - a[0] - ux, p[1] - a[1] - uy);
                assert!(
                    dx * dx + dy * dy >= r2 * (1.0 - 1e-9),
                    "{p:?} inside triangle {t}"
                );
            }
        }
    }

    #[test]
    fn normals_face_up() {
        let mesh = delaunay_25d(&pseudo_random(100, 4), MaxEdge::Unlimited)
            .unwrap()
            .mesh;
        for t in 0..mesh.triangles.len() {
            assert!(mesh.normal(t)[2] > 0.0);
        }
    }

    #[test]
    fn max_edge_drops_triangles_across_a_gap() {
        // Two 10 x 10 patches 1 apart (spacing 0.1).
        let mut points = grid(10, 0.0);
        points.extend(grid(10, 1.9));
        let all = delaunay_25d(&points, MaxEdge::Unlimited).unwrap();
        assert!(all.mesh.triangles.len() > 2 * 2 * 81);
        let out = delaunay_25d(&points, MaxEdge::Auto).unwrap();
        assert_eq!(out.mesh.triangles.len(), 2 * 2 * 81);
        assert_eq!(out.removed, all.mesh.triangles.len() - 2 * 2 * 81);
        assert!((out.max_edge - AUTO_EDGE_FACTOR * 0.1).abs() < 1e-9);
        for tri in &out.mesh.triangles {
            let sides = tri.map(|v| out.mesh.vertices[v as usize][0] > 1.0);
            assert!(sides.iter().all(|&s| s == sides[0]), "bridges the gap");
        }
        let fixed = delaunay_25d(&points, MaxEdge::Length(0.5)).unwrap();
        assert_eq!(fixed.mesh.triangles.len(), 2 * 2 * 81);
    }

    #[test]
    fn collinear_points_give_nothing() {
        let line: Vec<[f64; 3]> = (0..10).map(|i| [i as f64, 2.0 * i as f64, 0.0]).collect();
        assert_eq!(delaunay_25d(&line, MaxEdge::Unlimited), None);
        assert_eq!(delaunay_25d(&[], MaxEdge::Unlimited), None);
    }

    #[test]
    fn large_clouds_are_thinned_to_the_limit() {
        let cloud = PointCloud {
            positions: grid(100, 0.0),
            ..Default::default()
        };
        assert_eq!(thin_for_meshing(&cloud, 10_000), None);
        let (keep, voxel) = thin_for_meshing(&cloud, 2_000).unwrap();
        assert!(keep.len() <= 2_000 && keep.len() > 500, "{}", keep.len());
        assert!(voxel > 0.1);
    }
}
