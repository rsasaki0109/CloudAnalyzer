//! Surface normals from the `k` nearest neighbours of each point (PCA: the
//! direction of least spread), stored as the `nx`, `ny`, `nz` attributes.

use crate::kdtree::KdTree;
use crate::{Attribute, AttributeValues, PointCloud};

/// Attribute names of the normal components (as PLY and CloudCompare use).
pub const NORMAL_NAMES: [&str; 3] = ["nx", "ny", "nz"];

/// Which of the two opposite directions a normal takes.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum Orientation {
    /// Positive z (terrain, floors, roofs).
    Up,
    /// Away from this point (e.g. the centroid of a closed object).
    Away([f64; 3]),
}

/// Unit normal per point; zero where fewer than three neighbours span a
/// plane.
pub fn estimate_normals(points: &[[f64; 3]], k: usize, orientation: Orientation) -> Vec<[f32; 3]> {
    let Some(tree) = KdTree::new(points) else {
        return Vec::new();
    };
    let mut out = vec![[0.0f32; 3]; points.len()];
    for i in crate::distance::morton_order(points) {
        out[i] = normal_at(&tree, points, i, k, orientation);
    }
    out
}

/// Multi-threaded [`estimate_normals`] (feature `parallel`); identical results.
#[cfg(feature = "parallel")]
pub fn estimate_normals_par(
    points: &[[f64; 3]],
    k: usize,
    orientation: Orientation,
) -> Vec<[f32; 3]> {
    use rayon::prelude::*;
    let Some(tree) = KdTree::new(points) else {
        return Vec::new();
    };
    let order = crate::distance::morton_order(points);
    let parts: Vec<Vec<(usize, [f32; 3])>> = order
        .par_chunks(4096)
        .map(|chunk| {
            chunk
                .iter()
                .map(|&i| (i, normal_at(&tree, points, i, k, orientation)))
                .collect()
        })
        .collect();
    let mut out = vec![[0.0f32; 3]; points.len()];
    for (i, n) in parts.into_iter().flatten() {
        out[i] = n;
    }
    out
}

fn normal_at(
    tree: &KdTree,
    points: &[[f64; 3]],
    i: usize,
    k: usize,
    orientation: Orientation,
) -> [f32; 3] {
    let p = points[i];
    let hits = tree.nearest_k(&p, k.max(3));
    if hits.len() < 3 {
        return [0.0; 3];
    }
    let n = hits.len() as f64;
    let mut c = [0.0; 3];
    for &(j, _) in &hits {
        for a in 0..3 {
            c[a] += points[j][a] / n;
        }
    }
    let mut cov = [[0.0; 3]; 3];
    for &(j, _) in &hits {
        let d = [
            points[j][0] - c[0],
            points[j][1] - c[1],
            points[j][2] - c[2],
        ];
        for r in 0..3 {
            for s in 0..3 {
                cov[r][s] += d[r] * d[s];
            }
        }
    }
    let (values, vectors) = crate::icp::symmetric_eigen(cov);
    let smallest = (0..3)
        .min_by(|&a, &b| values[a].total_cmp(&values[b]))
        .unwrap_or(2);
    let middle = (0..3)
        .filter(|&a| a != smallest)
        .map(|a| values[a])
        .fold(f64::INFINITY, f64::min);
    let largest = values.iter().copied().fold(0.0, f64::max);
    if middle.is_nan() || middle <= 1e-12 * largest.max(f64::MIN_POSITIVE) {
        return [0.0; 3]; // collinear or repeated points: no plane
    }
    let mut v = [
        vectors[0][smallest],
        vectors[1][smallest],
        vectors[2][smallest],
    ];
    let flip = match orientation {
        Orientation::Up => v[2] < 0.0,
        Orientation::Away(from) => (0..3).map(|a| v[a] * (p[a] - from[a])).sum::<f64>() < 0.0,
    };
    if flip {
        v = v.map(|x| -x);
    }
    let len = (v[0] * v[0] + v[1] * v[1] + v[2] * v[2]).sqrt();
    v.map(|x| (x / len) as f32)
}

/// Store `normals` (one per point) as the `nx`, `ny`, `nz` attributes,
/// replacing earlier ones. Returns false if the count does not match.
pub fn set_normals(cloud: &mut PointCloud, normals: &[[f32; 3]]) -> bool {
    if normals.len() != cloud.len() {
        return false;
    }
    cloud
        .attributes
        .retain(|a| !NORMAL_NAMES.contains(&a.name.as_str()));
    for (axis, name) in NORMAL_NAMES.iter().enumerate() {
        cloud.attributes.push(Attribute {
            name: (*name).into(),
            values: AttributeValues::F32(normals.iter().map(|n| n[axis]).collect()),
        });
    }
    true
}

/// The normals stored on `cloud`, if it has all three components.
pub fn normals(cloud: &PointCloud) -> Option<Vec<[f32; 3]>> {
    let get = |name: &str| match &cloud.attribute(name)?.values {
        AttributeValues::F32(v) => Some(v.as_slice()),
        AttributeValues::U8(_) => None,
    };
    let (x, y, z) = (get("nx")?, get("ny")?, get("nz")?);
    Some((0..cloud.len()).map(|i| [x[i], y[i], z[i]]).collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_tilted_plane_gets_its_normal_facing_up() {
        // z = 0.5 x: normal (-0.5, 0, 1) / |..|.
        let points: Vec<[f64; 3]> = (0..2500)
            .map(|k| {
                let (x, y) = ((k % 50) as f64 * 0.1, (k / 50) as f64 * 0.1);
                [x, y, 0.5 * x]
            })
            .collect();
        let normals = estimate_normals(&points, 10, Orientation::Up);
        let expect = [-0.5 / 1.25f64.sqrt(), 0.0, 1.0 / 1.25f64.sqrt()];
        for n in &normals {
            for a in 0..3 {
                assert!((n[a] as f64 - expect[a]).abs() < 1e-4, "{n:?}");
            }
        }
    }

    #[test]
    fn a_sphere_points_outward_from_its_centre() {
        let mut points = Vec::new();
        for i in 0..40 {
            for j in 0..80 {
                let (t, f) = (
                    std::f64::consts::PI * (i as f64 + 0.5) / 40.0,
                    std::f64::consts::TAU * j as f64 / 80.0,
                );
                points.push([
                    5.0 + t.sin() * f.cos(),
                    5.0 + t.sin() * f.sin(),
                    5.0 + t.cos(),
                ]);
            }
        }
        let normals = estimate_normals(&points, 12, Orientation::Away([5.0, 5.0, 5.0]));
        for (p, n) in points.iter().zip(&normals) {
            let radial = (0..3).map(|a| (p[a] - 5.0) * n[a] as f64).sum::<f64>();
            assert!(radial > 0.9, "{radial}");
        }
    }

    #[test]
    fn stored_as_attributes() {
        let mut cloud = PointCloud {
            positions: vec![[0.0; 3]; 2],
            colors: None,
            attributes: Vec::new(),
        };
        assert!(set_normals(&mut cloud, &[[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]]));
        assert!(set_normals(&mut cloud, &[[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]));
        assert_eq!(cloud.attributes.len(), 3);
        assert_eq!(
            normals(&cloud).unwrap(),
            vec![[0.0, 1.0, 0.0], [1.0, 0.0, 0.0]]
        );
        assert!(!set_normals(&mut cloud, &[[0.0; 3]]));
        // Collinear points have no normal.
        let line: Vec<[f64; 3]> = (0..10).map(|i| [i as f64, 0.0, 0.0]).collect();
        assert!(
            estimate_normals(&line, 5, Orientation::Up)
                .iter()
                .all(|n| *n == [0.0; 3])
        );
    }

    #[cfg(feature = "parallel")]
    #[test]
    fn parallel_matches_serial() {
        let points: Vec<[f64; 3]> = (0..20_000)
            .map(|k| {
                let (x, y) = ((k % 200) as f64 * 0.1, (k / 200) as f64 * 0.1);
                [x, y, (x * 0.3).sin() + (y * 0.2).cos()]
            })
            .collect();
        assert_eq!(
            estimate_normals_par(&points, 8, Orientation::Up),
            estimate_normals(&points, 8, Orientation::Up)
        );
    }
}
