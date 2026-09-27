//! Point cloud I/O and analysis core for CloudAnalyzer Web.
//!
//! Coordinates are kept in `f64` so georeferenced clouds (UTM, ECEF) survive
//! loading; renderers should subtract [`PointCloud::suggested_shift`] before
//! narrowing to `f32`.

pub mod distance;
pub mod filter;
pub mod ground;
pub mod icp;
pub mod io;
pub mod kdtree;
pub mod m3c2;
pub mod mesh;
pub mod octree;
pub mod volume;

pub use distance::{C2cPart, DistanceStats, cloud_to_cloud, cloud_to_mesh, partition_c2c};
pub use io::{Format, IoError, read, read_mesh};
pub use mesh::TriangleMesh;

/// Name of the LiDAR return-strength attribute.
pub const INTENSITY: &str = "intensity";
/// Name of the (ASPRS) class-code attribute.
pub const CLASSIFICATION: &str = "classification";

/// Values of a per-point attribute.
#[derive(Debug, Clone, PartialEq)]
pub enum AttributeValues {
    F32(Vec<f32>),
    U8(Vec<u8>),
}

impl AttributeValues {
    pub fn len(&self) -> usize {
        match self {
            Self::F32(v) => v.len(),
            Self::U8(v) => v.len(),
        }
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// The values at `indices`, in that order.
    pub fn select(&self, indices: &[usize]) -> Self {
        match self {
            Self::F32(v) => Self::F32(indices.iter().map(|&i| v[i]).collect()),
            Self::U8(v) => Self::U8(indices.iter().map(|&i| v[i]).collect()),
        }
    }

    /// Reorder `range` so that position `i` holds the old `range.start + order[i]`.
    pub fn permute_range(&mut self, range: std::ops::Range<usize>, order: &[u32]) {
        fn go<T: Copy>(v: &mut [T], order: &[u32]) {
            let old = v.to_vec();
            for (dst, &o) in v.iter_mut().zip(order) {
                *dst = old[o as usize];
            }
        }
        match self {
            Self::F32(v) => go(&mut v[range], order),
            Self::U8(v) => go(&mut v[range], order),
        }
    }

    pub(crate) fn swap(&mut self, a: usize, b: usize) {
        match self {
            Self::F32(v) => v.swap(a, b),
            Self::U8(v) => v.swap(a, b),
        }
    }
}

/// A named per-point value carried alongside positions (e.g. intensity).
#[derive(Debug, Clone, PartialEq)]
pub struct Attribute {
    pub name: String,
    pub values: AttributeValues,
}

/// An unordered set of 3D points with optional per-point RGB colors and
/// attributes (every attribute has one value per point).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PointCloud {
    pub positions: Vec<[f64; 3]>,
    pub colors: Option<Vec<[u8; 3]>>,
    pub attributes: Vec<Attribute>,
}

/// Axis-aligned bounding box.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Aabb {
    pub min: [f64; 3],
    pub max: [f64; 3],
}

impl Aabb {
    pub fn center(&self) -> [f64; 3] {
        std::array::from_fn(|i| 0.5 * (self.min[i] + self.max[i]))
    }

    pub fn diagonal(&self) -> f64 {
        (0..3)
            .map(|i| (self.max[i] - self.min[i]).powi(2))
            .sum::<f64>()
            .sqrt()
    }
}

impl PointCloud {
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    pub fn attribute(&self, name: &str) -> Option<&Attribute> {
        self.attributes.iter().find(|a| a.name == name)
    }

    pub fn bounds(&self) -> Option<Aabb> {
        let first = *self.positions.first()?;
        let mut aabb = Aabb {
            min: first,
            max: first,
        };
        for p in &self.positions[1..] {
            for (i, &v) in p.iter().enumerate() {
                aabb.min[i] = aabb.min[i].min(v);
                aabb.max[i] = aabb.max[i].max(v);
            }
        }
        Some(aabb)
    }

    /// Points inside (or, with `inside == false`, outside) the axis-aligned
    /// box `[min, max]`, with their colors. Boundary points count as inside.
    pub fn crop(&self, min: [f64; 3], max: [f64; 3], inside: bool) -> PointCloud {
        let keep: Vec<usize> = (0..self.len())
            .filter(|&i| {
                let p = self.positions[i];
                (0..3).all(|a| p[a] >= min[a] && p[a] <= max[a]) == inside
            })
            .collect();
        self.select(&keep)
    }

    /// The points at `indices` (in that order), with their colors and attributes.
    pub fn select(&self, keep: &[usize]) -> PointCloud {
        PointCloud {
            positions: keep.iter().map(|&i| self.positions[i]).collect(),
            colors: self
                .colors
                .as_ref()
                .map(|c| keep.iter().map(|&i| c[i]).collect()),
            attributes: self
                .attributes
                .iter()
                .map(|a| Attribute {
                    name: a.name.clone(),
                    values: a.values.select(keep),
                })
                .collect(),
        }
    }

    /// A shift that brings the cloud near the origin, rounded so that shifted
    /// coordinates stay readable. Zero when the cloud is already small.
    pub fn suggested_shift(&self) -> [f64; 3] {
        const LARGE: f64 = 1.0e4;
        let Some(aabb) = self.bounds() else {
            return [0.0; 3];
        };
        let far = aabb
            .min
            .iter()
            .chain(aabb.max.iter())
            .any(|v| v.abs() > LARGE);
        if !far {
            return [0.0; 3];
        }
        aabb.center().map(|c| (c / 100.0).round() * 100.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn crop_keeps_attributes() {
        let cloud = PointCloud {
            positions: vec![[0.0; 3], [5.0; 3], [0.5; 3]],
            colors: None,
            attributes: vec![Attribute {
                name: CLASSIFICATION.into(),
                values: AttributeValues::U8(vec![2, 6, 9]),
            }],
        };
        let inside = cloud.crop([0.0; 3], [1.0; 3], true);
        assert_eq!(
            inside.attribute(CLASSIFICATION).unwrap().values,
            AttributeValues::U8(vec![2, 9])
        );
    }

    #[test]
    fn crop_keeps_points_and_colors_inside_the_box() {
        let cloud = PointCloud {
            positions: vec![
                [0.0, 0.0, 0.0],
                [1.0, 1.0, 1.0],
                [2.0, 0.5, 0.5],
                [0.5, 0.5, 3.0],
            ],
            colors: Some(vec![[1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4]]),
            attributes: Vec::new(),
        };
        let inside = cloud.crop([0.0; 3], [1.0; 3], true);
        assert_eq!(inside.positions, vec![[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]]);
        assert_eq!(inside.colors, Some(vec![[1, 1, 1], [2, 2, 2]]));
        let outside = cloud.crop([0.0; 3], [1.0; 3], false);
        assert_eq!(outside.colors, Some(vec![[3, 3, 3], [4, 4, 4]]));
    }
}
