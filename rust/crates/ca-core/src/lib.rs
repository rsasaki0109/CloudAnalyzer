//! Point cloud I/O and analysis core for CloudAnalyzer Web.
//!
//! Coordinates are kept in `f64` so georeferenced clouds (UTM, ECEF) survive
//! loading; renderers should subtract [`PointCloud::suggested_shift`] before
//! narrowing to `f32`.

pub mod distance;
pub mod io;
pub mod kdtree;
pub mod octree;

pub use distance::{C2cPart, DistanceStats, cloud_to_cloud, partition_c2c};
pub use io::{Format, IoError, read};

/// An unordered set of 3D points with optional per-point RGB colors.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PointCloud {
    pub positions: Vec<[f64; 3]>,
    pub colors: Option<Vec<[u8; 3]>>,
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
