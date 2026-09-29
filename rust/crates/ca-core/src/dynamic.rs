//! Dynamic points by visibility, as in rsasaki0109/dynamic-3d-object-removal
//! and Removert: a point one scan saw is dynamic when other scans taken
//! from nearby poses shortly before or after saw *through* it, their beam
//! in that direction reaching further, so nothing stood there then. A
//! surface they hit instead (the "revert" guard) keeps it: a parked car is
//! seen again, a passing one is not.
//!
//! Each scan is turned into a range image (the nearest return per azimuth
//! and elevation bin, in its own frame); every point of the neighbouring
//! scans is looked up in it. Dynamic points then gather into objects per
//! scan: stray ones revert, and the rest of an object joins.
//!
//! On SemanticKITTI 07 (moving classes, scans thinned to 0.4 m as the web
//! app does, ground-truth poses) the defaults score precision 0.80, recall
//! 0.58, F1 0.67, keeping 99.9 % of the static points; with KISS-ICP
//! odometry for the poses, F1 0.66.

use crate::PointCloud;
use crate::icp::Rigid;

#[derive(Debug, Clone, Copy)]
pub struct VisibilityParams {
    /// Scans this many keyframes before and after vote on a scan's points.
    pub window: usize,
    /// A beam counts as seeing through a point when it reaches this much
    /// further, and as hitting it within this much (metres).
    pub margin: f64,
    /// ... plus this fraction of the point's range: far returns are sparse
    /// and land in coarse bins, so they need more room.
    pub margin_ratio: f64,
    /// Fewest see-through votes for a dynamic point.
    pub min_see_through: usize,
    /// Range image bin size (degrees).
    pub resolution_deg: f64,
    /// Points further than this (metres) are not judged: sparse returns there.
    pub max_range: f64,
    /// Then, per scan: dynamic points closer than this (metres) form
    /// objects; objects of fewer than `min_object` points revert to static
    /// (stray votes), and static points this close to an object join it
    /// (the rest of a car whose every point did not get the votes). 0 skips.
    pub object_link: f64,
    pub min_object: usize,
}

impl Default for VisibilityParams {
    fn default() -> Self {
        Self {
            window: 10,
            margin: 0.5,
            margin_ratio: 0.02,
            min_see_through: 3,
            resolution_deg: 3.0,
            max_range: 50.0,
            object_link: 0.7,
            min_object: 15,
        }
    }
}

/// Elevations the range images cover (degrees): past a spinning LiDAR's field.
const ELEVATION: (f64, f64) = (-35.0, 20.0);

struct RangeImage {
    cols: usize,
    rows: usize,
    resolution: f64,
    ranges: Vec<f32>,
}

impl RangeImage {
    fn new(scan: &PointCloud, resolution_deg: f64) -> Self {
        let resolution = resolution_deg.to_radians();
        let cols = (std::f64::consts::TAU / resolution).ceil() as usize;
        let rows = ((ELEVATION.1 - ELEVATION.0).to_radians() / resolution).ceil() as usize;
        let mut image = RangeImage {
            cols,
            rows,
            resolution,
            ranges: vec![f32::INFINITY; cols * rows],
        };
        for p in &scan.positions {
            if let Some((bin, r)) = image.bin(p) {
                let slot = &mut image.ranges[bin];
                *slot = slot.min(r as f32);
            }
        }
        image
    }

    /// The bin of a point in this scan's frame, and its range.
    fn bin(&self, p: &[f64; 3]) -> Option<(usize, f64)> {
        let r = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
        if r < 1e-6 {
            return None;
        }
        let azimuth = p[1].atan2(p[0]) + std::f64::consts::PI;
        let elevation = (p[2] / r).asin() - ELEVATION.0.to_radians();
        let col = ((azimuth / self.resolution) as usize).min(self.cols - 1);
        let row = elevation / self.resolution;
        if row < 0.0 || row >= self.rows as f64 {
            return None;
        }
        Some((row as usize * self.cols + col, r))
    }
}

/// Per scan, which of its points are dynamic. `scans[i]` was taken at
/// `poses[i]` (each in its own frame); scans are judged against their
/// `window` neighbours on either side.
pub fn dynamic_points(
    poses: &[Rigid],
    scans: &[Option<&PointCloud>],
    params: &VisibilityParams,
) -> Vec<Vec<bool>> {
    let images: Vec<Option<RangeImage>> = scans
        .iter()
        .map(|s| s.map(|s| RangeImage::new(s, params.resolution_deg)))
        .collect();
    let inverse: Vec<Rigid> = poses.iter().map(crate::pose_graph::inverse).collect();
    scans
        .iter()
        .enumerate()
        .map(|(i, scan)| {
            let Some(scan) = scan else {
                return Vec::new();
            };
            let lo = i.saturating_sub(params.window);
            let hi = (i + params.window).min(scans.len() - 1);
            // From scan i's frame into each neighbour's.
            let neighbours: Vec<(Rigid, &RangeImage)> = (lo..=hi)
                .filter(|&j| j != i)
                .filter_map(|j| Some((inverse[j].compose(&poses[i]), images[j].as_ref()?)))
                .collect();
            scan.positions
                .iter()
                .map(|p| {
                    let own = (p[0] * p[0] + p[1] * p[1] + p[2] * p[2]).sqrt();
                    if own > params.max_range {
                        return false;
                    }
                    let (mut through, mut hits) = (0usize, 0usize);
                    for (to_j, image) in &neighbours {
                        let q = to_j.apply(p);
                        let Some((bin, r)) = image.bin(&q) else {
                            continue;
                        };
                        if r > params.max_range {
                            continue;
                        }
                        let seen = f64::from(image.ranges[bin]);
                        if !seen.is_finite() {
                            continue;
                        }
                        let margin = params.margin + params.margin_ratio * r;
                        if seen > r + margin {
                            through += 1;
                        } else if (seen - r).abs() <= margin {
                            hits += 1;
                        }
                    }
                    through >= params.min_see_through && through > hits
                })
                .collect::<Vec<bool>>()
        })
        .zip(scans)
        .map(|(flags, scan)| match scan {
            Some(scan) if params.object_link > 0.0 && !flags.is_empty() => {
                objects(scan, flags, params.object_link, params.min_object)
            }
            _ => flags,
        })
        .collect()
}

/// Dynamic points as whole objects (see [`VisibilityParams::object_link`]).
fn objects(scan: &PointCloud, flags: Vec<bool>, link: f64, min_object: usize) -> Vec<bool> {
    let dynamic: Vec<usize> = (0..flags.len()).filter(|&i| flags[i]).collect();
    if dynamic.is_empty() {
        return flags;
    }
    let points: Vec<[f64; 3]> = dynamic.iter().map(|&i| scan.positions[i]).collect();
    let clusters = crate::cluster::euclidean_clusters(&points, link, min_object.max(1));
    let kept: Vec<[f64; 3]> = points
        .iter()
        .zip(&clusters.labels)
        .filter(|&(_, &l)| l != crate::cluster::NOISE)
        .map(|(p, _)| *p)
        .collect();
    let Some(tree) = crate::kdtree::KdTree::new(&kept) else {
        return vec![false; flags.len()];
    };
    let mut guess = None;
    scan.positions
        .iter()
        .map(|p| {
            let hit = tree.nearest(p, guess);
            guess = Some(hit);
            hit.distance_sq <= link * link
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn cloud(points: Vec<[f64; 3]>) -> PointCloud {
        PointCloud {
            positions: points,
            ..PointCloud::default()
        }
    }

    /// A wall 20 m ahead (x = 20), seen from poses stepping along y.
    fn wall() -> Vec<[f64; 3]> {
        let mut out = Vec::new();
        for y in -40..=40 {
            for z in -10..=10 {
                out.push([20.0, y as f64 * 0.5, z as f64 * 0.2]);
            }
        }
        out
    }

    #[test]
    fn a_passing_object_is_dynamic_and_the_wall_is_not() {
        // Five scans; in the middle one a box 10 m ahead hides part of the wall.
        let poses: Vec<Rigid> = (0..5)
            .map(|k| Rigid {
                translation: [0.0, k as f64 * 0.2, 0.0],
                ..Rigid::IDENTITY
            })
            .collect();
        let mut middle = wall();
        middle.retain(|p| p[1].abs() > 2.0 || p[2].abs() > 1.0);
        let mut object = Vec::new();
        for y in -8..=8 {
            for z in -4..=4 {
                object.push([10.0, y as f64 * 0.12, z as f64 * 0.12]);
            }
        }
        let first_object = middle.len();
        middle.extend(&object);
        let scans: Vec<PointCloud> = (0..5)
            .map(|k| {
                if k == 2 {
                    cloud(middle.clone())
                } else {
                    cloud(wall())
                }
            })
            .collect();
        let refs: Vec<Option<&PointCloud>> = scans.iter().map(Some).collect();
        let flags = dynamic_points(&poses, &refs, &VisibilityParams::default());
        let moving = &flags[2][first_object..];
        let still = &flags[2][..first_object];
        assert!(
            moving.iter().filter(|&&d| d).count() * 10 >= moving.len() * 9,
            "object mostly dynamic"
        );
        assert!(
            still.iter().filter(|&&d| d).count() * 100 <= still.len(),
            "wall stays static"
        );
        // The other scans see only the wall: nothing dynamic.
        assert!(flags[0].iter().all(|&d| !d));
    }

    #[test]
    fn a_parked_object_seen_by_every_scan_stays() {
        let poses: Vec<Rigid> = (0..5)
            .map(|k| Rigid {
                translation: [0.0, k as f64 * 0.2, 0.0],
                ..Rigid::IDENTITY
            })
            .collect();
        let mut with_car = wall();
        for y in -8..=8 {
            for z in -4..=4 {
                with_car.push([10.0, y as f64 * 0.12, z as f64 * 0.12]);
            }
        }
        // Every scan sees the parked object (in world coordinates it stays put).
        let scans: Vec<PointCloud> = poses
            .iter()
            .map(|pose| {
                let back = crate::pose_graph::inverse(pose);
                cloud(with_car.iter().map(|p| back.apply(p)).collect())
            })
            .collect();
        let refs: Vec<Option<&PointCloud>> = scans.iter().map(Some).collect();
        let flags = dynamic_points(&poses, &refs, &VisibilityParams::default());
        let dynamic: usize = flags.iter().map(|f| f.iter().filter(|&&d| d).count()).sum();
        assert_eq!(dynamic, 0);
    }
}
