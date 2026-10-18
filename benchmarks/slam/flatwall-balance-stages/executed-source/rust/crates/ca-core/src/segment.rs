//! Selecting points inside a polygon drawn on the screen (lasso segmentation).

use crate::{AttributeValues, CLASSIFICATION, PointCloud};

/// What a lasso selects: points whose projection falls inside `polygon`.
pub struct Lasso<'a> {
    /// Row-major 4x4 matrix from original coordinates to clip space
    /// (projection · view · shift), in `f64` so georeferenced points project
    /// without losing precision.
    pub clip_from_world: [f64; 16],
    /// Polygon vertices in normalized device coordinates (x, y in -1..1).
    pub polygon: &'a [[f64; 2]],
    /// Only points inside this box (original coordinates) count, as when the
    /// view is clipped.
    pub clip_box: Option<([f64; 3], [f64; 3])>,
    /// Points of these classes are hidden, so never selected.
    pub hidden_classes: &'a [u8],
}

/// Whether each point is selected by the lasso. Points behind the camera
/// are never selected.
pub fn lasso_mask(cloud: &PointCloud, lasso: &Lasso) -> Vec<bool> {
    let m = &lasso.clip_from_world;
    let classes = match cloud.attribute(CLASSIFICATION).map(|a| &a.values) {
        Some(AttributeValues::U8(c)) if !lasso.hidden_classes.is_empty() => Some(c),
        _ => None,
    };
    let bounds = polygon_bounds(lasso.polygon);
    cloud
        .positions
        .iter()
        .enumerate()
        .map(|(i, p)| {
            if let Some((min, max)) = lasso.clip_box
                && (0..3).any(|a| p[a] < min[a] || p[a] > max[a])
            {
                return false;
            }
            if classes.is_some_and(|c| lasso.hidden_classes.contains(&c[i])) {
                return false;
            }
            let row = |r: usize| {
                m[r * 4] * p[0] + m[r * 4 + 1] * p[1] + m[r * 4 + 2] * p[2] + m[r * 4 + 3]
            };
            let w = row(3);
            if w <= 0.0 {
                return false;
            }
            let (x, y) = (row(0) / w, row(1) / w);
            let Some((lo, hi)) = bounds else { return false };
            x >= lo[0] && x <= hi[0] && y >= lo[1] && y <= hi[1] && inside(lasso.polygon, x, y)
        })
        .collect()
}

fn polygon_bounds(polygon: &[[f64; 2]]) -> Option<([f64; 2], [f64; 2])> {
    if polygon.len() < 3 {
        return None;
    }
    let mut lo = polygon[0];
    let mut hi = polygon[0];
    for v in polygon {
        for a in 0..2 {
            lo[a] = lo[a].min(v[a]);
            hi[a] = hi[a].max(v[a]);
        }
    }
    Some((lo, hi))
}

/// Even-odd rule, so a self-crossing lasso behaves predictably.
fn inside(polygon: &[[f64; 2]], x: f64, y: f64) -> bool {
    let mut odd = false;
    let mut j = polygon.len() - 1;
    for i in 0..polygon.len() {
        let ([xi, yi], [xj, yj]) = (polygon[i], polygon[j]);
        if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
            odd = !odd;
        }
        j = i;
    }
    odd
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Attribute;

    /// Looking down -z from above: clip x = world x, y = world y, w = 1.
    const TOP: [f64; 16] = [
        1.0, 0.0, 0.0, 0.0, //
        0.0, 1.0, 0.0, 0.0, //
        0.0, 0.0, 0.0, 0.0, //
        0.0, 0.0, 0.0, 1.0,
    ];

    fn square(half: f64) -> Vec<[f64; 2]> {
        vec![[-half, -half], [half, -half], [half, half], [-half, half]]
    }

    fn cloud() -> PointCloud {
        PointCloud {
            positions: vec![
                [0.0, 0.0, 0.0],
                [0.4, 0.4, 5.0],
                [0.9, 0.0, 0.0],
                [-0.2, 0.1, -3.0],
            ],
            colors: None,
            attributes: vec![Attribute {
                name: CLASSIFICATION.into(),
                values: AttributeValues::U8(vec![2, 2, 2, 7]),
            }],
        }
    }

    #[test]
    fn selects_points_projecting_inside_the_polygon() {
        let polygon = square(0.5);
        let lasso = Lasso {
            clip_from_world: TOP,
            polygon: &polygon,
            clip_box: None,
            hidden_classes: &[],
        };
        assert_eq!(lasso_mask(&cloud(), &lasso), [true, true, false, true]);
    }

    #[test]
    fn skips_clipped_and_hidden_points() {
        let polygon = square(0.5);
        let lasso = Lasso {
            clip_from_world: TOP,
            polygon: &polygon,
            clip_box: Some(([-1.0; 3], [1.0, 1.0, 1.0])),
            hidden_classes: &[7],
        };
        assert_eq!(lasso_mask(&cloud(), &lasso), [true, false, false, false]);
    }

    #[test]
    fn concave_polygon_uses_its_outline() {
        // A "U": the notch between the arms is outside.
        let polygon = [
            [-1.0, -1.0],
            [1.0, -1.0],
            [1.0, 1.0],
            [0.5, 1.0],
            [0.5, 0.0],
            [-0.5, 0.0],
            [-0.5, 1.0],
            [-1.0, 1.0],
        ];
        let cloud = PointCloud {
            positions: vec![[0.0, 0.5, 0.0], [0.0, -0.5, 0.0], [0.75, 0.5, 0.0]],
            ..Default::default()
        };
        let lasso = Lasso {
            clip_from_world: TOP,
            polygon: &polygon,
            clip_box: None,
            hidden_classes: &[],
        };
        assert_eq!(lasso_mask(&cloud, &lasso), [false, true, true]);
    }

    #[test]
    fn points_behind_the_camera_are_not_selected() {
        let mut m = TOP;
        m[14] = -1.0; // w = -z
        let polygon = square(10.0);
        let lasso = Lasso {
            clip_from_world: m,
            polygon: &polygon,
            clip_box: None,
            hidden_classes: &[],
        };
        let cloud = PointCloud {
            positions: vec![[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]],
            ..Default::default()
        };
        assert_eq!(lasso_mask(&cloud, &lasso), [true, false]);
    }
}
