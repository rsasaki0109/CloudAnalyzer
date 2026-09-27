//! Ground extraction with the Cloth Simulation Filter (CSF; Zhang et al.,
//! "An Easy-to-Use Airborne LiDAR Data Filtering Method Based on Cloth
//! Simulation", Remote Sensing 2016).
//!
//! The cloud is turned upside down and a cloth of particles is dropped onto
//! it from above. Gravity pulls each particle down until it reaches the
//! (inverted) terrain under it; springs between neighbouring particles keep
//! the cloth from sagging into the pits that buildings and trees leave in
//! the inverted surface. Points close to the settled cloth are ground.

use crate::PointCloud;

/// How stiff the cloth is: stiffer cloth bridges larger objects but cannot
/// follow steep terrain.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rigidness {
    /// Flat terrain with large buildings (stiffest).
    Flat,
    /// Gentle relief.
    Relief,
    /// Steep slopes (softest).
    Steep,
}

impl Rigidness {
    /// Constraint passes per step (CSF's `rigidness` 3 / 2 / 1).
    fn passes(self) -> usize {
        match self {
            Self::Flat => 3,
            Self::Relief => 2,
            Self::Steep => 1,
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct CsfParams {
    /// Distance between cloth particles (the paper's `cloth_resolution`).
    pub cloth_resolution: f64,
    /// Points within this vertical distance of the cloth are ground.
    pub class_threshold: f64,
    pub rigidness: Rigidness,
    pub time_step: f64,
    pub max_iterations: usize,
}

impl Default for CsfParams {
    fn default() -> Self {
        Self {
            cloth_resolution: 1.0,
            class_threshold: 0.5,
            rigidness: Rigidness::Relief,
            time_step: 0.65,
            max_iterations: 500,
        }
    }
}

/// Whether each point is ground, in the cloud's order. Returns `None` for an
/// empty cloud or a non-positive resolution.
pub fn csf(cloud: &PointCloud, params: CsfParams) -> Option<Vec<bool>> {
    let res = params.cloth_resolution;
    if cloud.is_empty() || res.is_nan() || res <= 0.0 {
        return None;
    }
    let bounds = cloud.bounds()?;
    // Pad the cloth by two particles so the border does not curl onto the data.
    let origin = [bounds.min[0] - 2.0 * res, bounds.min[1] - 2.0 * res];
    let nx = ((bounds.max[0] - origin[0]) / res).ceil() as usize + 3;
    let ny = ((bounds.max[1] - origin[1]) / res).ceil() as usize + 3;
    if nx.checked_mul(ny)? > 50_000_000 {
        return None; // resolution far too fine for the extent
    }
    let node = |p: &[f64; 3]| {
        let i = (((p[0] - origin[0]) / res).round() as usize).min(nx - 1);
        let j = (((p[1] - origin[1]) / res).round() as usize).min(ny - 1);
        j * nx + i
    };

    // Inverted height under each particle: the highest inverted point (i.e.
    // the lowest original point) that falls onto it.
    let mut floor = vec![f64::NEG_INFINITY; nx * ny];
    for p in &cloud.positions {
        let k = node(p);
        floor[k] = floor[k].max(-p[2]);
    }
    fill_from_neighbours(&mut floor, nx, ny);

    let start = -bounds.min[2] + 2.0 * res; // above every inverted point
    let mut height = vec![start; nx * ny];
    let mut previous = height.clone();
    let mut movable = vec![true; nx * ny];
    // Gravity scaled like CSF: displacement = g * dt^2 per step.
    let step = 0.2 * params.time_step * params.time_step;
    for _ in 0..params.max_iterations {
        let mut largest = 0.0f64;
        for k in 0..height.len() {
            if !movable[k] {
                continue;
            }
            let h = height[k];
            // Verlet integration with a little damping.
            let next = h + (h - previous[k]) * 0.99 - step;
            previous[k] = h;
            height[k] = next;
        }
        for _ in 0..params.rigidness.passes() {
            satisfy_springs(&mut height, &movable, nx, ny);
        }
        for k in 0..height.len() {
            if movable[k] && height[k] <= floor[k] {
                height[k] = floor[k];
                previous[k] = floor[k];
                movable[k] = false;
            }
            largest = largest.max((height[k] - previous[k]).abs());
        }
        // An absolute threshold like CSF's: scaling it by the resolution
        // would stop coarse cloths before they had fallen at all.
        if largest < 0.005 {
            break;
        }
    }

    // Ground: close to the cloth (bilinearly interpolated) in inverted height.
    let cloth_at = |x: f64, y: f64| {
        let fx = ((x - origin[0]) / res).clamp(0.0, (nx - 1) as f64);
        let fy = ((y - origin[1]) / res).clamp(0.0, (ny - 1) as f64);
        let (i, j) = (
            (fx.floor() as usize).min(nx - 2),
            (fy.floor() as usize).min(ny - 2),
        );
        let (tx, ty) = (fx - i as f64, fy - j as f64);
        let h = |i: usize, j: usize| height[j * nx + i];
        (1.0 - ty) * ((1.0 - tx) * h(i, j) + tx * h(i + 1, j))
            + ty * ((1.0 - tx) * h(i, j + 1) + tx * h(i + 1, j + 1))
    };
    Some(
        cloud
            .positions
            .iter()
            .map(|p| (-p[2] - cloth_at(p[0], p[1])).abs() <= params.class_threshold)
            .collect(),
    )
}

/// Pull each pair of 4-neighbours towards each other (half each if both
/// move, all the way if only one does).
fn satisfy_springs(height: &mut [f64], movable: &[bool], nx: usize, ny: usize) {
    let mut relax = |a: usize, b: usize| {
        let d = height[b] - height[a];
        match (movable[a], movable[b]) {
            (true, true) => {
                height[a] += 0.5 * d * 0.5;
                height[b] -= 0.5 * d * 0.5;
            }
            (true, false) => height[a] += 0.5 * d,
            (false, true) => height[b] -= 0.5 * d,
            (false, false) => {}
        }
    };
    for j in 0..ny {
        for i in 0..nx {
            let k = j * nx + i;
            if i + 1 < nx {
                relax(k, k + 1);
            }
            if j + 1 < ny {
                relax(k, k + nx);
            }
        }
    }
}

/// Give particles with no points under them the floor of the nearest
/// particle that has some (breadth-first), so the cloth still lands there.
fn fill_from_neighbours(floor: &mut [f64], nx: usize, ny: usize) {
    let mut queue: std::collections::VecDeque<usize> =
        (0..floor.len()).filter(|&k| floor[k].is_finite()).collect();
    while let Some(k) = queue.pop_front() {
        let (i, j) = (k % nx, k / nx);
        let mut visit = |n: usize| {
            if !floor[n].is_finite() {
                floor[n] = floor[k];
                queue.push_back(n);
            }
        };
        if i > 0 {
            visit(k - 1);
        }
        if i + 1 < nx {
            visit(k + 1);
        }
        if j > 0 {
            visit(k - nx);
        }
        if j + 1 < ny {
            visit(k + nx);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A gently rolling 60 x 60 m terrain with a 10 x 10 x 8 m building and
    /// a tree crown, sampled every 0.25 m. Returns the cloud and which
    /// points are truly ground.
    fn scene() -> (PointCloud, Vec<bool>) {
        let terrain = |x: f64, y: f64| 0.8 * (x / 12.0).sin() + 0.5 * (y / 9.0).cos();
        let mut positions = Vec::new();
        let mut truth = Vec::new();
        for j in 0..240 {
            for i in 0..240 {
                let (x, y) = (i as f64 * 0.25, j as f64 * 0.25);
                let ground = terrain(x, y);
                let on_building = (20.0..30.0).contains(&x) && (20.0..30.0).contains(&y);
                let in_tree = (x - 45.0).powi(2) + (y - 15.0).powi(2) < 16.0;
                if on_building {
                    positions.push([x, y, ground + 8.0]);
                    truth.push(false);
                } else if in_tree {
                    positions.push([x, y, ground + 6.0 + 0.3 * ((x * 7.0).sin())]);
                    truth.push(false);
                } else {
                    positions.push([x, y, ground]);
                    truth.push(true);
                }
            }
        }
        let cloud = PointCloud {
            positions,
            colors: None,
            attributes: Vec::new(),
        };
        (cloud, truth)
    }

    #[test]
    fn separates_ground_from_building_and_tree() {
        let (cloud, truth) = scene();
        let ground = csf(&cloud, CsfParams::default()).unwrap();
        let wrong = ground.iter().zip(&truth).filter(|(g, t)| g != t).count();
        // Allow a thin misclassified rim along object edges.
        assert!(
            wrong * 100 < cloud.len(),
            "{wrong} of {} misclassified",
            cloud.len()
        );
        // Every roof point is off-ground.
        let roof_as_ground = cloud
            .positions
            .iter()
            .zip(&ground)
            .filter(|(p, g)| **g && (21.0..29.0).contains(&p[0]) && (21.0..29.0).contains(&p[1]))
            .count();
        assert_eq!(roof_as_ground, 0);
    }

    #[test]
    fn coarse_cloth_on_a_large_sparse_scene() {
        // 3 x 3 km, a point every 20 m, 150 m of relief, a 60 m tall block.
        let mut positions = Vec::new();
        let mut truth = Vec::new();
        for j in 0..150 {
            for i in 0..150 {
                let (x, y) = (i as f64 * 20.0, j as f64 * 20.0);
                let ground = 75.0 * (x / 900.0).sin() * (y / 1200.0).cos() + 400.0;
                let tower = (1000.0..1200.0).contains(&x) && (1000.0..1200.0).contains(&y);
                positions.push([x, y, if tower { ground + 60.0 } else { ground }]);
                truth.push(!tower);
            }
        }
        let cloud = PointCloud {
            positions,
            colors: None,
            attributes: Vec::new(),
        };
        let params = CsfParams {
            cloth_resolution: 40.0,
            class_threshold: 3.0,
            ..CsfParams::default()
        };
        let ground = csf(&cloud, params).unwrap();
        let wrong = ground.iter().zip(&truth).filter(|(g, t)| g != t).count();
        assert!(
            wrong * 50 < cloud.len(),
            "{wrong} of {} misclassified",
            cloud.len()
        );
    }

    #[test]
    fn flat_ground_is_all_ground() {
        let positions: Vec<[f64; 3]> = (0..2500)
            .map(|k| [(k % 50) as f64, (k / 50) as f64, 3.0])
            .collect();
        let cloud = PointCloud {
            positions,
            colors: None,
            attributes: Vec::new(),
        };
        assert!(
            csf(&cloud, CsfParams::default())
                .unwrap()
                .iter()
                .all(|&g| g)
        );
    }

    #[test]
    fn rejects_bad_input() {
        assert!(csf(&PointCloud::default(), CsfParams::default()).is_none());
        let one = PointCloud {
            positions: vec![[0.0; 3]],
            colors: None,
            attributes: Vec::new(),
        };
        let params = CsfParams {
            cloth_resolution: 0.0,
            ..CsfParams::default()
        };
        assert!(csf(&one, params).is_none());
    }
}
