//! Ground extraction with the Cloth Simulation Filter (CSF; Zhang et al.,
//! "An Easy-to-Use Airborne LiDAR Data Filtering Method Based on Cloth
//! Simulation", Remote Sensing 2016).
//!
//! The cloud is turned upside down and a cloth of particles is dropped onto
//! it from above. Gravity pulls each particle down until it reaches the
//! (inverted) terrain under it; springs between neighbouring particles keep
//! the cloth from sagging into the pits that buildings and trees leave in
//! the inverted surface. Points close to the settled cloth are ground.
//!
//! The simulation follows the CSF library as bundled with CloudCompare's
//! CSF plugin, and gives the same classification (checked by
//! `scripts/validate_cloudcompare.py`).

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
    /// Spring correction factors `(single, double)` of the CSF library for
    /// its `rigidness` 3 / 2 / 1: the share of a height difference a particle
    /// takes when its neighbour is fixed, and each takes when both move.
    fn moves(self) -> (f64, f64) {
        match self {
            Self::Flat => (0.657, 0.468),
            Self::Relief => (0.51, 0.42),
            Self::Steep => (0.3, 0.3),
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
///
/// Follows the reference implementation (the CSF library, also used by
/// CloudCompare's CSF plugin): each particle stops at the height of the
/// point nearest to it, gravity and damping as in its Verlet step, springs
/// corrected with its rigidness-dependent factors, and iteration stops when
/// the cloth moves less than 0.005.
pub fn csf(cloud: &PointCloud, params: CsfParams) -> Option<Vec<bool>> {
    let res = params.cloth_resolution;
    if cloud.is_empty() || res.is_nan() || res <= 0.0 {
        return None;
    }
    let bounds = cloud.bounds()?;
    const BUFFER: usize = 2;
    let origin = [
        bounds.min[0] - BUFFER as f64 * res,
        bounds.min[1] - BUFFER as f64 * res,
    ];
    let nx = ((bounds.max[0] - bounds.min[0]) / res).floor() as usize + 2 * BUFFER;
    let ny = ((bounds.max[1] - bounds.min[1]) / res).floor() as usize + 2 * BUFFER;
    if nx.checked_mul(ny)? > 50_000_000 {
        return None; // resolution far too fine for the extent
    }

    // Heights are of the inverted cloud (-z). Each particle stops at the
    // height of the point nearest to it horizontally.
    let mut target = vec![f64::NEG_INFINITY; nx * ny];
    let mut nearest = vec![f64::INFINITY; nx * ny];
    for p in &cloud.positions {
        let col = ((p[0] - origin[0]) / res + 0.5) as usize;
        let row = ((p[1] - origin[1]) / res + 0.5) as usize;
        if col >= nx || row >= ny {
            continue;
        }
        let k = row * nx + col;
        let d = (p[0] - (origin[0] + col as f64 * res)).powi(2)
            + (p[1] - (origin[1] + row as f64 * res)).powi(2);
        if d < nearest[k] {
            nearest[k] = d;
            target[k] = -p[2];
        }
    }
    fill_by_scanline(&mut target, nx, ny);

    let start = -bounds.min[2] + 0.05; // just above every inverted point
    let mut height = vec![start; nx * ny];
    let mut previous = height.clone();
    let mut movable = vec![true; nx * ny];
    let dt2 = params.time_step * params.time_step;
    // Gravity times dt², added to each Verlet step.
    let gravity_step = -0.2 * dt2;
    let (single, double) = params.rigidness.moves();
    for _ in 0..params.max_iterations {
        for k in 0..height.len() {
            if movable[k] {
                let h = height[k];
                height[k] = h + (h - previous[k]) * (1.0 - DAMPING) + gravity_step;
                previous[k] = h;
            }
        }
        satisfy_springs(&mut height, &movable, nx, ny, single, double);
        let mut largest = 0.0f64;
        for k in 0..height.len() {
            if movable[k] {
                largest = largest.max((previous[k] - height[k]).abs());
            }
        }
        for k in 0..height.len() {
            if height[k] < target[k] {
                if movable[k] {
                    height[k] = target[k];
                }
                movable[k] = false;
            }
        }
        if largest != 0.0 && largest < 0.005 {
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
            .map(|p| (-p[2] - cloth_at(p[0], p[1])).abs() < params.class_threshold)
            .collect(),
    )
}

/// Velocity damping of the Verlet step.
const DAMPING: f64 = 0.01;

/// Pull each particle and its springs' other ends towards each other: by
/// `double` of the height difference each when both move, by `single` when
/// only one does. As in the CSF library, every particle is connected to its
/// 8 neighbours and to the 8 particles two steps away, and corrects each of
/// its springs in turn (so every spring is corrected from both ends), in the
/// library's order.
fn satisfy_springs(
    height: &mut [f64],
    movable: &[bool],
    nx: usize,
    ny: usize,
    single: f64,
    double: f64,
) {
    let (w, h) = (nx as isize, ny as isize);
    for y in 0..h {
        for x in 0..w {
            let k = (y * w + x) as usize;
            for (dx, dy) in spring_ends(x, y, w, h) {
                let m = ((y + dy) * w + x + dx) as usize;
                let d = height[m] - height[k];
                match (movable[k], movable[m]) {
                    (true, true) => {
                        height[k] += d * double;
                        height[m] -= d * double;
                    }
                    (true, false) => height[k] += d * single,
                    (false, true) => height[m] -= d * single,
                    (false, false) => {}
                }
            }
        }
    }
}

/// Offsets to the other ends of the springs of particle `(x, y)` on a `w` x
/// `h` cloth, in the order the CSF library creates them: its loop over
/// `(x, y)` connects each particle to `(x+s, y)`, `(x, y+s)`, `(x+s, y+s)`
/// and `(x+s, y)` to `(x, y+s)`, first for s = 1, then for s = 2.
fn spring_ends(x: isize, y: isize, w: isize, h: isize) -> impl Iterator<Item = (isize, isize)> {
    [1isize, 2].into_iter().flat_map(move |s| {
        let (left, down, right, up) = (x >= s, y >= s, x < w - s, y < h - s);
        [
            (left && down, (-s, -s)),
            (left, (-s, 0)),
            (left && up, (-s, s)),
            (down, (0, -s)),
            (down && right, (s, -s)),
            (right, (s, 0)),
            (up, (0, s)),
            (right && up, (s, s)),
        ]
        .into_iter()
        .filter_map(|(ok, d)| ok.then_some(d))
    })
}

/// Give particles with no point near them the height of the first particle
/// that has one, scanning right, left, down and up along the grid (as the
/// library does), then breadth-first for any still unset.
fn fill_by_scanline(target: &mut [f64], nx: usize, ny: usize) {
    let known: Vec<bool> = target.iter().map(|h| h.is_finite()).collect();
    let source = target.to_vec();
    for j in 0..ny {
        for i in 0..nx {
            let k = j * nx + i;
            if known[k] {
                continue;
            }
            let found = (i + 1..nx)
                .map(|x| j * nx + x)
                .chain((0..i).rev().map(|x| j * nx + x))
                .chain((0..j).rev().map(|y| y * nx + i))
                .chain((j + 1..ny).map(|y| y * nx + i))
                .find(|&m| known[m]);
            if let Some(m) = found {
                target[k] = source[m];
            }
        }
    }
    fill_from_neighbours(target, nx, ny);
}

/// Give particles still without a height that of the nearest particle with
/// one (breadth-first).
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
        // CloudCompare's CSF plugin (2.13, relief, resolution 1, threshold 0.5)
        // marks exactly these 55,207 points as ground.
        assert_eq!(ground.iter().filter(|&&g| g).count(), 55_207);
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
        // 3 x 3 km, a point every 20 m, 20 m of relief, a 60 m tall block.
        // (With much more relief the stiff cloth cannot follow the terrain,
        // in CloudCompare's CSF plugin as here.)
        let mut positions = Vec::new();
        let mut truth = Vec::new();
        for j in 0..150 {
            for i in 0..150 {
                let (x, y) = (i as f64 * 20.0, j as f64 * 20.0);
                let ground = 10.0 * (x / 900.0).sin() * (y / 1200.0).cos() + 400.0;
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
