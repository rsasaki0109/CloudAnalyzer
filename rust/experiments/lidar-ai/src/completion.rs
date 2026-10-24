//! Fixed, opt-in completion experiment. Original nearest-only decisions stay.
//! Seeds are immutable original owners; added points never become new seeds.
use crate::projection::{self, CLASSES, HEIGHT, MAX_POINTS, PIXELS, Projection, WIDTH};
use serde::Serialize;
use std::collections::HashMap;

pub const POLICY: &str = "near-surface-two-seeds-v1";
pub const RADIUS_M: f64 = 0.15;
pub const MIN_SEED_SEPARATION_M: f64 = 0.02;
pub const SEED_SCORE: f32 = 0.99;
pub const CONFLICT_SCORE: f32 = 0.9;
pub const MAX_NEIGHBORS: usize = 256;

#[derive(Debug, Default, Serialize)]
pub struct CompletionReport {
    pub original_removed: usize,
    pub added: usize,
    pub total_removed: usize,
    pub retained: usize,
    pub distance_rejected: usize,
    pub crowded_rejected: usize,
    pub conflict_rejected: usize,
    pub support_rejected: usize,
}
#[derive(Clone, Copy)]
struct Owner {
    index: usize,
    class: usize,
    score: f32,
}
fn cell(p: &[f32; 4]) -> (i32, i32, i32) {
    (
        (f64::from(p[0]) / RADIUS_M).floor() as i32,
        (f64::from(p[1]) / RADIUS_M).floor() as i32,
        (f64::from(p[2]) / RADIUS_M).floor() as i32,
    )
}
fn squared(a: &[f32; 4], b: &[f32; 4]) -> f64 {
    (0..3)
        .map(|i| (f64::from(a[i]) - f64::from(b[i])).powi(2))
        .sum()
}
/// Same bounded spherical pixel convention as the original native projection.
fn pixel(p: &[f32; 4]) -> Option<usize> {
    if !p.iter().all(|v| v.is_finite()) {
        return None;
    }
    let [x, y, z, _] = p.map(f64::from);
    let r = x.hypot(y).hypot(z);
    if !(1e-6..=100.).contains(&r) {
        return None;
    }
    let pitch = (z / r).asin();
    let (lower, upper) = ((-25_f64).to_radians(), 3_f64.to_radians());
    if !(lower..=upper).contains(&pitch) {
        return None;
    }
    let col = ((0.5 * (-y.atan2(x) / std::f64::consts::PI + 1.) * WIDTH as f64).floor() as usize)
        .min(WIDTH - 1);
    let row =
        (((upper - pitch) / (upper - lower) * HEIGHT as f64).floor() as usize).min(HEIGHT - 1);
    Some(row * WIDTH + col)
}

pub fn complete(
    points: &[[f32; 4]],
    projected: &Projection,
    probabilities: &[f32],
    targets: &[usize],
    threshold: f32,
) -> Result<(Vec<bool>, CompletionReport), String> {
    if points.is_empty()
        || points.len() > MAX_POINTS
        || projected.report.points != points.len()
        || projected.owners.len() != PIXELS
        || projected
            .owners
            .iter()
            .flatten()
            .any(|&i| i >= points.len())
    {
        return Err("completion requires a matching original projection".into());
    }
    let (mut mask, baseline) =
        projection::removal_mask(projected, probabilities, targets, threshold)?;
    let mut owners = vec![None; PIXELS];
    let mut grid: HashMap<(i32, i32, i32), Vec<Owner>> = HashMap::new();
    let mut classified = vec![false; points.len()];
    for (pix, index) in projected.owners.iter().enumerate() {
        if let Some(index) = *index {
            classified[index] = true;
            let mut best = (0, -1_f32);
            for class in 0..CLASSES {
                let value = probabilities[class * PIXELS + pix];
                if value > best.1 {
                    best = (class, value);
                }
            }
            let owner = Owner {
                index,
                class: best.0,
                score: best.1,
            };
            owners[pix] = Some(owner);
            if owner.score >= CONFLICT_SCORE {
                grid.entry(cell(&points[index])).or_default().push(owner);
            }
        }
    }
    let mut report = CompletionReport {
        original_removed: baseline.removed,
        ..Default::default()
    };
    for (i, point) in points.iter().enumerate() {
        if classified[i] {
            continue;
        }
        let Some(pix) = pixel(point) else {
            continue;
        };
        let Some(first) = owners[pix] else {
            continue;
        };
        // Both seeds must also satisfy the baseline selection. No chains.
        if !mask[first.index] || first.score < SEED_SCORE {
            continue;
        }
        if squared(point, &points[first.index]) > RADIUS_M * RADIUS_M {
            report.distance_rejected += 1;
            continue;
        }
        let (x, y, z) = cell(point);
        let mut neighborhood = Vec::new();
        let mut crowded = false;
        'cells: for dx in -1..=1 {
            for dy in -1..=1 {
                for dz in -1..=1 {
                    if let Some(list) = grid.get(&(x + dx, y + dy, z + dz)) {
                        if neighborhood.len() + list.len() > MAX_NEIGHBORS {
                            crowded = true;
                            break 'cells;
                        }
                        neighborhood.extend_from_slice(list);
                    }
                }
            }
        }
        if crowded {
            report.crowded_rejected += 1;
            continue;
        }
        let mut conflict = false;
        let mut support = false;
        // Never stop at the first supporter: a later conflicting owner vetoes it.
        for neighbor in neighborhood {
            if squared(point, &points[neighbor.index]) > RADIUS_M * RADIUS_M {
                continue;
            }
            if neighbor.class != first.class {
                conflict = true;
                break;
            }
            if neighbor.index != first.index
                && neighbor.score >= SEED_SCORE
                && mask[neighbor.index]
                && squared(&points[first.index], &points[neighbor.index])
                    >= MIN_SEED_SEPARATION_M * MIN_SEED_SEPARATION_M
            {
                support = true;
            }
        }
        if conflict {
            report.conflict_rejected += 1;
        } else if !support {
            report.support_rejected += 1;
        } else {
            mask[i] = true;
            report.added += 1;
        }
    }
    report.total_removed = report.original_removed + report.added;
    report.retained = points.len() - report.total_removed;
    Ok((mask, report))
}

#[cfg(test)]
mod tests {
    use super::*;
    fn scores(p: &Projection, changes: &[(usize, usize, f32)]) -> Vec<f32> {
        let mut values = vec![0.; CLASSES * PIXELS];
        values[..PIXELS].fill(1.);
        for &(index, class, score) in changes {
            let pixel = p.owners.iter().position(|&o| o == Some(index)).unwrap();
            values[pixel] = 1. - score;
            values[class * PIXELS + pixel] = score;
        }
        values
    }
    #[test]
    fn two_separated_original_seeds_complete_only_close_collisions_and_never_chain() {
        let points = [
            [10., 0., 0., 0.5],
            [10.02, 0., 0., 0.5],
            [10., 0.05, 0., 0.5],
            [10.14, 0., 0., 0.5],
            [10.27, 0., 0., 0.5],
            [0., 0., 10., 0.5],
            [f32::NAN, 0., 0., 0.5],
        ];
        let p = projection::project(&points).unwrap();
        let values = scores(&p, &[(0, 1, 0.999), (2, 1, 0.999)]);
        let (mask, r) = complete(&points, &p, &values, &[1], 0.9).unwrap();
        assert_eq!(mask, [true, true, true, true, false, false, false]);
        assert_eq!((r.original_removed, r.added, r.total_removed), (2, 2, 4));
    }
    #[test]
    fn a_later_conflicting_owner_vetoes_a_good_pair() {
        let points = [
            [10., 0., 0., 0.5],
            [10.02, 0., 0., 0.5],
            [10., 0.05, 0., 0.5],
            [10., -0.06, 0., 0.5],
        ];
        let p = projection::project(&points).unwrap();
        let values = scores(&p, &[(0, 1, 0.999), (2, 1, 0.999), (3, 9, 0.95)]);
        let (mask, r) = complete(&points, &p, &values, &[1], 0.9).unwrap();
        assert_eq!(mask, [true, false, true, false]);
        assert_eq!(r.conflict_rejected, 1);
        let values = scores(&p, &[(0, 1, 0.999), (2, 1, 0.999), (3, 9, 0.8)]);
        assert!(complete(&points, &p, &values, &[1], 0.9).unwrap().0[1]);
    }
    #[test]
    fn low_confidence_single_and_duplicate_position_seeds_do_not_complete() {
        let points = [
            [10., 0., 0., 0.5],
            [10.02, 0., 0., 0.5],
            [10., 0.05, 0., 0.5],
        ];
        let p = projection::project(&points).unwrap();
        for changes in [
            vec![(0, 1, 0.999)],
            vec![(0, 1, 0.999), (2, 1, 0.98)],
            vec![(0, 1, 0.98), (2, 1, 0.999)],
        ] {
            assert!(
                !complete(&points, &p, &scores(&p, &changes), &[1], 0.9)
                    .unwrap()
                    .0[1]
            );
        }
        let points = [
            [-10., 0., 0., 0.5],
            [-10., -0., 0., 0.5],
            [-10.02, 0., 0., 0.5],
        ];
        let p = projection::project(&points).unwrap();
        assert!(
            !complete(
                &points,
                &p,
                &scores(&p, &[(0, 1, 0.999), (1, 1, 0.999)]),
                &[1],
                0.9
            )
            .unwrap()
            .0[2]
        );
    }
    #[test]
    fn malformed_probability_and_projection_contracts_fail_without_a_mask() {
        let points = [[10., 0., 0., 0.5]];
        let p = projection::project(&points).unwrap();
        assert!(complete(&points, &p, &[], &[1], 0.9).is_err());
        assert!(complete(&[], &p, &scores(&p, &[(0, 1, 0.999)]), &[1], 0.9).is_err());
    }
    #[test]
    fn crowded_cells_abstain_rather_than_ignoring_a_potential_conflict() {
        let mut points = vec![[0.5, 0., 0., 0.5], [0.501, 0., 0., 0.5]];
        for y in -10..=10 {
            for z in -20..=0 {
                points.push([0.5, y as f32 * 0.005, z as f32 * 0.005, 0.5]);
            }
        }
        let p = projection::project(&points).unwrap();
        let changes: Vec<_> = p.owners.iter().flatten().map(|&i| (i, 1, 0.999)).collect();
        let (mask, r) = complete(&points, &p, &scores(&p, &changes), &[1], 0.9).unwrap();
        assert!(!mask[1]);
        assert!(r.crowded_rejected > 0);
    }
}
