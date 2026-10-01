//! Local geometry fitting, independent of any reference map. Candidate source
//! positions are retained; only XY is fitted, with at most 0.5 m displacement.

use super::ExtractedRoad;

fn solve(mut a: [[f64; 4]; 3]) -> Option<[f64; 3]> {
    for k in 0..3 {
        let pivot = (k..3).max_by(|&i, &j| a[i][k].abs().total_cmp(&a[j][k].abs()))?;
        if a[pivot][k].abs() < 1e-9 {
            return None;
        }
        a.swap(k, pivot);
        let scale = a[k][k];
        for v in &mut a[k][k..] {
            *v /= scale;
        }
        let pivot_row = a[k];
        for (i, row) in a.iter_mut().enumerate() {
            if i == k {
                continue;
            }
            let scale = row[k];
            for (value, pivot) in row[k..].iter_mut().zip(&pivot_row[k..]) {
                *value -= scale * pivot;
            }
        }
    }
    Some([a[0][3], a[1][3], a[2][3]])
}

fn local(points: &[[f64; 3]], k: usize) -> Option<[f64; 2]> {
    let start = k.saturating_sub(2);
    let end = (k + 2).min(points.len() - 1);
    let weight = |i: usize| (3 - k.abs_diff(i)) as f64;
    let sum: f64 = (start..=end).map(weight).sum();
    let mean: [f64; 2] =
        std::array::from_fn(|j| (start..=end).map(|i| points[i][j] * weight(i)).sum::<f64>() / sum);
    let mut xx = 0.0;
    let mut xy = 0.0;
    let mut yy = 0.0;
    for (i, p) in points.iter().enumerate().take(end + 1).skip(start) {
        let x = p[0] - mean[0];
        let y = p[1] - mean[1];
        xx += weight(i) * x * x;
        xy += weight(i) * x * y;
        yy += weight(i) * y * y;
    }
    if xx + yy < 1e-12 {
        return None;
    }
    let angle = 0.5 * (2.0 * xy).atan2(xx - yy);
    let axis = [angle.cos(), angle.sin()];
    let normal = [-axis[1], axis[0]];
    let project =
        |p: [f64; 3], dir: [f64; 2]| (p[0] - mean[0]) * dir[0] + (p[1] - mean[1]) * dir[1];
    let xp = project(points[k], axis);
    let mut yp = 0.0;
    if end - start >= 3 {
        let scale = (start..=end)
            .map(|i| project(points[i], axis).abs())
            .fold(0.0, f64::max)
            .max(1e-6);
        let mut matrix = [[0.0; 4]; 3];
        for (i, &p) in points.iter().enumerate().take(end + 1).skip(start) {
            let x = project(p, axis) / scale;
            let y = project(p, normal);
            let row = [1.0, x, x * x];
            for a in 0..3 {
                for b in 0..3 {
                    matrix[a][b] += weight(i) * row[a] * row[b];
                }
                matrix[a][3] += weight(i) * row[a] * y;
            }
        }
        let beta = solve(matrix)?;
        let x = xp / scale;
        yp = beta[0] + beta[1] * x + beta[2] * x * x;
    }
    Some([
        mean[0] + axis[0] * xp + normal[0] * yp,
        mean[1] + axis[1] * xp + normal[1] * yp,
    ])
}

pub(super) fn fit(road: &mut ExtractedRoad) -> (usize, f64) {
    if road.reference.len() < 3 {
        return (0, 0.0);
    }
    let original = road.boundaries.clone();
    for (line, source) in road.boundaries.iter_mut().zip(&original) {
        for (k, p) in line.iter_mut().enumerate() {
            let Some(target) = local(source, k) else {
                continue;
            };
            let delta = [target[0] - p[0], target[1] - p[1]];
            let length = delta[0].hypot(delta[1]);
            if length <= 1e-6 {
                continue;
            }
            let scale = (0.5 / length).min(1.0);
            p[0] += delta[0] * scale;
            p[1] += delta[1] * scale;
        }
    }
    // Reject fits that cross neighboring lines at a sampled cross-section.
    for k in 0..road.reference.len() {
        let a = road.reference[k.saturating_sub(1)];
        let b = road.reference[(k + 1).min(road.reference.len() - 1)];
        let length = (b[0] - a[0]).hypot(b[1] - a[1]).max(1e-12);
        let n = [-(b[1] - a[1]) / length, (b[0] - a[0]) / length];
        if road.boundaries.windows(2).any(|pair| {
            (pair[0][k][0] - pair[1][k][0]) * n[0] + (pair[0][k][1] - pair[1][k][1]) * n[1] <= 0.1
        }) {
            for (line, source) in road.boundaries.iter_mut().zip(&original) {
                line[k] = source[k];
            }
        }
    }
    let mut count = 0;
    let mut maximum: f64 = 0.0;
    for (line, source) in road.boundaries.iter().zip(&original) {
        for (p, q) in line.iter().zip(source) {
            let distance = (p[0] - q[0]).hypot(p[1] - q[1]);
            if distance > 1e-6 {
                count += 1;
                maximum = maximum.max(distance);
            }
        }
    }
    if count > 0 {
        road.source_boundaries = Some(original);
    }
    (count, maximum)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector_map::Evidence;

    fn road(line: Vec<[f64; 3]>) -> ExtractedRoad {
        ExtractedRoad {
            reference: line.iter().map(|p| [p[0], 0.0, p[2]]).collect(),
            evidence: vec![vec![Evidence::Curb; line.len()]],
            boundaries: vec![line],
            source_boundaries: None,
        }
    }

    #[test]
    fn fitting_reduces_noise_without_flattening_a_bend_or_changing_sources() {
        let curve = |x: f64| 0.01 * (x - 20.0).powi(2);
        let source: Vec<_> = (0..=20)
            .map(|k| {
                let x = k as f64 * 2.0;
                [x, curve(x) + if k % 2 == 0 { 0.2 } else { -0.2 }, x * 0.03]
            })
            .collect();
        let mut r = road(source.clone());
        let labels = r.evidence.clone();
        let (count, maximum) = fit(&mut r);
        assert!(count > 0 && maximum <= 0.5);
        assert_eq!(r.source_boundaries.as_ref().unwrap()[0], source);
        assert_eq!(r.evidence, labels);
        let mean_error = r.boundaries[0][3..18]
            .iter()
            .map(|p| (p[1] - curve(p[0])).abs())
            .sum::<f64>()
            / 15.0;
        assert!(mean_error < 0.1, "{mean_error}");
        for (p, q) in r.boundaries[0].iter().zip(&source) {
            assert_eq!(p[2], q[2]);
            assert!((p[0] - q[0]).hypot(p[1] - q[1]) <= 0.5 + 1e-12);
        }
    }

    #[test]
    fn fitting_caps_large_outliers_and_leaves_straight_lines_unchanged() {
        let line: Vec<_> = (0..=10).map(|k| [k as f64 * 2.0, 0.0, 4.0]).collect();
        let mut straight = road(line.clone());
        assert_eq!(fit(&mut straight), (0, 0.0));
        assert_eq!(straight.boundaries[0], line);
        assert!(straight.source_boundaries.is_none());
        let mut outlier = line;
        outlier[5][1] = 4.0;
        let mut r = road(outlier);
        let (_, maximum) = fit(&mut r);
        assert!((maximum - 0.5).abs() < 1e-12);
        assert_eq!(r.boundaries[0][5][2], 4.0);
    }
}
