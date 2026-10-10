//! Fit lateral deviations in the trajectory frame, without a reference map.
//! Source positions/heights are retained and XY movement is capped at 0.5 m.
use super::{Evidence, ExtractedRoad};

/// SPD pentadiagonal Cholesky solve; O(n) time and storage.
/// Each row stores its diagonal and the two lower diagonals.
fn solve(mut a: Vec<[f64; 3]>, mut rhs: Vec<f64>) -> Option<Vec<f64>> {
    for k in 0..a.len() {
        if k >= 2 {
            a[k][2] /= a[k - 2][0];
        }
        if k >= 1 {
            a[k][1] = (a[k][1] - a[k][2] * a[k - 1][1]) / a[k - 1][0];
        }
        let diagonal = a[k][0] - a[k][1].powi(2) - a[k][2].powi(2);
        if !diagonal.is_finite() || diagonal <= 0.0 {
            return None;
        }
        a[k][0] = diagonal.sqrt();
        if k >= 1 {
            rhs[k] -= a[k][1] * rhs[k - 1];
        }
        if k >= 2 {
            rhs[k] -= a[k][2] * rhs[k - 2];
        }
        rhs[k] /= a[k][0];
    }
    for k in (0..a.len()).rev() {
        if k + 1 < a.len() {
            rhs[k] -= a[k + 1][1] * rhs[k + 1];
        }
        if k + 2 < a.len() {
            rhs[k] -= a[k + 2][2] * rhs[k + 2];
        }
        rhs[k] /= a[k][0];
    }
    rhs.iter().all(|v| v.is_finite()).then_some(rhs)
}

/// Protect observed sharp corners with locally straight support on both sides.
/// An isolated spike fails straightness; inferred zigzags are not corners.
fn corners(points: &[[f64; 3]], labels: &[Evidence]) -> Vec<bool> {
    let mut result = vec![false; points.len()];
    if points.len() < 5 {
        return result;
    }
    for k in 2..points.len() - 2 {
        if labels[k - 2..=k + 2].contains(&Evidence::WidthPrior) {
            continue;
        }
        let chord = |a: usize, b: usize| {
            let d = [points[b][0] - points[a][0], points[b][1] - points[a][1]];
            let length = d[0].hypot(d[1]);
            (length > 0.1).then_some([d[0] / length, d[1] / length])
        };
        let (Some(incoming), Some(outgoing)) = (chord(k - 2, k), chord(k, k + 2)) else {
            continue;
        };
        let angle = (incoming[0] * outgoing[1] - incoming[1] * outgoing[0])
            .atan2(incoming[0] * outgoing[0] + incoming[1] * outgoing[1])
            .abs();
        let straight = |a: usize, mid: usize, direction: [f64; 2]| {
            let d = [points[mid][0] - points[a][0], points[mid][1] - points[a][1]];
            (d[0] * direction[1] - d[1] * direction[0]).abs() <= 0.1
        };
        if angle >= 35.0_f64.to_radians()
            && straight(k - 2, k - 1, incoming)
            && straight(k, k + 1, outgoing)
        {
            result[k] = true;
        }
    }
    result
}

fn corrections(
    reference: &[[f64; 3]],
    normals: &[[f64; 2]],
    points: &[[f64; 3]],
    labels: &[Evidence],
) -> Option<Vec<f64>> {
    let offsets: Vec<_> = points
        .iter()
        .zip(reference)
        .zip(normals)
        .map(|((p, r), n)| (p[0] - r[0]) * n[0] + (p[1] - r[1]) * n[1])
        .collect();
    let weights = labels.iter().map(|label| match label {
        Evidence::Intensity | Evidence::RgbPaint => 2.0,
        Evidence::Curb => 1.0,
        Evidence::SupportEdge => 0.5,
        Evidence::WidthPrior => 0.25,
    });
    let mut matrix: Vec<_> = weights.map(|w| [w, 0.0, 0.0]).collect();
    let mut rhs = vec![0.0; points.len()];
    let corner = corners(points, labels);
    let mut fixed = vec![false; points.len()];
    for (k, &is_corner) in corner.iter().enumerate() {
        if is_corner {
            fixed[k - 2..=k + 2].fill(true);
        }
    }
    for k in 1..points.len() - 1 {
        if corner[k] {
            continue;
        }
        let distance = |a: usize, b: usize| {
            (reference[a][0] - reference[b][0])
                .hypot(reference[a][1] - reference[b][1])
                .max(0.1)
        };
        let a = distance(k - 1, k);
        let b = distance(k, k + 1);
        let scale = 2.0 / (a + b);
        let row = [scale / a, -scale / a - scale / b, scale / b];
        let curvature: f64 = row
            .iter()
            .zip(&offsets[k - 1..=k + 1])
            .map(|(c, v)| c * v)
            .sum();
        // Physical strength (2 m)^4; variable sampling uses metre distances.
        for (j, &c) in row.iter().enumerate() {
            matrix[k - 1 + j][0] += 16.0 * c * c;
            rhs[k - 1 + j] -= 16.0 * c * curvature;
        }
        matrix[k][1] += 16.0 * row[0] * row[1];
        matrix[k + 1][1] += 16.0 * row[1] * row[2];
        matrix[k + 1][2] += 16.0 * row[0] * row[2];
    }
    // Corrections of observed corner neighbourhoods are exactly zero.
    for k in 0..matrix.len() {
        for d in 1..=2 {
            if k >= d && (fixed[k] || fixed[k - d]) {
                matrix[k][d] = 0.0;
            }
        }
        if fixed[k] {
            matrix[k][0] = 1.0;
            rhs[k] = 0.0;
        }
    }
    solve(matrix, rhs)
}

pub(super) fn fit(road: &mut ExtractedRoad) -> (usize, f64) {
    if road.reference.len() < 3 {
        return (0, 0.0);
    }
    let normals: Vec<_> = (0..road.reference.len())
        .map(|k| {
            let a = road.reference[k.saturating_sub(1)];
            let b = road.reference[(k + 1).min(road.reference.len() - 1)];
            let length = (b[0] - a[0]).hypot(b[1] - a[1]).max(1e-12);
            [-(b[1] - a[1]) / length, (b[0] - a[0]) / length]
        })
        .collect();
    let original = road.boundaries.clone();
    for ((line, source), labels) in road
        .boundaries
        .iter_mut()
        .zip(&original)
        .zip(&road.evidence)
    {
        let Some(delta) = corrections(&road.reference, &normals, source, labels) else {
            continue;
        };
        for ((p, &d), n) in line.iter_mut().zip(&delta).zip(&normals) {
            let d = d.clamp(-0.5, 0.5);
            if d.abs() > 1e-6 {
                p[0] += n[0] * d;
                p[1] += n[1] * d;
            }
        }
    }
    // Reject fits that cross neighbouring lines at a sampled cross-section.
    for (k, n) in normals.iter().enumerate() {
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

    #[test]
    fn supported_sharp_corner_is_not_rounded_like_an_inferred_kink() {
        let source: Vec<_> = (0..=12)
            .map(|k| {
                let x = k as f64 * 2.0;
                [x, (x - 12.0).max(0.0), 4.0]
            })
            .collect();
        let mut measured = road(source.clone());
        fit(&mut measured);
        assert_eq!(measured.boundaries[0], source);
        let mut inferred = road(source);
        inferred.evidence[0].fill(Evidence::WidthPrior);
        assert!(fit(&mut inferred).0 > 0);
        assert!(inferred.boundaries[0][6][1] > 0.1);
        assert_eq!(inferred.evidence[0][6], Evidence::WidthPrior);
    }

    #[test]
    fn reference_curve_and_parallel_arc_are_preserved() {
        let arc = |radius: f64| {
            (0..21)
                .map(|k| {
                    let angle = k as f64 * 0.05;
                    [radius * angle.cos(), radius * angle.sin(), k as f64 * 0.01]
                })
                .collect::<Vec<_>>()
        };
        let mut r = road(arc(20.0));
        r.reference = arc(20.0);
        let source = r.boundaries.clone();
        assert_eq!(fit(&mut r), (0, 0.0));
        assert_eq!(r.boundaries, source);
        r.boundaries = vec![arc(22.0)];
        fit(&mut r);
        for (k, p) in r.boundaries[0].iter().enumerate() {
            assert!((p[0].hypot(p[1]) - 22.0).abs() < 0.001);
            assert_eq!(p[2], k as f64 * 0.01);
        }
    }

    #[test]
    fn irregular_station_spacing_and_large_rotated_coordinates_keep_the_same_fit() {
        let source: Vec<_> = (0..17)
            .map(|k| {
                let x = k as f64 + (k as f64 * 0.7).sin() * 0.2;
                [x, 1.75 + if k % 2 == 0 { 0.2 } else { -0.2 }, 4.0]
            })
            .collect();
        let mut local = road(source);
        let mut rotated = local.clone();
        let transform = |p: [f64; 3]| {
            let angle = 1.3_f64;
            [
                50000.0 + angle.cos() * p[0] - angle.sin() * p[1],
                70000.0 + angle.sin() * p[0] + angle.cos() * p[1],
                p[2],
            ]
        };
        for p in &mut rotated.reference {
            *p = transform(*p);
        }
        for p in &mut rotated.boundaries[0] {
            *p = transform(*p);
        }
        fit(&mut local);
        fit(&mut rotated);
        for (a, b) in local.boundaries[0].iter().zip(&rotated.boundaries[0]) {
            let a = transform(*a);
            assert!((a[0] - b[0]).hypot(a[1] - b[1]) < 1e-8);
            assert_eq!(a[2], b[2]);
        }
    }

    #[test]
    fn different_evidence_weights_never_invert_neighbouring_boundaries() {
        let source: Vec<_> = (0..17)
            .map(|k| {
                [
                    k as f64 * 2.0,
                    1.0 + if k % 2 == 0 { 0.4 } else { -0.4 },
                    4.0,
                ]
            })
            .collect();
        let mut r = road(source.clone());
        r.boundaries
            .push(source.iter().map(|p| [p[0], p[1] - 0.2, p[2]]).collect());
        r.evidence[0].fill(Evidence::WidthPrior);
        r.evidence.push(vec![Evidence::Curb; source.len()]);
        fit(&mut r);
        for (a, b) in r.boundaries[0].iter().zip(&r.boundaries[1]) {
            assert!(a[1] - b[1] > 0.1);
        }
    }
}
