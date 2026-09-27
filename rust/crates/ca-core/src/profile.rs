//! Cross-section profiles: the points within a band around a polyline drawn
//! in the XY plane, with their distance along it.

/// For each point within `half_width` (horizontally) of the polyline
/// `line` (XY vertices), `(index, distance along the line)`, in cloud order.
/// A point is measured against its nearest segment; points beyond the ends
/// are left out. Returns nothing for fewer than two distinct vertices.
pub fn profile(points: &[[f64; 3]], line: &[[f64; 2]], half_width: f64) -> Vec<(u32, f64)> {
    // Segments with their start distance, skipping repeated vertices.
    let mut segments: Vec<([f64; 2], [f64; 2], f64, f64)> = Vec::new();
    let mut along = 0.0;
    for pair in line.windows(2) {
        let (a, b) = (pair[0], pair[1]);
        let d = [b[0] - a[0], b[1] - a[1]];
        let len = d[0].hypot(d[1]);
        if len > 0.0 && len.is_finite() {
            segments.push((a, [d[0] / len, d[1] / len], len, along));
            along += len;
        }
    }
    if segments.is_empty() || half_width.is_nan() || half_width < 0.0 {
        return Vec::new();
    }
    let (mut lo, mut hi) = ([f64::INFINITY; 2], [f64::NEG_INFINITY; 2]);
    for v in line {
        for a in 0..2 {
            lo[a] = lo[a].min(v[a] - half_width);
            hi[a] = hi[a].max(v[a] + half_width);
        }
    }
    let mut out = Vec::new();
    for (i, p) in points.iter().enumerate() {
        if p[0] < lo[0] || p[0] > hi[0] || p[1] < lo[1] || p[1] > hi[1] {
            continue;
        }
        let mut best: Option<(f64, f64)> = None; // (offset, along)
        for &(a, dir, len, start) in &segments {
            let v = [p[0] - a[0], p[1] - a[1]];
            let t = v[0] * dir[0] + v[1] * dir[1];
            if !(0.0..=len).contains(&t) {
                continue;
            }
            let offset = (v[0] * dir[1] - v[1] * dir[0]).abs();
            if offset <= half_width && best.is_none_or(|(o, _)| offset < o) {
                best = Some((offset, start + t));
            }
        }
        if let Some((_, d)) = best {
            out.push((i as u32, d));
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid() -> Vec<[f64; 3]> {
        (0..10_000)
            .map(|k| {
                [
                    (k % 100) as f64 * 0.1,
                    (k / 100) as f64 * 0.1,
                    (k % 7) as f64,
                ]
            })
            .collect()
    }

    #[test]
    fn keeps_the_band_along_a_straight_line() {
        let points = grid();
        // Along y = 5 from x = 1 to x = 8, 0.05 either side: one grid row.
        let hits = profile(&points, &[[1.0, 5.0], [8.0, 5.0]], 0.05);
        assert_eq!(hits.len(), 71);
        for &(i, d) in &hits {
            let p = points[i as usize];
            assert!((p[1] - 5.0).abs() < 1e-9);
            assert!((d - (p[0] - 1.0)).abs() < 1e-9);
        }
        assert!(hits.windows(2).all(|w| w[0].0 < w[1].0));
    }

    #[test]
    fn distance_accumulates_around_a_corner() {
        let points = grid();
        let hits = profile(&points, &[[1.0, 1.0], [5.0, 1.0], [5.0, 4.0]], 0.01);
        // 41 points on the first leg, 30 more on the second (the corner once).
        assert_eq!(hits.len(), 71);
        let end = hits.iter().map(|h| h.1).fold(0.0, f64::max);
        assert!((end - 7.0).abs() < 1e-9);
        let corner = points
            .iter()
            .position(|p| p[0] == 5.0 && (p[1] - 1.0).abs() < 1e-9)
            .unwrap();
        let at = hits.iter().find(|h| h.0 as usize == corner).unwrap().1;
        assert!((at - 4.0).abs() < 1e-9);
    }

    #[test]
    fn degenerate_lines_give_nothing() {
        let points = grid();
        assert!(profile(&points, &[[1.0, 1.0]], 1.0).is_empty());
        assert!(profile(&points, &[[1.0, 1.0], [1.0, 1.0]], 1.0).is_empty());
        assert!(profile(&points, &[[1.0, 1.0], [2.0, 1.0]], -1.0).is_empty());
    }
}
