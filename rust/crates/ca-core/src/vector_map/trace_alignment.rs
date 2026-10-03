//! Conservative source-only translation of straight operator traces.
use super::{BuildOptions, SurfaceIndex, quantile};
use crate::PointCloud;
use serde::Serialize;

#[derive(Debug, Clone, Serialize)]
pub struct TraceAlignmentReport {
    pub applied: bool,
    pub reason: String,
    pub sampled_sections: usize,
    pub curb_pair_sections: usize,
    pub ambiguous_sections: usize,
    pub longest_supported_run: usize,
    pub median_lateral_shift_m: Option<f64>,
    pub interquartile_range_m: Option<f64>,
    pub shift_xy: [f64; 2],
}

/// The band edges must each have TWO bounded raised outside bins and an inside
/// road bin. Missing returns, walls and coverage limits cannot close a band.
fn curb(surface: &[Option<f64>], edge: usize, outward: isize, o: &BuildOptions) -> bool {
    let Some(base) = surface[edge] else {
        return false;
    };
    let at = |step: isize| {
        edge.checked_add_signed(step)
            .and_then(|i| surface.get(i).copied().flatten())
            .map(|z| z - base)
    };
    (1..=2).all(|q| at(outward * q).is_some_and(|z| z >= o.curb_height && z <= o.curb_height + 0.3))
        && (1..=2).any(|q| at(-outward * q).is_some_and(|z| z.abs() <= o.curb_height.max(0.05)))
}

pub(super) fn align(
    cloud: &PointCloud,
    line: &mut [[f64; 3]],
    o: &BuildOptions,
) -> TraceAlignmentReport {
    let mut report = TraceAlignmentReport {
        applied: false,
        reason: "no stable paired curb corridor".into(),
        sampled_sections: line.len(),
        curb_pair_sections: 0,
        ambiguous_sections: 0,
        longest_supported_run: 0,
        median_lateral_shift_m: None,
        interquartile_range_m: None,
        shift_xy: [0.0; 2],
    };
    let a = line[0];
    let b = line[line.len() - 1];
    let length = (b[0] - a[0]).hypot(b[1] - a[1]);
    if length < 0.1 {
        report.reason = "trace is not straight".into();
        return report;
    }
    let dir = [(b[0] - a[0]) / length, (b[1] - a[1]) / length];
    if line.windows(2).any(|pair| {
        let dx = pair[1][0] - pair[0][0];
        let dy = pair[1][1] - pair[0][1];
        (dx * dir[0] + dy * dir[1]) / dx.hypot(dy) < 5_f64.to_radians().cos()
    }) {
        report.reason = "trace is not straight within five degrees".into();
        return report;
    }
    let width = (o.forward_lanes + o.backward_lanes) as f64 * o.lane_width;
    let left = if o.left_hand_traffic {
        o.lane_width * 0.5
    } else {
        width - o.lane_width * 0.5
    };
    let reach = 2.0 * width;
    let index = SurfaceIndex::new(cloud, line, reach + o.half_window);
    let mut shifts = Vec::new();
    let mut run = 0;
    for &p in line.iter() {
        let bins = index.slice(p, dir, -reach, reach, o);
        let surface: Vec<_> = bins
            .iter()
            .map(|ids| {
                if ids.len() < o.min_bin_points {
                    return None;
                }
                quantile(
                    &mut ids
                        .iter()
                        .map(|&i| cloud.positions[i][2])
                        .collect::<Vec<_>>(),
                    0.2,
                )
            })
            .collect();
        let lateral = |i: usize| -reach + (i as f64 + 0.5) * o.bin_width;
        // Sensor pose Z is intentionally ignored. Compare bands to the low
        // source surface under the trace, so other ground levels cannot win.
        let ground = quantile(
            &mut surface
                .iter()
                .enumerate()
                .filter(|(i, _)| lateral(*i).abs() < 0.8)
                .filter_map(|(_, z)| *z)
                .collect::<Vec<_>>(),
            0.5,
        );
        let mut candidates = Vec::new();
        let mut start = 0;
        for end in 1..=surface.len() {
            if end < surface.len()
                && surface[end]
                    .zip(surface[end - 1])
                    .is_some_and(|(a, b)| (a - b).abs() <= o.curb_height.min(0.08))
            {
                continue;
            }
            if let Some(base) = surface[start] {
                let right = lateral(start);
                let outer_left = lateral(end - 1);
                let band_width = outer_left - right;
                if band_width + 1e-9 >= width
                    && band_width <= width + 2.0 * o.search_margin
                    && right <= o.lane_width * 0.5
                    && outer_left >= -o.lane_width * 0.5
                    && ground.is_some_and(|z| (z - base).abs() <= 0.3)
                    && curb(&surface, start, -1, o)
                    && curb(&surface, end - 1, 1, o)
                {
                    candidates.push(0_f64.clamp(
                        (right + width - left).min(outer_left - left),
                        outer_left - left,
                    ));
                }
            }
            start = end;
        }
        if candidates.len() == 1 {
            shifts.push(candidates[0]);
            report.curb_pair_sections += 1;
            run += 1;
            report.longest_supported_run = report.longest_supported_run.max(run);
        } else {
            report.ambiguous_sections += usize::from(candidates.len() > 1);
            run = 0;
        }
    }
    if report.ambiguous_sections > 0
        || report.curb_pair_sections * 2 <= line.len()
        || report.longest_supported_run < 3
    {
        report.reason =
            "paired curbs need a majority of sections and three consecutive observations".into();
        return report;
    }
    let shift = quantile(&mut shifts, 0.5).unwrap();
    let iqr = quantile(&mut shifts, 0.75).unwrap() - quantile(&mut shifts, 0.25).unwrap();
    report.median_lateral_shift_m = Some(shift);
    report.interquartile_range_m = Some(iqr);
    if iqr > 2.0 * o.bin_width + 1e-9 || shift.abs() > width {
        report.reason =
            "paired curb offsets are inconsistent or exceed configured road width".into();
        return report;
    }
    if shift.abs() < o.bin_width {
        report.reason = "configured lane width already fits inside the paired curbs".into();
        return report;
    }
    report.applied = true;
    report.reason =
        "minimum translation into a stable source curb pair; lane layout remains a reviewed prior"
            .into();
    report.shift_xy = [-dir[1] * shift, dir[0] * shift];
    for p in line {
        p[0] += report.shift_xy[0];
        p[1] += report.shift_xy[1];
    }
    report
}

#[cfg(test)]
mod tests {
    use super::*;
    fn scene(raised: f64, missing: bool, supported_length: f64) -> PointCloud {
        let mut cloud = PointCloud::default();
        for x in 0..=320 {
            for y in -140..=140 {
                let x = x as f64 * 0.1;
                let y = y as f64 * 0.1 + 0.03;
                if missing && y > 5.4 {
                    continue;
                }
                let curb = !(-1.8..=5.4).contains(&y) && x <= supported_length;
                cloud
                    .positions
                    .push([x, y, 2.0 + if curb { raised } else { 0.0 }]);
            }
        }
        cloud
    }
    #[test]
    fn source_pair_translates_xy_preserving_explicit_width_and_sensor_z() {
        let cloud = scene(0.2, false, 32.0);
        let mut line = super::super::resample(&[[2., 0., 80.], [30., 0., 80.]], 2.).unwrap();
        let report = align(&cloud, &mut line, &BuildOptions::default());
        assert!(report.applied, "{report:?}");
        assert!((report.shift_xy[1] - 3.55).abs() < 0.21);
        assert!(line.iter().all(|p| p[2] == 80.0));
        let before = cloud.positions.clone();
        let poses = [[2., 0., 80.], [30., 0., 80.]];
        let (_, legacy) = super::super::extract(&cloud, &poses, &BuildOptions::default()).unwrap();
        assert!(legacy.trace_alignment.is_none());
        let options = BuildOptions {
            align_trace_to_curbs: true,
            fit_source_surface: true,
            ..BuildOptions::default()
        };
        let (roads, report) = super::super::extract(&cloud, &poses, &options).unwrap();
        assert!(report.trace_alignment.unwrap().applied);
        assert!(!roads.is_empty());
        assert!(
            roads
                .iter()
                .all(|r| r.boundaries.len() == 3
                    && r.boundaries.iter().flatten().all(|p| p[2] < 2.3))
        );
        assert_eq!(cloud.positions, before);
    }
    #[test]
    fn one_curb_walls_missing_majority_and_curved_traces_are_held() {
        let original = super::super::resample(&[[2., 0., 50.], [30., 0., 50.]], 2.).unwrap();
        for cloud in [
            scene(0.2, true, 32.),
            scene(1.5, false, 32.),
            scene(0.2, false, 6.),
        ] {
            let mut line = original.clone();
            assert!(!align(&cloud, &mut line, &BuildOptions::default()).applied);
            assert_eq!(line, original);
        }
        let mut line = vec![[2., 0., 50.], [10., 0., 50.], [30., 5., 50.]];
        let original = line.clone();
        assert!(!align(&scene(0.2, false, 32.), &mut line, &BuildOptions::default()).applied);
        assert_eq!(line, original);
    }
    #[test]
    fn already_inside_and_too_wide_lane_priors_are_held() {
        let cloud = scene(0.2, false, 32.);
        let mut line = vec![[2., 3.6, 50.], [30., 3.6, 50.]];
        assert!(!align(&cloud, &mut line, &BuildOptions::default()).applied);
        let mut line = vec![[2., 0., 50.], [30., 0., 50.]];
        let options = BuildOptions {
            forward_lanes: 3,
            backward_lanes: 3,
            ..BuildOptions::default()
        };
        assert!(!align(&cloud, &mut line, &options).applied);
    }

    #[test]
    fn rotated_large_coordinates_right_hand_and_inconsistent_curbs() {
        let angle = 0.7_f64;
        let transform = |p: [f64; 3]| {
            [
                500_000.0 + p[0] * angle.cos() - p[1] * angle.sin(),
                4_000_000.0 + p[0] * angle.sin() + p[1] * angle.cos(),
                p[2],
            ]
        };
        let mut cloud = scene(0.2, false, 32.0);
        for p in &mut cloud.positions {
            p[1] = -p[1];
            *p = transform(*p);
        }
        let mut line =
            super::super::resample(&[transform([2., 0., 70.]), transform([30., 0., 70.])], 2.)
                .unwrap();
        let options = BuildOptions {
            left_hand_traffic: false,
            ..BuildOptions::default()
        };
        let report = align(&cloud, &mut line, &options);
        assert!(report.applied, "{report:?}");
        assert!((report.median_lateral_shift_m.unwrap() + 3.55).abs() < 0.21);
        assert!((report.shift_xy[0] - 3.55 * angle.sin()).abs() < 0.21);
        assert!(line.iter().all(|p| p[2] == 70.));
        let mut cloud = scene(0.2, false, 32.);
        for p in &mut cloud.positions {
            if p[0] > 16. {
                p[1] += 1.5;
            }
        }
        let mut line = super::super::resample(&[[2., 0., 70.], [30., 0., 70.]], 2.).unwrap();
        let original = line.clone();
        let report = align(&cloud, &mut line, &BuildOptions::default());
        assert!(!report.applied, "{report:?}");
        assert!(report.interquartile_range_m.unwrap() > 0.4);
        assert_eq!(line, original);
    }
}
