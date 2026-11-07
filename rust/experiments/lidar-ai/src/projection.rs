//! Bounded SalsaNext/HDL-64 single-scan input and conservative point mapping.
use serde::Serialize;

pub const HEIGHT: usize = 64;
pub const WIDTH: usize = 2048;
pub const PIXELS: usize = HEIGHT * WIDTH;
pub const CLASSES: usize = 20;
pub const MAX_POINTS: usize = 150_000;
pub const CLASS_NAMES: [&str; CLASSES] = [
    "unlabeled",
    "car",
    "bicycle",
    "motorcycle",
    "truck",
    "other-vehicle",
    "person",
    "bicyclist",
    "motorcyclist",
    "road",
    "parking",
    "sidewalk",
    "other-ground",
    "building",
    "fence",
    "vegetation",
    "trunk",
    "terrain",
    "pole",
    "traffic-sign",
];
const MEAN: [f32; 5] = [12.12, 10.88, 0.23, -1.04, 0.21];
const STD: [f32; 5] = [12.32, 11.47, 6.91, 0.86, 0.16];

#[derive(Debug, Default, Serialize)]
pub struct ProjectionReport {
    pub points: usize,
    pub projected: usize,
    pub occluded: usize,
    pub invalid: usize,
    pub outside_view: usize,
    pub outside_range: usize,
}

pub struct Projection {
    pub input: Vec<f32>,
    pub owners: Vec<Option<usize>>,
    pub report: ProjectionReport,
}

/// One untransformed sensor-frame XYZ + normalized-remission scan, not a fused map.
/// Returns outside [-25,3] elevation / 0..100 m stay unclassified.
/// Only the nearest point in a pixel can receive that pixel's model result.
pub fn project(points: &[[f32; 4]]) -> Result<Projection, String> {
    if points.is_empty() || points.len() > MAX_POINTS {
        return Err(format!("expected one scan with 1..={MAX_POINTS} points"));
    }
    let mut owners = vec![None; PIXELS];
    let mut ranges = vec![f64::INFINITY; PIXELS];
    let mut report = ProjectionReport {
        points: points.len(),
        ..Default::default()
    };
    let mut eligible = 0;
    for (i, point) in points.iter().enumerate() {
        if !point.iter().all(|v| v.is_finite()) {
            report.invalid += 1;
            continue;
        }
        if !(0.0..=1.0).contains(&point[3]) {
            return Err(
                "remission must already be in 0..1; no automatic intensity rescaling".into(),
            );
        }
        let [x, y, z, _] = point.map(f64::from);
        let range = x.hypot(y).hypot(z);
        if !(1e-6..=100.0).contains(&range) {
            report.outside_range += 1;
            continue;
        }
        let pitch = (z / range).asin();
        let lower = (-25_f64).to_radians();
        let upper = 3_f64.to_radians();
        if pitch < lower || pitch > upper {
            report.outside_view += 1;
            continue;
        }
        let yaw = -y.atan2(x);
        let col = ((0.5 * (yaw / std::f64::consts::PI + 1.0) * WIDTH as f64).floor() as usize)
            .min(WIDTH - 1);
        let row =
            (((upper - pitch) / (upper - lower) * HEIGHT as f64).floor() as usize).min(HEIGHT - 1);
        let pixel = row * WIDTH + col;
        eligible += 1;
        // Deterministic first owner for a distance tie; no labels for other returns.
        if range < ranges[pixel] {
            owners[pixel] = Some(i);
            ranges[pixel] = range;
        }
    }
    let mut input = vec![0.0; 5 * PIXELS];
    for (pixel, owner) in owners.iter().enumerate() {
        if let Some(i) = *owner {
            let [x, y, z, intensity] = points[i];
            let values = [ranges[pixel] as f32, x, y, z, intensity];
            for channel in 0..5 {
                input[channel * PIXELS + pixel] = (values[channel] - MEAN[channel]) / STD[channel];
            }
            report.projected += 1;
        }
    }
    report.occluded = eligible - report.projected;
    Ok(Projection {
        input,
        owners,
        report,
    })
}

#[derive(Debug, Serialize)]
pub struct MaskReport {
    pub class_counts: [usize; CLASSES],
    pub removed: usize,
    pub retained: usize,
    pub threshold: f32,
    pub target_classes: Vec<String>,
    pub unclassified_retained: usize,
}

/// Softmax is an explicit contract: invalid tensors fail before any mask is returned.
/// No k-NN label spreading; occluded/out-of-view/invalid returns always stay.
pub fn removal_mask(
    projection: &Projection,
    probabilities: &[f32],
    targets: &[usize],
    threshold: f32,
) -> Result<(Vec<bool>, MaskReport), String> {
    if !threshold.is_finite() || !(0.0..=1.0).contains(&threshold) {
        return Err("confidence threshold must be finite in 0..1".into());
    }
    if targets.is_empty() || targets.iter().any(|&c| c == 0 || c >= CLASSES) {
        return Err("select at least one named class; unlabeled cannot be removed".into());
    }
    if probabilities.len() != CLASSES * PIXELS {
        return Err("model must return [1,20,64,2048] probabilities".into());
    }
    let mut mask = vec![false; projection.report.points];
    let mut counts = [0; CLASSES];
    for pixel in 0..PIXELS {
        let mut sum = 0.0_f64;
        let mut best = (0, -1.0_f32);
        for class in 0..CLASSES {
            let p = probabilities[class * PIXELS + pixel];
            if !p.is_finite() || !(0.0..=1.0).contains(&p) {
                return Err(
                    "model returned non-finite/out-of-range probabilities; no removal selected"
                        .into(),
                );
            }
            sum += f64::from(p);
            if p > best.1 {
                best = (class, p);
            }
        }
        if (sum - 1.0).abs() > 0.002 {
            return Err("model output is not a normalized softmax; no removal selected".into());
        }
        if let Some(i) = projection.owners[pixel] {
            counts[best.0] += 1;
            mask[i] = targets.contains(&best.0) && best.1 >= threshold;
        }
    }
    let removed = mask.iter().filter(|&&v| v).count();
    let report = MaskReport {
        class_counts: counts,
        removed,
        retained: mask.len() - removed,
        threshold,
        target_classes: targets.iter().map(|&c| CLASS_NAMES[c].to_owned()).collect(),
        unclassified_retained: mask.len() - projection.report.projected,
    };
    Ok((mask, report))
}

pub fn read_scan(bytes: &[u8]) -> Result<Vec<[f32; 4]>, String> {
    if bytes.is_empty() || !bytes.len().is_multiple_of(16) || bytes.len() / 16 > MAX_POINTS {
        return Err("expected little-endian KITTI XYZI records, at most 150000 points".into());
    }
    Ok(bytes
        .as_chunks::<16>()
        .0
        .iter()
        .map(|b| {
            std::array::from_fn(|i| f32::from_le_bytes(b[i * 4..i * 4 + 4].try_into().unwrap()))
        })
        .collect())
}

/// Select original records without reserializing or replacing any retained attribute.
pub fn split_scan(bytes: &[u8], mask: &[bool]) -> Result<(Vec<u8>, Vec<u8>), String> {
    if !bytes.len().is_multiple_of(16) || bytes.len() / 16 != mask.len() {
        return Err("mask does not match the original scan".into());
    }
    let mut kept = Vec::new();
    let mut removed = Vec::new();
    for (point, &remove) in bytes.as_chunks::<16>().0.iter().zip(mask) {
        if remove {
            removed.extend_from_slice(point);
        } else {
            kept.extend_from_slice(point);
        }
    }
    Ok((kept, removed))
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn projection_keeps_only_nearest_owners_and_reports_every_other_return() {
        let points = [
            [10., 0., 0., 0.5],
            [20., 0., 0., 0.6],
            [0., 0., 10., 0.5],
            [0.; 4],
            [f32::NAN, 0., 0., 0.5],
            [200., 0., 0., 0.5],
            [10., 0., 0., 0.7],
        ];
        let p = project(&points).unwrap();
        assert_eq!(
            p.owners.iter().flatten().copied().collect::<Vec<_>>(),
            vec![0]
        );
        assert_eq!(
            (
                p.report.projected,
                p.report.occluded,
                p.report.invalid,
                p.report.outside_view,
                p.report.outside_range
            ),
            (1, 2, 1, 1, 2)
        );
        let pixel = p.owners.iter().position(|v| *v == Some(0)).unwrap();
        assert!((p.input[pixel] - (10. - MEAN[0]) / STD[0]).abs() < 1e-6);
        assert!((p.input[4 * PIXELS + pixel] - (0.5 - MEAN[4]) / STD[4]).abs() < 1e-6);
        assert_eq!(p.input.iter().filter(|&&v| v != 0.).count(), 5);
        assert!(project(&[[1., 0., 0., 18000.]]).is_err());
        assert!(project(&[]).is_err());
        assert!(read_scan(&[0; 15]).is_err());
        assert!(read_scan(&vec![0; 16 * (MAX_POINTS + 1)]).is_err());
    }
    #[test]
    fn confidence_targets_collisions_and_invalid_softmax_do_not_remove_other_points() {
        let points = [[10., 0., 0., 0.5], [20., 0., 0., 0.6], [0., 10., 0., 0.5]];
        let p = project(&points).unwrap();
        let pixel = p.owners.iter().position(|v| *v == Some(0)).unwrap();
        let mut scores = vec![0.; CLASSES * PIXELS];
        scores[..PIXELS].fill(1.);
        scores[pixel] = 0.05;
        scores[PIXELS + pixel] = 0.95;
        let (mask, r) = removal_mask(&p, &scores, &[1], 0.9).unwrap();
        assert_eq!(mask, vec![true, false, false]);
        assert_eq!((r.removed, r.retained, r.unclassified_retained), (1, 2, 1));
        assert!(
            removal_mask(&p, &scores, &[1], 0.99)
                .unwrap()
                .0
                .iter()
                .all(|&v| !v)
        );
        assert!(removal_mask(&p, &scores, &[0], 0.9).is_err());
        assert!(removal_mask(&p, &scores, &[1], f32::NAN).is_err());
        scores[1] = f32::NAN;
        assert!(removal_mask(&p, &scores, &[1], 0.9).is_err());
        scores[1] = 0.;
        assert!(removal_mask(&p, &scores, &[1], 0.9).is_err());
    }
    #[test]
    fn yaw_seam_sensor_axes_and_view_limits_do_not_alias_distant_returns() {
        let at_pitch = |degrees: f64| {
            let pitch = degrees.to_radians();
            [
                (10.0 * pitch.cos()) as f32,
                0.0,
                (10.0 * pitch.sin()) as f32,
                0.5,
            ]
        };
        let points = [
            [-10.0, 0.0, 0.0, 0.5],
            [-10.0, -0.0, 0.0, 0.5],
            [0.0, 10.0, 0.0, 0.5],
            [0.0, -10.0, 0.0, 0.5],
            at_pitch(2.99),
            at_pitch(3.01),
            at_pitch(-24.99),
            at_pitch(-25.01),
        ];
        let p = project(&points).unwrap();
        let column = |id| p.owners.iter().position(|&v| v == Some(id)).unwrap() % WIDTH;
        assert_eq!(
            (column(0), column(1), column(2), column(3)),
            (0, WIDTH - 1, WIDTH / 4, 3 * WIDTH / 4)
        );
        assert_eq!(p.report.projected, 6);
        assert_eq!(p.report.outside_view, 2);
        assert!(!p.owners.contains(&Some(5)) && !p.owners.contains(&Some(7)));
    }

    #[test]
    fn split_preserves_every_original_record_and_retained_nan_payload() {
        let words = [0x7fc00001_u32, 0x80000000, 0, 0x3f000000, 1, 2, 3, 4];
        let bytes: Vec<_> = words.into_iter().flat_map(u32::to_le_bytes).collect();
        let (kept, removed) = split_scan(&bytes, &[false, true]).unwrap();
        assert_eq!(kept, bytes[..16]);
        assert_eq!(removed, bytes[16..]);
        assert!(split_scan(&bytes, &[false]).is_err());
        assert!(read_scan(&bytes).unwrap()[0][0].is_nan());
    }
}
