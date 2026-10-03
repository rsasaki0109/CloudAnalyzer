//! Conservative longitudinal RGB paint observations in the trajectory frame.
//! Brightness alone is insufficient: require dark flanks along the whole slice.
use super::{BuildOptions, Candidate, Evidence, quantile};
use crate::PointCloud;

pub(super) struct Frame {
    pub position: [f64; 3],
    pub direction: [f64; 2],
    pub lateral_min: f64,
    pub ground: f64,
}

pub(super) fn candidates(
    cloud: &PointCloud,
    bins: &[Vec<usize>],
    surface: &[Option<f64>],
    frame: Frame,
    o: &BuildOptions,
) -> Vec<Candidate> {
    let Frame {
        position: p,
        direction: dir,
        lateral_min: low,
        ground,
    } = frame;
    let Some(rgb) = cloud.colors.as_ref() else {
        return vec![];
    };
    let brightness = |i: usize| f64::from(*rgb[i].iter().min().unwrap());
    let on_surface = |i: usize, z: f64| (cloud.positions[i][2] - z).abs() < 0.12;
    let mut values: Vec<_> = bins
        .iter()
        .zip(surface)
        .flat_map(|(ids, z)| {
            ids.iter().filter_map(|&i| {
                z.filter(|z| (*z - ground).abs() < 0.3 && on_surface(i, *z))
                    .map(|_| brightness(i))
            })
        })
        .collect();
    let (Some(base), Some(peak)) = (quantile(&mut values, 0.1), quantile(&mut values, 0.995))
    else {
        return vec![];
    };
    // An absolute RGB contrast prevents nearly uniform asphalt/shadows from
    // being promoted just because a relative percentile has a small range.
    if peak - base < 40.0 {
        return vec![];
    }
    let threshold = base + 0.7 * (peak - base);
    let mut profile = vec![[None; 4]; bins.len()];
    for (j, ids) in bins.iter().enumerate() {
        let Some(z) = surface[j].filter(|z| (*z - ground).abs() < 0.3) else {
            continue;
        };
        let mut bands = [vec![], vec![], vec![], vec![]];
        for &i in ids {
            if !on_surface(i, z) {
                continue;
            }
            let q = cloud.positions[i];
            let along = (q[0] - p[0]) * dir[0] + (q[1] - p[1]) * dir[1];
            let k =
                (((along + o.half_window) / (2.0 * o.half_window) * 4.0).floor() as usize).min(3);
            bands[k].push(brightness(i));
        }
        for k in 0..4 {
            if bands[k].len() >= o.min_bin_points {
                profile[j][k] = quantile(&mut bands[k], 0.75);
            }
        }
    }
    let bright = |j: usize| profile[j].iter().all(|s| s.is_some_and(|s| s >= threshold));
    let dark = |j: usize| {
        profile[j]
            .iter()
            .all(|s| s.is_some_and(|s| s < base + 0.4 * (peak - base)))
    };
    let mut result = vec![];
    let mut j = 1;
    while j + 1 < bins.len() {
        if !bright(j) {
            j += 1;
            continue;
        }
        let start = j;
        while j + 1 < bins.len() && bright(j) {
            j += 1;
        }
        let end = j - 1;
        if (end - start + 1) as f64 * o.bin_width > 0.6 || !dark(start - 1) || !dark(j) {
            continue;
        }
        result.push(Candidate {
            lateral: low + (start + end + 1) as f64 * 0.5 * o.bin_width,
            z: surface[start].unwrap(),
            evidence: Evidence::RgbPaint,
        });
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;

    fn detect(paint: impl Fn(f64, f64) -> [u8; 3], gap: bool) -> Vec<Candidate> {
        let mut cloud = PointCloud::default();
        let mut rgb = vec![];
        let mut bins = vec![vec![]; 20];
        for k in 0..80 {
            let x = -1.975 + k as f64 * 0.05;
            for (j, bin) in bins.iter_mut().enumerate() {
                let y = -2.0 + (j as f64 + 0.5) * 0.2;
                if gap && j == 9 {
                    continue;
                }
                bin.push(cloud.positions.len());
                cloud.positions.push([x, y, 3.0]);
                rgb.push(paint(x, y));
            }
        }
        cloud.colors = Some(rgb);
        candidates(
            &cloud,
            &bins,
            &[Some(3.0); 20],
            Frame {
                position: [0.0, 0.0, 99.0],
                direction: [1.0, 0.0],
                lateral_min: -2.0,
                ground: 3.0,
            },
            &BuildOptions::default(),
        )
    }

    #[test]
    fn supported_longitudinal_white_paint_is_observed() {
        let result = detect(
            |_, y| {
                if (y - 0.1).abs() < 0.01 {
                    [230; 3]
                } else {
                    [40; 3]
                }
            },
            false,
        );
        assert_eq!(result.len(), 1);
        assert!((result[0].lateral - 0.1).abs() < 1e-9);
        assert_eq!(result[0].z, 3.0);
        assert_eq!(result[0].evidence, Evidence::RgbPaint);
    }

    #[test]
    fn transverse_bars_crossing_stripes_and_missing_flanks_are_held() {
        for repeated in [false, true] {
            assert!(
                detect(
                    |x, _| if if repeated {
                        (x * 2.0).sin() > 0.0
                    } else {
                        x.abs() < 0.2
                    } {
                        [230; 3]
                    } else {
                        [40; 3]
                    },
                    false
                )
                .is_empty()
            );
        }
        assert!(
            detect(
                |_, y| if (y - 0.1).abs() < 0.01 {
                    [230; 3]
                } else {
                    [40; 3]
                },
                true
            )
            .is_empty()
        );
        assert!(detect(|_, _| [230; 3], false).is_empty());
        assert!(detect(|_, y| if y.abs() < 0.8 { [230; 3] } else { [40; 3] }, false).is_empty());
        assert!(
            detect(
                |_, y| if (y - 0.1).abs() < 0.01 {
                    [250, 40, 40]
                } else {
                    [40; 3]
                },
                false
            )
            .is_empty()
        );
        assert!(
            detect(
                |_, y| if (y - 0.1).abs() < 0.01 {
                    [60; 3]
                } else {
                    [40; 3]
                },
                false
            )
            .is_empty()
        );
    }
}
