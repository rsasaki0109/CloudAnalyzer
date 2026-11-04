//! Trajectory loading and ATE/RPE evaluation.

use super::*;

/// A trajectory: timed positions and optional orientations (see
/// [`ca_core::trajectory`]).
#[wasm_bindgen]
pub struct TrajectoryData {
    pub(crate) inner: ca_core::trajectory::Trajectory,
}

#[wasm_bindgen]
impl TrajectoryData {
    /// The layout (`"tum"`, `"kitti"` or `"csv"`) of a trajectory file, from
    /// its name and first bytes, or `undefined` for other (point) files.
    pub fn detect(name: &str, head: &str) -> Option<String> {
        use ca_core::trajectory::Format;
        let format = match ca_core::trajectory::detect(name, head)? {
            Format::Tum => "tum",
            Format::Kitti => "kitti",
            Format::Csv => "csv",
        };
        Some(format.into())
    }

    pub fn parse(text: &str, format: &str) -> Result<TrajectoryData, JsError> {
        use ca_core::trajectory::Format;
        let format = match format {
            "tum" => Format::Tum,
            "kitti" => Format::Kitti,
            "csv" => Format::Csv,
            other => {
                return Err(JsError::new(&format!(
                    "unknown trajectory format {other:?}"
                )));
            }
        };
        let inner = ca_core::trajectory::parse(text, format).map_err(|e| JsError::new(&e.0))?;
        Ok(TrajectoryData { inner })
    }

    /// Rebuild a parsed trajectory from its arrays (orientations may be empty).
    #[wasm_bindgen(constructor)]
    pub fn new(
        timestamps: &[f64],
        positions: &[f64],
        orientations: &[f64],
    ) -> Result<TrajectoryData, JsError> {
        let n = timestamps.len();
        if positions.len() != 3 * n || !(orientations.is_empty() || orientations.len() == 4 * n) {
            return Err(JsError::new("trajectory arrays do not match"));
        }
        Ok(TrajectoryData {
            inner: ca_core::trajectory::Trajectory {
                timestamps: timestamps.to_vec(),
                positions: positions.as_chunks::<3>().0.to_vec(),
                orientations: (!orientations.is_empty())
                    .then(|| orientations.as_chunks::<4>().0.to_vec()),
            },
        })
    }

    pub fn timestamps(&self) -> Vec<f64> {
        self.inner.timestamps.clone()
    }

    /// Interleaved xyz.
    pub fn positions(&self) -> Vec<f64> {
        self.inner.positions.iter().flatten().copied().collect()
    }

    /// Interleaved quaternions `[x, y, z, w]`, if the file has orientations.
    pub fn orientations(&self) -> Option<Vec<f64>> {
        self.inner
            .orientations
            .as_ref()
            .map(|q| q.iter().flatten().copied().collect())
    }
}

/// Result of [`evaluate_trajectory`]: errors per matched pose (ATE) and per
/// pose pair (RPE), and the alignment.
#[wasm_bindgen]
pub struct TrajectoryEvaluation {
    inner: ca_core::trajectory::Evaluation,
}

#[wasm_bindgen]
impl TrajectoryEvaluation {
    /// Times of the matched reference poses.
    pub fn timestamps(&self) -> Vec<f64> {
        self.inner.timestamps.clone()
    }

    /// The matched estimate after alignment, interleaved xyz.
    pub fn estimate(&self) -> Vec<f64> {
        self.inner.estimate.iter().flatten().copied().collect()
    }

    /// The matched reference poses, interleaved xyz.
    pub fn reference(&self) -> Vec<f64> {
        self.inner.reference.iter().flatten().copied().collect()
    }

    /// A series: `ate`, `ate_rotation`, `rpe`, `rpe_rotation` or
    /// `rpe_percent` (rotations in degrees); `undefined` when not available.
    pub fn values(&self, name: &str) -> Option<Vec<f64>> {
        let e = &self.inner;
        match name {
            "ate" => Some(e.ate.clone()),
            "ate_rotation" => e.ate_rotation.clone(),
            "rpe" => Some(e.rpe_translation.clone()),
            "rpe_rotation" => e.rpe_rotation.clone(),
            "rpe_percent" => e.rpe_percent.clone(),
            _ => None,
        }
    }

    /// `[count, rmse, mean, median, std, min, max]` of a series (see [`Self::values`]).
    pub fn stats(&self, name: &str) -> Option<Vec<f64>> {
        let s = ca_core::trajectory::Stats::of(&self.values(name)?)?;
        Some(vec![
            s.count as f64,
            s.rmse,
            s.mean,
            s.median,
            s.std,
            s.min,
            s.max,
        ])
    }

    /// Row-major 4x4 transform taking the estimate onto the reference.
    pub fn matrix(&self) -> Vec<f64> {
        self.inner.alignment.to_matrix().to_vec()
    }

    #[wasm_bindgen(getter)]
    pub fn scale(&self) -> f64 {
        self.inner.alignment.scale
    }

    #[wasm_bindgen(getter, js_name = endpointDrift)]
    pub fn endpoint_drift(&self) -> f64 {
        self.inner.endpoint_drift
    }

    #[wasm_bindgen(getter, js_name = referenceLength)]
    pub fn reference_length(&self) -> f64 {
        self.inner.reference_length
    }

    #[wasm_bindgen(getter, js_name = estimateLength)]
    pub fn estimate_length(&self) -> f64 {
        self.inner.estimate_length
    }
}

/// ATE / RPE of `estimate` against `reference`: poses matched within
/// `max_time_delta` seconds, `alignment` one of `none`, `origin`, `se3` or
/// `sim3`, RPE over `delta` frames or, with `delta_unit == "m"`, metres.
#[wasm_bindgen(js_name = evaluateTrajectory)]
pub fn evaluate_trajectory(
    estimate: &TrajectoryData,
    reference: &TrajectoryData,
    max_time_delta: f64,
    alignment: &str,
    delta: f64,
    delta_unit: &str,
) -> Result<TrajectoryEvaluation, JsError> {
    use ca_core::trajectory::{Alignment, EvalParams, RpeDelta};
    let alignment = match alignment {
        "none" => Alignment::None,
        "origin" => Alignment::Origin,
        "se3" => Alignment::Se3,
        "sim3" => Alignment::Sim3,
        other => return Err(JsError::new(&format!("unknown alignment {other:?}"))),
    };
    let rpe_delta = if delta_unit == "m" {
        RpeDelta::Meters(delta)
    } else {
        RpeDelta::Frames(delta.max(0.0) as usize)
    };
    let params = EvalParams {
        max_time_delta,
        alignment,
        rpe_delta,
    };
    ca_core::trajectory::evaluate(&estimate.inner, &reference.inner, &params)
        .map(|inner| TrajectoryEvaluation { inner })
        .map_err(|e| JsError::new(&e.0))
}
