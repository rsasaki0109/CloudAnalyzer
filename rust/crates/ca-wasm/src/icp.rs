//! ICP registration and point-pair alignment.

use super::*;

/// Outcome of [`register_icp`].
#[wasm_bindgen]
pub struct IcpOutcome {
    #[wasm_bindgen(readonly, js_name = correspondenceTemporal)]
    pub correspondence_temporal: bool,
    #[wasm_bindgen(readonly, js_name = temporalQueries)]
    pub temporal_queries: usize,
    #[wasm_bindgen(readonly, js_name = temporalScratchBytes)]
    pub temporal_scratch_bytes: usize,
    matrix: [f64; 16],
    #[wasm_bindgen(readonly, js_name = rmsInitial)]
    pub rms_initial: f64,
    #[wasm_bindgen(readonly, js_name = rmsFinal)]
    pub rms_final: f64,
    #[wasm_bindgen(readonly)]
    pub iterations: usize,
    #[wasm_bindgen(readonly)]
    pub converged: bool,
}

#[wasm_bindgen]
impl IcpOutcome {
    /// Row-major 4x4 transform taking the moving cloud onto the reference.
    pub fn matrix(&self) -> Vec<f64> {
        self.matrix.to_vec()
    }
}

/// Register `moving` onto `reference` with ICP. The clouds are not changed;
/// apply the result with [`Cloud::transform`].
#[wasm_bindgen(js_name = registerIcp)]
#[allow(clippy::too_many_arguments)]
pub fn register_icp(
    moving: &Cloud,
    reference: &Cloud,
    max_iterations: usize,
    overlap: f64,
    match_centroids: bool,
    point_to_plane: bool,
    simd: Option<bool>,
    temporal: Option<bool>,
) -> Result<IcpOutcome, JsError> {
    let params = IcpParams {
        metric: if point_to_plane {
            IcpMetric::PointToPlane
        } else {
            IcpMetric::PointToPoint
        },
        max_iterations,
        overlap,
        match_centroids,
        ..IcpParams::default()
    };
    if moving.inner.is_empty() {
        return Err(JsError::new(
            "ICP needs non-empty clouds and at least 3 pairs",
        ));
    }
    let mut prepared = ca_core::icp::IcpPrepared::new(&reference.inner)
        .ok_or_else(|| JsError::new("ICP needs non-empty clouds and at least 3 pairs"))?;
    let temporal = temporal.unwrap_or(false);
    let result = prepared
        .compute_with_correspondence(&moving.inner, params, simd.unwrap_or(false), temporal)
        .ok_or_else(|| JsError::new("ICP needs non-empty clouds and at least 3 pairs"))?;
    Ok(IcpOutcome {
        correspondence_temporal: temporal,
        temporal_queries: prepared.temporal_queries(),
        temporal_scratch_bytes: prepared.temporal_scratch_bytes(),
        matrix: result.transform.to_matrix(),
        rms_initial: result.rms_initial,
        rms_final: result.rms_final,
        iterations: result.iterations,
        converged: result.converged,
    })
}

/// Immutable owned ICP reference and reusable lazy normals.
#[wasm_bindgen]
pub struct IcpPrepared {
    inner: ca_core::icp::IcpPrepared,
}

#[wasm_bindgen]
impl IcpPrepared {
    #[wasm_bindgen(constructor)]
    pub fn new(reference: &Cloud) -> Result<Self, JsError> {
        Ok(Self {
            inner: ca_core::icp::IcpPrepared::new(&reference.inner)
                .ok_or_else(|| JsError::new("ICP needs a non-empty reference"))?,
        })
    }

    #[wasm_bindgen(getter, js_name = retainedBytes)]
    pub fn retained_bytes(&self) -> usize {
        self.inner.retained_bytes()
    }

    #[allow(clippy::too_many_arguments)]
    pub fn compute(
        &mut self,
        moving: &Cloud,
        max_iterations: usize,
        overlap: f64,
        match_centroids: bool,
        point_to_plane: bool,
        simd: Option<bool>,
        temporal: Option<bool>,
    ) -> Result<IcpOutcome, JsError> {
        let params = IcpParams {
            metric: if point_to_plane {
                IcpMetric::PointToPlane
            } else {
                IcpMetric::PointToPoint
            },
            max_iterations,
            overlap,
            match_centroids,
            ..IcpParams::default()
        };
        let result = self
            .inner
            .compute_with_correspondence(
                &moving.inner,
                params,
                simd.unwrap_or(false),
                temporal.unwrap_or(false),
            )
            .ok_or_else(|| JsError::new("ICP needs non-empty clouds and at least 3 pairs"))?;
        Ok(IcpOutcome {
            correspondence_temporal: temporal.unwrap_or(false),
            temporal_queries: self.inner.temporal_queries(),
            temporal_scratch_bytes: self.inner.temporal_scratch_bytes(),
            matrix: result.transform.to_matrix(),
            rms_initial: result.rms_initial,
            rms_final: result.rms_final,
            iterations: result.iterations,
            converged: result.converged,
        })
    }
}

/// The rigid motion taking picked `moving` points onto their `reference`
/// points (flat xyz triples, pair i in both): the row-major 4x4 matrix, the
/// RMS, then each pair's residual distance.
#[wasm_bindgen(js_name = alignPairs)]
pub fn align_pairs(moving: &[f64], reference: &[f64]) -> Result<Vec<f64>, JsError> {
    let triples = |v: &[f64]| -> Vec<[f64; 3]> { v.as_chunks::<3>().0.to_vec() };
    let out = ca_core::icp::align_pairs(&triples(moving), &triples(reference))
        .ok_or_else(|| JsError::new("need at least three pairs, not all on one line"))?;
    let mut values = out.transform.to_matrix().to_vec();
    values.push(out.rms);
    values.extend(out.residuals);
    Ok(values)
}
