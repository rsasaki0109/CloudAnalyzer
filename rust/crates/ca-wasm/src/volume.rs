//! Cut and fill volume between two surfaces.

use super::*;

enum OwnedSurface {
    Points(Vec<[f64; 3]>),
    Mesh(TriangleMesh),
    Constant(f64),
}

/// One side of a cut/fill volume computation (see [`compute_volume`]).
#[wasm_bindgen]
pub struct VolumeSurface {
    inner: OwnedSurface,
}

#[wasm_bindgen]
impl VolumeSurface {
    #[wasm_bindgen(js_name = fromCloud)]
    pub fn from_cloud(cloud: &Cloud) -> VolumeSurface {
        VolumeSurface {
            inner: OwnedSurface::Points(cloud.inner.positions.clone()),
        }
    }

    #[wasm_bindgen(js_name = fromMesh)]
    pub fn from_mesh(mesh: &Mesh) -> VolumeSurface {
        VolumeSurface {
            inner: OwnedSurface::Mesh(mesh.inner.clone()),
        }
    }

    /// A horizontal plane at height `z` (original coordinates).
    pub fn constant(z: f64) -> VolumeSurface {
        VolumeSurface {
            inner: OwnedSurface::Constant(z),
        }
    }
}

impl VolumeSurface {
    fn surface(&self) -> ca_core::volume::Surface<'_> {
        use ca_core::volume::Surface;
        match &self.inner {
            OwnedSurface::Points(p) => Surface::Points(p),
            OwnedSurface::Mesh(m) => Surface::Mesh(m),
            OwnedSurface::Constant(z) => Surface::Constant(*z),
        }
    }
}

/// Strict user-defined height screening; quantities are full differences.
#[wasm_bindgen]
pub struct HeightReview {
    cells: Vec<f64>,
    #[wasm_bindgen(readonly, js_name = cellLimit)]
    pub cell_limit: usize,
    #[wasm_bindgen(readonly)]
    pub threshold: f64,
    #[wasm_bindgen(readonly, js_name = riseCells)]
    pub rise_cells: usize,
    #[wasm_bindgen(readonly, js_name = fallCells)]
    pub fall_cells: usize,
    #[wasm_bindgen(readonly, js_name = withinCells)]
    pub within_cells: usize,
    #[wasm_bindgen(readonly, js_name = riseArea)]
    pub rise_area: f64,
    #[wasm_bindgen(readonly, js_name = fallArea)]
    pub fall_area: f64,
    #[wasm_bindgen(readonly, js_name = riseVolume)]
    pub rise_volume: f64,
    #[wasm_bindgen(readonly, js_name = fallVolume)]
    pub fall_volume: f64,
}

#[wasm_bindgen]
impl HeightReview {
    /// Packed [row-major index, world x, world y, before, after, difference].
    #[wasm_bindgen(js_name = takeCells)]
    pub fn take_cells(&mut self) -> Vec<f64> {
        std::mem::take(&mut self.cells)
    }
}

/// Result of [`compute_volume`].
#[wasm_bindgen]
pub struct VolumeOutput {
    #[wasm_bindgen(readonly)]
    pub added: f64,
    #[wasm_bindgen(readonly)]
    pub removed: f64,
    #[wasm_bindgen(readonly, js_name = addedArea)]
    pub added_area: f64,
    #[wasm_bindgen(readonly, js_name = removedArea)]
    pub removed_area: f64,
    #[wasm_bindgen(readonly, js_name = matchedCells)]
    pub matched_cells: usize,
    #[wasm_bindgen(readonly, js_name = totalCells)]
    pub total_cells: usize,
    #[wasm_bindgen(readonly, js_name = outsideRegionCells)]
    pub outside_region_cells: usize,
    #[wasm_bindgen(readonly)]
    pub cell: f64,
    cells: Option<Cloud>,
    height_review: Option<HeightReview>,
}

#[wasm_bindgen]
impl VolumeOutput {
    #[wasm_bindgen(js_name = takeHeightReview)]
    pub fn take_height_review(&mut self) -> Option<HeightReview> {
        self.height_review.take()
    }

    /// The compared cells as an indexed cloud: one point per cell with both
    /// heights, at the `after` height (or `before` where `after` is missing),
    /// with the height difference in the `height_difference` attribute.
    #[wasm_bindgen(js_name = takeCells)]
    pub fn take_cells(&mut self) -> Result<Cloud, JsError> {
        self.cells
            .take()
            .ok_or_else(|| JsError::new("cells already taken"))
    }
}

fn volume_shared_grid(values: &[f64], cell: f64) -> Result<ca_core::volume::Grid, JsError> {
    if values.len() != 4
        || !values.iter().all(|v| v.is_finite())
        || values[2] < 1.0
        || values[3] < 1.0
        || values[2].fract() != 0.0
        || values[3].fract() != 0.0
        || values[2] * values[3] > 1_000_000.0
    {
        return Err(JsError::new(
            "shared grid needs finite min X/Y and positive integer dimensions within 1,000,000 cells",
        ));
    }
    let grid = ca_core::volume::Grid {
        min: [values[0], values[1]],
        cell,
        nx: values[2] as usize,
        ny: values[3] as usize,
    };
    if !grid.valid() {
        return Err(JsError::new("invalid shared volume grid"));
    }
    Ok(grid)
}

/// Bounded raw-observation intersection; owns no input point clouds.
#[wasm_bindgen]
pub struct ObservedVolumeSupport {
    grid: ca_core::volume::Grid,
    height: ca_core::volume::CellHeight,
    mask: Vec<u8>,
}
#[wasm_bindgen]
impl ObservedVolumeSupport {
    #[wasm_bindgen(constructor)]
    pub fn new(
        cell: f64,
        height: &str,
        shared_grid: Vec<f64>,
    ) -> Result<ObservedVolumeSupport, JsError> {
        use ca_core::volume::CellHeight;
        let height = match height {
            "mean" => CellHeight::Mean,
            "min" => CellHeight::Min,
            "max" => CellHeight::Max,
            _ => return Err(JsError::new("unknown volume cell height")),
        };
        let grid = volume_shared_grid(&shared_grid, cell)?;
        Ok(Self {
            grid,
            height,
            mask: vec![1; grid.nx * grid.ny],
        })
    }
    pub fn intersect(&mut self, surface: &VolumeSurface) -> Result<usize, JsError> {
        ca_core::volume::intersect_observed_support(
            surface.surface(),
            &self.grid,
            self.height,
            &mut self.mask,
        )
        .ok_or_else(|| {
            JsError::new("observed support requires raw point surfaces and a valid mask")
        })
    }
    #[wasm_bindgen(js_name = takeMask)]
    pub fn take_mask(&mut self) -> Vec<u8> {
        std::mem::take(&mut self.mask)
    }
}

/// Cut/fill volume between two surfaces on a grid of `cell`-sized squares.
/// `height` is `"mean"`, `"min"` or `"max"`; `fill_empty` interpolates cells
/// a surface does not cover.
/// `shared_grid`, when supplied, is [lower-left X, lower-left Y, nx, ny].
#[wasm_bindgen(js_name = computeVolume)]
#[allow(clippy::too_many_arguments)] // Keep the existing positional JS API compatible.
pub fn compute_volume(
    before: &VolumeSurface,
    after: &VolumeSurface,
    cell: f64,
    height: &str,
    fill_empty: bool,
    region: Option<Vec<f64>>,
    summary_only: bool,
    shared_grid: Option<Vec<f64>>,
    support_mask: Option<Vec<u8>>,
    height_threshold: Option<f64>,
) -> Result<VolumeOutput, JsError> {
    use ca_core::volume::{CellHeight, VolumeParams};
    if height_threshold.is_some_and(|v| !v.is_finite() || v < 0.0) {
        return Err(JsError::new(
            "height screening threshold must be finite and nonnegative",
        ));
    }
    let height = match height {
        "mean" => CellHeight::Mean,
        "min" => CellHeight::Min,
        "max" => CellHeight::Max,
        other => return Err(JsError::new(&format!("unknown cell height {other:?}"))),
    };
    let params = VolumeParams {
        cell,
        height,
        fill_empty,
    };
    let region = region
        .map(|b| {
            <[f64; 4]>::try_from(b)
                .map_err(|_| JsError::new("volume region needs four world XY bounds"))
        })
        .transpose()?;
    let result = if let Some(values) = shared_grid {
        let grid = volume_shared_grid(&values, cell)?;
        ca_core::volume::volume_on_grid_with_support(
            before.surface(),
            after.surface(),
            params,
            region,
            grid,
            support_mask.as_deref(),
        )
    } else {
        if support_mask.is_some() {
            return Err(JsError::new("observed support requires a shared grid"));
        }
        ca_core::volume::volume_in_region(before.surface(), after.surface(), params, region)
    };
    let r = result.ok_or_else(|| {
        JsError::new("need a valid volume grid, region and at least one cloud or mesh")
    })?;
    let height_review = height_threshold
        .map(|threshold| {
            let h = r.height_review(threshold).ok_or_else(|| {
                JsError::new("height screening cannot be represented on these surfaces")
            })?;
            Ok::<_, JsError>(HeightReview {
                cells: h
                    .cells
                    .iter()
                    .flat_map(|c| [c.index as f64, c.x, c.y, c.before, c.after, c.difference])
                    .collect(),
                cell_limit: ca_core::volume::HEIGHT_REVIEW_CELL_LIMIT,
                threshold: h.threshold,
                rise_cells: h.rise_cells,
                fall_cells: h.fall_cells,
                within_cells: h.within_cells,
                rise_area: h.rise_area,
                fall_area: h.fall_area,
                rise_volume: h.rise_volume,
                fall_volume: h.fall_volume,
            })
        })
        .transpose()?;
    let mut positions = Vec::with_capacity(if summary_only { 0 } else { r.matched_cells });
    let mut difference = Vec::with_capacity(if summary_only { 0 } else { r.matched_cells });
    if !summary_only {
        for j in 0..r.grid.ny {
            for i in 0..r.grid.nx {
                let k = j * r.grid.nx + i;
                let d = r.after[k] - r.before[k];
                if d.is_nan() {
                    continue;
                }
                let [x, y] = r.grid.center(i, j);
                positions.push([x, y, r.after[k]]);
                difference.push(d as f32);
            }
        }
    }
    let cells = if positions.is_empty() {
        None
    } else {
        let mut inner = PointCloud {
            positions,
            colors: None,
            attributes: vec![ca_core::Attribute {
                name: "height_difference".into(),
                values: AttributeValues::F32(difference),
            }],
        };
        let lod = build_lod(&mut inner)?;
        Some(Cloud {
            inner,
            lod,
            layout: None,
        })
    };
    Ok(VolumeOutput {
        added: r.added,
        removed: r.removed,
        added_area: r.added_area,
        removed_area: r.removed_area,
        matched_cells: r.matched_cells,
        total_cells: r.total_cells,
        outside_region_cells: r.grid.nx * r.grid.ny - r.total_cells,
        cell: r.grid.cell,
        cells,
        height_review,
    })
}
