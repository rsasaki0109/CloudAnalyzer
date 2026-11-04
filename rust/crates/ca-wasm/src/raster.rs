//! DEM rasters and Delaunay meshes.

use super::*;

/// Result of [`Cloud::rasterize`]: the height grid and its cells as a cloud.
#[wasm_bindgen]
pub struct RasterOutput {
    #[wasm_bindgen(readonly)]
    pub nx: usize,
    #[wasm_bindgen(readonly)]
    pub ny: usize,
    /// Lower-left corner of the grid (original coordinates).
    #[wasm_bindgen(readonly, js_name = minX)]
    pub min_x: f64,
    #[wasm_bindgen(readonly, js_name = minY)]
    pub min_y: f64,
    #[wasm_bindgen(readonly)]
    pub cell: f64,
    #[wasm_bindgen(readonly, js_name = populatedCells)]
    pub populated_cells: usize,
    pub(crate) heights: Vec<f32>,
    pub(crate) cells: Option<Cloud>,
}

#[wasm_bindgen]
impl RasterOutput {
    /// Per-cell heights, row-major from the lowest y; NaN where empty.
    #[wasm_bindgen(js_name = takeHeights)]
    pub fn take_heights(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.heights)
    }

    /// One point per non-empty cell at its height, as an indexed cloud with
    /// the height in the `height` attribute (for coloring).
    #[wasm_bindgen(js_name = takeCells)]
    pub fn take_cells(&mut self) -> Result<Cloud, JsError> {
        self.cells
            .take()
            .ok_or_else(|| JsError::new("cells already taken"))
    }
}

/// A raster (see [`RasterOutput`]) as a single-band Float32 GeoTIFF.
#[wasm_bindgen(js_name = rasterGeotiff)]
pub fn raster_geotiff(
    heights: &[f32],
    nx: usize,
    ny: usize,
    min_x: f64,
    min_y: f64,
    cell: f64,
) -> Result<Vec<u8>, JsError> {
    if nx.checked_mul(ny) != Some(heights.len()) || cell.is_nan() || cell <= 0.0 {
        return Err(JsError::new("the heights do not match the grid"));
    }
    let grid = ca_core::volume::Grid {
        min: [min_x, min_y],
        cell,
        nx,
        ny,
    };
    Ok(ca_core::geotiff::write_geotiff(&grid, heights))
}

/// Result of [`Cloud::mesh_delaunay`].
#[wasm_bindgen]
pub struct DelaunayOutput {
    /// Points triangulated (fewer than the cloud's when it was thinned).
    #[wasm_bindgen(readonly)]
    pub points: usize,
    /// Voxel size the cloud was thinned with, 0 when it was not.
    #[wasm_bindgen(readonly)]
    pub voxel: f64,
    /// Longest horizontal edge kept (`Infinity` when all were kept).
    #[wasm_bindgen(readonly, js_name = maxEdge)]
    pub max_edge: f64,
    /// Triangles dropped for a longer edge.
    #[wasm_bindgen(readonly)]
    pub removed: usize,
    pub(crate) mesh: Option<Mesh>,
}

#[wasm_bindgen]
impl DelaunayOutput {
    #[wasm_bindgen(js_name = takeMesh)]
    pub fn take_mesh(&mut self) -> Result<Mesh, JsError> {
        self.mesh
            .take()
            .ok_or_else(|| JsError::new("mesh already taken"))
    }
}
