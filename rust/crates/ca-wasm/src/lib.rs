//! WebAssembly bindings for the CloudAnalyzer Web viewer.

use ca_core::icp::{IcpMetric, IcpParams, Rigid};
use ca_core::octree::{BUCKETS, BucketLayout, NO_CHILD, Octree, OctreeNode, OctreeParams};
use ca_core::{
    AttributeValues, CLASSIFICATION, DistanceStats, INTENSITY, PointCloud, TriangleMesh,
};
use wasm_bindgen::prelude::*;

mod bag;
mod copc_box;
mod odometry;
mod pose_graph;
mod vector_map;
pub use bag::{BagFile, BagMessage};
pub use copc_box::CopcBoxReader;
pub use odometry::{LidarOdometry, OdometryMap, Registration};
pub use pose_graph::{PoseGraphSession, register_scans};
pub use vector_map::VectorMapSession;

/// Numbers per node in [`Cloud::lod_nodes`].
const NODE_STRIDE: usize = 15;

/// A loaded point cloud. Coordinates stay in `f64` on the Rust side.
///
/// Points are reordered at load time into the order of a level-of-detail
/// octree, where each node's points are contiguous. Everything this type
/// returns per point (positions, colors, C2C distances) uses that order.
#[wasm_bindgen]
pub struct Cloud {
    inner: PointCloud,
    lod: Octree,
    /// Placement of the buckets during a parallel index build.
    layout: Option<BucketLayout>,
}

impl Cloud {
    fn orientation(&self, name: &str) -> Result<ca_core::normals::Orientation, JsError> {
        use ca_core::normals::Orientation;
        match name {
            "up" => Ok(Orientation::Up),
            "outward" => {
                let b = self
                    .inner
                    .bounds()
                    .ok_or_else(|| JsError::new("empty cloud"))?;
                Ok(Orientation::Away(b.center()))
            }
            other => Err(JsError::new(&format!("unknown orientation {other:?}"))),
        }
    }

    /// An unindexed copy of the points at `keep`.
    fn selected(&self, keep: &[usize]) -> Result<Cloud, JsError> {
        let inner = self.inner.select(keep);
        if inner.is_empty() {
            return Err(JsError::new("the filter removed every point"));
        }
        Ok(Cloud::unindexed(inner))
    }

    /// An unindexed copy with `segment` as its `source` attribute (see
    /// [`Segmentation`]) and, if given, `colors` as its RGB.
    fn segmented(&self, segment: Vec<u8>, colors: Option<Vec<[u8; 3]>>) -> Cloud {
        let mut inner = self.inner.clone();
        inner
            .attributes
            .retain(|a| a.name != ca_core::merge::SOURCE);
        inner.attributes.push(ca_core::Attribute {
            name: ca_core::merge::SOURCE.into(),
            values: AttributeValues::U8(segment),
        });
        if colors.is_some() {
            inner.colors = colors;
        }
        Cloud::unindexed(inner)
    }

    fn unindexed(inner: PointCloud) -> Cloud {
        Cloud {
            inner,
            lod: Octree {
                order: Vec::new(),
                nodes: Vec::new(),
                grid: OctreeParams::default().grid,
            },
            layout: None,
        }
    }
}

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
    #[wasm_bindgen(readonly)]
    pub cell: f64,
    cells: Option<Cloud>,
}

#[wasm_bindgen]
impl VolumeOutput {
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

/// Cut/fill volume between two surfaces on a grid of `cell`-sized squares.
/// `height` is `"mean"`, `"min"` or `"max"`; `fill_empty` interpolates cells
/// a surface does not cover.
#[wasm_bindgen(js_name = computeVolume)]
pub fn compute_volume(
    before: &VolumeSurface,
    after: &VolumeSurface,
    cell: f64,
    height: &str,
    fill_empty: bool,
) -> Result<VolumeOutput, JsError> {
    use ca_core::volume::{CellHeight, VolumeParams};
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
    let r = ca_core::volume::volume(before.surface(), after.surface(), params)
        .ok_or_else(|| JsError::new("need a positive cell size and at least one cloud or mesh"))?;
    let mut positions = Vec::with_capacity(r.matched_cells);
    let mut difference = Vec::with_capacity(r.matched_cells);
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
        cell: r.grid.cell,
        cells,
    })
}

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
    heights: Vec<f32>,
    cells: Option<Cloud>,
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
    mesh: Option<Mesh>,
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

/// Reads a point file piece by piece (see [`ca_core::io::PointStream`]), so
/// large files never sit in memory whole.
#[wasm_bindgen]
pub struct StreamLoader {
    inner: Option<ca_core::io::PointStream>,
}

#[wasm_bindgen]
impl StreamLoader {
    /// Bytes of the file start needed by [`StreamLoader::open`], or
    /// `undefined` if the header is not complete in `head` or the format
    /// cannot be streamed.
    #[wasm_bindgen(js_name = headerLength)]
    pub fn header_length(name: &str, head: &[u8]) -> Option<usize> {
        ca_core::io::PointStream::header_len(name, head)
    }

    /// Start streaming, or `undefined` for files that must be read whole.
    pub fn open(name: &str, head: &[u8]) -> Result<Option<StreamLoader>, JsError> {
        Ok(ca_core::io::PointStream::open(name, head)?.map(|s| StreamLoader { inner: Some(s) }))
    }

    fn stream(&mut self) -> Result<&mut ca_core::io::PointStream, JsError> {
        self.inner
            .as_mut()
            .ok_or_else(|| JsError::new("stream already finished"))
    }

    /// File offset of the first point record.
    #[wasm_bindgen(getter, js_name = dataOffset)]
    pub fn data_offset(&self) -> f64 {
        self.inner.as_ref().map_or(0.0, |s| s.data_offset() as f64)
    }

    /// Point records announced by the header.
    #[wasm_bindgen(getter, js_name = totalPoints)]
    pub fn total_points(&self) -> f64 {
        self.inner.as_ref().map_or(0.0, |s| s.total_points() as f64)
    }

    #[wasm_bindgen(js_name = setKeepEvery)]
    pub fn set_keep_every(&mut self, n: f64) -> Result<(), JsError> {
        self.stream()?.set_keep_every(n.max(1.0) as u64);
        Ok(())
    }

    /// Feed the next bytes of the record section.
    pub fn push(&mut self, bytes: &[u8]) -> Result<(), JsError> {
        self.stream()?.push(bytes);
        Ok(())
    }

    /// The cloud read so far (call [`Cloud::build_index`] on it).
    pub fn finish(&mut self) -> Result<Cloud, JsError> {
        let stream = self
            .inner
            .take()
            .ok_or_else(|| JsError::new("stream already finished"))?;
        Ok(Cloud::unindexed(stream.finish()?))
    }
}

/// Points announced by a file header (LAS, PLY, PCD), or `undefined`.
#[wasm_bindgen(js_name = announcedPoints)]
pub fn announced_points(name: &str, head: &[u8]) -> Option<f64> {
    ca_core::io::announced_points(name, head).map(|n| n as f64)
}

/// Name of every scan in an E57 file, in the order of its `source` values
/// (an empty string for an unnamed scan).
#[wasm_bindgen(js_name = e57ScanNames)]
pub fn e57_scan_names(bytes: &[u8]) -> Result<Vec<String>, JsError> {
    let names = ca_core::io::e57_scan_names(bytes)?;
    Ok(names.into_iter().map(Option::unwrap_or_default).collect())
}

#[wasm_bindgen]
impl Cloud {
    /// Parse a file; the format is detected from `name` and the leading bytes.
    /// Call [`Cloud::build_index`] before using any per-point output.
    pub fn parse(name: &str, bytes: &[u8]) -> Result<Cloud, JsError> {
        Ok(Cloud::unindexed(ca_core::read(name, bytes)?))
    }

    /// A cloud of `xyz` (three per point) with a per-point `intensity` and
    /// `time` within the scan (each empty for none), e.g. decoded from a ROS
    /// message. Call [`Cloud::build_index`] before using any per-point output.
    #[wasm_bindgen(js_name = fromXyz)]
    pub fn from_xyz(xyz: &[f32], intensity: &[f32], time: &[f32]) -> Result<Cloud, JsError> {
        if !xyz.len().is_multiple_of(3) {
            return Err(JsError::new("xyz needs three numbers per point"));
        }
        let n = xyz.len() / 3;
        if !intensity.is_empty() && intensity.len() != n {
            return Err(JsError::new("intensity needs one number per point"));
        }
        if !time.is_empty() && time.len() != n {
            return Err(JsError::new("time needs one number per point"));
        }
        let mut inner = PointCloud {
            positions: xyz
                .as_chunks::<3>()
                .0
                .iter()
                .map(|p| p.map(f64::from))
                .collect(),
            ..PointCloud::default()
        };
        if !intensity.is_empty() {
            inner.attributes.push(ca_core::Attribute {
                name: INTENSITY.to_string(),
                values: AttributeValues::F32(intensity.to_vec()),
            });
        }
        if !time.is_empty() {
            inner.attributes.push(ca_core::Attribute {
                name: ca_core::odometry::TIME.to_string(),
                values: AttributeValues::F32(time.to_vec()),
            });
        }
        Ok(Cloud::unindexed(inner))
    }

    /// Parse a whole file keeping every `keep_every`-th point (LAS/LAZ thin
    /// while decoding). Call [`Cloud::build_index`] afterwards.
    #[wasm_bindgen(js_name = parseThinned)]
    pub fn parse_thinned(name: &str, bytes: &[u8], keep_every: usize) -> Result<Cloud, JsError> {
        let inner = ca_core::io::read_thinned(name, bytes, keep_every)?;
        Ok(Cloud::unindexed(inner))
    }

    /// Build the level-of-detail octree, reordering the points (a separate
    /// step so callers can report progress).
    #[wasm_bindgen(js_name = buildIndex)]
    pub fn build_index(&mut self) -> Result<(), JsError> {
        self.lod = build_lod(&mut self.inner)?;
        Ok(())
    }

    /// A new cloud with the points inside (or outside) the box `[min, max]`,
    /// given in original coordinates. Returns an error if nothing is kept.
    pub fn crop(&self, min: &[f64], max: &[f64], inside: bool) -> Result<Cloud, JsError> {
        let (min, max): ([f64; 3], [f64; 3]) = (
            min.try_into()
                .map_err(|_| JsError::new("min must have 3 components"))?,
            max.try_into()
                .map_err(|_| JsError::new("max must have 3 components"))?,
        );
        let mut inner = self.inner.crop(min, max, inside);
        if inner.is_empty() {
            return Err(JsError::new("no points in the selection"));
        }
        let lod = build_lod(&mut inner)?;
        Ok(Cloud {
            inner,
            lod,
            layout: None,
        })
    }

    /// Which points a screen-space lasso selects (1) or not (0).
    /// `clip_from_world` is a row-major 4x4 matrix from original coordinates
    /// to clip space, `polygon` flat x, y pairs in normalized device
    /// coordinates, `clip_box` empty or min then max (original coordinates).
    #[wasm_bindgen(js_name = lassoMask)]
    pub fn lasso_mask(
        &self,
        clip_from_world: &[f64],
        polygon: &[f64],
        clip_box: &[f64],
        hidden_classes: &[u8],
    ) -> Result<Vec<u8>, JsError> {
        let clip_from_world: [f64; 16] = clip_from_world
            .try_into()
            .map_err(|_| JsError::new("the matrix must have 16 values"))?;
        let polygon: Vec<[f64; 2]> = polygon.as_chunks::<2>().0.to_vec();
        let clip_box = match clip_box {
            [] => None,
            [a, b, c, d, e, f] => Some(([*a, *b, *c], [*d, *e, *f])),
            _ => return Err(JsError::new("the clip box must have 6 values")),
        };
        let lasso = ca_core::segment::Lasso {
            clip_from_world,
            polygon: &polygon,
            clip_box,
            hidden_classes,
        };
        Ok(ca_core::segment::lasso_mask(&self.inner, &lasso)
            .into_iter()
            .map(u8::from)
            .collect())
    }

    /// A copy of the points whose `mask` entry equals `value`, not yet
    /// indexed. Errors if there are none.
    #[wasm_bindgen(js_name = selectMask)]
    pub fn select_mask(&self, mask: &[u8], value: u8) -> Result<Cloud, JsError> {
        if mask.len() != self.inner.len() {
            return Err(JsError::new("the mask does not match the cloud"));
        }
        let keep: Vec<usize> = (0..mask.len()).filter(|&i| mask[i] == value).collect();
        if keep.is_empty() {
            return Err(JsError::new("no points in the selection"));
        }
        Ok(Cloud::unindexed(self.inner.select(&keep)))
    }

    /// A filtered copy of the cloud (with colors and attributes), not yet
    /// indexed (call [`Cloud::build_index`] or build it on the pool):
    /// `"voxel"` keeps one point per voxel of edge `a`; `"random"` keeps `a`
    /// random points; `"sor"` drops statistical outliers with `a` neighbours
    /// and a `b` standard-deviation threshold; `"splat"` keeps Gaussian
    /// splats with opacity at least `a` and size at most `b`.
    pub fn filter(&self, op: &str, a: f64, b: f64) -> Result<Cloud, JsError> {
        use ca_core::filter;
        let keep = match op {
            "voxel" => filter::voxel_subsample(&self.inner, a),
            "random" => filter::random_subsample(&self.inner, a.max(0.0) as usize, 0x5eed),
            "spatial" => filter::spatial_subsample(&self.inner, a),
            "octree" => filter::octree_subsample(&self.inner, a.max(0.0) as u32),
            "sor" => filter::statistical_outliers(&self.inner, a.max(1.0) as usize, b),
            "splat" => filter::splat_cleanup(&self.inner, a, b)
                .ok_or_else(|| JsError::new("not Gaussian splats (no opacity and size)"))?,
            other => return Err(JsError::new(&format!("unknown filter {other:?}"))),
        };
        self.selected(&keep)
    }

    /// SOR from the statistic computed on the pool (see [`plan_sor`]): an
    /// unindexed copy without the points more than `ratio` standard
    /// deviations above the mean.
    #[wasm_bindgen(js_name = filterSor)]
    pub fn filter_sor(&self, means: &[f64], ratio: f64) -> Result<Cloud, JsError> {
        if means.len() != self.inner.len() {
            return Err(JsError::new("one statistic per point is needed"));
        }
        self.selected(&ca_core::filter::sor_keep(means, ratio))
    }

    /// Cross-section: the points within `half_width` (horizontally) of the
    /// polyline `line` (`x, y` pairs in original coordinates) with their
    /// distance along it; beyond about `max_points`, every n-th of them.
    pub fn profile(
        &self,
        line: &[f64],
        half_width: f64,
        max_points: usize,
    ) -> Result<ProfileHits, JsError> {
        if !line.len().is_multiple_of(2) {
            return Err(JsError::new("the line needs x, y pairs"));
        }
        let hits =
            ca_core::profile::profile(&self.inner.positions, line.as_chunks::<2>().0, half_width);
        let step = hits.len().div_ceil(max_points.max(1)).max(1);
        let kept: Vec<_> = hits.iter().step_by(step).collect();
        Ok(ProfileHits {
            along: kept.iter().map(|h| h.1).collect(),
            positions: kept
                .iter()
                .flat_map(|h| self.inner.positions[h.0 as usize])
                .collect(),
            total: hits.len(),
        })
    }

    /// Interleaved unit normals (octree order), or `undefined` without
    /// `nx`/`ny`/`nz` attributes.
    pub fn normals(&self) -> Option<Vec<f32>> {
        ca_core::normals::normals(&self.inner).map(|n| n.into_flattened())
    }

    /// Estimate normals from the `k` nearest neighbours of each point here,
    /// oriented `"up"` (+z) or `"outward"` (away from the bounding box
    /// centre), and store them as attributes.
    #[wasm_bindgen(js_name = estimateNormals)]
    pub fn estimate_normals(&mut self, k: usize, orientation: &str) -> Result<(), JsError> {
        let orientation = self.orientation(orientation)?;
        let normals = ca_core::normals::estimate_normals(&self.inner.positions, k, orientation);
        ca_core::normals::set_normals(&mut self.inner, &normals);
        Ok(())
    }

    /// Orientation for [`normals_of`]: `[mode, cx, cy, cz]` with mode 0 = up,
    /// 1 = away from the bounding box centre `c`.
    #[wasm_bindgen(js_name = normalsOrientation)]
    pub fn normals_orientation(&self, orientation: &str) -> Result<Vec<f64>, JsError> {
        Ok(match self.orientation(orientation)? {
            ca_core::normals::Orientation::Up => vec![0.0, 0.0, 0.0, 0.0],
            ca_core::normals::Orientation::Away(c) => vec![1.0, c[0], c[1], c[2]],
        })
    }

    /// Store normals computed elsewhere (interleaved, octree order).
    #[wasm_bindgen(js_name = setNormals)]
    pub fn set_normals(&mut self, normals: &[f32]) -> Result<(), JsError> {
        if !ca_core::normals::set_normals(&mut self.inner, normals.as_chunks::<3>().0)
            || !normals.len().is_multiple_of(3)
        {
            return Err(JsError::new("one normal per point is needed"));
        }
        Ok(())
    }

    /// Distinct values of the `u8` attribute `name` (e.g. `"classification"`
    /// or `"source"`), ascending, or `undefined` without that attribute.
    #[wasm_bindgen(js_name = splitValues)]
    pub fn split_values(&self, name: &str) -> Option<Vec<u8>> {
        let groups = ca_core::merge::split_by(&self.inner, name)?;
        Some(groups.into_iter().map(|(v, _)| v).collect())
    }

    /// An unindexed copy of the points whose attribute `name` is `value`.
    #[wasm_bindgen(js_name = splitPart)]
    pub fn split_part(&self, name: &str, value: u8) -> Result<Cloud, JsError> {
        let Some(ca_core::Attribute {
            values: AttributeValues::U8(values),
            ..
        }) = self.inner.attribute(name)
        else {
            return Err(JsError::new(&format!("no {name} attribute to split by")));
        };
        let keep: Vec<usize> = (0..values.len()).filter(|&i| values[i] == value).collect();
        self.selected(&keep)
    }

    /// Ground extraction (Cloth Simulation Filter) as a new cloud.
    /// `rigidness` is `"flat"`, `"relief"` or `"steep"`; `output` is
    /// `"classified"` (a copy with class 2 = ground, 1 = the rest),
    /// `"ground"` or `"objects"`.
    #[wasm_bindgen(js_name = extractGround)]
    pub fn extract_ground(
        &self,
        cloth_resolution: f64,
        class_threshold: f64,
        rigidness: &str,
        output: &str,
    ) -> Result<Cloud, JsError> {
        use ca_core::ground::{CsfParams, Rigidness, csf};
        let rigidness = match rigidness {
            "flat" => Rigidness::Flat,
            "relief" => Rigidness::Relief,
            "steep" => Rigidness::Steep,
            other => return Err(JsError::new(&format!("unknown rigidness {other:?}"))),
        };
        let params = CsfParams {
            cloth_resolution,
            class_threshold,
            rigidness,
            ..CsfParams::default()
        };
        let ground = csf(&self.inner, params)
            .ok_or_else(|| JsError::new("the cloth resolution is too fine or not positive"))?;
        let mut inner = match output {
            "classified" => {
                let mut copy = self.inner.clone();
                let classes: Vec<u8> = ground.iter().map(|&g| if g { 2 } else { 1 }).collect();
                copy.attributes.retain(|a| a.name != CLASSIFICATION);
                copy.attributes.push(ca_core::Attribute {
                    name: CLASSIFICATION.into(),
                    values: AttributeValues::U8(classes),
                });
                copy
            }
            "ground" | "objects" => {
                let want = output == "ground";
                let keep: Vec<usize> = (0..ground.len()).filter(|&i| ground[i] == want).collect();
                self.inner.select(&keep)
            }
            other => return Err(JsError::new(&format!("unknown output {other:?}"))),
        };
        if inner.is_empty() {
            return Err(JsError::new("no points in the requested output"));
        }
        let lod = build_lod(&mut inner)?;
        Ok(Cloud {
            inner,
            lod,
            layout: None,
        })
    }

    /// Rasterize into a height grid of `cell`-sized squares (a DEM / DSM).
    /// `height` is `"mean"`, `"min"`, `"max"` or `"percentile"` (of
    /// `percentile`, 0-100); `class` restricts to one class code.
    pub fn rasterize(
        &self,
        cell: f64,
        height: &str,
        percentile: f64,
        fill_empty: bool,
        class: Option<u8>,
    ) -> Result<RasterOutput, JsError> {
        use ca_core::raster::{RasterHeight, RasterParams};
        let height = match height {
            "mean" => RasterHeight::Mean,
            "min" => RasterHeight::Min,
            "max" => RasterHeight::Max,
            "percentile" => RasterHeight::Percentile(percentile),
            other => return Err(JsError::new(&format!("unknown cell height {other:?}"))),
        };
        let params = RasterParams {
            cell,
            height,
            fill_empty,
            class,
        };
        let r = ca_core::raster::rasterize(&self.inner, params).ok_or_else(|| {
            JsError::new(match class {
                Some(_) => "no points of that class, or the cell size is not positive",
                None => "need a positive cell size (and percentile within 0-100)",
            })
        })?;
        let g = r.grid;
        let mut positions = Vec::new();
        let mut values = Vec::new();
        for j in 0..g.ny {
            for i in 0..g.nx {
                let h = r.heights[j * g.nx + i];
                if !h.is_nan() {
                    let [x, y] = g.center(i, j);
                    positions.push([x, y, h]);
                    values.push(h as f32);
                }
            }
        }
        let mut inner = PointCloud {
            positions,
            colors: None,
            attributes: vec![ca_core::Attribute {
                name: "height".into(),
                values: AttributeValues::F32(values),
            }],
        };
        let lod = build_lod(&mut inner)?;
        Ok(RasterOutput {
            nx: g.nx,
            ny: g.ny,
            min_x: g.min[0],
            min_y: g.min[1],
            cell: g.cell,
            populated_cells: r.populated_cells,
            heights: r.heights.iter().map(|&h| h as f32).collect(),
            cells: Some(Cloud {
                inner,
                lod,
                layout: None,
            }),
        })
    }

    /// RANSAC shape detection: up to `max_shapes` `"plane"`s, `"sphere"`s
    /// or `"cylinder"`s with at least `min_support` points within
    /// `threshold`, using the cloud's normals (estimated here if it has
    /// none). The result's cloud is an unindexed copy whose `source`
    /// attribute is the segment of each point (the shapes, then the rest);
    /// with `recolor`, its RGB colors show the segments.
    #[wasm_bindgen(js_name = detectShapes)]
    pub fn detect_shapes(
        &self,
        primitive: &str,
        threshold: f64,
        min_support: usize,
        max_shapes: usize,
        recolor: bool,
    ) -> Result<Segmentation, JsError> {
        use ca_core::shapes::{Primitive, RansacParams, Shape};
        let (primitive, kind) = match primitive {
            "plane" => (Primitive::Plane, SEGMENT_PLANE),
            "sphere" => (Primitive::Sphere, SEGMENT_SPHERE),
            "cylinder" => (Primitive::Cylinder, SEGMENT_CYLINDER),
            other => return Err(JsError::new(&format!("unknown shape {other:?}"))),
        };
        if threshold.is_nan() || threshold <= 0.0 {
            return Err(JsError::new("the distance threshold must be positive"));
        }
        let normals = ca_core::normals::normals(&self.inner).unwrap_or_else(|| {
            ca_core::normals::estimate_normals(
                &self.inner.positions,
                12,
                ca_core::normals::Orientation::Up,
            )
        });
        let params = RansacParams {
            primitive,
            threshold,
            min_support,
            // Segment numbers are bytes, and the rest takes one.
            max_shapes: max_shapes.min(u8::MAX as usize),
            ..RansacParams::default()
        };
        let found = ca_core::shapes::detect_shapes(&self.inner.positions, &normals, &params);
        let mut out = Segmentation::new(found.len());
        let mut segment = vec![found.len() as u8; self.inner.len()];
        for (k, d) in found.iter().enumerate() {
            for &i in &d.indices {
                segment[i] = k as u8;
            }
            let params = match d.shape {
                Shape::Plane { normal, d } => {
                    [normal[0], normal[1], normal[2], d, 0.0, 0.0, 0.0, 0.0]
                }
                Shape::Sphere { center, radius } => {
                    [center[0], center[1], center[2], radius, 0.0, 0.0, 0.0, 0.0]
                }
                Shape::Cylinder {
                    point,
                    axis,
                    radius,
                    length,
                } => [
                    point[0], point[1], point[2], axis[0], axis[1], axis[2], radius, length,
                ],
            };
            out.push(kind, d.indices.len(), params, d.rms, segment_color(k));
        }
        let rest = self.inner.len() - found.iter().map(|d| d.indices.len()).sum::<usize>();
        if rest > 0 {
            out.push(SEGMENT_REST, rest, [0.0; 8], f64::NAN, REST_COLOR);
        }
        let colors = recolor.then(|| segment.iter().map(|&s| out.color(s as usize)).collect());
        out.cloud = Some(self.segmented(segment, colors));
        Ok(out)
    }

    /// Euclidean clustering: points within `epsilon` of each other form a
    /// cluster, and clusters under `min_size` points are noise. Like
    /// [`Cloud::detect_shapes`], the segments are the clusters (largest
    /// first; past 254 of them the rest are one segment), then the noise;
    /// `recolor` colors every cluster on its own.
    pub fn clusters(
        &self,
        epsilon: f64,
        min_size: usize,
        recolor: bool,
    ) -> Result<Segmentation, JsError> {
        use ca_core::cluster::{NOISE, euclidean_clusters};
        if epsilon.is_nan() || epsilon <= 0.0 {
            return Err(JsError::new("the cluster distance must be positive"));
        }
        let positions = &self.inner.positions;
        let clusters = euclidean_clusters(positions, epsilon, min_size);
        let own = clusters.sizes.len().min(u8::MAX as usize - 1);
        let others = own < clusters.sizes.len();
        let noise = own as u8 + others as u8;
        let segment: Vec<u8> = clusters
            .labels
            .iter()
            .map(|&l| match l {
                NOISE => noise,
                l if (l as usize) < own => l as u8,
                _ => own as u8,
            })
            .collect();
        // Centroid and extent of each segment.
        let segments = noise as usize + 1;
        let mut sum = vec![[0.0f64; 3]; segments];
        let mut lo = vec![[f64::INFINITY; 3]; segments];
        let mut hi = vec![[f64::NEG_INFINITY; 3]; segments];
        let mut count = vec![0usize; segments];
        for (p, &s) in positions.iter().zip(&segment) {
            let s = s as usize;
            count[s] += 1;
            for a in 0..3 {
                sum[s][a] += p[a];
                lo[s][a] = lo[s][a].min(p[a]);
                hi[s][a] = hi[s][a].max(p[a]);
            }
        }
        let mut out = Segmentation::new(clusters.sizes.len());
        for s in (0..segments).filter(|&s| count[s] > 0) {
            let m = count[s] as f64;
            let params = [
                sum[s][0] / m,
                sum[s][1] / m,
                sum[s][2] / m,
                hi[s][0] - lo[s][0],
                hi[s][1] - lo[s][1],
                hi[s][2] - lo[s][2],
                0.0,
                0.0,
            ];
            let (kind, color) = match s {
                s if s < own => (SEGMENT_CLUSTER, segment_color(s)),
                s if s == own && others => (SEGMENT_OTHER_CLUSTERS, OTHER_COLOR),
                _ => (SEGMENT_REST, REST_COLOR),
            };
            out.push(kind, count[s], params, f64::NAN, color);
        }
        let colors = recolor.then(|| {
            clusters
                .labels
                .iter()
                .map(|&l| match l {
                    NOISE => REST_COLOR,
                    l => segment_color(l as usize),
                })
                .collect()
        });
        out.cloud = Some(self.segmented(segment, colors));
        Ok(out)
    }

    /// 2.5D Delaunay mesh: the points triangulated in the XY plane, heights
    /// kept. Triangles with a horizontal edge longer than `max_edge` are
    /// dropped (`undefined`: 4x the median edge; 0 keeps them all). A cloud
    /// of more than `max_points` is first thinned with a voxel filter.
    #[wasm_bindgen(js_name = meshDelaunay)]
    pub fn mesh_delaunay(
        &self,
        max_edge: Option<f64>,
        max_points: usize,
    ) -> Result<DelaunayOutput, JsError> {
        use ca_core::delaunay::{MaxEdge, delaunay_25d, thin_for_meshing};
        let max_edge = match max_edge {
            None => MaxEdge::Auto,
            Some(l) if l == 0.0 || l == f64::INFINITY => MaxEdge::Unlimited,
            Some(l) if l > 0.0 => MaxEdge::Length(l),
            Some(_) => return Err(JsError::new("the max edge length must not be negative")),
        };
        let thinned = thin_for_meshing(&self.inner, max_points);
        let points = match &thinned {
            Some((keep, _)) => keep.iter().map(|&i| self.inner.positions[i]).collect(),
            None => self.inner.positions.clone(),
        };
        let out = delaunay_25d(&points, max_edge)
            .ok_or_else(|| JsError::new("the points are collinear in XY"))?;
        if out.mesh.triangles.is_empty() {
            return Err(JsError::new(
                "every triangle is longer than the max edge length",
            ));
        }
        Ok(DelaunayOutput {
            points: points.len(),
            voxel: thinned.map_or(0.0, |(_, voxel)| voxel),
            max_edge: out.max_edge,
            removed: out.removed,
            mesh: Some(Mesh { inner: out.mesh }),
        })
    }

    /// Apply a rigid transform (row-major 4x4) to every point and rebuild the
    /// octree, which also changes the point order.
    pub fn transform(&mut self, matrix: &[f64]) -> Result<(), JsError> {
        let matrix: &[f64; 16] = matrix
            .try_into()
            .map_err(|_| JsError::new("matrix must have 16 entries"))?;
        let rigid = Rigid::from_matrix(matrix);
        for p in &mut self.inner.positions {
            *p = rigid.apply(p);
        }
        self.lod = build_lod(&mut self.inner)?;
        Ok(())
    }

    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.inner.len()
    }

    /// `[minX, minY, minZ, maxX, maxY, maxZ]`.
    pub fn bounds(&self) -> Vec<f64> {
        self.inner
            .bounds()
            .map(|b| b.min.into_iter().chain(b.max).collect())
            .unwrap_or_default()
    }

    /// Shift that brings this cloud near the origin (zero for small coordinates).
    #[wasm_bindgen(js_name = suggestedShift)]
    pub fn suggested_shift(&self) -> Vec<f64> {
        self.inner.suggested_shift().to_vec()
    }

    /// Interleaved `xyz` positions minus `shift`, narrowed to `f32` for rendering.
    pub fn positions(&self, shift: &[f64]) -> Result<Vec<f32>, JsError> {
        let shift: [f64; 3] = shift
            .try_into()
            .map_err(|_| JsError::new("shift must have 3 components"))?;
        let mut out = Vec::with_capacity(self.inner.len() * 3);
        for p in &self.inner.positions {
            out.extend([
                (p[0] - shift[0]) as f32,
                (p[1] - shift[1]) as f32,
                (p[2] - shift[2]) as f32,
            ]);
        }
        Ok(out)
    }

    /// Exact `f64` coordinates of point `index` (octree order), or
    /// `undefined` when out of range.
    pub fn point(&self, index: usize) -> Option<Vec<f64>> {
        self.inner.positions.get(index).map(|p| p.to_vec())
    }

    /// Interleaved `xyz` at full `f64` precision (octree order), e.g. to
    /// split a C2M job across workers.
    #[wasm_bindgen(js_name = rawPositions)]
    pub fn raw_positions(&self) -> Vec<f64> {
        self.inner.positions.as_flattened().to_vec()
    }

    /// Serialize the cloud as `"ply"` (binary), `"las"`, `"laz"`, `"csv"` or
    /// `"e57"`, with an optional scalar field (one value per point, in the
    /// cloud's order; E57 keeps only XYZ, intensity and RGB). Points are
    /// written in octree order, not the original file order.
    pub fn export(
        &self,
        format: &str,
        scalar_name: Option<String>,
        scalar: Option<Vec<f32>>,
    ) -> Result<Vec<u8>, JsError> {
        let field = match (&scalar_name, &scalar) {
            (Some(name), Some(values)) => Some(ca_core::io::ScalarField { name, values }),
            (None, None) => None,
            _ => return Err(JsError::new("scalar name and values go together")),
        };
        let fields = field.as_slice();
        match format {
            "ply" => ca_core::io::write_ply(&self.inner, fields),
            "las" => ca_core::io::write_las(&self.inner, fields, false),
            "laz" => ca_core::io::write_las(&self.inner, fields, true),
            "csv" => ca_core::io::write_csv(&self.inner, fields),
            "e57" => ca_core::io::write_e57(&self.inner),
            other => return Err(JsError::new(&format!("unknown export format {other:?}"))),
        }
        .map_err(|e| JsError::new(&e))
    }

    /// The significant changes of an M3C2 result grouped into objects (see
    /// `ca_core::m3c2::changed_objects`): per object, largest first,
    /// `[count, centroid xyz, min xyz, max xyz, mean change]`. Each point
    /// also gets a `change_object` field: its object's rank from 1, or NaN.
    #[wasm_bindgen(js_name = changedObjects)]
    pub fn changed_objects(
        &mut self,
        min_change: f64,
        link: f64,
        min_points: usize,
    ) -> Result<Vec<f64>, JsError> {
        let change: Vec<f64> = match self.inner.attribute("m3c2_distance").map(|a| &a.values) {
            Some(AttributeValues::F32(v)) => v.iter().map(|&d| f64::from(d)).collect(),
            _ => return Err(JsError::new("not an M3C2 result (no m3c2_distance)")),
        };
        let significant: Vec<bool> = match self.inner.attribute("significant").map(|a| &a.values) {
            Some(AttributeValues::U8(v)) => v.iter().map(|&s| s != 0).collect(),
            Some(AttributeValues::F32(v)) => v.iter().map(|&s| s > 0.0).collect(),
            _ => return Err(JsError::new("not an M3C2 result (no significant)")),
        };
        let (objects, labels) = ca_core::m3c2::changed_objects(
            &self.inner.positions,
            &change,
            &significant,
            min_change,
            link,
            min_points,
        );
        self.inner.attributes.retain(|a| a.name != "change_object");
        self.inner.attributes.push(ca_core::Attribute {
            name: "change_object".into(),
            values: AttributeValues::F32(
                labels
                    .iter()
                    .map(|&l| {
                        if l == ca_core::cluster::NOISE {
                            f32::NAN
                        } else {
                            (l + 1) as f32
                        }
                    })
                    .collect(),
            ),
        });
        Ok(objects
            .iter()
            .flat_map(|o| {
                let mut row = vec![o.count as f64];
                row.extend(o.centroid);
                row.extend(o.min);
                row.extend(o.max);
                row.push(o.mean_change);
                row
            })
            .collect())
    }

    /// Values of a named per-point attribute as `f32` (octree order), or
    /// `undefined` when the cloud does not have it.
    pub fn attribute(&self, name: &str) -> Option<Vec<f32>> {
        match &self.inner.attribute(name)?.values {
            AttributeValues::F32(v) => Some(v.clone()),
            AttributeValues::U8(v) => Some(v.iter().map(|&x| x as f32).collect()),
        }
    }

    /// Names of the attributes that make sense as scalar fields: every
    /// float attribute except the normal components, plus intensity.
    #[wasm_bindgen(js_name = scalarNames)]
    pub fn scalar_names(&self) -> Vec<String> {
        self.inner
            .attributes
            .iter()
            .filter(|a| {
                a.name == INTENSITY
                    || (matches!(a.values, AttributeValues::F32(_))
                        && !ca_core::normals::NORMAL_NAMES.contains(&a.name.as_str()))
            })
            .map(|a| a.name.clone())
            .collect()
    }

    /// Store `values` (one per point, octree order) as a float attribute,
    /// replacing one of the same name, so exports carry it.
    #[wasm_bindgen(js_name = setAttribute)]
    pub fn set_attribute(&mut self, name: &str, values: Vec<f32>) -> Result<(), JsError> {
        if values.len() != self.inner.len() {
            return Err(JsError::new("need one value per point"));
        }
        if name.is_empty()
            || name == CLASSIFICATION
            || ca_core::normals::NORMAL_NAMES.contains(&name)
        {
            return Err(JsError::new(&format!(
                "{name:?} cannot be used as a field name"
            )));
        }
        self.inner.attributes.retain(|a| a.name != name);
        self.inner.attributes.push(ca_core::Attribute {
            name: name.into(),
            values: AttributeValues::F32(values),
        });
        Ok(())
    }

    /// Z of every point as `f32` (octree order).
    #[wasm_bindgen(js_name = heights)]
    pub fn heights(&self) -> Vec<f32> {
        self.inner.positions.iter().map(|p| p[2] as f32).collect()
    }

    /// Per-point intensity (octree order), or `undefined` when the file has none.
    pub fn intensity(&self) -> Option<Vec<f32>> {
        match &self.inner.attribute(INTENSITY)?.values {
            AttributeValues::F32(v) => Some(v.clone()),
            AttributeValues::U8(v) => Some(v.iter().map(|&x| x as f32).collect()),
        }
    }

    /// Per-point class codes (octree order), or `undefined` when the file has none.
    pub fn classification(&self) -> Option<Vec<u8>> {
        match &self.inner.attribute(CLASSIFICATION)?.values {
            AttributeValues::U8(v) => Some(v.clone()),
            AttributeValues::F32(v) => Some(v.iter().map(|&x| x.clamp(0.0, 255.0) as u8).collect()),
        }
    }

    /// Interleaved `rgb` bytes, or `undefined` when the file has no colors.
    pub fn colors(&self) -> Option<Vec<u8>> {
        self.inner
            .colors
            .as_ref()
            .map(|c| c.as_flattened().to_vec())
    }

    /// Octree nodes, 15 numbers each: `start, count, minX, minY, minZ, size,
    /// level, child0..child7` (children are node indices or -1). `min` is in
    /// original coordinates; subtract the shift used for `positions`.
    #[wasm_bindgen(js_name = lodNodes)]
    pub fn lod_nodes(&self) -> Vec<f64> {
        nodes_to_flat(&self.lod.nodes)
    }

    /// Root cube of a parallel index build: `[minX, minY, minZ, size]`.
    /// Build it in three steps: [`bucket_chunk`] on contiguous slices from
    /// [`Cloud::positions_range`] / [`Cloud::colors_range`]; [`build_bucket`]
    /// on each bucket's pieces concatenated in slice order; then
    /// [`Cloud::begin_buckets`], [`Cloud::put_bucket`] for each and
    /// [`Cloud::finish_buckets`].
    #[wasm_bindgen(js_name = indexCube)]
    pub fn index_cube(&self) -> Vec<f64> {
        let (lo, size) = ca_core::octree::root_cube(&self.inner.positions);
        vec![lo[0], lo[1], lo[2], size]
    }

    /// Interleaved `xyz` of points `start..end` (a copy).
    #[wasm_bindgen(js_name = positionsRange)]
    pub fn positions_range(&self, start: usize, end: usize) -> Vec<f64> {
        let end = end.min(self.inner.len());
        self.inner.positions[start.min(end)..end]
            .as_flattened()
            .to_vec()
    }

    /// Interleaved `rgb` of points `start..end`, if the cloud has colors.
    #[wasm_bindgen(js_name = colorsRange)]
    pub fn colors_range(&self, start: usize, end: usize) -> Option<Vec<u8>> {
        let end = end.min(self.inner.len());
        self.inner
            .colors
            .as_ref()
            .map(|c| c[start.min(end)..end].as_flattened().to_vec())
    }

    /// Start placing built buckets; per bucket, its size and the counts
    /// returned by [`build_bucket`].
    #[wasm_bindgen(js_name = beginBuckets)]
    pub fn begin_buckets(
        &mut self,
        cube: &[f64],
        sizes: &[u32],
        root_kept: &[u32],
        level1_kept: &[u32],
    ) -> Result<(), JsError> {
        let (lo, size) = cube_of(cube)?;
        let arr = |v: &[u32]| -> Result<[u32; BUCKETS], JsError> {
            v.try_into()
                .map_err(|_| JsError::new("expected one count per bucket"))
        };
        let layout = BucketLayout::new(
            lo,
            size,
            OctreeParams::default(),
            arr(sizes)?,
            arr(root_kept)?,
            arr(level1_kept)?,
        )
        .filter(|l| l.len() == self.inner.len())
        .ok_or_else(|| JsError::new("bucket counts do not match the cloud"))?;
        self.layout = Some(layout);
        Ok(())
    }

    /// Place bucket `key` as returned by [`build_bucket`]; `order` gives the
    /// cloud index of each of its points.
    #[wasm_bindgen(js_name = putBucket)]
    pub fn put_bucket(
        &mut self,
        key: usize,
        positions: &[f64],
        colors: Option<Vec<u8>>,
        order: &[u32],
        nodes: &[f64],
    ) -> Result<(), JsError> {
        let nodes = nodes_from_flat(nodes)?;
        let layout = self
            .layout
            .as_mut()
            .ok_or_else(|| JsError::new("beginBuckets was not called"))?;
        layout
            .put(
                key,
                &mut self.inner.positions,
                self.inner.colors.as_deref_mut(),
                positions.as_chunks::<3>().0,
                colors.as_ref().map(|c| c.as_chunks::<3>().0),
                order,
                nodes,
            )
            .ok_or_else(|| JsError::new("bucket does not match its counts"))
    }

    /// Finish a parallel index build once every bucket was placed.
    #[wasm_bindgen(js_name = finishBuckets)]
    pub fn finish_buckets(&mut self) -> Result<(), JsError> {
        let layout = self
            .layout
            .take()
            .ok_or_else(|| JsError::new("beginBuckets was not called"))?;
        let mut lod = layout
            .finish(&mut self.inner.attributes)
            .ok_or_else(|| JsError::new("a bucket is missing"))?;
        lod.order = Vec::new();
        self.lod = lod;
        Ok(())
    }

    /// Subsampling lattice resolution per node edge; a node's point spacing is
    /// about `size / lodGrid`.
    #[wasm_bindgen(getter, js_name = lodGrid)]
    pub fn lod_grid(&self) -> u32 {
        self.lod.grid
    }
}

fn nodes_to_flat(nodes: &[OctreeNode]) -> Vec<f64> {
    let mut out = Vec::with_capacity(nodes.len() * NODE_STRIDE);
    for n in nodes {
        out.extend([n.start as f64, n.count as f64]);
        out.extend(n.min);
        out.extend([n.size, n.level as f64]);
        out.extend(
            n.children
                .map(|c| if c == NO_CHILD { -1.0 } else { c as f64 }),
        );
    }
    out
}

fn nodes_from_flat(flat: &[f64]) -> Result<Vec<OctreeNode>, JsError> {
    if !flat.len().is_multiple_of(NODE_STRIDE) {
        return Err(JsError::new("node table has the wrong length"));
    }
    Ok(flat
        .as_chunks::<NODE_STRIDE>()
        .0
        .iter()
        .map(|n| OctreeNode {
            start: n[0] as u32,
            count: n[1] as u32,
            min: [n[2], n[3], n[4]],
            size: n[5],
            level: n[6] as u8,
            children: std::array::from_fn(|c| {
                if n[7 + c] < 0.0 {
                    NO_CHILD
                } else {
                    n[7 + c] as u32
                }
            }),
        })
        .collect())
}

fn cube_of(cube: &[f64]) -> Result<([f64; 3], f64), JsError> {
    match cube {
        &[x, y, z, size] if size > 0.0 => Ok(([x, y, z], size)),
        _ => Err(JsError::new("cube must be [minX, minY, minZ, size]")),
    }
}

fn colors_of(colors: Option<Vec<u8>>, points: usize) -> Result<Option<Vec<[u8; 3]>>, JsError> {
    match colors {
        Some(c) if c.len() != 3 * points => Err(JsError::new("colors do not match positions")),
        c => Ok(c.map(|c| c.as_chunks::<3>().0.to_vec())),
    }
}

/// Points reordered off the main worker, with what the reordering produced.
#[wasm_bindgen]
pub struct Reordered {
    positions: Vec<[f64; 3]>,
    colors: Option<Vec<[u8; 3]>>,
    order: Vec<u32>,
    counts: Vec<u32>,
    nodes: Vec<OctreeNode>,
}

#[wasm_bindgen]
impl Reordered {
    pub fn positions(&self) -> Vec<f64> {
        self.positions.as_flattened().to_vec()
    }

    pub fn colors(&self) -> Option<Vec<u8>> {
        self.colors.as_ref().map(|c| c.as_flattened().to_vec())
    }

    /// `order[i]` is the input index of the point now at `i`.
    pub fn order(&self) -> Vec<u32> {
        self.order.clone()
    }

    /// [`bucket_chunk`]: points per bucket. [`build_bucket`]: the root's and
    /// the level-1 node's share of the bucket.
    pub fn counts(&self) -> Vec<u32> {
        self.counts.clone()
    }

    /// [`build_bucket`]: the subtree's node table (see [`Cloud::lod_nodes`]),
    /// ranges relative to the subtree.
    pub fn nodes(&self) -> Vec<f64> {
        nodes_to_flat(&self.nodes)
    }
}

/// Step 1 of a parallel index build (see [`Cloud::index_cube`]): sort a
/// slice of the cloud by bucket, keeping the order within each bucket.
#[wasm_bindgen(js_name = bucketChunk)]
pub fn bucket_chunk(
    positions: &[f64],
    colors: Option<Vec<u8>>,
    cube: &[f64],
) -> Result<Reordered, JsError> {
    let (lo, size) = cube_of(cube)?;
    let points = positions.as_chunks::<3>().0;
    let colors = colors_of(colors, points.len())?;
    let (order, counts) = ca_core::octree::bucket_order(points, lo, size);
    Ok(Reordered {
        positions: order.iter().map(|&i| points[i as usize]).collect(),
        colors: colors.map(|c| order.iter().map(|&i| c[i as usize]).collect()),
        order,
        counts: counts.to_vec(),
        nodes: Vec::new(),
    })
}

/// Step 2 of a parallel index build: one bucket's points (concatenated from
/// every slice in order) reordered into `[root's | level-1 node's | subtree]`.
#[wasm_bindgen(js_name = buildBucket)]
pub fn build_bucket(
    positions: &[f64],
    colors: Option<Vec<u8>>,
    cube: &[f64],
    key: usize,
) -> Result<Reordered, JsError> {
    let (lo, size) = cube_of(cube)?;
    let mut points = positions.as_chunks::<3>().0.to_vec();
    let mut colors = colors_of(colors, points.len())?;
    let bucket = ca_core::octree::build_bucket(
        &mut points,
        colors.as_deref_mut(),
        lo,
        size,
        key,
        OctreeParams::default(),
    )
    .ok_or_else(|| JsError::new("invalid bucket"))?;
    Ok(Reordered {
        positions: points,
        colors,
        order: bucket.order,
        counts: vec![bucket.root_kept as u32, bucket.level1_kept as u32],
        nodes: bucket.nodes,
    })
}

/// Reorder `cloud` in place into octree order (colors follow) and return
/// the octree without the permutation, which the viewer does not need.
fn build_lod(cloud: &mut PointCloud) -> Result<Octree, JsError> {
    let mut lod = Octree::build_for_cloud(cloud, OctreeParams::default())
        .ok_or_else(|| JsError::new("cloud is empty or too large"))?;
    lod.order = Vec::new();
    Ok(lod)
}

/// Outcome of [`register_icp`].
#[wasm_bindgen]
pub struct IcpOutcome {
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
pub fn register_icp(
    moving: &Cloud,
    reference: &Cloud,
    max_iterations: usize,
    overlap: f64,
    match_centroids: bool,
    point_to_plane: bool,
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
    let result = ca_core::icp::icp(&moving.inner, &reference.inner, params)
        .ok_or_else(|| JsError::new("ICP needs non-empty clouds and at least 3 pairs"))?;
    Ok(IcpOutcome {
        matrix: result.transform.to_matrix(),
        rms_initial: result.rms_initial,
        rms_final: result.rms_final,
        iterations: result.iterations,
        converged: result.converged,
    })
}

/// M3C2 change from `reference` (cloud 1) to `compared` (cloud 2) at core
/// points taken from `compared` (all points, or one per `core_spacing`
/// voxel). Returns the core points as a new indexed cloud carrying the
/// `m3c2_distance` and `lod95` (NaN where undefined) and `significant`
/// (1/0) attributes.
/// Map quality of `estimate` against the ground-truth map `truth` by
/// voxel Gaussians (see `ca_core::map_quality`): `[AWD, SCS, voxels]`.
#[wasm_bindgen(js_name = mapQuality)]
pub fn map_quality(estimate: &Cloud, truth: &Cloud, voxel: f64, min_points: usize) -> Vec<f64> {
    let s = ca_core::map_quality::voxel_scores(
        &estimate.inner.positions,
        &truth.inner.positions,
        voxel,
        min_points,
    );
    vec![s.awd, s.scs, s.voxels as f64]
}

#[wasm_bindgen(js_name = computeM3c2)]
pub fn compute_m3c2(
    compared: &Cloud,
    reference: &Cloud,
    normal_radius: f64,
    projection_radius: f64,
    max_depth: f64,
    core_spacing: f64,
) -> Result<Cloud, JsError> {
    use ca_core::m3c2::{M3c2Params, m3c2};
    let core = if core_spacing > 0.0 {
        compared.inner.select(&ca_core::filter::voxel_subsample(
            &compared.inner,
            core_spacing,
        ))
    } else {
        compared.inner.clone()
    };
    let params = M3c2Params {
        normal_radius,
        projection_radius,
        max_depth,
        ..M3c2Params::default()
    };
    let r = m3c2(
        &core.positions,
        &reference.inner.positions,
        &compared.inner.positions,
        params,
    )
    .ok_or_else(|| JsError::new("M3C2 needs non-empty clouds and positive radii"))?;
    let mut inner = core;
    inner
        .attributes
        .retain(|a| !matches!(a.name.as_str(), "m3c2_distance" | "lod95" | "significant"));
    inner.attributes.push(ca_core::Attribute {
        name: "m3c2_distance".into(),
        values: AttributeValues::F32(r.distance.iter().map(|&d| d as f32).collect()),
    });
    inner.attributes.push(ca_core::Attribute {
        name: "lod95".into(),
        values: AttributeValues::F32(r.lod95.iter().map(|&d| d as f32).collect()),
    });
    inner.attributes.push(ca_core::Attribute {
        name: "significant".into(),
        values: AttributeValues::U8(r.significant.iter().map(|&s| u8::from(s)).collect()),
    });
    let lod = build_lod(&mut inner)?;
    Ok(Cloud {
        inner,
        lod,
        layout: None,
    })
}

/// Result of a cloud-to-cloud distance computation.
#[wasm_bindgen]
pub struct C2cResult {
    distances: Vec<f32>,
    stats: DistanceStats,
}

#[wasm_bindgen]
impl C2cResult {
    /// Per-point distances of the compared cloud, in its point order.
    pub fn distances(&self) -> Vec<f32> {
        self.distances.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn count(&self) -> usize {
        self.stats.count
    }
    #[wasm_bindgen(getter)]
    pub fn min(&self) -> f64 {
        self.stats.min
    }
    #[wasm_bindgen(getter)]
    pub fn max(&self) -> f64 {
        self.stats.max
    }
    #[wasm_bindgen(getter)]
    pub fn mean(&self) -> f64 {
        self.stats.mean
    }
    #[wasm_bindgen(getter)]
    pub fn rms(&self) -> f64 {
        self.stats.rms
    }
    #[wasm_bindgen(getter, js_name = stdDev)]
    pub fn std_dev(&self) -> f64 {
        self.stats.std_dev
    }
    #[wasm_bindgen(getter)]
    pub fn median(&self) -> f64 {
        self.stats.median
    }
}

/// Distance from every point of `compared` to its nearest neighbour in `reference`.
#[wasm_bindgen(js_name = cloudToCloud)]
pub fn cloud_to_cloud(compared: &Cloud, reference: &Cloud) -> Result<C2cResult, JsError> {
    let distances = ca_core::cloud_to_cloud(&compared.inner, &reference.inner)
        .ok_or_else(|| JsError::new("reference cloud is empty"))?;
    summarize_distances(&distances)
}

fn from_flat(xyz: &[f64]) -> Result<PointCloud, JsError> {
    if !xyz.len().is_multiple_of(3) {
        return Err(JsError::new("positions length must be a multiple of 3"));
    }
    Ok(PointCloud {
        positions: xyz.as_chunks::<3>().0.to_vec(),
        colors: None,
        attributes: Vec::new(),
    })
}

/// A triangle mesh (PLY with faces, OBJ, STL), used as a C2M reference.
#[wasm_bindgen]
pub struct Mesh {
    inner: TriangleMesh,
}

#[wasm_bindgen]
impl Mesh {
    /// Parse a mesh file, or return `undefined` when the file is a point
    /// cloud (e.g. a PLY without faces).
    pub fn parse(name: &str, bytes: &[u8]) -> Result<Option<Mesh>, JsError> {
        Ok(ca_core::read_mesh(name, bytes)?.map(|inner| Mesh { inner }))
    }

    #[wasm_bindgen(getter, js_name = vertexCount)]
    pub fn vertex_count(&self) -> usize {
        self.inner.vertices.len()
    }

    #[wasm_bindgen(getter, js_name = triangleCount)]
    pub fn triangle_count(&self) -> usize {
        self.inner.triangles.len()
    }

    fn as_cloud(&self) -> PointCloud {
        PointCloud {
            positions: self.inner.vertices.clone(),
            colors: None,
            attributes: Vec::new(),
        }
    }

    /// `[minX, minY, minZ, maxX, maxY, maxZ]`.
    pub fn bounds(&self) -> Vec<f64> {
        self.as_cloud()
            .bounds()
            .map(|b| b.min.into_iter().chain(b.max).collect())
            .unwrap_or_default()
    }

    #[wasm_bindgen(js_name = suggestedShift)]
    pub fn suggested_shift(&self) -> Vec<f64> {
        self.as_cloud().suggested_shift().to_vec()
    }

    /// Interleaved vertex `xyz` minus `shift`, narrowed to `f32` for rendering.
    pub fn positions(&self, shift: &[f64]) -> Result<Vec<f32>, JsError> {
        let shift: [f64; 3] = shift
            .try_into()
            .map_err(|_| JsError::new("shift must have 3 components"))?;
        let mut out = Vec::with_capacity(self.inner.vertices.len() * 3);
        for p in &self.inner.vertices {
            out.extend([
                (p[0] - shift[0]) as f32,
                (p[1] - shift[1]) as f32,
                (p[2] - shift[2]) as f32,
            ]);
        }
        Ok(out)
    }

    /// Triangle vertex indices, three per triangle.
    pub fn indices(&self) -> Vec<u32> {
        self.inner.triangles.as_flattened().to_vec()
    }

    /// Interleaved vertex `xyz` at full precision, for [`mesh_distances`].
    #[wasm_bindgen(js_name = rawVertices)]
    pub fn raw_vertices(&self) -> Vec<f64> {
        self.inner.vertices.as_flattened().to_vec()
    }

    /// Serialize the mesh as `"ply"` (binary) or `"obj"`.
    pub fn export(&self, format: &str) -> Result<Vec<u8>, JsError> {
        match format {
            "ply" => Ok(ca_core::io::write_mesh_ply(&self.inner)),
            "obj" => Ok(ca_core::io::write_obj(&self.inner)),
            other => Err(JsError::new(&format!("unknown mesh format {other:?}"))),
        }
    }
}

/// Distance from every point of `compared` to `mesh` (C2M); negative behind
/// the closest triangle when `signed`.
#[wasm_bindgen(js_name = cloudToMesh)]
pub fn cloud_to_mesh(compared: &Cloud, mesh: &Mesh, signed: bool) -> Result<C2cResult, JsError> {
    let distances = ca_core::cloud_to_mesh(&compared.inner.positions, &mesh.inner, signed)
        .ok_or_else(|| JsError::new("mesh has no triangles"))?;
    summarize_distances(&distances)
}

/// C2M distances for a batch of interleaved `xyz` queries against a mesh
/// given as raw vertices and triangle indices. Used by the worker pool.
#[wasm_bindgen(js_name = meshDistances)]
pub fn mesh_distances(
    vertices: &[f64],
    indices: &[u32],
    queries: &[f64],
    signed: bool,
) -> Result<Vec<f64>, JsError> {
    if !vertices.len().is_multiple_of(3) || !indices.len().is_multiple_of(3) {
        return Err(JsError::new("vertices and indices must come in triples"));
    }
    let mut mesh = TriangleMesh {
        vertices: vertices.as_chunks::<3>().0.to_vec(),
        triangles: indices.as_chunks::<3>().0.to_vec(),
    };
    mesh.validate();
    let points = from_flat(queries)?.positions;
    ca_core::cloud_to_mesh(&points, &mesh, signed)
        .ok_or_else(|| JsError::new("mesh has no triangles"))
}

/// Nearest-neighbour distances for a batch of interleaved `xyz` queries.
/// Used by the worker pool, where each worker handles one slice of the
/// compared cloud against its own copy of the reference.
/// SOR split into spatially compact parts for the worker pool; see
/// [`ca_core::filter::local_knn`].
#[wasm_bindgen]
pub struct SorPlan {
    split: ca_core::filter::KnnSplit,
}

#[wasm_bindgen]
impl SorPlan {
    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.split.parts.len()
    }

    /// Cloud indices of part `k`.
    pub fn indices(&self, k: usize) -> Vec<u32> {
        self.split.parts[k].clone()
    }

    /// Interleaved `xyz` of part `k`'s points (a copy).
    pub fn points(&self, cloud: &Cloud, k: usize) -> Vec<f64> {
        self.split.parts[k]
            .iter()
            .flat_map(|&i| cloud.inner.positions[i as usize])
            .collect()
    }

    /// Where the parts lie, for [`SorPart::local`].
    pub fn regions(&self) -> Vec<f64> {
        self.split.regions.to_flat()
    }
}

#[wasm_bindgen(js_name = planSor)]
pub fn plan_sor(cloud: &Cloud, parts: usize) -> SorPlan {
    // The octree order already groups nearby points: split along it when
    // the cloud is indexed, instead of sorting.
    let split = if cloud.lod.nodes.is_empty() {
        ca_core::filter::split_for_knn(&cloud.inner.positions, parts)
    } else {
        ca_core::filter::split_octree_for_knn(&cloud.inner.positions, &cloud.lod.nodes, parts, 4)
    };
    SorPlan { split }
}

/// One part of a split SOR job, kept on a pool worker between its two steps.
#[wasm_bindgen]
pub struct SorPart {
    part: ca_core::filter::KnnPart,
}

/// Outcome of [`SorPart::local`].
#[wasm_bindgen]
pub struct SorLocal {
    local: ca_core::filter::LocalKnn,
    open_points: Vec<f64>,
}

#[wasm_bindgen]
impl SorLocal {
    /// The statistic per point of the part, NaN where still open.
    pub fn means(&self) -> Vec<f64> {
        self.local.means.clone()
    }

    /// Part-local indices of the open points.
    pub fn open(&self) -> Vec<u32> {
        self.local.open.clone()
    }

    /// Interleaved `xyz` of the open points.
    #[wasm_bindgen(js_name = openPoints)]
    pub fn open_points(&self) -> Vec<f64> {
        self.open_points.clone()
    }

    /// `k + 1` ascending squared distances per open point.
    pub fn candidates(&self) -> Vec<f64> {
        self.local.candidates.clone()
    }

    /// Per open point, the bit mask of the other parts to ask.
    pub fn reach(&self) -> Vec<u32> {
        self.local.reach.clone()
    }
}

#[wasm_bindgen]
impl SorPart {
    /// Index the points of one part (see [`plan_sor`]).
    #[wasm_bindgen(constructor)]
    pub fn new(points: &[f64]) -> SorPart {
        SorPart {
            part: ca_core::filter::KnnPart::new(points.as_chunks::<3>().0.to_vec()),
        }
    }

    /// Step 1: the statistic where this part's own points settle it.
    pub fn local(&self, k: usize, own: usize, regions: &[f64]) -> Result<SorLocal, JsError> {
        let regions = ca_core::filter::Regions::from_flat(regions)
            .ok_or_else(|| JsError::new("invalid SOR regions"))?;
        let local = self.part.local(k, own, &regions);
        let open_points = local
            .open
            .iter()
            .flat_map(|&i| self.part_point(i as usize))
            .collect();
        Ok(SorLocal { local, open_points })
    }

    /// Step 2: `k + 1` ascending squared distances from each query to this
    /// part's points.
    pub fn within(&self, queries: &[f64], k: usize) -> Vec<f64> {
        self.part.within(queries.as_chunks::<3>().0, k)
    }
}

impl SorPart {
    fn part_point(&self, i: usize) -> [f64; 3] {
        self.part.point(i)
    }
}

/// Normals of one part of a cloud on a pool worker, from that part's own
/// points (see [`Cloud::normals_orientation`] for `orientation`).
#[wasm_bindgen(js_name = normalsOf)]
pub fn normals_of(points: &[f64], k: usize, orientation: &[f64]) -> Result<Vec<f32>, JsError> {
    use ca_core::normals::Orientation;
    let orientation = match orientation {
        [0.0, ..] => Orientation::Up,
        &[1.0, x, y, z] => Orientation::Away([x, y, z]),
        _ => return Err(JsError::new("invalid orientation")),
    };
    Ok(
        ca_core::normals::estimate_normals(points.as_chunks::<3>().0, k, orientation)
            .into_flattened(),
    )
}

/// Builds one cloud from several (see [`ca_core::merge::Merger`]).
#[wasm_bindgen]
pub struct CloudMerger {
    merger: ca_core::merge::Merger,
}

#[wasm_bindgen]
impl CloudMerger {
    #[wasm_bindgen(constructor)]
    pub fn new() -> CloudMerger {
        CloudMerger {
            merger: ca_core::merge::Merger::new(),
        }
    }

    /// Append a cloud; `r, g, b` colors its points if others have colors
    /// and it has none.
    pub fn add(&mut self, cloud: &Cloud, r: u8, g: u8, b: u8) -> Result<(), JsError> {
        if self.merger.add(&cloud.inner, [r, g, b]) {
            Ok(())
        } else {
            Err(JsError::new("at most 256 clouds can be merged"))
        }
    }

    /// The merged cloud, not yet indexed, with a `source` attribute (the
    /// index of the cloud each point came from).
    pub fn finish(self) -> Cloud {
        Cloud::unindexed(self.merger.finish())
    }
}

impl Default for CloudMerger {
    fn default() -> Self {
        Self::new()
    }
}

/// Reads a COPC file node by node (see [`ca_core::io::copc`]): open it from
/// its first bytes, feed it the hierarchy pages it asks for, then the
/// decoded nodes, and finish into a cloud.
#[wasm_bindgen]
pub struct CopcReader {
    header: ca_core::io::copc::CopcHeader,
    selector: ca_core::io::copc::NodeSelector,
    points: ca_core::io::copc::CopcPoints,
}

#[wasm_bindgen]
impl CopcReader {
    /// Whether these first bytes (at least 400) are a COPC file.
    #[wasm_bindgen(js_name = isCopc)]
    pub fn is_copc(head: &[u8]) -> bool {
        ca_core::io::copc::CopcHeader::is_copc(head)
    }

    /// Bytes from the start needed by [`CopcReader::open`].
    #[wasm_bindgen(js_name = headerLength)]
    pub fn header_length(head: &[u8]) -> Option<usize> {
        ca_core::io::copc::CopcHeader::needed(head)
    }

    pub fn open(head: &[u8]) -> Result<CopcReader, JsError> {
        let header =
            ca_core::io::copc::CopcHeader::parse(head).map_err(|e| JsError::new(&e.to_string()))?;
        Ok(CopcReader {
            selector: ca_core::io::copc::NodeSelector::new(header.root_page),
            header,
            points: Default::default(),
        })
    }

    #[wasm_bindgen(getter, js_name = totalPoints)]
    pub fn total_points(&self) -> f64 {
        self.header.total_points as f64
    }

    /// Hierarchy pages still needed to know every node down to `level`, as
    /// `offset, size` pairs.
    #[wasm_bindgen(js_name = pagesFor)]
    pub fn pages_for(&self, level: i32) -> Vec<f64> {
        self.selector
            .pages_for(level)
            .into_iter()
            .flat_map(|(o, s)| [o as f64, s as f64])
            .collect()
    }

    #[wasm_bindgen(js_name = addPage)]
    pub fn add_page(&mut self, offset: f64, bytes: &[u8]) {
        self.selector.add_page(offset as u64, bytes);
    }

    /// Points in the nodes at `level` (once its pages are added).
    #[wasm_bindgen(js_name = levelPoints)]
    pub fn level_points(&self, level: i32) -> f64 {
        self.selector.level_points(level) as f64
    }

    #[wasm_bindgen(js_name = deeperThan)]
    pub fn deeper_than(&self, level: i32) -> bool {
        self.selector.deeper_than(level)
    }

    /// Nodes down to `level`, largest first, as `offset, size, points` triples.
    #[wasm_bindgen(js_name = nodesTo)]
    pub fn nodes_to(&self, level: i32) -> Vec<f64> {
        self.selector
            .nodes_to(level)
            .into_iter()
            .flat_map(|e| [e.offset as f64, e.byte_size as f64, e.point_count as f64])
            .collect()
    }

    /// Add nodes decoded by [`decode_copc_nodes`] (its arrays).
    #[wasm_bindgen(js_name = addDecoded)]
    pub fn add_decoded(
        &mut self,
        positions: &[f64],
        colors: Option<Vec<u16>>,
        intensity: Vec<f32>,
        classification: Vec<u8>,
    ) -> Result<(), JsError> {
        let positions = positions.as_chunks::<3>().0.to_vec();
        let n = positions.len();
        if intensity.len() != n
            || classification.len() != n
            || colors.as_ref().is_some_and(|c| c.len() != 3 * n)
        {
            return Err(JsError::new("decoded COPC arrays differ in length"));
        }
        self.points.extend(ca_core::io::copc::CopcPoints {
            positions,
            colors: colors.map(|c| c.as_chunks::<3>().0.to_vec()),
            intensity,
            classification,
        });
        Ok(())
    }

    /// The cloud (not yet indexed).
    pub fn finish(self) -> Result<Cloud, JsError> {
        if self.points.is_empty() {
            return Err(JsError::new("no points were read"));
        }
        Ok(Cloud::unindexed(self.points.into_cloud()))
    }
}

/// Points decoded from COPC nodes on a pool worker.
#[wasm_bindgen]
pub struct DecodedCopc {
    points: ca_core::io::copc::CopcPoints,
}

#[wasm_bindgen]
impl DecodedCopc {
    pub fn positions(&self) -> Vec<f64> {
        self.points.positions.as_flattened().to_vec()
    }

    /// 16-bit RGB, if the file has colors.
    pub fn colors(&self) -> Option<Vec<u16>> {
        self.points
            .colors
            .as_ref()
            .map(|c| c.as_flattened().to_vec())
    }

    pub fn intensity(&self) -> Vec<f32> {
        self.points.intensity.clone()
    }

    pub fn classification(&self) -> Vec<u8> {
        self.points.classification.clone()
    }
}

/// Decode COPC nodes: `chunks` is their compressed data back to back and
/// `counts` their point counts; `head` the file's first bytes.
#[wasm_bindgen(js_name = decodeCopcNodes)]
pub fn decode_copc_nodes(
    head: &[u8],
    chunks: &[u8],
    sizes: &[u32],
    counts: &[u32],
) -> Result<DecodedCopc, JsError> {
    let header =
        ca_core::io::copc::CopcHeader::parse(head).map_err(|e| JsError::new(&e.to_string()))?;
    let mut points = ca_core::io::copc::CopcPoints::default();
    let mut at = 0usize;
    for (&size, &count) in sizes.iter().zip(counts) {
        let chunk = chunks
            .get(at..at + size as usize)
            .ok_or_else(|| JsError::new("COPC chunk out of range"))?;
        points.extend(
            header
                .decode_node(chunk, count as usize)
                .map_err(|e| JsError::new(&e.to_string()))?,
        );
        at += size as usize;
    }
    Ok(DecodedCopc { points })
}

/// Reads a plain LAS/LAZ file chunk by chunk (see
/// [`ca_core::io::las_chunks`]): open it from its first bytes, feed it the
/// ranges it asks for (the LAZ chunk table), have the chunks decoded with
/// [`decode_las_chunks`] (e.g. on the worker pool), add them in file order
/// and finish into a cloud.
#[wasm_bindgen]
pub struct LasReader {
    layout: ca_core::io::las_chunks::LasLayout,
    points: ca_core::io::RawLasPoints,
}

#[wasm_bindgen]
impl LasReader {
    /// Bytes from the start needed by [`LasReader::open`], or `undefined`
    /// if `head` is not LAS or too short to tell.
    #[wasm_bindgen(js_name = headerLength)]
    pub fn header_length(head: &[u8]) -> Option<usize> {
        ca_core::io::las_chunks::LasLayout::header_len(head)
    }

    /// Start reading a file of `file_size` bytes, or `undefined` when it is
    /// not LAS or its points cannot be read chunk by chunk.
    pub fn open(head: &[u8], file_size: f64) -> Result<Option<LasReader>, JsError> {
        let layout = ca_core::io::las_chunks::LasLayout::open(head, file_size as u64)?;
        Ok(layout.map(|layout| LasReader {
            layout,
            points: Default::default(),
        }))
    }

    /// The next byte range to read as `[offset, length]`, or empty once the
    /// chunks are known.
    pub fn needs(&self) -> Vec<f64> {
        self.layout
            .needs()
            .map_or(Vec::new(), |(o, n)| vec![o as f64, n as f64])
    }

    /// The bytes of the range [`LasReader::needs`] asked for.
    pub fn supply(&mut self, bytes: &[u8]) -> Result<(), JsError> {
        Ok(self.layout.supply(bytes)?)
    }

    #[wasm_bindgen(getter, js_name = totalPoints)]
    pub fn total_points(&self) -> f64 {
        self.layout.total_points() as f64
    }

    /// Bytes of the file start that [`decode_las_chunks`] needs.
    #[wasm_bindgen(getter, js_name = dataOffset)]
    pub fn data_offset(&self) -> usize {
        self.layout.data_offset()
    }

    /// Every chunk as `offset, size, count, first` (file order).
    pub fn chunks(&self) -> Vec<f64> {
        self.layout
            .chunks()
            .iter()
            .flat_map(|c| {
                [
                    c.offset as f64,
                    c.size as f64,
                    c.count as f64,
                    c.first as f64,
                ]
            })
            .collect()
    }

    /// Add chunks decoded by [`decode_las_chunks`] (its arrays), in file order.
    #[wasm_bindgen(js_name = addDecoded)]
    pub fn add_decoded(
        &mut self,
        positions: &[f64],
        colors: Option<Vec<u16>>,
        intensity: Vec<f32>,
        classification: Vec<u8>,
    ) -> Result<(), JsError> {
        let positions = positions.as_chunks::<3>().0.to_vec();
        let n = positions.len();
        if intensity.len() != n
            || classification.len() != n
            || colors.as_ref().is_some_and(|c| c.len() != 3 * n)
        {
            return Err(JsError::new("decoded LAS arrays differ in length"));
        }
        self.points.extend(ca_core::io::RawLasPoints {
            positions,
            colors: colors.map(|c| c.as_chunks::<3>().0.to_vec()),
            intensity,
            classification,
        });
        Ok(())
    }

    /// Right shift from the file's 16-bit colors to 8 bits (0 or 8), as
    /// decided over the points added so far.
    #[wasm_bindgen(getter, js_name = colorShift)]
    pub fn color_shift(&self) -> u32 {
        self.points.color_shift()
    }

    /// The cloud (not yet indexed).
    pub fn finish(self) -> Result<Cloud, JsError> {
        if self.points.is_empty() {
            return Err(JsError::new("no points were read"));
        }
        Ok(Cloud::unindexed(self.points.into_cloud()))
    }
}

/// Points decoded from LAS/LAZ chunks, and each chunk's bounds.
#[wasm_bindgen]
pub struct DecodedLas {
    inner: ca_core::io::las_chunks::DecodedChunks,
}

#[wasm_bindgen]
impl DecodedLas {
    pub fn positions(&self) -> Vec<f64> {
        self.inner.points.positions.as_flattened().to_vec()
    }

    /// 16-bit RGB, if the file has colors.
    pub fn colors(&self) -> Option<Vec<u16>> {
        self.inner
            .points
            .colors
            .as_ref()
            .map(|c| c.as_flattened().to_vec())
    }

    pub fn intensity(&self) -> Vec<f32> {
        self.inner.points.intensity.clone()
    }

    pub fn classification(&self) -> Vec<u8> {
        self.inner.points.classification.clone()
    }

    /// `min x, y, z, max x, y, z` per chunk, over all of its points.
    pub fn bounds(&self) -> Vec<f64> {
        self.inner.bounds.as_flattened().to_vec()
    }
}

/// Decode LAS/LAZ chunks lying back to back in `bytes` (from the first
/// chunk's offset); `chunks` is `offset, size, count, first` per chunk as
/// from [`LasReader::chunks`] and `head` the file's first
/// [`LasReader::data_offset`] bytes. Keeps every `keep_every`-th point of
/// the file.
#[wasm_bindgen(js_name = decodeLasChunks)]
pub fn decode_las_chunks(
    head: &[u8],
    bytes: &[u8],
    chunks: &[f64],
    keep_every: f64,
) -> Result<DecodedLas, JsError> {
    let decoder = ca_core::io::las_chunks::ChunkDecoder::new(head)?;
    let chunks: Vec<_> = chunks
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| ca_core::io::las_chunks::LasChunk {
            offset: c[0] as u64,
            size: c[1] as u64,
            count: c[2] as u64,
            first: c[3] as u64,
        })
        .collect();
    let inner = decoder.decode(bytes, &chunks, keep_every.max(1.0) as u64)?;
    Ok(DecodedLas { inner })
}

/// Points of a cross-section (see [`Cloud::profile`]).
#[wasm_bindgen]
pub struct ProfileHits {
    along: Vec<f64>,
    positions: Vec<f64>,
    /// Points in the band before thinning.
    #[wasm_bindgen(readonly)]
    pub total: usize,
}

#[wasm_bindgen]
impl ProfileHits {
    /// Distance of each kept point along the line.
    pub fn along(&self) -> Vec<f64> {
        self.along.clone()
    }

    /// Interleaved `xyz` of the kept points, in original coordinates.
    pub fn positions(&self) -> Vec<f64> {
        self.positions.clone()
    }
}

/// Kinds of segment in a [`Segmentation`].
const SEGMENT_PLANE: u8 = 0;
const SEGMENT_SPHERE: u8 = 1;
const SEGMENT_CYLINDER: u8 = 2;
const SEGMENT_CLUSTER: u8 = 3;
/// The clusters past the 254 largest, together.
const SEGMENT_OTHER_CLUSTERS: u8 = 4;
/// Points in no shape, or cluster noise.
const SEGMENT_REST: u8 = 5;
/// Numbers per segment in [`Segmentation::params`].
const SEGMENT_PARAMS: usize = 8;
const REST_COLOR: [u8; 3] = [128, 128, 128];
const OTHER_COLOR: [u8; 3] = [200, 200, 200];

/// A distinct color for segment `k`: hues a golden angle apart.
fn segment_color(k: usize) -> [u8; 3] {
    let h = (0.08 + k as f64 * 0.618_033_988_749_895).fract() * 6.0;
    let (s, v) = (0.7, 0.95);
    let f = h.fract();
    let (p, q, t) = (v * (1.0 - s), v * (1.0 - s * f), v * (1.0 - s * (1.0 - f)));
    let rgb = match h as u32 {
        0 => [v, t, p],
        1 => [q, v, p],
        2 => [p, v, t],
        3 => [p, q, v],
        4 => [t, p, v],
        _ => [v, p, q],
    };
    rgb.map(|c| (c * 255.0).round() as u8)
}

/// Shapes or clusters found in a cloud (see [`Cloud::detect_shapes`] and
/// [`Cloud::clusters`]): a copy of the cloud whose `source` attribute is
/// each point's segment, and a table with one row per segment, in segment
/// order.
#[wasm_bindgen]
pub struct Segmentation {
    cloud: Option<Cloud>,
    kinds: Vec<u8>,
    counts: Vec<u32>,
    params: Vec<f64>,
    rms: Vec<f64>,
    colors: Vec<u8>,
    /// Shapes or clusters found (all clusters, also those past 254).
    #[wasm_bindgen(readonly)]
    pub found: usize,
}

impl Segmentation {
    fn new(found: usize) -> Self {
        Self {
            cloud: None,
            kinds: Vec::new(),
            counts: Vec::new(),
            params: Vec::new(),
            rms: Vec::new(),
            colors: Vec::new(),
            found,
        }
    }

    fn push(
        &mut self,
        kind: u8,
        count: usize,
        params: [f64; SEGMENT_PARAMS],
        rms: f64,
        color: [u8; 3],
    ) {
        self.kinds.push(kind);
        self.counts.push(count as u32);
        self.params.extend(params);
        self.rms.push(rms);
        self.colors.extend(color);
    }

    fn color(&self, segment: usize) -> [u8; 3] {
        std::array::from_fn(|c| self.colors[segment * 3 + c])
    }
}

#[wasm_bindgen]
impl Segmentation {
    /// The segmented copy (unindexed); can be taken once.
    #[wasm_bindgen(js_name = takeCloud)]
    pub fn take_cloud(&mut self) -> Result<Cloud, JsError> {
        self.cloud
            .take()
            .ok_or_else(|| JsError::new("cloud already taken"))
    }

    /// Kind of each segment: 0 plane, 1 sphere, 2 cylinder, 3 cluster, 4 the
    /// clusters past the 254 largest, 5 the rest (or noise).
    pub fn kinds(&self) -> Vec<u8> {
        self.kinds.clone()
    }

    /// Points in each segment.
    pub fn counts(&self) -> Vec<u32> {
        self.counts.clone()
    }

    /// Eight numbers per segment, in original coordinates. Plane: unit
    /// normal and `d` (`n · p + d = 0`); sphere: centre and radius;
    /// cylinder: axis midpoint, unit axis, radius and length; clusters:
    /// centroid and extent.
    pub fn params(&self) -> Vec<f64> {
        self.params.clone()
    }

    /// RMS distance of each shape's points to it (NaN for other segments).
    pub fn rms(&self) -> Vec<f64> {
        self.rms.clone()
    }

    /// Display color of each segment, interleaved RGB.
    pub fn colors(&self) -> Vec<u8> {
        self.colors.clone()
    }
}

/// Run the heavy kernels once on a small synthetic cloud. Browsers first
/// run WebAssembly with a baseline compiler and optimize a function only
/// after it has been busy, and never in the middle of a call; one long call
/// on a large cloud would run unoptimized from start to end. Calling this
/// when the page is idle gets the optimized code compiled beforehand.
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

#[wasm_bindgen(js_name = warmUp)]
pub fn warm_up() {
    let n = 60_000;
    let positions: Vec<[f64; 3]> = (0..n)
        .map(|i| {
            let (x, y) = ((i % 250) as f64 * 0.1, (i / 250) as f64 * 0.1);
            [x, y, (x * 0.7).sin() + (y * 0.3).cos()]
        })
        .collect();
    let mut cloud = PointCloud {
        positions: positions.clone(),
        colors: Some(vec![[1, 2, 3]; n]),
        attributes: Vec::new(),
    };
    let params = OctreeParams {
        max_leaf: 2_000,
        ..OctreeParams::default()
    };
    let _ = Octree::build_bucketed(&mut cloud, params);
    let _ = Octree::build_for_cloud(&mut cloud.clone(), params);
    let _ = ca_core::io::write_las(&cloud, &[], true);
    let queries: Vec<[f64; 3]> = positions
        .iter()
        .step_by(3)
        .map(|p| [p[0] + 0.03, p[1], p[2] + 0.1])
        .collect();
    let _ = ca_core::cloud_to_cloud(
        &PointCloud {
            positions: queries.clone(),
            colors: None,
            attributes: Vec::new(),
        },
        &cloud,
    );
    let mesh = TriangleMesh {
        vertices: positions[..2_500].to_vec(),
        triangles: (0..9u32)
            .flat_map(|j| {
                (0..249u32).flat_map(move |i| {
                    let k = j * 250 + i;
                    [[k, k + 1, k + 250], [k + 1, k + 251, k + 250]]
                })
            })
            .collect(),
    };
    let _ = ca_core::cloud_to_mesh(&queries[..5_000], &mesh, true);
    let split = ca_core::filter::split_for_knn(&positions, 2);
    let part: Vec<[f64; 3]> = split.parts[0]
        .iter()
        .map(|&i| positions[i as usize])
        .collect();
    let part = ca_core::filter::KnnPart::new(part);
    let local = part.local(8, 0, &split.regions);
    let _ = part.within(&queries[..2_000], 8);
    let _ = ca_core::filter::sor_keep(&local.means, 1.0);
    let _ = ca_core::filter::spatial_subsample(&cloud, 0.05);
    let _ = ca_core::filter::octree_subsample(&cloud, 6);
    let _ = ca_core::raster::rasterize(
        &cloud,
        ca_core::raster::RasterParams {
            cell: 0.25,
            height: ca_core::raster::RasterHeight::Percentile(50.0),
            fill_empty: true,
            class: None,
        },
    );
    let _ = ca_core::delaunay::delaunay_25d(&positions, ca_core::delaunay::MaxEdge::Auto);
    let normals = ca_core::normals::estimate_normals(
        &positions[..20_000],
        12,
        ca_core::normals::Orientation::Up,
    );
    for primitive in [
        ca_core::shapes::Primitive::Plane,
        ca_core::shapes::Primitive::Cylinder,
    ] {
        let params = ca_core::shapes::RansacParams {
            primitive,
            threshold: 0.05,
            min_support: 2_000,
            max_shapes: 2,
            candidates: 100,
            ..Default::default()
        };
        let _ = ca_core::shapes::detect_shapes(&positions[..20_000], &normals, &params);
    }
    let _ = ca_core::cluster::euclidean_clusters(&positions[..20_000], 0.15, 10);
}

#[wasm_bindgen(js_name = nearestDistances)]
pub fn nearest_distances(reference: &[f64], queries: &[f64]) -> Result<Vec<f64>, JsError> {
    ca_core::cloud_to_cloud(&from_flat(queries)?, &from_flat(reference)?)
        .ok_or_else(|| JsError::new("reference cloud is empty"))
}

/// A C2C job split into independent, spatially compact parts
/// (see [`ca_core::partition_c2c`]). Each part can run on its own worker via
/// [`nearest_distances`], and its results belong at [`C2cPlan::take_query_indices`].
#[wasm_bindgen]
pub struct C2cPlan {
    parts: Vec<(Vec<u32>, Vec<f64>, Vec<f64>)>,
}

#[wasm_bindgen]
impl C2cPlan {
    #[wasm_bindgen(getter)]
    pub fn length(&self) -> usize {
        self.parts.len()
    }

    // The getters below hand the buffers over instead of copying them, so
    // each may be called once per part.

    /// Indices into the compared cloud for part `k`.
    #[wasm_bindgen(js_name = takeQueryIndices)]
    pub fn take_query_indices(&mut self, k: usize) -> Vec<u32> {
        std::mem::take(&mut self.parts[k].0)
    }

    /// Interleaved `xyz` of the compared points of part `k`.
    #[wasm_bindgen(js_name = takeQueries)]
    pub fn take_queries(&mut self, k: usize) -> Vec<f64> {
        std::mem::take(&mut self.parts[k].1)
    }

    /// Interleaved `xyz` of the reference points part `k` needs.
    #[wasm_bindgen(js_name = takeReference)]
    pub fn take_reference(&mut self, k: usize) -> Vec<f64> {
        std::mem::take(&mut self.parts[k].2)
    }
}

/// Split a C2C job into up to `parts` pieces for parallel workers.
#[wasm_bindgen(js_name = planCloudToCloud)]
pub fn plan_cloud_to_cloud(
    compared: &Cloud,
    reference: &Cloud,
    parts: usize,
) -> Result<C2cPlan, JsError> {
    let gather = |cloud: &PointCloud, idx: &[u32]| -> Vec<f64> {
        idx.iter()
            .flat_map(|&i| cloud.positions[i as usize])
            .collect()
    };
    let parts = ca_core::partition_c2c(&compared.inner, &reference.inner, parts)
        .ok_or_else(|| JsError::new("reference cloud is empty"))?
        .into_iter()
        .map(|part| {
            let queries = gather(&compared.inner, &part.queries);
            let reference = gather(&reference.inner, &part.reference);
            (part.queries, queries, reference)
        })
        .collect();
    Ok(C2cPlan { parts })
}

/// Summary statistics (and `f32` copies for coloring) of per-point distances.
#[wasm_bindgen(js_name = summarizeDistances)]
pub fn summarize_distances(distances: &[f64]) -> Result<C2cResult, JsError> {
    let stats = DistanceStats::from_distances(distances)
        .ok_or_else(|| JsError::new("compared cloud is empty"))?;
    Ok(C2cResult {
        distances: distances.iter().map(|&d| d as f32).collect(),
        stats,
    })
}

/// A trajectory: timed positions and optional orientations (see
/// [`ca_core::trajectory`]).
#[wasm_bindgen]
pub struct TrajectoryData {
    inner: ca_core::trajectory::Trajectory,
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
