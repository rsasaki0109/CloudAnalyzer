//! WebAssembly bindings for the CloudAnalyzer Web viewer.

use ca_core::icp::{IcpMetric, IcpParams, Rigid};
use ca_core::octree::{NO_CHILD, Octree, OctreeNode, OctreeParams, PendingSubtree};
use ca_core::{
    AttributeValues, CLASSIFICATION, DistanceStats, INTENSITY, PointCloud, TriangleMesh,
};
use wasm_bindgen::prelude::*;

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
    /// Subtrees still to be built during a parallel index build.
    pending: Vec<PendingSubtree>,
}

impl Cloud {
    fn unindexed(inner: PointCloud) -> Cloud {
        Cloud {
            inner,
            lod: Octree {
                order: Vec::new(),
                nodes: Vec::new(),
                grid: OctreeParams::default().grid,
            },
            pending: Vec::new(),
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
            pending: Vec::new(),
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

#[wasm_bindgen]
impl Cloud {
    /// Parse a file; the format is detected from `name` and the leading bytes.
    /// Call [`Cloud::build_index`] before using any per-point output.
    pub fn parse(name: &str, bytes: &[u8]) -> Result<Cloud, JsError> {
        Ok(Cloud::unindexed(ca_core::read(name, bytes)?))
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
            pending: Vec::new(),
        })
    }

    /// A filtered copy of the cloud (with colors and attributes):
    /// `"voxel"` keeps one point per voxel of edge `a`; `"random"` keeps `a`
    /// random points; `"sor"` drops statistical outliers with `a` neighbours
    /// and a `b` standard-deviation threshold.
    pub fn filter(&self, op: &str, a: f64, b: f64) -> Result<Cloud, JsError> {
        use ca_core::filter;
        let keep = match op {
            "voxel" => filter::voxel_subsample(&self.inner, a),
            "random" => filter::random_subsample(&self.inner, a.max(0.0) as usize, 0x5eed),
            "sor" => filter::statistical_outliers(&self.inner, a.max(1.0) as usize, b),
            other => return Err(JsError::new(&format!("unknown filter {other:?}"))),
        };
        let mut inner = self.inner.select(&keep);
        if inner.is_empty() {
            return Err(JsError::new("the filter removed every point"));
        }
        let lod = build_lod(&mut inner)?;
        Ok(Cloud {
            inner,
            lod,
            pending: Vec::new(),
        })
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
            pending: Vec::new(),
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

    /// Serialize the cloud as `"ply"` (binary) or `"csv"`, with an optional
    /// scalar field (one value per point, in the cloud's order). Points are
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
            "csv" => ca_core::io::write_csv(&self.inner, fields),
            other => return Err(JsError::new(&format!("unknown export format {other:?}"))),
        }
        .map_err(|e| JsError::new(&e))
    }

    /// Values of a named per-point attribute as `f32` (octree order), or
    /// `undefined` when the cloud does not have it.
    pub fn attribute(&self, name: &str) -> Option<Vec<f32>> {
        match &self.inner.attribute(name)?.values {
            AttributeValues::F32(v) => Some(v.clone()),
            AttributeValues::U8(v) => Some(v.iter().map(|&x| x as f32).collect()),
        }
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

    /// Start a parallel index build: build the octree levels above
    /// `split_level` here and return the remaining subtrees, 7 numbers each
    /// (`start, end, minX, minY, minZ, size, level`). Build each with
    /// [`build_subtree`] on [`Cloud::subtree_positions`] /
    /// [`Cloud::subtree_colors`], hand it back with [`Cloud::finish_subtree`],
    /// then call [`Cloud::end_index`].
    #[wasm_bindgen(js_name = startIndex)]
    pub fn start_index(&mut self, split_level: u8) -> Result<Vec<f64>, JsError> {
        let (lod, pending) =
            Octree::build_partial(&mut self.inner, OctreeParams::default(), split_level)
                .ok_or_else(|| JsError::new("cloud is empty or too large"))?;
        self.lod = lod;
        self.pending = pending;
        Ok(self
            .pending
            .iter()
            .flat_map(|p| {
                [
                    p.start as f64,
                    p.end as f64,
                    p.min[0],
                    p.min[1],
                    p.min[2],
                    p.size,
                    p.level as f64,
                ]
            })
            .collect())
    }

    /// Interleaved `xyz` of pending subtree `k` (a copy).
    #[wasm_bindgen(js_name = subtreePositions)]
    pub fn subtree_positions(&self, k: usize) -> Vec<f64> {
        let p = &self.pending[k];
        self.inner.positions[p.start as usize..p.end as usize]
            .as_flattened()
            .to_vec()
    }

    /// Interleaved `rgb` of pending subtree `k`, if the cloud has colors.
    #[wasm_bindgen(js_name = subtreeColors)]
    pub fn subtree_colors(&self, k: usize) -> Option<Vec<u8>> {
        let p = &self.pending[k];
        self.inner
            .colors
            .as_ref()
            .map(|c| c[p.start as usize..p.end as usize].as_flattened().to_vec())
    }

    /// Store a subtree built by [`build_subtree`]: its reordered points,
    /// colors, node table (in the [`Cloud::lod_nodes`] layout) and the
    /// permutation it applied, which reorders the attributes kept here.
    #[wasm_bindgen(js_name = finishSubtree)]
    pub fn finish_subtree(
        &mut self,
        k: usize,
        positions: &[f64],
        colors: Option<Vec<u8>>,
        nodes: &[f64],
        order: &[u32],
    ) -> Result<(), JsError> {
        let p = self.pending[k];
        let range = p.start as usize..p.end as usize;
        if positions.len() != 3 * range.len() {
            return Err(JsError::new("subtree has the wrong number of points"));
        }
        self.inner.positions[range.clone()].copy_from_slice(positions.as_chunks::<3>().0);
        if let (Some(dst), Some(src)) = (self.inner.colors.as_mut(), colors) {
            dst[range.clone()].copy_from_slice(src.as_chunks::<3>().0);
        }
        if order.len() != range.len() {
            return Err(JsError::new("subtree order has the wrong length"));
        }
        // The slice is small enough to stay in cache, so this gather is cheap.
        for attribute in &mut self.inner.attributes {
            attribute.values.permute_range(range.clone(), order);
        }
        let subtree = Octree {
            order: Vec::new(),
            nodes: nodes_from_flat(nodes)?,
            grid: self.lod.grid,
        };
        self.lod.graft(&p, subtree);
        Ok(())
    }

    /// Finish a parallel index build started with [`Cloud::start_index`].
    #[wasm_bindgen(js_name = endIndex)]
    pub fn end_index(&mut self) {
        self.pending.clear();
        self.lod.order = Vec::new();
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

/// A subtree built off the main worker: reordered points, colors and nodes.
#[wasm_bindgen]
pub struct Subtree {
    cloud: PointCloud,
    nodes: Vec<OctreeNode>,
    order: Vec<u32>,
}

#[wasm_bindgen]
impl Subtree {
    pub fn positions(&self) -> Vec<f64> {
        self.cloud.positions.as_flattened().to_vec()
    }

    pub fn colors(&self) -> Option<Vec<u8>> {
        self.cloud
            .colors
            .as_ref()
            .map(|c| c.as_flattened().to_vec())
    }

    /// Node table in the [`Cloud::lod_nodes`] layout, ranges relative to the subtree.
    pub fn nodes(&self) -> Vec<f64> {
        nodes_to_flat(&self.nodes)
    }

    /// `order[i]` is the input index of the point now at `i`.
    pub fn order(&self) -> Vec<u32> {
        self.order.clone()
    }
}

/// Build one pending subtree (see [`Cloud::start_index`]) on a pool worker.
/// `job` is the subtree's 7-number description.
#[wasm_bindgen(js_name = buildSubtree)]
pub fn build_subtree(
    positions: &[f64],
    colors: Option<Vec<u8>>,
    job: &[f64],
) -> Result<Subtree, JsError> {
    let job: &[f64; 7] = job
        .try_into()
        .map_err(|_| JsError::new("subtree job must have 7 numbers"))?;
    let pending = PendingSubtree {
        node: 0,
        start: job[0] as u32,
        end: job[1] as u32,
        min: [job[2], job[3], job[4]],
        size: job[5],
        level: job[6] as u8,
    };
    let mut cloud = PointCloud {
        positions: positions.as_chunks::<3>().0.to_vec(),
        colors: colors.map(|c| c.as_chunks::<3>().0.to_vec()),
        attributes: Vec::new(),
    };
    if cloud
        .colors
        .as_ref()
        .is_some_and(|c| c.len() != cloud.positions.len())
    {
        return Err(JsError::new("colors do not match positions"));
    }
    let tree = Octree::build_subtree(
        &mut cloud.positions,
        cloud.colors.as_deref_mut(),
        &pending,
        OctreeParams::default(),
    )
    .ok_or_else(|| JsError::new("empty subtree"))?;
    Ok(Subtree {
        cloud,
        nodes: tree.nodes,
        order: tree.order,
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
        pending: Vec::new(),
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
