//! The point cloud type and its viewer-facing operations.

use super::*;

/// A loaded point cloud. Coordinates stay in `f64` on the Rust side.
///
/// Points are reordered at load time into the order of a level-of-detail
/// octree, where each node's points are contiguous. Everything this type
/// returns per point (positions, colors, C2C distances) uses that order.
#[wasm_bindgen]
pub struct Cloud {
    pub(crate) inner: PointCloud,
    pub(crate) lod: Octree,
    /// Placement of the buckets during a parallel index build.
    pub(crate) layout: Option<BucketLayout>,
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

    pub(crate) fn unindexed(inner: PointCloud) -> Cloud {
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
