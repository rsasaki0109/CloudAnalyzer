//! Neighbour-based operations: statistical outlier removal, GPU k-NN buffers and normals.

use super::*;

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
    #[wasm_bindgen(js_name = prepareHintBounds)]
    pub fn prepare_hint_bounds(&mut self) -> Result<usize, JsError> {
        self.part
            .prepare_hint_bounds()
            .ok_or_else(|| JsError::new("empty neighbourhood part"))?;
        Ok(self.part.retained_bytes())
    }
    #[wasm_bindgen(js_name = gpuHintBuffers)]
    pub fn gpu_hint_buffers(&mut self) -> Result<GpuKnnBuffers, JsError> {
        let (nodes, queries) = self
            .part
            .gpu_hint_buffers()
            .ok_or_else(|| JsError::new("empty GPU neighbourhood part"))?;
        Ok(GpuKnnBuffers {
            nodes,
            queries,
            points: Vec::new(),
        })
    }

    #[wasm_bindgen(js_name = gpuCandidateBuffers)]
    pub fn gpu_candidate_buffers(&mut self) -> Result<GpuKnnBuffers, JsError> {
        let mut buffers = self.gpu_hint_buffers()?;
        buffers.points = self
            .part
            .gpu_candidate_points()
            .ok_or_else(|| JsError::new("empty GPU neighbourhood part"))?;
        Ok(buffers)
    }

    #[wasm_bindgen(js_name = normalsCandidates)]
    pub fn normals_candidates(
        &self,
        k: usize,
        orientation: &[f64],
        candidates: &[u32],
    ) -> Result<Vec<f32>, JsError> {
        self.part
            .normals_candidates(k, normal_orientation(orientation)?, candidates)
            .map(Vec::into_flattened)
            .ok_or_else(|| JsError::new("invalid neighbourhood candidates"))
    }

    #[wasm_bindgen(js_name = localCandidates)]
    pub fn local_candidates(
        &self,
        k: usize,
        own: usize,
        regions: &[f64],
        candidates: &[u32],
    ) -> Result<SorLocal, JsError> {
        let regions = ca_core::filter::Regions::from_flat(regions)
            .ok_or_else(|| JsError::new("invalid SOR regions"))?;
        let local = self
            .part
            .local_candidates(k, own, &regions, candidates)
            .ok_or_else(|| JsError::new("invalid neighbourhood candidates"))?;
        let open_points = local
            .open
            .iter()
            .flat_map(|&i| self.part_point(i as usize))
            .collect();
        Ok(SorLocal { local, open_points })
    }

    #[wasm_bindgen(js_name = normalsHinted)]
    pub fn normals_hinted(
        &self,
        k: usize,
        orientation: &[f64],
        candidates: &[u32],
    ) -> Result<Vec<f32>, JsError> {
        self.part
            .normals_hinted(k, normal_orientation(orientation)?, candidates)
            .map(Vec::into_flattened)
            .ok_or_else(|| JsError::new("invalid neighbourhood hints"))
    }

    #[wasm_bindgen(js_name = normalsCpuHints)]
    pub fn normals_cpu_hints(
        &mut self,
        k: usize,
        orientation: &[f64],
    ) -> Result<Vec<f32>, JsError> {
        let candidates = self
            .part
            .cpu_hint_candidates(k.max(3))
            .ok_or_else(|| JsError::new("empty neighbourhood part"))?;
        self.normals_hinted(k, orientation, &candidates)
    }

    #[wasm_bindgen(js_name = localHinted)]
    pub fn local_hinted(
        &self,
        k: usize,
        own: usize,
        regions: &[f64],
        candidates: &[u32],
    ) -> Result<SorLocal, JsError> {
        let regions = ca_core::filter::Regions::from_flat(regions)
            .ok_or_else(|| JsError::new("invalid SOR regions"))?;
        let local = self
            .part
            .local_hinted(k, own, &regions, candidates)
            .ok_or_else(|| JsError::new("invalid neighbourhood hints"))?;
        let open_points = local
            .open
            .iter()
            .flat_map(|&i| self.part_point(i as usize))
            .collect();
        Ok(SorLocal { local, open_points })
    }

    #[wasm_bindgen(js_name = localCpuHints)]
    pub fn local_cpu_hints(
        &mut self,
        k: usize,
        own: usize,
        regions: &[f64],
    ) -> Result<SorLocal, JsError> {
        let candidates = self
            .part
            .cpu_hint_candidates(k.saturating_add(1))
            .ok_or_else(|| JsError::new("empty neighbourhood part"))?;
        self.local_hinted(k, own, regions, &candidates)
    }

    /// Retain immutable coordinates, tree and query order for repeated fits.
    pub fn prepared(points: &[f64]) -> SorPart {
        SorPart {
            part: ca_core::filter::KnnPart::prepared(points.as_chunks::<3>().0.to_vec()),
        }
    }

    #[wasm_bindgen(getter, js_name = retainedBytes)]
    pub fn retained_bytes(&self) -> usize {
        self.part.retained_bytes()
    }

    /// Part-local normals using this preparation's exact tree and query order.
    pub fn normals(&self, k: usize, orientation: &[f64]) -> Result<Vec<f32>, JsError> {
        let orientation = normal_orientation(orientation)?;
        Ok(self.part.normals(k, orientation).into_flattened())
    }

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

#[wasm_bindgen]
pub struct GpuKnnBuffers {
    nodes: Vec<u32>,
    queries: Vec<f32>,
    points: Vec<f32>,
}

#[wasm_bindgen]
impl GpuKnnBuffers {
    #[wasm_bindgen(js_name = takePoints)]
    pub fn take_points(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.points)
    }
    #[wasm_bindgen(js_name = takeNodes)]
    pub fn take_nodes(&mut self) -> Vec<u32> {
        std::mem::take(&mut self.nodes)
    }
    #[wasm_bindgen(js_name = takeQueries)]
    pub fn take_queries(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.queries)
    }
}

#[wasm_bindgen]
impl SorPart {
    #[wasm_bindgen(js_name = withinWithScratch)]
    pub fn within_with_scratch(&self, queries: &[f64], k: usize, reuse: bool) -> Vec<f64> {
        self.part
            .within_with_options(queries.as_chunks::<3>().0, k, reuse, reuse)
    }

    #[wasm_bindgen(js_name = withinWithBounds)]
    pub fn within_with_bounds(&self, queries: &[f64], k: usize, bounds: bool) -> Vec<f64> {
        self.part
            .within_with_bounds(queries.as_chunks::<3>().0, k, true, false, bounds)
    }

    #[wasm_bindgen(js_name = normalsWithBounds)]
    pub fn normals_with_bounds(
        &self,
        k: usize,
        orientation: &[f64],
        bounds: bool,
    ) -> Result<Vec<f32>, JsError> {
        Ok(self
            .part
            .normals_with_bounds(k, normal_orientation(orientation)?, bounds)
            .into_flattened())
    }

    #[wasm_bindgen(js_name = localWithBounds)]
    pub fn local_with_bounds(
        &self,
        k: usize,
        own: usize,
        regions: &[f64],
        bounds: bool,
    ) -> Result<SorLocal, JsError> {
        let regions = ca_core::filter::Regions::from_flat(regions)
            .ok_or_else(|| JsError::new("invalid SOR regions"))?;
        let local = self.part.local_with_options(k, own, &regions, true, bounds);
        let open_points = local
            .open
            .iter()
            .flat_map(|&i| self.part_point(i as usize))
            .collect();
        Ok(SorLocal { local, open_points })
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
    let orientation = normal_orientation(orientation)?;
    Ok(
        ca_core::normals::estimate_normals(points.as_chunks::<3>().0, k, orientation)
            .into_flattened(),
    )
}

/// Explicit scalar/allocating comparison for neighbourhood/PCA verification.
#[wasm_bindgen(js_name = normalsOfWithScratch)]
pub fn normals_of_with_scratch(
    points: &[f64],
    k: usize,
    orientation: &[f64],
    reuse: bool,
) -> Result<Vec<f32>, JsError> {
    let orientation = normal_orientation(orientation)?;
    Ok(ca_core::normals::estimate_normals_with_options(
        points.as_chunks::<3>().0,
        k,
        orientation,
        reuse,
        reuse,
    )
    .into_flattened())
}

fn normal_orientation(orientation: &[f64]) -> Result<ca_core::normals::Orientation, JsError> {
    use ca_core::normals::Orientation;
    match orientation {
        [0.0, ..] => Ok(Orientation::Up),
        &[1.0, x, y, z] => Ok(Orientation::Away([x, y, z])),
        _ => Err(JsError::new("invalid orientation")),
    }
}
