//! COPC and LAS readers and chunk decoding.

use super::*;

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
