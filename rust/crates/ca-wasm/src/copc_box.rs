//! Bounded full-density spatial selection for a small browser working cloud.

use ca_core::io::copc::{CopcHeader, CopcPoints};
use ca_core::io::copc_query::{CopcQuery, QueryItem, QueryLimits};
use wasm_bindgen::prelude::*;

use super::Cloud;

const MAX_SAFE: f64 = 9_007_199_254_740_991.0;
const MAX_SELECTED: usize = 1_000_000;

fn integer(value: f64) -> Result<u64, JsError> {
    if !value.is_finite() || value < 0.0 || value > MAX_SAFE || value.fract() != 0.0 {
        return Err(JsError::new(
            "byte offsets/sizes must be safe nonnegative integers",
        ));
    }
    Ok(value as u64)
}

fn error(value: ca_core::IoError) -> JsError {
    JsError::new(&value.to_string())
}

#[wasm_bindgen]
pub struct CopcBoxReader {
    header: CopcHeader,
    query: CopcQuery,
    points: CopcPoints,
    limit: usize,
}

#[wasm_bindgen]
impl CopcBoxReader {
    pub fn open(
        head: &[u8],
        file_size: f64,
        bounds: &[f64],
        max_points: usize,
    ) -> Result<Self, JsError> {
        if max_points == 0 || max_points > MAX_SELECTED {
            return Err(JsError::new("selection point limit must be in 1..1000000"));
        }
        let bounds: [f64; 6] = bounds
            .try_into()
            .map_err(|_| JsError::new("box requires six coordinates"))?;
        let header = CopcHeader::parse(head).map_err(error)?;
        if header.total_points as u128 > MAX_SAFE as u128 {
            return Err(JsError::new(
                "source point count exceeds exact JavaScript integer range",
            ));
        }
        let query = CopcQuery::new(&header, integer(file_size)?, bounds, QueryLimits::default())
            .map_err(error)?;
        let points = CopcPoints {
            colors: (header.point_format() >= 7).then(Vec::new),
            ..Default::default()
        };
        Ok(Self {
            header,
            query,
            points,
            limit: max_points,
        })
    }

    /// One item [kind, offset, size, count]; kind 0=page, 1=node; empty=done.
    #[wasm_bindgen(js_name = nextItem)]
    pub fn next_item(&self) -> Vec<f64> {
        match self.query.next_item() {
            Some(QueryItem::Page { offset, size }) => vec![0., offset as f64, size as f64, 0.],
            Some(QueryItem::Node(entry)) => vec![
                1.,
                entry.offset as f64,
                entry.byte_size as f64,
                entry.point_count as f64,
            ],
            None => Vec::new(),
        }
    }

    #[wasm_bindgen(js_name = supplyPage)]
    pub fn supply_page(&mut self, offset: f64, page: &[u8]) -> Result<(), JsError> {
        self.query
            .supply_page(integer(offset)?, page)
            .map_err(error)
    }

    /// Decode exactly one pending node and atomically append selected points.
    #[wasm_bindgen(js_name = supplyNode)]
    pub fn supply_node(&mut self, offset: f64, compressed: &[u8]) -> Result<(), JsError> {
        let Some(QueryItem::Node(entry)) = self.query.next_item() else {
            return Err(JsError::new("no pending point node"));
        };
        if integer(offset)? != entry.offset || compressed.len() as u64 != entry.byte_size as u64 {
            return Err(JsError::new("compressed bytes do not match pending node"));
        }
        // The query has already checked compressed/raw node quotas. The raw
        // decoder preflights encoded layer lengths before allocating from them.
        let decoded = self
            .header
            .decode_node(compressed, entry.point_count as usize)
            .map_err(error)?;
        let selected = decoded
            .positions
            .iter()
            .filter(|&&p| self.query.contains(p))
            .count();
        if selected > self.limit - self.points.len() {
            return Err(JsError::new(&format!(
                "full-density box exceeds {} selected points; use a smaller box",
                self.limit
            )));
        }
        let allocation = |e: std::collections::TryReserveError| {
            JsError::new(&format!("selection allocation: {e}"))
        };
        self.points
            .positions
            .try_reserve_exact(selected)
            .map_err(allocation)?;
        self.points
            .intensity
            .try_reserve_exact(selected)
            .map_err(allocation)?;
        self.points
            .classification
            .try_reserve_exact(selected)
            .map_err(allocation)?;
        if let Some(colors) = &mut self.points.colors {
            colors.try_reserve_exact(selected).map_err(allocation)?;
        }
        // Allocation/count errors leave the node and selected output unchanged.
        self.query.advance_node().map_err(error)?;
        for (i, &position) in decoded.positions.iter().enumerate() {
            if self.query.contains(position) {
                self.points.positions.push(position);
                self.points.intensity.push(decoded.intensity[i]);
                self.points.classification.push(decoded.classification[i]);
                if let (Some(out), Some(colors)) = (&mut self.points.colors, &decoded.colors) {
                    out.push(colors[i]);
                }
            }
        }
        Ok(())
    }

    #[wasm_bindgen(getter, js_name = selectedPoints)]
    pub fn selected_points(&self) -> usize {
        self.points.len()
    }

    /// Browser cloud attributes follow normal viewer normalization (8bit RGB).
    pub fn finish(self) -> Result<Cloud, JsError> {
        if self.query.next_item().is_some() {
            return Err(JsError::new("full-density query is not complete"));
        }
        if self.points.is_empty() {
            return Err(JsError::new("no points inside full-density box"));
        }
        Ok(Cloud::unindexed(self.points.into_cloud()))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn full_density_fixture_box_matches_every_original_level() {
        let bytes = include_bytes!("../../../../web/e2e/fixtures/small.copc.laz");
        let needed = CopcHeader::needed(bytes).unwrap();
        let header = CopcHeader::parse(&bytes[..needed]).unwrap();
        let bounds = [8., 8., -1., 20., 24., 10.];
        let mut reader =
            CopcBoxReader::open(&bytes[..needed], bytes.len() as f64, &bounds, MAX_SELECTED)
                .unwrap();
        let (offset, size) = header.root_page;
        let entries =
            ca_core::io::copc::parse_page(&bytes[offset as usize..(offset + size) as usize]);
        let mut expected = Vec::new();
        for entry in entries.iter().filter(|e| e.point_count > 0) {
            let decoded = header
                .decode_node(
                    &bytes[entry.offset as usize..entry.offset as usize + entry.byte_size as usize],
                    entry.point_count as usize,
                )
                .unwrap();
            expected.extend(
                decoded
                    .positions
                    .into_iter()
                    .filter(|p| (0..3).all(|a| p[a] >= bounds[a] && p[a] <= bounds[a + 3])),
            );
        }
        while let Some(item) = reader.query.next_item() {
            let (offset, size) = match item {
                QueryItem::Page { offset, size } => (offset, size),
                QueryItem::Node(entry) => (entry.offset, entry.byte_size as u64),
            };
            let data = &bytes[offset as usize..(offset + size) as usize];
            match item {
                QueryItem::Page { .. } => reader.supply_page(offset as f64, data).unwrap(),
                QueryItem::Node(_) => reader.supply_node(offset as f64, data).unwrap(),
            }
        }
        let mut actual = reader.finish().unwrap().inner.positions;
        expected.sort_by(|a, b| a.partial_cmp(b).unwrap());
        actual.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(actual, expected);
        assert_eq!(actual.len(), 1965);
    }
}
