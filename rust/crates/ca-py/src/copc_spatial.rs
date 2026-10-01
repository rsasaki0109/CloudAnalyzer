//! Bounded full-density traversal. IO and output ownership stay with Python.

use ca_core::io::copc::CopcHeader;
use ca_core::io::copc_query::{CopcQuery, QueryItem, QueryLimits};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

#[pyclass(module = "cloudanalyzer_core._core")]
pub struct CopcSpatialQuery {
    header: CopcHeader,
    query: CopcQuery,
    raw_limit: u64,
}

#[pymethods]
impl CopcSpatialQuery {
    #[new]
    #[pyo3(signature = (head, file_size, bounds, page_bytes=1048576, compressed_node_bytes=16777216, raw_node_bytes=33554432, pending_entries=16384, page_depth=32))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        head: &[u8],
        file_size: u64,
        bounds: [f64; 6],
        page_bytes: u64,
        compressed_node_bytes: u64,
        raw_node_bytes: u64,
        pending_entries: usize,
        page_depth: usize,
    ) -> PyResult<Self> {
        let header = CopcHeader::parse(head).map_err(super::io_err)?;
        let query = CopcQuery::new(
            &header,
            file_size,
            bounds,
            QueryLimits {
                page_bytes,
                compressed_node_bytes,
                raw_node_bytes,
                pending_entries,
                page_depth,
            },
        )
        .map_err(super::io_err)?;
        Ok(Self {
            header,
            query,
            raw_limit: raw_node_bytes,
        })
    }

    #[getter]
    fn total_points(&self) -> u64 {
        self.header.total_points
    }

    #[getter]
    fn record_length(&self) -> usize {
        self.header.record_len()
    }

    /// (kind, offset, bytes, points); kind is "page" or "node".
    fn next_item(&self) -> Option<(&'static str, u64, u64, u64)> {
        self.query.next_item().map(|item| match item {
            QueryItem::Page { offset, size } => ("page", offset, size, 0),
            QueryItem::Node(entry) => (
                "node",
                entry.offset,
                entry.byte_size as u64,
                entry.point_count as u64,
            ),
        })
    }

    fn supply_page(&mut self, offset: u64, page: &[u8]) -> PyResult<()> {
        self.query.supply_page(offset, page).map_err(super::io_err)
    }

    /// Decode only the pending node, without acknowledging it.
    fn decode_records<'py>(
        &self,
        py: Python<'py>,
        offset: u64,
        compressed: &[u8],
    ) -> PyResult<Bound<'py, PyBytes>> {
        let Some(QueryItem::Node(entry)) = self.query.next_item() else {
            return Err(PyValueError::new_err("no pending point node"));
        };
        if offset != entry.offset || compressed.len() as u64 != entry.byte_size as u64 {
            return Err(PyValueError::new_err(
                "compressed bytes do not match the pending node",
            ));
        }
        let raw = py
            .detach(|| {
                self.header
                    .decode_records(compressed, entry.point_count as usize, self.raw_limit)
            })
            .map_err(super::io_err)?;
        Ok(PyBytes::new(py, &raw))
    }

    fn advance_node(&mut self) -> PyResult<()> {
        self.query.advance_node().map_err(super::io_err)
    }
}
