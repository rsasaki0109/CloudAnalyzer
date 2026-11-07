//! Streaming loaders and file inspection.

use super::*;

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
