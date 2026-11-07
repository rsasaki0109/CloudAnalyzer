//! Merging clouds.

use super::*;

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
