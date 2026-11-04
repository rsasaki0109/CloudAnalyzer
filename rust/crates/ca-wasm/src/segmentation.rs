//! Shape and cluster segmentation colouring.

use super::*;

/// Kinds of segment in a [`Segmentation`].
pub(crate) const SEGMENT_PLANE: u8 = 0;
pub(crate) const SEGMENT_SPHERE: u8 = 1;
pub(crate) const SEGMENT_CYLINDER: u8 = 2;
pub(crate) const SEGMENT_CLUSTER: u8 = 3;
/// The clusters past the 254 largest, together.
pub(crate) const SEGMENT_OTHER_CLUSTERS: u8 = 4;
/// Points in no shape, or cluster noise.
pub(crate) const SEGMENT_REST: u8 = 5;
/// Numbers per segment in [`Segmentation::params`].
const SEGMENT_PARAMS: usize = 8;
pub(crate) const REST_COLOR: [u8; 3] = [128, 128, 128];
pub(crate) const OTHER_COLOR: [u8; 3] = [200, 200, 200];

/// A distinct color for segment `k`: hues a golden angle apart.
pub(crate) fn segment_color(k: usize) -> [u8; 3] {
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
    pub(crate) cloud: Option<Cloud>,
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
    pub(crate) fn new(found: usize) -> Self {
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

    pub(crate) fn push(
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

    pub(crate) fn color(&self, segment: usize) -> [u8; 3] {
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
