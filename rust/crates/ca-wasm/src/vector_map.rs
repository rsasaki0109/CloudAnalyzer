//! A vector map (lanes, stop lines, signals, crosswalks) drawn over the
//! clouds, built and edited with vectormap-rs and exported as Lanelet2 for
//! Autoware. Everything crosses the boundary as JSON: edits are vectormap
//! commands, the view is plain polylines in the map's local frame.

use serde_json::{Value, json};
use vectormap_core::{Command, Map, Point2, Polyline3, Side};
use vectormap_io::lanelet2::{self, LoadOptions, SaveOptions};
use vectormap_io::{autoware, json as irjson};
use vectormap_validation::{ValidationOptions, validate};
use wasm_bindgen::prelude::*;

/// Edits kept for undo.
const UNDO_DEPTH: usize = 100;

#[wasm_bindgen]
pub struct VectorMapSession {
    map: Map,
    undo: Vec<Map>,
}

fn error(e: impl std::fmt::Display) -> JsError {
    JsError::new(&e.to_string())
}

fn points(line: &Polyline3) -> Value {
    line.points.iter().map(|p| json!([p.x, p.y, p.z])).collect()
}

#[wasm_bindgen]
impl VectorMapSession {
    /// An empty map.
    #[wasm_bindgen(constructor)]
    pub fn new() -> VectorMapSession {
        VectorMapSession {
            map: Map::new(),
            undo: Vec::new(),
        }
    }

    /// Replace the map with a Lanelet2 file (`.osm`) or vectormap IR
    /// (`.json`). Returns the load issues as JSON.
    pub fn open(&mut self, name: &str, text: &str) -> Result<String, JsError> {
        let loaded = if name.to_ascii_lowercase().ends_with(".json") {
            irjson::from_str(text).map_err(error)?
        } else {
            lanelet2::read_str(text, &LoadOptions::default()).map_err(error)?
        };
        self.map = loaded.map;
        self.undo.clear();
        serde_json::to_string(&loaded.issues).map_err(error)
    }

    /// Start over with an empty map.
    pub fn clear(&mut self) {
        self.push_undo();
        self.map = Map::new();
    }

    /// Apply a JSON list of vectormap commands, all or nothing. Returns the
    /// change sets as JSON.
    pub fn apply(&mut self, commands: &str) -> Result<String, JsError> {
        let commands: Vec<Command> = serde_json::from_str(commands).map_err(error)?;
        let before = self.map.clone();
        let changes = self.map.apply_all(&commands).map_err(error)?;
        self.undo.push(before);
        if self.undo.len() > UNDO_DEPTH {
            self.undo.remove(0);
        }
        serde_json::to_string(&changes).map_err(error)
    }

    /// Draft roads from cloud cross-sections along original-coordinate poses.
    /// All generated roads are one undo step; construction failures keep the map.
    #[wasm_bindgen(js_name = buildFromTrajectory)]
    pub fn build_from_trajectory(
        &mut self,
        cloud: &crate::Cloud,
        positions: &[f64],
        options: &str,
    ) -> Result<String, JsError> {
        if !positions.len().is_multiple_of(3) {
            return Err(JsError::new(
                "trajectory positions must contain XYZ triples",
            ));
        }
        let o: ca_core::vector_map::BuildOptions = serde_json::from_str(options).map_err(error)?;
        let poses: Vec<_> = positions
            .as_chunks::<3>()
            .0
            .iter()
            .map(|p| [p[0], p[1], p[2]])
            .collect();
        let before = self.map.clone();
        let report =
            ca_core::vector_map::build(&mut self.map, &cloud.inner, &poses, &o).map_err(error)?;
        self.undo.push(before);
        if self.undo.len() > UNDO_DEPTH {
            self.undo.remove(0);
        }
        serde_json::to_string(&report).map_err(error)
    }

    /// Undo the last edit; false if there is none.
    pub fn undo(&mut self) -> bool {
        match self.undo.pop() {
            Some(map) => {
                self.map = map;
                true
            }
            None => false,
        }
    }

    #[wasm_bindgen(getter, js_name = undoDepth)]
    pub fn undo_depth(&self) -> usize {
        self.undo.len()
    }

    /// What to draw, as JSON: lanes (their boundaries oriented along the
    /// lane, centreline, links), boundaries with their kinds, stop lines,
    /// crosswalks and signals, in the map's local frame (metres).
    pub fn view(&self) -> String {
        let map = &self.map;
        let lanes: Vec<Value> = map
            .lanes()
            .filter_map(|lane| {
                let left = map.oriented_boundary(lane.id, Side::Left)?;
                let right = map.oriented_boundary(lane.id, Side::Right)?;
                let center = map.centerline(lane.id)?;
                Some(json!({
                    "id": lane.id,
                    "kind": lane.kind,
                    "left": points(&left),
                    "right": points(&right),
                    "center": points(&center),
                    "successors": map.successors(lane.id),
                    "predecessors": map.predecessors(lane.id),
                    "leftNeighbor": map.neighbor(lane.id, Side::Left),
                    "rightNeighbor": map.neighbor(lane.id, Side::Right),
                    "speedLimit": lane.speed_limit.map(|s| s.kmh),
                    "turn": lane.turn_direction,
                    "oneWay": lane.one_way,
                }))
            })
            .collect();
        let boundaries: Vec<Value> = map
            .boundaries()
            .map(|b| json!({"id": b.id, "kind": b.kind, "points": points(&b.geometry)}))
            .collect();
        let stop_lines: Vec<Value> = map
            .stop_lines()
            .map(|s| json!({"id": s.id, "points": points(&s.geometry)}))
            .collect();
        let crosswalks: Vec<Value> = map
            .crosswalks()
            .map(|c| {
                let outline = c.outline();
                let ring: Vec<Value> = outline
                    .points
                    .iter()
                    .map(|p| json!([p.x, p.y, p.z]))
                    .collect();
                json!({"id": c.id, "outline": ring})
            })
            .collect();
        let signals: Vec<Value> = map
            .traffic_signals()
            .map(|s| json!({"id": s.id, "points": points(&s.geometry), "height": s.height}))
            .collect();
        json!({
            "lanes": lanes,
            "boundaries": boundaries,
            "stopLines": stop_lines,
            "crosswalks": crosswalks,
            "signals": signals,
            "georeferenced": map.metadata().georeference.is_some(),
        })
        .to_string()
    }

    /// The lane nearest to `(x, y)` as JSON (`null` for an empty map):
    /// its id, distance, station, lateral offset and whether the point is
    /// inside it.
    #[wasm_bindgen(js_name = nearestLane)]
    pub fn nearest_lane(&self, x: f64, y: f64) -> String {
        serde_json::to_string(&self.map.find_nearest_lane(Point2::new(x, y)))
            .unwrap_or_else(|_| "null".into())
    }

    /// Everything about one lane as JSON (`null` if there is none).
    #[wasm_bindgen(js_name = laneInfo)]
    pub fn lane_info(&self, lane: u64) -> String {
        serde_json::to_string(&self.map.lane_info(vectormap_core::LaneId(lane)))
            .unwrap_or_else(|_| "null".into())
    }

    /// Issues as JSON, most severe first; `autoware` adds the Autoware
    /// compatibility checks.
    pub fn validate(&self, autoware_checks: bool) -> String {
        let mut report = validate(&self.map, &ValidationOptions::default());
        if autoware_checks {
            report.issues.extend(autoware::check(&self.map));
            report.issues.sort_by_key(|i| std::cmp::Reverse(i.severity));
        }
        serde_json::to_string(&report.issues).unwrap_or_else(|_| "[]".into())
    }

    /// The map as Lanelet2 (Autoware profile when `autoware`), as JSON:
    /// `{osm, projectorInfo, issues}`.
    #[wasm_bindgen(js_name = exportLanelet2)]
    pub fn export_lanelet2(&self, autoware_profile: bool) -> String {
        let options = if autoware_profile {
            SaveOptions::autoware()
        } else {
            SaveOptions::default()
        };
        let (osm, issues) = lanelet2::write_string(&self.map, &options);
        json!({
            "osm": osm,
            "projectorInfo": autoware::projector_info_yaml(self.map.metadata().georeference),
            "issues": issues,
        })
        .to_string()
    }

    /// The map as vectormap IR JSON (lossless).
    #[wasm_bindgen(js_name = toJson)]
    pub fn to_json(&self) -> String {
        irjson::to_string(&self.map)
    }
}

impl VectorMapSession {
    fn push_undo(&mut self) {
        self.undo.push(self.map.clone());
        if self.undo.len() > UNDO_DEPTH {
            self.undo.remove(0);
        }
    }
}

impl Default for VectorMapSession {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_edit_undo_and_export() {
        let mut s = VectorMapSession::new();
        let changes = s
            .apply(
                r#"[{"op": "build_road", "reference": [[0,0,1],[40,0,1]],
                    "lanes": [{"width": 3.5}, {"width": 3.5, "direction": "backward"}],
                    "segment_length": 20, "speed_limit": {"kmh": 30}}]"#,
            )
            .unwrap();
        assert!(changes.contains("\"created\""));
        let view: Value = serde_json::from_str(&s.view()).unwrap();
        assert_eq!(view["lanes"].as_array().unwrap().len(), 4);
        assert_eq!(view["boundaries"].as_array().unwrap().len(), 6);
        let nearest: Value = serde_json::from_str(&s.nearest_lane(5.0, 1.0)).unwrap();
        assert_eq!(nearest["inside"], true);
        let lane = nearest["lane"].as_u64().unwrap();
        s.apply(&format!(
            r#"[{{"op": "add_traffic_signal", "lanes": [{lane}]}}]"#
        ))
        .unwrap();
        let view: Value = serde_json::from_str(&s.view()).unwrap();
        assert_eq!(view["signals"].as_array().unwrap().len(), 1);
        assert_eq!(view["stopLines"].as_array().unwrap().len(), 1);

        let issues: Value = serde_json::from_str(&s.validate(true)).unwrap();
        assert!(
            issues
                .as_array()
                .unwrap()
                .iter()
                .all(|i| i["severity"] != "error"),
            "{issues}"
        );
        let export: Value = serde_json::from_str(&s.export_lanelet2(true)).unwrap();
        assert_eq!(export["projectorInfo"], "projector_type: Local\n");
        let osm = export["osm"].as_str().unwrap().to_string();

        assert!(s.undo());
        let view: Value = serde_json::from_str(&s.view()).unwrap();
        assert!(view["signals"].as_array().unwrap().is_empty());

        // The export reads back with the same lanes.
        let mut t = VectorMapSession::new();
        t.open("lanelet2_map.osm", &osm).unwrap();
        let view: Value = serde_json::from_str(&t.view()).unwrap();
        assert_eq!(view["lanes"].as_array().unwrap().len(), 4);
        assert_eq!(view["signals"].as_array().unwrap().len(), 1);
    }
}
