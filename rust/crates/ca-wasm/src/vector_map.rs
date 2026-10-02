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
        if self.map != before {
            self.undo.push(before);
            if self.undo.len() > UNDO_DEPTH {
                self.undo.remove(0);
            }
        }
        serde_json::to_string(&report).map_err(error)
    }

    /// Preview ground-supported junction drafts without changing the map or undo history.
    #[wasm_bindgen(js_name = previewJunctions)]
    pub fn preview_junctions(
        &self,
        cloud: &crate::Cloud,
        options: &str,
    ) -> Result<String, JsError> {
        let o = serde_json::from_str(options).map_err(error)?;
        let report =
            ca_core::vector_map::junctions::propose(&self.map, &cloud.inner, &o).map_err(error)?;
        serde_json::to_string(&report).map_err(error)
    }

    /// Add selected geometric connections as one Undo step, rechecking support.
    #[wasm_bindgen(js_name = connectJunctions)]
    pub fn connect_junctions(
        &mut self,
        cloud: &crate::Cloud,
        options: &str,
        pairs: &str,
    ) -> Result<String, JsError> {
        let o = serde_json::from_str(options).map_err(error)?;
        let pairs: Option<Vec<[u64; 2]>> = serde_json::from_str(pairs).map_err(error)?;
        let before = self.map.clone();
        let report = ca_core::vector_map::junctions::connect(
            &mut self.map,
            &cloud.inner,
            &o,
            pairs.as_deref(),
        )
        .map_err(error)?;
        if self.map != before {
            self.undo.push(before);
            if self.undo.len() > UNDO_DEPTH {
                self.undo.remove(0);
            }
        }
        serde_json::to_string(&report).map_err(error)
    }

    /// Measure a user-identified signal head; additions are one undo step.
    #[wasm_bindgen(js_name = measureSignal)]
    pub fn measure_signal(
        &mut self,
        cloud: &crate::Cloud,
        options: &str,
        preview: bool,
    ) -> Result<String, JsError> {
        let o = serde_json::from_str(options).map_err(error)?;
        let before = self.map.clone();
        let report = if preview {
            ca_core::vector_map::signals::measure(&self.map, &cloud.inner, &o)
        } else {
            ca_core::vector_map::signals::add(&mut self.map, &cloud.inner, &o)
        }
        .map_err(error)?;
        if self.map != before {
            self.undo.push(before);
            if self.undo.len() > UNDO_DEPTH {
                self.undo.remove(0);
            }
        }
        serde_json::to_string(&report).map_err(error)
    }

    /// Preview measured ground bands, or add an explicitly confirmed crossing.
    #[wasm_bindgen(js_name = measureCrosswalk)]
    pub fn measure_crosswalk(
        &mut self,
        cloud: &crate::Cloud,
        options: &str,
        preview: bool,
    ) -> Result<String, JsError> {
        let o = serde_json::from_str(options).map_err(error)?;
        let before = self.map.clone();
        let report = if preview {
            ca_core::vector_map::crosswalks::propose(&self.map, &cloud.inner, &o)
        } else {
            ca_core::vector_map::crosswalks::add(&mut self.map, &cloud.inner, &o)
        }
        .map_err(error)?;
        if self.map != before {
            self.undo.push(before);
            if self.undo.len() > UNDO_DEPTH {
                self.undo.remove(0);
            }
        }
        serde_json::to_string(&report).map_err(error)
    }

    /// Explicitly edit existing crossing/signal geometry as one Undo step.
    #[wasm_bindgen(js_name = editFeatureGeometry)]
    pub fn edit_feature_geometry(&mut self, options: &str) -> Result<String, JsError> {
        let o = serde_json::from_str(options).map_err(error)?;
        let before = self.map.clone();
        let report =
            ca_core::vector_map::feature_editing::edit(&mut self.map, &o).map_err(error)?;
        if report.changed {
            self.undo.push(before);
            if self.undo.len() > UNDO_DEPTH {
                self.undo.remove(0);
            }
        }
        serde_json::to_string(&report).map_err(error)
    }

    /// Search the generated roads' surroundings without manual feature boxes.
    #[wasm_bindgen(js_name = discoverFeatures)]
    pub fn discover_features(
        &self,
        cloud: &crate::Cloud,
        options: &str,
    ) -> Result<String, JsError> {
        let o = serde_json::from_str(options).map_err(error)?;
        serde_json::to_string(
            &ca_core::vector_map::discovery::propose(&self.map, &cloud.inner, &o).map_err(error)?,
        )
        .map_err(error)
    }

    /// Add explicit reviewed classifications/associations as one atomic Undo step.
    #[wasm_bindgen(js_name = confirmFeatures)]
    pub fn confirm_features(
        &mut self,
        cloud: &crate::Cloud,
        options: &str,
        confirmations: &str,
    ) -> Result<String, JsError> {
        let o = serde_json::from_str(options).map_err(error)?;
        let confirmations: Vec<_> = serde_json::from_str(confirmations).map_err(error)?;
        let before = self.map.clone();
        let report =
            ca_core::vector_map::discovery::add(&mut self.map, &cloud.inner, &o, &confirmations)
                .map_err(error)?;
        if self.map != before {
            self.undo.push(before);
            if self.undo.len() > UNDO_DEPTH {
                self.undo.remove(0);
            }
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
                    "leftRef": {"id": lane.left.boundary, "reversed": lane.left.reversed},
                    "rightRef": {"id": lane.right.boundary, "reversed": lane.right.reversed},
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
            .map(|s| json!({"id": s.id, "points": points(&s.geometry),"geometrySource":s.attributes.get("cloudanalyzer_geometry_source").or_else(||s.attributes.get_prefixed("lanelet2","cloudanalyzer_geometry_source")).unwrap_or("imported_or_manual")}))
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
                let paint_bands = c
                    .attributes
                    .get("cloudanalyzer_paint_bands")
                    .or_else(|| {
                        c.attributes
                            .get_prefixed("lanelet2", "cloudanalyzer_paint_bands")
                    })
                    .map(|text| {
                        if text.len() > 8192 {
                            return vec![];
                        }
                        serde_json::from_str::<Vec<Vec<[f64; 3]>>>(text)
                            .ok()
                            .filter(|v| {
                                v.len() <= 32
                                    && v.iter().all(|b| {
                                        b.len() == 4 && b.iter().flatten().all(|x| x.is_finite())
                                    })
                            })
                            .unwrap_or_default()
                    });
                json!({"id": c.id, "outline": ring, "paintBands":paint_bands,
                    "editable": c.polygon.is_none() && c.left_edge.points.len() >= 2 && c.right_edge.points.len() >= 2 && ring.len() <= 256,
                    "geometrySource": c.attributes.get("cloudanalyzer_geometry_source").or_else(|| c.attributes.get_prefixed("lanelet2", "cloudanalyzer_geometry_source")).unwrap_or("imported_or_manual")})
            })
            .collect();
        let signals: Vec<Value> = map
            .traffic_signals()
            .map(|s| json!({"id": s.id, "points": points(&s.geometry), "height": s.height,
                "geometrySource": s.attributes.get("cloudanalyzer_geometry_source").or_else(|| s.attributes.get_prefixed("lanelet2", "cloudanalyzer_geometry_source")).unwrap_or("imported_or_manual")}))
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
    #[test]
    fn curved_paint_envelope_confirm_export_reload_edit_and_undo() {
        let mut source = ca_core::PointCloud {
            colors: Some(vec![]),
            ..Default::default()
        };
        for i in 0..=200 {
            for j in 0..=160 {
                let x = -10. + i as f64 * 0.1;
                let y = -8. + j as f64 * 0.1;
                let bright = (-8.0..8.0).contains(&x)
                    && (x + 8.) % 1. < 0.5
                    && (y - 0.2 * x - 0.015 * x * x).abs() <= 3. + 0.05 * x;
                source.positions.push([x, y, 2. + 0.02 * x]);
                source
                    .colors
                    .as_mut()
                    .unwrap()
                    .push([if bright { 210 } else { 70 }; 3]);
            }
        }
        let cloud = crate::Cloud::unindexed(source);
        let mut s = super::VectorMapSession::new();
        s.apply(r#"[{"op":"build_road","reference":[[-10,0,2],[10,0,2]],"lanes":[{"width":3.5,"direction":"forward"}]}]"#).unwrap();
        let lane = s.map.lanes().next().unwrap().id;
        let options =
            serde_json::json!({"min":[-11,-9,1.5],"max":[11,9,2.5],"lanes":[lane]}).to_string();
        let before = s.to_json();
        let preview: serde_json::Value =
            serde_json::from_str(&s.measure_crosswalk(&cloud, &options, true).unwrap()).unwrap();
        assert_eq!(s.to_json(), before);
        assert!(
            preview["candidates"][0]["outline"]
                .as_array()
                .unwrap()
                .len()
                > 4
        );
        s.measure_crosswalk(&cloud, &options, false).unwrap();
        let walk = s.map.crosswalks().next().unwrap().clone();
        let added = s.to_json();
        let saved: serde_json::Value = serde_json::from_str(&s.export_lanelet2(true)).unwrap();
        let mut restored = super::VectorMapSession::new();
        restored
            .open("curved.osm", saved["osm"].as_str().unwrap())
            .unwrap();
        let loaded = restored.map.crosswalk(walk.id).unwrap().clone();
        assert_eq!(loaded.left_edge.points.len(), walk.left_edge.points.len());
        assert_eq!(loaded.right_edge.points.len(), walk.right_edge.points.len());
        // Lanelet2 normalizes travel direction; all envelope vertices and
        // segments must survive, allowing reversal of the entire edge.
        for (a, b) in [
            (&loaded.left_edge.points, &walk.left_edge.points),
            (&loaded.right_edge.points, &walk.right_edge.points),
        ] {
            let equal = |x: &vectormap_core::Point3, y: &vectormap_core::Point3| {
                (x.x - y.x).hypot(x.y - y.y) < 1e-7 && (x.z - y.z).abs() < 1e-7
            };
            assert!(
                a.iter().zip(b).all(|(x, y)| equal(x, y))
                    || a.iter().zip(b.iter().rev()).all(|(x, y)| equal(x, y))
            );
        }
        let imported = restored.to_json();
        restored.measure_crosswalk(&cloud, &options, false).unwrap();
        assert_eq!(restored.to_json(), imported);
        let mut vertices: Vec<_> = loaded
            .outline()
            .points
            .iter()
            .map(|p| [p.x, p.y, p.z])
            .collect();
        vertices[0][2] += 0.02;
        restored
            .edit_feature_geometry(
                &serde_json::json!({"kind":"crosswalk","id":walk.id,"points":vertices}).to_string(),
            )
            .unwrap();
        assert!(restored.undo());
        assert_eq!(restored.to_json(), imported);
        assert!(s.undo());
        assert_eq!(s.to_json(), before);
        assert_ne!(added, before);
    }

    #[test]
    fn source_only_scene_discovery_confirm_edit_roundtrip_and_undo() {
        let mut source = ca_core::PointCloud {
            colors: Some(vec![]),
            ..Default::default()
        };
        for i in 0..=400 {
            for j in 0..=100 {
                let x = -5.0 + i as f64 * 0.1;
                let y = -5.0 + j as f64 * 0.1;
                let white = ((8.0..12.0).contains(&x) && (x - 8.0) % 1.0 < 0.5
                    || (20.0..20.6).contains(&x))
                    && y.abs() <= 3.0;
                source.positions.push([x, y, 2.0]);
                source
                    .colors
                    .as_mut()
                    .unwrap()
                    .push([if white { 220 } else { 60 }; 3]);
            }
        }
        for i in 0..=24 {
            for j in 0..=12 {
                source
                    .positions
                    .push([25.0, -0.6 + i as f64 * 0.05, 6.0 + j as f64 * 0.05]);
                source.colors.as_mut().unwrap().push([80; 3]);
            }
        }
        let cloud = crate::Cloud::unindexed(source);
        let mut s = super::VectorMapSession::new();
        s.build_from_trajectory(&cloud, &[-5.0, 0.0, 4.0, 35.0, 0.0, 4.0], "{}")
            .unwrap();
        let before = s.to_json();
        let depth = s.undo.len();
        let r: serde_json::Value =
            serde_json::from_str(&s.discover_features(&cloud, "{}").unwrap()).unwrap();
        assert_eq!(s.to_json(), before);
        assert_eq!(s.undo.len(), depth);
        let lane = s.map.lanes().next().unwrap().id;
        let confirmations:Vec<_>=r["candidates"].as_array().unwrap().iter().map(|c|serde_json::json!({"candidate":c["id"],"key":c["key"],"lanes":[lane],"classification":match c["evidence"]["kind"].as_str().unwrap(){"repeated_paint"=>"crosswalk","bright_bar"=>"stop_line",_=>"vehicle_signal"}})).collect();
        let confirmed = serde_json::to_string(&confirmations).unwrap();
        s.confirm_features(&cloud, "{}", &confirmed).unwrap();
        assert_eq!(s.undo.len(), depth + 1);
        let added = s.to_json();
        s.confirm_features(&cloud, "{}", &confirmed).unwrap();
        assert_eq!(s.to_json(), added);
        assert_eq!(s.undo.len(), depth + 1);
        let stop = s.map.stop_lines().next().expect("measured stop line");
        let id = stop.id;
        let mut points: Vec<_> = stop
            .geometry
            .points
            .iter()
            .map(|p| [p.x, p.y, p.z])
            .collect();
        points[0][0] += 0.1;
        s.edit_feature_geometry(
            &serde_json::json!({"kind":"stop_line","id":id,"points":points}).to_string(),
        )
        .unwrap();
        assert_eq!(s.undo.len(), depth + 2);
        assert!(s.undo());
        assert_eq!(s.to_json(), added);
        let saved: serde_json::Value = serde_json::from_str(&s.export_lanelet2(true)).unwrap();
        let mut reopened = super::VectorMapSession::new();
        reopened
            .open("map.osm", saved["osm"].as_str().unwrap())
            .unwrap();
        let imported = reopened.to_json();
        reopened.confirm_features(&cloud, "{}", &confirmed).unwrap();
        assert_eq!(reopened.to_json(), imported);
        assert_eq!(reopened.undo.len(), 0);
        assert_eq!(
            reopened.map.crosswalks().count(),
            s.map.crosswalks().count()
        );
        assert_eq!(reopened.map.traffic_signals().count(), 1);
        assert!(s.undo());
        assert_eq!(s.to_json(), before);
    }

    #[test]
    fn paint_preview_add_replay_osm_roundtrip_and_undo() {
        let mut s = super::VectorMapSession::new();
        s.apply(r#"[{"op":"build_road","reference":[[-5,0,2],[5,0,2]],"lanes":[{"width":3.5,"direction":"forward"}]}]"#).unwrap();
        let lane = s.map.lanes().next().unwrap().id;
        let mut points = ca_core::PointCloud {
            colors: Some(vec![]),
            ..Default::default()
        };
        for i in 0..=200 {
            for j in 0..=160 {
                let x = -5.0 + i as f64 * 0.05;
                let y = -4.0 + j as f64 * 0.05;
                points.positions.push([x, y, 2.0]);
                let paint = (-2.0..2.0).contains(&x) && (x + 2.0) % 1.0 < 0.5 && y.abs() <= 3.0;
                points
                    .colors
                    .as_mut()
                    .unwrap()
                    .push([if paint { 210 } else { 70 }; 3]);
            }
        }
        let cloud = crate::Cloud::unindexed(points);
        let options = serde_json::json!({"min":[-5.1,-4.1,1.9],"max":[5.1,4.1,2.1],"lanes":[lane]})
            .to_string();
        let before = s.to_json();
        let depth = s.undo.len();
        let preview: serde_json::Value =
            serde_json::from_str(&s.measure_crosswalk(&cloud, &options, true).unwrap()).unwrap();
        assert_eq!(preview["candidates"][0]["stripe_count"], 4);
        assert_eq!(s.to_json(), before);
        assert_eq!(s.undo.len(), depth);
        s.measure_crosswalk(&cloud, &options, false).unwrap();
        assert_eq!(s.undo.len(), depth + 1);
        let added = s.to_json();
        s.measure_crosswalk(&cloud, &options, false).unwrap();
        assert_eq!(s.to_json(), added);
        assert_eq!(s.undo.len(), depth + 1);
        let exported: serde_json::Value = serde_json::from_str(&s.export_lanelet2(true)).unwrap();
        assert!(
            exported["osm"]
                .as_str()
                .unwrap()
                .contains("point_cloud_brightness_stripes")
        );
        let mut restored = super::VectorMapSession::new();
        restored
            .open("map.osm", exported["osm"].as_str().unwrap())
            .unwrap();
        let imported = restored.to_json();
        let view: serde_json::Value = serde_json::from_str(&restored.view()).unwrap();
        assert_eq!(
            view["crosswalks"][0]["paintBands"]
                .as_array()
                .unwrap()
                .len(),
            4
        );
        let replay: serde_json::Value =
            serde_json::from_str(&restored.measure_crosswalk(&cloud, &options, false).unwrap())
                .unwrap();
        assert!(replay["reused"].is_number());
        assert_eq!(restored.to_json(), imported);
        assert_eq!(restored.undo.len(), 0);
        let crossing = restored.map.crosswalks().next().unwrap();
        let id = crossing.id;
        let bands = crossing
            .attributes
            .get_prefixed("lanelet2", "cloudanalyzer_paint_bands")
            .unwrap()
            .to_string();
        let mut vertices: Vec<_> = crossing
            .outline()
            .points
            .iter()
            .map(|p| [p.x, p.y, p.z])
            .collect();
        vertices[0][0] += 0.1;
        let edit = serde_json::json!({"kind":"crosswalk","id":id,"points":vertices}).to_string();
        restored.edit_feature_geometry(&edit).unwrap();
        assert_eq!(restored.undo.len(), 1);
        let changed = restored.to_json();
        restored.edit_feature_geometry(&edit).unwrap();
        assert_eq!(restored.to_json(), changed);
        assert_eq!(restored.undo.len(), 1);
        restored.measure_crosswalk(&cloud, &options, false).unwrap();
        assert_eq!(restored.to_json(), changed); // measuring again preserves the explicit edit
        let saved: serde_json::Value =
            serde_json::from_str(&restored.export_lanelet2(true)).unwrap();
        let mut reopened = super::VectorMapSession::new();
        reopened
            .open("edited.osm", saved["osm"].as_str().unwrap())
            .unwrap();
        let walk = reopened.map.crosswalk(id).unwrap();
        assert_eq!(
            walk.attributes
                .get_prefixed("lanelet2", "cloudanalyzer_geometry_source"),
            Some("point_cloud_brightness_stripes_user_edited")
        );
        assert_eq!(
            walk.attributes
                .get_prefixed("lanelet2", "cloudanalyzer_paint_bands"),
            Some(bands.as_str())
        );
        assert!(restored.undo());
        assert_eq!(restored.to_json(), imported);
        assert!(s.undo());
        assert_eq!(s.to_json(), before);
    }

    #[test]
    fn signal_measurement_preview_add_replay_and_undo_are_atomic() {
        let mut s = super::VectorMapSession::new();
        s.apply(r#"[{"op":"build_road","reference":[[0,0,2],[0,10,2]],"lanes":[{"width":3.5,"direction":"forward"}]}]"#).unwrap();
        let lane = s.map.lanes().next().unwrap().id;
        let mut points = ca_core::PointCloud::default();
        for x in 0..=24 {
            for z in 0..=10 {
                points
                    .positions
                    .push([-0.6 + x as f64 * 0.05, 11.0, 7.0 + z as f64 * 0.05]);
            }
        }
        let cloud = crate::Cloud::unindexed(points);
        let options =
            serde_json::json!({"min":[-0.7,10.9,6.9],"max":[0.7,11.1,7.6],"lanes":[lane]})
                .to_string();
        let before = s.to_json();
        let depth = s.undo.len();
        s.measure_signal(&cloud, &options, true).unwrap();
        assert_eq!(s.to_json(), before);
        assert_eq!(s.undo.len(), depth);
        s.measure_signal(&cloud, &options, false).unwrap();
        assert_eq!(s.undo.len(), depth + 1);
        let added = s.to_json();
        s.measure_signal(&cloud, &options, false).unwrap();
        assert_eq!(s.to_json(), added);
        assert_eq!(s.undo.len(), depth + 1);
        let exported: serde_json::Value = serde_json::from_str(&s.export_lanelet2(true)).unwrap();
        let mut restored = super::VectorMapSession::new();
        restored
            .open("map.osm", exported["osm"].as_str().unwrap())
            .unwrap();
        let imported = restored.to_json();
        let replay: serde_json::Value =
            serde_json::from_str(&restored.measure_signal(&cloud, &options, false).unwrap())
                .unwrap();
        assert!(replay["reused"].is_number());
        assert_eq!(restored.to_json(), imported);
        assert_eq!(restored.undo.len(), 0);
        let signal = restored.map.traffic_signals().next().unwrap();
        let id = signal.id;
        let mut vertices: Vec<_> = signal
            .geometry
            .points
            .iter()
            .map(|p| [p.x, p.y, p.z])
            .collect();
        vertices[0][2] += 0.1;
        restored
            .edit_feature_geometry(
                &serde_json::json!({"kind":"signal","id":id,"points":vertices,"height":0.7})
                    .to_string(),
            )
            .unwrap();
        assert_eq!(restored.undo.len(), 1);
        let saved: serde_json::Value =
            serde_json::from_str(&restored.export_lanelet2(true)).unwrap();
        let mut reopened = super::VectorMapSession::new();
        reopened
            .open("edited.osm", saved["osm"].as_str().unwrap())
            .unwrap();
        assert_eq!(reopened.map.traffic_signal(id).unwrap().height, Some(0.7));
        assert!(reopened.map.traffic_signal(id).unwrap().bulbs.is_empty());
        assert!(restored.undo());
        assert_eq!(restored.to_json(), imported);
        assert!(s.undo());
        assert_eq!(s.to_json(), before);
    }
    use super::*;

    #[test]
    fn graded_junction_survives_export_reload_and_exact_undo() {
        let mut s = VectorMapSession::new();
        s.apply(
            r#"[
          {"op":"build_road","reference":[[-20,0,0.8],[-10,0,1.4]],"lanes":[{"width":3.5}]},
          {"op":"build_road","reference":[[0,0,2],[10,0,2.6]],"lanes":[{"width":3.5}]}
        ]"#,
        )
        .unwrap();
        s.undo.clear();
        let before = s.to_json();
        let mut ground = ca_core::PointCloud::default();
        for x in -105..=55 {
            for y in -20..=20 {
                let x = x as f64 * 0.2;
                ground.positions.push([x, y as f64 * 0.2, 2.0 + x * 0.06]);
            }
        }
        let cloud = crate::Cloud::unindexed(ground);
        let preview: Value =
            serde_json::from_str(&s.preview_junctions(&cloud, "{}").unwrap()).unwrap();
        assert_eq!(preview["candidates"].as_array().unwrap().len(), 1);
        assert_eq!(s.to_json(), before);
        assert!(s.undo.is_empty());
        let report: Value =
            serde_json::from_str(&s.connect_junctions(&cloud, "{}", "null").unwrap()).unwrap();
        let id = vectormap_core::LaneId(report["added"][0].as_u64().unwrap());
        let original = s.map.centerline(id).unwrap();
        assert!(original.points.last().unwrap().z - original.points[0].z > 0.3);
        let saved: Value = serde_json::from_str(&s.export_lanelet2(true)).unwrap();
        let mut reopened = VectorMapSession::new();
        reopened
            .open("graded.osm", saved["osm"].as_str().unwrap())
            .unwrap();
        assert_eq!(reopened.map.centerline(id), Some(original));
        assert_eq!(
            reopened
                .map
                .lane(id)
                .unwrap()
                .attributes
                .get_prefixed("lanelet2", "cloudanalyzer_review_required"),
            Some("yes")
        );
        assert_eq!(s.undo.len(), 1);
        assert!(s.undo());
        assert_eq!(s.to_json(), before);
    }

    #[test]
    fn junction_preview_and_replay_do_not_add_undo_but_branch_batch_does() {
        let mut s = VectorMapSession::new();
        s.apply(r#"[
          {"op":"build_road","reference":[[-20,0,2],[-10,0,2]],"lanes":[{"width":3.5}],"speed_limit":{"kmh":20}},
          {"op":"build_road","reference":[[0,10,2],[0,20,2]],"lanes":[{"width":3.5}],"speed_limit":{"kmh":20}},
          {"op":"build_road","reference":[[0,-10,2],[0,-20,2]],"lanes":[{"width":3.5}],"speed_limit":{"kmh":20}}
        ]"#).unwrap();
        s.undo.clear();
        let mut ground = ca_core::PointCloud::default();
        for x in -110..=10 {
            for y in -110..=110 {
                ground.positions.push([x as f64 * 0.2, y as f64 * 0.2, 2.0]);
            }
        }
        let cloud = crate::Cloud::unindexed(ground);
        let before = s.to_json();
        let preview: Value =
            serde_json::from_str(&s.preview_junctions(&cloud, "{}").unwrap()).unwrap();
        assert_eq!(preview["candidates"].as_array().unwrap().len(), 2);
        assert_eq!(s.to_json(), before);
        assert_eq!(s.undo.len(), 0);
        s.connect_junctions(&cloud, "{}", "[]").unwrap();
        assert_eq!(s.undo.len(), 0);
        let report: Value =
            serde_json::from_str(&s.connect_junctions(&cloud, "{}", "null").unwrap()).unwrap();
        assert_eq!(report["added"].as_array().unwrap().len(), 2);
        assert_eq!(s.undo.len(), 1);
        let after = s.to_json();
        s.connect_junctions(&cloud, "{}", "null").unwrap();
        assert_eq!(s.to_json(), after);
        assert_eq!(s.undo.len(), 1);
        assert!(s.undo());
        assert_eq!(s.to_json(), before);
        assert!(!s.undo());
    }

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
