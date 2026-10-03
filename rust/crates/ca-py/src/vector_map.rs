//! File-based draft generation; returns artifacts without writing destinations.
use numpy::{PyReadonlyArray2, PyUntypedArrayMethods};
use pyo3::{exceptions::PyValueError, prelude::*};
use serde_json::{Value, json};
use vectormap_core::{GeoReference, Map};
use vectormap_io::{autoware, lanelet2};

/// Build draft roads and return JSON containing the map, projector and report.
/// Input positions must already use the same metre frame; metadata does not
/// transform the cloud or the trajectory. Reference geometry is never copied.
#[pyfunction]
#[pyo3(signature = (cloud, trajectory, options="{}", reference_map=None, georeference=None, existing_map=None))]
pub fn build_vector_map(
    py: Python<'_>,
    cloud: &str,
    trajectory: &str,
    options: &str,
    reference_map: Option<&str>,
    georeference: Option<&str>,
    existing_map: Option<&str>,
) -> PyResult<String> {
    py.detach(|| {
        generate(
            cloud,
            trajectory,
            options,
            reference_map,
            georeference,
            existing_map,
        )
    })
    .map_err(PyValueError::new_err)
}

fn generate(
    cloud: &str,
    trajectory: &str,
    options: &str,
    reference_map: Option<&str>,
    georeference: Option<&str>,
    existing_map: Option<&str>,
) -> Result<String, String> {
    let read = |path: &str| std::fs::read_to_string(path).map_err(|e| format!("{path}: {e}"));
    if reference_map.is_some() && georeference.is_some() {
        return Err("choose reference_map or explicit georeference, not both".into());
    }
    if existing_map.is_some() && (reference_map.is_some() || georeference.is_some()) {
        return Err("existing_map retains its own coordinates; choose it without reference_map or georeference".into());
    }
    let parameters = serde_json::from_str(options).map_err(|e| e.to_string())?;
    let mut map = Map::new();
    let mut import_issues = Vec::new();
    if let Some(path) = existing_map {
        let loaded = if path.to_ascii_lowercase().ends_with(".json") {
            vectormap_io::json::from_str(&read(path)?)
        } else {
            lanelet2::read_str(&read(path)?, &Default::default())
        }
        .map_err(|e| e.to_string())?;
        if let Some(issue) = loaded
            .issues
            .iter()
            .find(|i| i.severity == vectormap_core::Severity::Error)
        {
            return Err(format!("cannot retain existing_map: {}", issue.message));
        }
        map = loaded.map;
        import_issues = loaded.issues;
    }
    if let Some(path) = reference_map {
        let reference =
            lanelet2::read_str(&read(path)?, &Default::default()).map_err(|e| e.to_string())?;
        map.metadata_mut().georeference = reference.map.metadata().georeference;
    }
    if let Some(text) = georeference {
        let geo: GeoReference = serde_json::from_str(text).map_err(|e| e.to_string())?;
        if !geo.origin.lat.is_finite()
            || !(-80.0..=84.0).contains(&geo.origin.lat)
            || !geo.origin.lon.is_finite()
            || !(-180.0..=180.0).contains(&geo.origin.lon)
            || !geo.origin.alt.is_finite()
        {
            return Err(
                "georeference requires finite latitude [-80,84], longitude [-180,180] and altitude"
                    .into(),
            );
        }
        map.metadata_mut().georeference = Some(geo);
    }
    let data = std::fs::read(cloud).map_err(|e| format!("{cloud}: {e}"))?;
    let cloud = ca_core::read(cloud, &data).map_err(|e| e.to_string())?;
    let text = read(trajectory)?;
    let format = ca_core::trajectory::detect(trajectory, &text)
        .ok_or("trajectory must be TUM, KITTI, or timestamped XYZ CSV")?;
    let poses = ca_core::trajectory::parse(&text, format).map_err(|e| e.to_string())?;
    let extraction = ca_core::vector_map::build(&mut map, &cloud, &poses.positions, &parameters)
        .map_err(|e| e.to_string())?;
    artifacts(
        &map,
        json!(import_issues),
        json!({"status":"draft","extraction":extraction}),
    )
}

/// Preview or add junction connection drafts without changing existing geometry.
#[pyfunction]
#[pyo3(signature = (cloud, vector_map, options="{}", lane_pairs=None, preview_only=false))]
pub fn connect_vector_map_junctions(
    py: Python<'_>,
    cloud: &str,
    vector_map: &str,
    options: &str,
    lane_pairs: Option<&str>,
    preview_only: bool,
) -> PyResult<String> {
    py.detach(|| {
        let parameters = serde_json::from_str(options).map_err(|e| e.to_string())?;
        let selected: Option<Vec<[u64; 2]>> = lane_pairs
            .map(serde_json::from_str)
            .transpose()
            .map_err(|e| e.to_string())?;
        let text = std::fs::read_to_string(vector_map).map_err(|e| format!("{vector_map}: {e}"))?;
        let loaded = if vector_map.to_ascii_lowercase().ends_with(".json") {
            vectormap_io::json::from_str(&text)
        } else {
            lanelet2::read_str(&text, &Default::default())
        }
        .map_err(|e| e.to_string())?;
        if let Some(issue) = loaded
            .issues
            .iter()
            .find(|i| i.severity == vectormap_core::Severity::Error)
        {
            return Err(format!("cannot retain vector_map: {}", issue.message));
        }
        let data = std::fs::read(cloud).map_err(|e| format!("{cloud}: {e}"))?;
        let cloud = ca_core::read(cloud, &data).map_err(|e| e.to_string())?;
        let mut map = loaded.map;
        let junctions = if preview_only {
            ca_core::vector_map::junctions::propose(&map, &cloud, &parameters)
        } else {
            ca_core::vector_map::junctions::connect(
                &mut map,
                &cloud,
                &parameters,
                selected.as_deref(),
            )
        }
        .map_err(|e| e.to_string())?;
        artifacts(
            &map,
            json!(loaded.issues),
            json!({
                "status": if preview_only {"preview"} else {"draft"}, "junctions":junctions,
            }),
        )
    })
    .map_err(PyValueError::new_err)
}

/// Compatibility file reader: loads the complete input cloud before measurement.
#[pyfunction]
#[pyo3(signature = (cloud, vector_map, options, preview_only=true))]
pub fn measure_vector_map_signal(
    py: Python<'_>,
    cloud: &str,
    vector_map: &str,
    options: &str,
    preview_only: bool,
) -> PyResult<String> {
    py.detach(|| {
        let data = std::fs::read(cloud).map_err(|e| format!("{cloud}: {e}"))?;
        let cloud = ca_core::read(cloud, &data).map_err(|e| e.to_string())?;
        signal_artifacts(&cloud, vector_map, options, preview_only)
    })
    .map_err(PyValueError::new_err)
}

/// Fit at most 200,000 selected finite XYZ points, retaining f64 source coordinates.
#[pyfunction]
#[pyo3(signature = (points, vector_map, options, preview_only=true))]
pub fn measure_vector_map_signal_points(
    py: Python<'_>,
    points: PyReadonlyArray2<f64>,
    vector_map: &str,
    options: &str,
    preview_only: bool,
) -> PyResult<String> {
    let shape = points.shape();
    if shape[1] != 3 || shape[0] > 200_000 {
        return Err(PyValueError::new_err(
            "signal points must have shape (N, 3), at most 200000 points",
        ));
    }
    if !points.as_array().iter().all(|v| v.is_finite()) {
        return Err(PyValueError::new_err("signal points must be finite"));
    }
    let cloud = crate::cloud_of(crate::points(&points)?);
    py.detach(|| signal_artifacts(&cloud, vector_map, options, preview_only))
        .map_err(PyValueError::new_err)
}

fn signal_artifacts(
    cloud: &ca_core::PointCloud,
    vector_map: &str,
    options: &str,
    preview_only: bool,
) -> Result<String, String> {
    let parameters = serde_json::from_str(options).map_err(|e| e.to_string())?;
    let text = std::fs::read_to_string(vector_map).map_err(|e| format!("{vector_map}: {e}"))?;
    let loaded = if vector_map.to_ascii_lowercase().ends_with(".json") {
        vectormap_io::json::from_str(&text)
    } else {
        lanelet2::read_str(&text, &Default::default())
    }
    .map_err(|e| e.to_string())?;
    if let Some(issue) = loaded
        .issues
        .iter()
        .find(|i| i.severity == vectormap_core::Severity::Error)
    {
        return Err(format!("cannot retain vector_map: {}", issue.message));
    }
    let mut map = loaded.map;
    let measurement = if preview_only {
        ca_core::vector_map::signals::measure(&map, cloud, &parameters)
    } else {
        ca_core::vector_map::signals::add(&mut map, cloud, &parameters)
    }
    .map_err(|e| e.to_string())?;
    artifacts(
        &map,
        json!(loaded.issues),
        json!({"status":if preview_only {"preview"} else {"draft"},"signal":measurement}),
    )
}

/// Attribute-preserving compatibility reader: loads the complete source cloud.
#[pyfunction]
#[pyo3(signature = (cloud, vector_map, options, preview_only=true))]
pub fn measure_vector_map_crosswalk(
    py: Python<'_>,
    cloud: &str,
    vector_map: &str,
    options: &str,
    preview_only: bool,
) -> PyResult<String> {
    py.detach(|| {
        let parameters = serde_json::from_str(options).map_err(|e| e.to_string())?;
        let text = std::fs::read_to_string(vector_map).map_err(|e| format!("{vector_map}: {e}"))?;
        let loaded = if vector_map.to_ascii_lowercase().ends_with(".json") {
            vectormap_io::json::from_str(&text)
        } else {
            lanelet2::read_str(&text, &Default::default())
        }
        .map_err(|e| e.to_string())?;
        if let Some(issue) = loaded
            .issues
            .iter()
            .find(|i| i.severity == vectormap_core::Severity::Error)
        {
            return Err(format!("cannot retain vector_map: {}", issue.message));
        }
        let cloud = ca_core::read(
            cloud,
            &std::fs::read(cloud).map_err(|e| format!("{cloud}: {e}"))?,
        )
        .map_err(|e| e.to_string())?;
        let mut map = loaded.map;
        let measurement = if preview_only {
            ca_core::vector_map::crosswalks::propose(&map, &cloud, &parameters)
        } else {
            ca_core::vector_map::crosswalks::add(&mut map, &cloud, &parameters)
        }
        .map_err(|e| e.to_string())?;
        artifacts(
            &map,
            json!(loaded.issues),
            json!({"status": if preview_only {"preview"} else {"draft"}, "crosswalk":measurement}),
        )
    })
    .map_err(PyValueError::new_err)
}

/// Attribute-preserving whole-file discovery; does not require feature boxes.
#[pyfunction]
#[pyo3(signature=(cloud,vector_map=None,options="{}",confirmations=None))]
pub fn discover_vector_map_features(
    py: Python<'_>,
    cloud: &str,
    vector_map: Option<&str>,
    options: &str,
    confirmations: Option<&str>,
) -> PyResult<String> {
    py.detach(||{
        let options=serde_json::from_str(options).map_err(|e|e.to_string())?;
        let confirmed:Option<Vec<ca_core::vector_map::discovery::Confirmation>>=confirmations.map(serde_json::from_str).transpose().map_err(|e|e.to_string())?;
        let mut map=Map::new(); let mut issues=vec![];
        if let Some(path)=vector_map {
            let text=std::fs::read_to_string(path).map_err(|e|format!("{path}: {e}"))?;
            let loaded=if path.to_ascii_lowercase().ends_with(".json"){vectormap_io::json::from_str(&text)}else{lanelet2::read_str(&text,&Default::default())}.map_err(|e|e.to_string())?;
            if let Some(issue)=loaded.issues.iter().find(|i|i.severity==vectormap_core::Severity::Error) {return Err(format!("cannot retain vector_map: {}",issue.message));}
            map=loaded.map; issues=loaded.issues;
        }
        let cloud=ca_core::read(cloud,&std::fs::read(cloud).map_err(|e|format!("{cloud}: {e}"))?).map_err(|e|e.to_string())?;
        let discovery=ca_core::vector_map::discovery::propose(&map,&cloud,&options).map_err(|e|e.to_string())?;
        let additions=confirmed.as_ref().map(|c|ca_core::vector_map::discovery::add(&mut map,&cloud,&options,c)).transpose().map_err(|e|e.to_string())?;
        artifacts(&map,json!(issues),json!({"status":if confirmed.is_some(){"draft"}else{"preview"},"discovery":discovery,"additions":additions}))
    }).map_err(PyValueError::new_err)
}

/// Check source coverage without modifying or exporting the input map.
#[pyfunction]
pub fn audit_vector_map_quality(py: Python<'_>, cloud: &str, vector_map: &str) -> PyResult<String> {
    py.detach(|| {
        let text = std::fs::read_to_string(vector_map).map_err(|e| e.to_string())?;
        let loaded = if vector_map.to_ascii_lowercase().ends_with(".json") {
            vectormap_io::json::from_str(&text)
        } else {
            lanelet2::read_str(&text, &Default::default())
        }
        .map_err(|e| e.to_string())?;
        if loaded
            .issues
            .iter()
            .any(|i| i.severity == vectormap_core::Severity::Error)
        {
            return Err("resolve map import errors before auditing source coverage".to_string());
        }
        let cloud = ca_core::read(cloud, &std::fs::read(cloud).map_err(|e| e.to_string())?)
            .map_err(|e| e.to_string())?;
        let quality =
            ca_core::vector_map::quality::audit(&loaded.map, &cloud).map_err(|e| e.to_string())?;
        serde_json::to_string(&json!({"quality":quality,"import_issues":loaded.issues,
            "validation":vectormap_validation::validate(&loaded.map,&Default::default())}))
        .map_err(|e| e.to_string())
    })
    .map_err(PyValueError::new_err)
}

fn artifacts(map: &Map, import_issues: Value, mut report: Value) -> Result<String, String> {
    let (osm, export_issues) = lanelet2::write_string(map, &lanelet2::SaveOptions::autoware());
    if let Some(issue) = export_issues
        .iter()
        .find(|issue| issue.severity == vectormap_core::Severity::Error)
    {
        return Err(format!(
            "cannot export this coordinate frame: {}",
            issue.message
        ));
    }
    let mut issues = autoware::check(map);
    issues.extend(export_issues);
    let validation = vectormap_validation::validate(map, &Default::default());
    report["autoware_issues"] = json!(issues);
    report["import_issues"] = import_issues;
    report["validation"] = json!(validation);
    report["georeference"] = json!(map.metadata().georeference);
    Ok(json!({
        "osm":osm, "projector_info":autoware::projector_info_yaml(map.metadata().georeference),
        "map_json":vectormap_io::json::to_string(map),
        "report":report
    })
    .to_string())
}

/// Inspect or explicitly edit equipment associations without point-cloud copies.
#[pyfunction]
#[pyo3(signature=(vector_map, options=None))]
pub fn edit_vector_map_relations(
    py: Python<'_>,
    vector_map: &str,
    options: Option<&str>,
) -> PyResult<String> {
    py.detach(|| -> Result<String, String> {
        let text = std::fs::read_to_string(vector_map).map_err(|e| e.to_string())?;
        let loaded = if vector_map.to_ascii_lowercase().ends_with(".json") {
            vectormap_io::json::from_str(&text).map_err(|e| e.to_string())?
        } else {
            lanelet2::read_str(&text, &Default::default()).map_err(|e| e.to_string())?
        };
        if let Some(issue) = loaded
            .issues
            .iter()
            .find(|i| i.severity == vectormap_core::Severity::Error)
        {
            return Err(format!("cannot retain vector_map: {}", issue.message));
        }
        let mut map = loaded.map;
        let edits = options
            .map(serde_json::from_str::<ca_core::vector_map::relations::LinkEdit>)
            .transpose()
            .map_err(|e| e.to_string())?;
        let edit = edits
            .as_ref()
            .map(|o| ca_core::vector_map::relations::edit(&mut map, o))
            .transpose()
            .map_err(|e| e.to_string())?;
        artifacts(
            &map,
            json!(loaded.issues),
            json!({"edit":edit,"relationships":ca_core::vector_map::relations::inspect(&map)}),
        )
    })
    .map_err(PyValueError::new_err)
}

/// Preview geometric targets, or explicitly adopt one current-map candidate.
#[pyfunction]
#[pyo3(signature=(vector_map, rule_id, adoption=None))]
pub fn propose_vector_map_relations(
    py: Python<'_>,
    vector_map: &str,
    rule_id: u64,
    adoption: Option<&str>,
) -> PyResult<String> {
    py.detach(|| -> Result<String, String> {
        let text = std::fs::read_to_string(vector_map).map_err(|e| e.to_string())?;
        let loaded = if vector_map.to_ascii_lowercase().ends_with(".json") {
            vectormap_io::json::from_str(&text).map_err(|e| e.to_string())?
        } else { lanelet2::read_str(&text, &Default::default()).map_err(|e| e.to_string())? };
        if let Some(issue) = loaded.issues.iter().find(|i| i.severity == vectormap_core::Severity::Error) {
            return Err(format!("cannot retain vector_map: {}", issue.message));
        }
        let mut map = loaded.map;
        let proposal = ca_core::vector_map::relation_proposals::propose(&map, rule_id).map_err(|e|e.to_string())?;
        let edit = if let Some(text) = adoption {
            let o: ca_core::vector_map::relation_proposals::Adoption = serde_json::from_str(text).map_err(|e|e.to_string())?;
            if o.rule_id != rule_id { return Err("adoption rule must match the requested rule".into()); }
            Some(ca_core::vector_map::relation_proposals::adopt(&mut map, &o).map_err(|e|e.to_string())?)
        } else { None };
        artifacts(&map, json!(loaded.issues), json!({"proposal":proposal,"edit":edit,"relationships":ca_core::vector_map::relations::inspect(&map)}))
    }).map_err(PyValueError::new_err)
}
