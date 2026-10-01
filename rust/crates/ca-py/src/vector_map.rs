//! File-based draft generation; returns artifacts without writing destinations.
use pyo3::{exceptions::PyValueError, prelude::*};
use serde_json::json;
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
    let (osm, export_issues) = lanelet2::write_string(&map, &lanelet2::SaveOptions::autoware());
    if let Some(issue) = export_issues
        .iter()
        .find(|issue| issue.severity == vectormap_core::Severity::Error)
    {
        return Err(format!(
            "cannot export this coordinate frame: {}",
            issue.message
        ));
    }
    let mut issues = autoware::check(&map);
    issues.extend(export_issues);
    let validation = vectormap_validation::validate(&map, &Default::default());
    Ok(json!({
        "osm":osm, "projector_info":autoware::projector_info_yaml(map.metadata().georeference),
        "map_json":vectormap_io::json::to_string(&map),
        "report":{"status":"draft","extraction":extraction,"autoware_issues":issues,"import_issues":import_issues,
            "validation":validation,"georeference":map.metadata().georeference}
    })
    .to_string())
}
