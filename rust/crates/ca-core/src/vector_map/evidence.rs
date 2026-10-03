//! Read-only build snapshots; never stored in the map or its exports.
use super::{Evidence, ExtractedRoad};
use serde::Serialize;
use std::collections::BTreeSet;
use vectormap_core::{BuiltRoad, LaneDirection, Map, RoadLane};

const MAX_VERTICES: usize = 100_000;
const MAX_PROFILES: usize = 10_000;

#[derive(Debug, Clone, Serialize)]
pub struct BoundaryEvidenceProfile {
    pub boundary_ids: Vec<u64>,
    pub geometry: Vec<[f64; 3]>,
    pub evidence: Vec<Evidence>,
    pub source_before_fitting: Vec<[f64; 3]>,
}

#[derive(Debug, Clone, Default, Serialize)]
pub struct BuildEvidence {
    pub profiles: Vec<BoundaryEvidenceProfile>,
    pub limited: bool,
    #[serde(skip)]
    vertices: usize,
}

impl BuildEvidence {
    pub(super) fn can_accept(&mut self, road: &ExtractedRoad) -> bool {
        let vertices: usize = road.boundaries.iter().map(Vec::len).sum();
        let fits = self.vertices + vertices <= MAX_VERTICES
            && self.profiles.len() + road.boundaries.len() <= MAX_PROFILES;
        self.limited |= !fits;
        fits
    }
    pub(super) fn capture(
        &mut self,
        map: &Map,
        built: &BuiltRoad,
        lanes: &[RoadLane],
        road: ExtractedRoad,
    ) {
        if built.lanes.iter().map(Vec::len).sum::<usize>() > MAX_PROFILES {
            self.limited = true;
            return;
        }
        for (j, geometry) in road.boundaries.into_iter().enumerate() {
            let column = j.min(lanes.len() - 1);
            let along_left = j < lanes.len();
            let forward = lanes[column].direction == LaneDirection::Forward;
            let ids: BTreeSet<_> = built.lanes[column]
                .iter()
                .filter_map(|id| map.lane(*id))
                .map(|lane| {
                    if along_left == forward {
                        lane.left.boundary.0
                    } else {
                        lane.right.boundary.0
                    }
                })
                .collect();
            self.vertices += geometry.len();
            self.profiles.push(BoundaryEvidenceProfile {
                boundary_ids: ids.into_iter().collect(),
                source_before_fitting: road
                    .source_boundaries
                    .as_ref()
                    .map_or_else(|| geometry.clone(), |s| s[j].clone()),
                geometry,
                evidence: road.evidence[j].clone(),
            });
        }
    }
}
