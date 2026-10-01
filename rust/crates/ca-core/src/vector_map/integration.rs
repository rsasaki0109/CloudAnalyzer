//! Conservative interval union with existing lanes. Geometry and traffic rules
//! in the existing map are authoritative; repeat observations never overwrite them.

use std::collections::{HashMap, HashSet};
use vectormap_core::{
    LaneDirection, LaneId, LaneKind, Map, Point2, Point3, Polyline3, RoadLane, Side,
};

use super::{BuildError, BuildReport, ExtractedRoad};

const CELL: f64 = 2.0;
const HEIGHT: f64 = 0.3;
const HEADING_COS: f64 = 0.9659258262890683; // 15 degrees

struct Lane {
    id: LaneId,
    center: Polyline3,
    left: Polyline3,
    right: Polyline3,
}

struct Index {
    lanes: Vec<Lane>,
    cells: HashMap<(i64, i64), Vec<usize>>,
    links: HashSet<(LaneId, LaneId)>,
}

fn cell(p: Point2) -> (i64, i64) {
    ((p.x / CELL).floor() as i64, (p.y / CELL).floor() as i64)
}

// Fixed resolution avoids changing the comparison when section cuts insert
// boundary vertices. Explicit existing centerlines remain authoritative.
fn comparison_center(left: &Polyline3, right: &Polyline3) -> Polyline3 {
    let n = (left.length().max(right.length()) / 0.5).ceil() as usize + 1;
    let l = left.resample_count(n);
    let r = right.resample_count(n);
    Polyline3::new(
        l.points
            .iter()
            .zip(&r.points)
            .map(|(a, b)| a.lerp(*b, 0.5))
            .collect(),
    )
}

impl Index {
    fn new(map: &Map, join_sections: bool) -> Self {
        let mut lanes = Vec::new();
        let mut cells: HashMap<_, Vec<_>> = HashMap::new();
        for lane in map
            .lanes()
            .filter(|l| l.kind == LaneKind::Driving && l.turn_direction.is_none())
        {
            let (Some(center), Some(left), Some(right)) = (
                map.centerline(lane.id),
                map.oriented_boundary(lane.id, Side::Left),
                map.oriented_boundary(lane.id, Side::Right),
            ) else {
                continue;
            };
            if center.points.len() < 2 || left.points.len() < 2 || right.points.len() < 2 {
                continue;
            }
            lanes.push(Lane {
                id: lane.id,
                center,
                left,
                right,
            });
        }
        if join_sections {
            // A cross-section need not cut both edges at the same station.
            // Compare their complete, uniquely connected chains so a point
            // beside a cut is not rejected just because it is on the next edge
            // piece. This is a query view; the original map is never changed.
            let mut joined = Vec::new();
            let mut remaining: HashMap<_, _> = lanes.into_iter().map(|l| (l.id, l)).collect();
            let mut starts: Vec<_> = remaining.keys().copied().collect();
            starts.sort_unstable_by_key(|&id| {
                let p = map.predecessors(id);
                (p.len() == 1 && map.successors(p[0]).len() == 1, id)
            });
            for id in starts {
                let Some(mut chain) = remaining.remove(&id) else {
                    continue;
                };
                let mut explicit_center = map.lane(id).unwrap().centerline.is_some();
                let mut tail = id;
                while let [next] = map.successors(tail) {
                    if map.predecessors(*next) != [tail] {
                        break;
                    }
                    let Some(piece) = remaining.get(next) else {
                        break;
                    };
                    if chain
                        .left
                        .points
                        .last()
                        .unwrap()
                        .distance(piece.left.points[0])
                        > 1e-4
                        || chain
                            .right
                            .points
                            .last()
                            .unwrap()
                            .distance(piece.right.points[0])
                            > 1e-4
                    {
                        break;
                    }
                    let piece = remaining.remove(next).unwrap();
                    explicit_center |= map.lane(*next).unwrap().centerline.is_some();
                    chain.center = chain.center.concat(&piece.center);
                    chain.left = chain.left.concat(&piece.left);
                    chain.right = chain.right.concat(&piece.right);
                    tail = *next;
                }
                if !explicit_center {
                    chain.center = comparison_center(&chain.left, &chain.right);
                }
                joined.push(chain);
            }
            lanes = joined;
        }
        for (index, lane) in lanes.iter().enumerate() {
            // Sample each segment, including its endpoints. The queried 3x3
            // cells cover the allowed <=1 m discrepancy without a global scan.
            let mut seen = HashSet::new();
            for pair in lane.center.points.windows(2) {
                let n = pair[0].xy().distance(pair[1].xy()).ceil().max(1.0) as usize;
                for k in 0..=n {
                    let key = cell(pair[0].lerp(pair[1], k as f64 / n as f64).xy());
                    if seen.insert(key) {
                        cells.entry(key).or_default().push(index);
                    }
                }
            }
        }
        let links = map
            .lanes()
            .flat_map(|l| map.successors(l.id).iter().map(move |&to| (l.id, to)))
            .collect();
        Self {
            lanes,
            cells,
            links,
        }
    }

    fn matches(
        &self,
        boundaries: &[[f64; 3]],
        centers: &[Polyline3],
        directions: &[[f64; 2]],
        lanes: &[RoadLane],
        tolerance: f64,
    ) -> Option<Vec<LaneId>> {
        let mut matched = Vec::new();
        for (j, spec) in lanes.iter().enumerate() {
            let (left, right, sign) = match spec.direction {
                LaneDirection::Forward => (boundaries[j], boundaries[j + 1], 1.0),
                LaneDirection::Backward => (boundaries[j + 1], boundaries[j], -1.0),
            };
            let point = Point3::new(
                (left[0] + right[0]) / 2.0,
                (left[1] + right[1]) / 2.0,
                (left[2] + right[2]) / 2.0,
            );
            let point = centers[j].nearest_point(point.xy())?.point;
            let (x, y) = cell(point.xy());
            let mut candidates = Vec::new();
            for a in x - 1..=x + 1 {
                for b in y - 1..=y + 1 {
                    if let Some(ids) = self.cells.get(&(a, b)) {
                        candidates.extend(ids);
                    }
                }
            }
            candidates.sort_unstable();
            candidates.dedup();
            let mut valid: Vec<_> = candidates
                .into_iter()
                .filter_map(|&i| {
                    let lane = &self.lanes[i];
                    if matched.contains(&lane.id) {
                        return None;
                    }
                    let near = lane.center.nearest_point(point.xy())?;
                    if near.distance > tolerance || (near.point.z - point.z).abs() > HEIGHT {
                        return None;
                    }
                    let (ld, rd) = if sign > 0.0 {
                        (directions[j], directions[j + 1])
                    } else {
                        (directions[j + 1], directions[j])
                    };
                    for (line, p, dir) in [(&lane.left, left, ld), (&lane.right, right, rd)] {
                        let near = line.nearest_point(Point2::new(p[0], p[1]))?;
                        if near.distance > tolerance || (near.point.z - p[2]).abs() > HEIGHT {
                            return None;
                        }
                        // Compare corresponding edges, not the trajectory's
                        // heading against an arc-length-derived lane centre.
                        // At a vertex either incident edge can be the nearest.
                        let aligned = (near.segment.saturating_sub(1)
                            ..=(near.segment + 1).min(line.points.len() - 2))
                            .any(|k| {
                                let a = line.points[k];
                                let b = line.points[k + 1];
                                let segment = Polyline3::new(vec![a, b]);
                                let n = segment.nearest_point(Point2::new(p[0], p[1])).unwrap();
                                let norm = (b.x - a.x).hypot(b.y - a.y) * dir[0].hypot(dir[1]);
                                n.distance <= tolerance
                                    && (n.point.z - p[2]).abs() <= HEIGHT
                                    && norm > 1e-8
                                    && sign * ((b.x - a.x) * dir[0] + (b.y - a.y) * dir[1]) / norm
                                        >= HEADING_COS
                            });
                        if !aligned {
                            return None;
                        }
                    }
                    Some((near.distance, lane.id))
                })
                .collect();
            valid.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
            let best = *valid.first()?;
            // At an existing section cut, both pieces legitimately match. Any
            // other competing geometry must be in the same connected chain.
            let mut seen = HashSet::from([best.1]);
            loop {
                let before = seen.len();
                for &(_, id) in &valid {
                    if seen.iter().any(|&other| {
                        self.links.contains(&(id, other)) || self.links.contains(&(other, id))
                    }) {
                        seen.insert(id);
                    }
                }
                if seen.len() == before {
                    break;
                }
            }
            if seen.len() != valid.len() {
                return None;
            }
            matched.push(best.1);
        }
        Some(matched)
    }
}

/// Compare both endpoints and the midpoint of every incoming interval. Lanes
/// may span existing section cuts, provided the corresponding topology links.
pub(super) fn uncovered(
    map: &Map,
    road: ExtractedRoad,
    lanes: &[RoadLane],
    tolerance: f64,
    report: &mut BuildReport,
) -> Vec<ExtractedRoad> {
    if map.lanes().next().is_none() {
        return vec![road];
    }
    let index = Index::new(map, true);
    let lines: Vec<_> = road
        .boundaries
        .iter()
        .map(|line| Polyline3::new(line.iter().map(|p| Point3::new(p[0], p[1], p[2])).collect()))
        .collect();
    let centers: Vec<_> = lines
        .windows(2)
        .map(|pair| comparison_center(&pair[0], &pair[1]))
        .collect();
    let mut covered = Vec::new();
    for k in 0..road.reference.len() - 1 {
        let a = road.reference[k];
        let b = road.reference[k + 1];
        let distance = (b[0] - a[0]).hypot(b[1] - a[1]);
        let directions: Vec<_> = road
            .boundaries
            .iter()
            .map(|line| [line[k + 1][0] - line[k][0], line[k + 1][1] - line[k][1]])
            .collect();
        let section = |t: f64| -> Vec<[f64; 3]> {
            road.boundaries
                .iter()
                .map(|line| std::array::from_fn(|j| line[k][j] * (1.0 - t) + line[k + 1][j] * t))
                .collect()
        };
        let found = [0.0, 0.5, 1.0]
            .map(|t| index.matches(&section(t), &centers, &directions, lanes, tolerance));
        let reused = if let [Some(start), Some(mid), Some(end)] = found {
            start
                .iter()
                .zip(&mid)
                .zip(&end)
                .zip(lanes)
                .all(|(((a, m), b), lane)| {
                    let connected = |x, y| {
                        x == y
                            || match lane.direction {
                                LaneDirection::Forward => map.successors(x).contains(&y),
                                LaneDirection::Backward => map.successors(y).contains(&x),
                            }
                    };
                    connected(*a, *m) && connected(*m, *b)
                })
        } else {
            false
        };
        covered.push(reused);
        if reused {
            report.reused_intervals += 1;
            report.reused_length += distance;
        }
    }
    let slice = |start: usize, end: usize| ExtractedRoad {
        reference: road.reference[start..=end].to_vec(),
        boundaries: road
            .boundaries
            .iter()
            .map(|line| line[start..=end].to_vec())
            .collect(),
        evidence: road
            .evidence
            .iter()
            .map(|line| line[start..=end].to_vec())
            .collect(),
        source_boundaries: road.source_boundaries.as_ref().map(|lines| {
            lines
                .iter()
                .map(|line| line[start..=end].to_vec())
                .collect()
        }),
    };
    let mut parts = Vec::new();
    let mut start = None;
    for (k, &reused) in covered.iter().enumerate() {
        if !reused && start.is_none() {
            start = Some(k);
        }
        if reused && let Some(a) = start.take() {
            parts.push(slice(a, k));
        }
    }
    if let Some(a) = start {
        parts.push(slice(a, road.reference.len() - 1));
    }
    parts
}

/// Connect coincident directed ends after an extension. Crossings, nearby
/// parallel lanes, and nonzero gaps do not create links here.
pub(super) fn link_touching(map: &mut Map, created: &[LaneId]) -> Result<usize, BuildError> {
    if created.is_empty() {
        return Ok(0);
    }
    let new: HashSet<_> = created.iter().copied().collect();
    let index = Index::new(map, false);
    let mut links = Vec::new();
    for from in &index.lanes {
        if !map.successors(from.id).is_empty() {
            continue;
        }
        let (x, y) = cell(from.center.points.last().unwrap().xy());
        let mut candidates: Vec<usize> = Vec::new();
        for a in x - 1..=x + 1 {
            for b in y - 1..=y + 1 {
                if let Some(ids) = index.cells.get(&(a, b)) {
                    candidates.extend(ids);
                }
            }
        }
        candidates.sort_unstable();
        candidates.dedup();
        for i in candidates {
            let to = &index.lanes[i];
            if from.id == to.id
                || (!new.contains(&from.id) && !new.contains(&to.id))
                || !map.predecessors(to.id).is_empty()
            {
                continue;
            }
            let aligned = [&from.left, &from.right]
                .into_iter()
                .zip([&to.left, &to.right])
                .all(|(a, b)| a.points.last().unwrap().distance(b.points[0]) < 1e-4);
            let a = from.center.points[from.center.points.len() - 2];
            let b = *from.center.points.last().unwrap();
            let c = to.center.points[0];
            let d = to.center.points[1];
            let norm = (b.x - a.x).hypot(b.y - a.y) * (d.x - c.x).hypot(d.y - c.y);
            if aligned
                && norm > 1e-8
                && ((b.x - a.x) * (d.x - c.x) + (b.y - a.y) * (d.y - c.y)) / norm > HEADING_COS
            {
                links.push((from.id, to.id));
            }
        }
    }
    let mut outgoing: HashMap<LaneId, usize> = HashMap::new();
    let mut incoming: HashMap<LaneId, usize> = HashMap::new();
    for &(a, b) in &links {
        *outgoing.entry(a).or_default() += 1;
        *incoming.entry(b).or_default() += 1;
    }
    links.retain(|&(a, b)| outgoing[&a] == 1 && incoming[&b] == 1);
    for &(from, to) in &links {
        map.connect(from, to)
            .map_err(|e| BuildError(e.to_string()))?;
    }
    Ok(links.len())
}
