"""CloudAnalyzer as MCP tools, for AI agents (``ca mcp``; ``pip install "cloudanalyzer[mcp]"``).

The tools fix and compare SLAM maps with the Rust core (as the web app does,
natively and on all cores) and evaluate clouds and trajectories. Each returns
the JSON report its ``ca`` command prints; paths are local to the machine the
server runs on. Register it with an MCP client, e.g. Claude Code:

    claude mcp add cloudanalyzer -- ca mcp
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from ca.vector_map import build_vector_map, connect_vector_map_junctions, measure_vector_map_signal, measure_vector_map_crosswalk, discover_vector_map_features, edit_vector_map_relations, propose_vector_map_relations
from ca.mapping_job import start_mapping_job, inspect_mapping_job, propose_mapping_corridors, inspect_mapping_corridors, generate_mapping_geometry, inspect_mapping_geometry, generate_mapping_corridor_lanes, diagnose_mapping_candidate, generate_mapping_candidate, select_mapping_candidate
from ca.mapping_run import start_mapping_run, inspect_mapping_run, advance_mapping_run
from ca.mapping_revision import continue_mapping_run
from ca.mapping_plan_apply import apply_supported_hd_plan
from ca.mapping_bundle import export_mapping_run, export_mapping_preview, inspect_mapping_bundle

INSTRUCTIONS = """\
CloudAnalyzer fixes and measures LiDAR point clouds and SLAM maps on this machine.

After finishing a mapping run, export_mapping_run copies the exact delivered pair
and final audits to a new portable review ZIP. Supply source-data attribution and
a byte limit. inspect_mapping_bundle verifies every member without the original
logs, run directories or native core. Read review.diagnosis and the full audits;
integrity checks do not prove accuracy. The ZIP cannot resume a mapping job.
For a large point map, export_mapping_preview retains a bounded display-only
subset of exact source records and the complete HD map. Saved audits still refer
to the original full point map, which stays outside that ZIP. Do not use the
preview as the source of new audits or generation. A map fitting the point cap
uses the exact full-map package instead; inspect the returned schema/provenance.

To repair another region of a finished map, use continue_mapping_run with a new
output directory, explicit budget and reason. Candidate 1 retains the exact delivered
pair without spending an attempt. inspect_hd_plan checks observed missing reference
intervals against both ground estimators, width requirements and retained HD occupancy
before lane generation. It performs no lane export or automatic adoption.
After choosing 1..8 fully supported nonambiguous intervals on retained plan pages,
apply_supported_hd_plan executes their HD-only draft, combined retention-checked
patch and baseline comparison with an explicit endpoint-link policy. Completed
stages are reused on retry; failed attempts are not silently repeated. Read its
comparison and holds, then finish_retry explicitly or retain the baseline. It
never adopts the parent automatically or establishes independent accuracy.
repair_hd starts a gap-only HD child with the
exact retained point map and source proposal, sharing the repair budget and requiring
a combined patch before adoption. Local-density child patches preserve earlier
lanes/connections and outside point records. inspect_protected_density previews a box while retaining every old lane hull plus its saved source-audit radius at all
heights; retry_local_density uses that exact preview. It can exclude useful new
evidence and keeps all four audit gates. Keep referenced prior runs accessible.

For the complete agent-driven mapping loop, prefer start_mapping_run(source,
out_dir, layout_hypothesis). Supply the user's explicit unverified lane layout
once, without predefined corridor IDs. It generates the point map and proposals,
then returns decision guidance. Continue autonomously using inspect_mapping_run
and advance_mapping_run: inspect candidate evidence, draft complete include/defer
choices, read automatic lane export and both diagnoses, retry within the fixed
layout/budget, and finish with both artifact sets and unresolved holds. Use the
refine action with association=trajectory_containing when off-path overlapping
bands fragment the route. It re-extracts once at unchanged source thresholds;
original observations/drafts remain saved, and new proposal IDs need fresh inspection.
After a lane draft, use inspect_connections then explicit connect pairs to test short
source-supported gaps between consecutive drive pieces. The fixed single-forward-lane
layout, recorded path, both ground estimators and reopened OSM topology are checked.
Read local route spans and global longest route separately; connections do not increase
original corridor extent or prove permitted turns. Failed links retain the prior draft.
For a combined local HD patch, inspect_connection_region explicitly supplies an
HD-only bounds_xy envelope containing the unchanged point-update box, with sides
at most 20 m. Only links touching new repair lanes are offered; all geometry must
stay inside that HD envelope. Inspect exact geometry and connect seen pairs.
Point records, retained geometry/edges, width and all four source-audit protocols
stay fixed. No scope expansion occurs implicitly; failures preserve the prior pair.
Connections can extend audited connected or gap-patched drafts: inherit all existing
lanes and directed edges, select only new inspected pairs, and reject any new failures
on retained source samples in either estimator or saved format.
Use inspect_gaps to read missing source intervals and aligned raw-return neighborhoods.
For a thinning hypothesis, retry_pointcloud explicitly reduces scan/map voxel sizes
once while freezing corrected motion, retained frames, filter policy and thresholds.
Remaining HD attempts transfer to a child run; inspect its fresh proposal IDs and
draft there. compare_retry reports actual gained AND lost source intervals, all four
audits and global route spans. Finish the child, then explicitly finish_retry after
comparison, or finish the root baseline if the trial is worse. No automatic adoption.
Alternatively use inspect_unused_frames after gap inspection to find unused raw
frames observing missing intervals, with interpolated corrected-pose hypotheses
and consistency checks against both neighboring retained scans. retry_frames
explicitly fuses inspected eligible IDs, keeping thinning and existing corrected
poses fixed. ICP's alternative correction is not applied and does not certify
accuracy. Reference and expanded fusion graphs/trajectories are both retained.
For a local point update, inspect_local_points with seen gap IDs and explicit
bounds_xy, then retry_local_frames with the returned preview_file and eligible
frame IDs. Alternatively inspect_local_density with seen gap IDs and bounds_xy,
then retry_local_density with that exact preview_file and bounded thinning options;
this keeps the original retained frames and needs no unused-frame inspection.
Both generate the full fusion candidate but apply only that XY column
at all heights, preserving outside PLY records and attributes exactly. Existing HD
source support must not regress. Use a combined HD patch inside the same box before
compare_retry and finish_retry; finish the original root map to reject the trial.
This shares the single root point-map retry and transferred HD budget; compare
actual audited maps before explicitly delivering a trial or baseline pair.
For a local HD repair, draft only inspected missing ranges in the retry child.
When an isolated addition has a boundary height mismatch, inspect_heights pages
offer adjacent interior vertices inside the frozen local point-update box.
Use edit_heights with that exact preview_file and seen vertices, each with an
explicit nonzero delta_z_m at most 0.1 m and reason. Boundary XY/endpoints, all
unchosen Z, IDs, semantics, routes and point-map files remain fixed. This spends
one shared attempt and reserves one for the combined patch; one height trial is
allowed per child. All addition traces need full support in four unchanged
audits, unchanged sample counts and an exact OSM roundtrip. Original source
curves remain observations; height edits are hypotheses, not accuracy claims.
Failures retain the trial and audits without publishing a replacement draft.
Use inspect_patch with gap IDs inspected through the root, examine exact geometric
endpoint pairs, then patch_gaps with explicit pair decisions ([] when isolated).
This keeps original lane geometry, IDs, metadata and connections, adds only inside
selected gaps, and audits all retained/new traces against the actual child point map.
New lanes need full support from both estimators in IR and reopened OSM; no new
failure locations are allowed on retained lanes. A patch spends one transferred
HD attempt. Point replacement is local only for explicit local retry strategies;
HD source support does not establish legal connectivity.
Compare and finish explicitly; held patches retain their full checks and baseline.
Use the returned revision for every action. Interrupted processing actions can resume without
replaying completed stages. Do not stop after startup or a single failed trial;
finish with a useful retained draft or explain why no HD draft can be generated.
No LLM is embedded: you are the reasoning agent. Only initial assumptions need
operator input; per-piece lane JSON is bound automatically from the fixed layout.

For an agent-controlled raw-recording-to-both-maps job, use start_mapping_job with
a NEW output directory. It makes the point-cloud map and corrected trajectory and
records hashes and reports. Read inspect_mapping_job, then propose_mapping_corridors
to find low-surface bands and geometric corridor candidates before assigning lanes.
Read inspect_mapping_corridors (paged index or candidate geometry) for widths and
edge evidence. Paired curb profiles can suggest a width; coverage gaps and search
limits cannot. Branches, absent ground anchors and different levels remain unresolved.
The proposal stage is cached and does not spend HD attempts. It does not infer
road use, lane count/direction, speed or automatically adopt geometry. After reading
the proposals, use generate_mapping_geometry with explicit include/defer decisions
and reasons to retain selected source curves in editable IR without lane assumptions.
Optional ranges use observed stations; included bands cannot overlap input stations.
Read inspect_mapping_geometry for curves, decisions and full-extent holds. Geometry
drafts use the shared HD attempt budget, preserve existing selections, infer no lanes
and are not selectable as audited lane maps. They publish IR and an evidence report;
no OSM is published because standalone unknown ways are dropped on Lanelet2 reload.
To test a lane layout inside those saved curves, use generate_mapping_corridor_lanes
with explicit source_span_hypothesis boundary policy and per-centre-curve lane specs.
Specify lane fractions, direction, kind=driving, one_way, speed and minimum width;
fractions partition the observed span as an unverified layout hypothesis. Outer
curves stay fixed; insufficient widths retain failed attempts without silently
widening or changing lane count. This uses one shared HD attempt and saves Lanelet2,
IR, projector and four source audits after verifying OSM lane/geometry reload.
Read diagnose_mapping_candidate; full-drive extent and semantic/source holds remain.
For an explicitly assumed lane-map hypothesis, call generate_mapping_candidate
with explicit road_options and a reason for each hypothesis. Road options require
forward_lanes, backward_lanes, left_hand_traffic, lane_width and speed_limit; other
build_vector_map fitting options are available. Read native failures, source audits,
export warnings and retained extent between trials. Use diagnose_mapping_candidate
to separate saved per-lane/trace height mismatches, insufficient returns and endpoint
holds without rerunning generation or spending attempts. These failures are evidence,
not proven root causes; inspect XY and level alignment before changing only Z.
New jobs also retain bounded problem locations and local low-return heights in
the original frame. Check problems_available/problems_limited before treating the
preview as complete; summary counts have an independent budget. Local heights may
belong to another level. New jobs retain both the legacy low-quantile audit and
ground_consensus evidence from the lowest spatially supported source layer. Read
both protocols: source_quality_passed requires both, and disagreement needs review.
local_ground_height experimentally uses that layer for seed heights; test it with
unchanged priors and extent, and do not interpret source support as accuracy.
Lane counts, permitted traffic,
width and speed are assumptions, not established by point-cloud support. Do not
reduce lane count or retained extent merely to raise a coverage score. Attempts
are bounded; failed trials preserve earlier artifacts. Select an audited candidate
with select_mapping_candidate and an explanation; low source support and unresolved
semantics remain explicit. Selection never certifies deployment readiness. No LLM
is embedded in these tools: the calling agent makes and records the decisions.

A SLAM session folder holds a poses file (g2o, or a KITTI / TUM trajectory) and one scan
per pose (PCD, PLY, LAS/LAZ, XYZ, KITTI .bin) named by frame number. Look at a folder with
session_layout first (it is quick). Raw scans without poses: run slam_odometry first, then
posegraph_fix with its trajectory. Then run posegraph_fix (loops, IMU gravity, dynamic
points, the fixed map) or posegraph_compare (two drives through the same places: what
changed). Those read every scan and take seconds to minutes on large drives. Outputs go to
out_dir; view_link(out_dir) gives a link that opens them in the CloudAnalyzer web app for the
person to look at.

build_vector_map drafts Autoware roads from a surveyed cloud and a recorded trajectory
already in the same metre frame. The trajectory must follow the outside forward lane.
Use a new out_dir; the tool will not replace existing results. Choose lane counts, widths,
traffic side and speed. A reference_map supplies only coordinate metadata; otherwise
specify a projection and origin, or use Local. Detected boundaries are candidates; read
the evidence counts, warnings and validation issues and review the draft geometry.
To append another pass, use existing_map (vector_map.json or Lanelet2 OSM), without
reference_map or projection options. It preserves existing geometry, IDs and rules,
reuses matching intervals and reports added/reused lengths. Both passes must already
share a coordinate frame; this does not align survey drift. Ambiguous overlaps need review.
connect_vector_map_junctions proposes ground-supported connections between open road ends,
including branches. Use preview_only=true and a new out_dir to inspect candidate geometry;
then call it with the original vector_map, a different new out_dir and selected lane_pairs.
Omitting lane_pairs adds all proposals; [] adds none. Prefer editable IR to retain imported
rules and metadata. Ground support cannot establish legal turns, lane clearance or signal
rules. Existing rules remain, but new connections require traffic-rule review.
measure_vector_map_signal fits housing geometry from a user-identified head's 3D box.
discover_vector_map_features searches generated road corridors, or all supported lower
surfaces with scope=ground_surface and no map, without manually placing feature boxes.
Preview into a NEW out_dir. Inspect unconfirmed paint/panel proposals over source points;
then submit explicit candidate/key/classification/lanes confirmations against the
original map and source. Nearby lanes are suggestions only. This loads the whole local
attribute-bearing scene; export a smaller scene for large sources. No automatic object
identity, lamp state, stop sign or signal/stop-line link is inferred. Limited previews
are reported explicitly (64 highest-support proposals per evidence type).
measure_vector_map_crosswalk previews measured bright ground bands and adds a
user-confirmed crossing with explicitly selected lane IDs. Retained RGB or
intensity is required; paint patterns alone do not classify an object.
Supply original-coordinate bounds and explicitly confirmed controlled lane IDs. Preview
first, then add with preview_only=false in another new out_dir. This measures shape;
it does not classify an unlabelled object or infer lamps, stop lines or controlled lanes.
Local LAS/LAZ/CSV stream a filtered box; local/HTTP(S) COPC reads all overlapping levels.
At most 200000 selected XYZ points are retained. Other signal formats still read whole files.
After generation, edit with the separate vectormap MCP server: register it with
`claude mcp add vectormap -- vectormap mcp <out_dir>/lanelet2_map.osm`.
Use view_link([cloud, out_dir]) to show the point cloud and Lanelet2 map together.

tile_copc reads full-density local/HTTP COPC into bounded XY tile packs and a SQLite
journal, preserving raw LAS fields and source identities. Supply source-coordinate grid,
origin, optional box and halo. Use a new out_dir; resume=true requires the same source/options.
stop_after_nodes bounds a call and commits a resumable prefix. Core counts exclude halo copies.
export_copc_tile streams one committed tile to new LAS/LAZ; include_halo adds explicit flags
and source identities. Packs are internal artifacts, not files the viewer can directly open.
Physical ten-billion-point processing is not benchmarked; finite halo cannot establish exact
global neighbors/ICP. Read the returned small summary and manifest rather than load all packs.
"""


def session_layout(folder: str) -> dict[str, Any]:
    """Look at a SLAM session folder without loading it: the poses file, how many poses and
    scans, whether scans match poses by frame number, a KITTI calib.txt, and folders or files
    that look like IMU gravity (OXTS or 'frame ux uy uz')."""
    from ca.posegraph_fix import SCAN_SUFFIXES, match_scans, poses_file, read_trajectory

    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(folder)
    poses = poses_file(root)
    scans = sorted(p for p in root.iterdir() if p.suffix.lower() in SCAN_SUFFIXES)
    out: dict[str, Any] = {
        "folder": str(root),
        "poses_file": poses.name if poses else None,
        "scans": len(scans),
        "scan_formats": sorted({p.suffix.lower() for p in scans}),
        "kitti_calib": (root / "calib.txt").exists(),
    }
    if poses is not None and poses.suffix.lower() != ".g2o":
        trajectory, stamps = read_trajectory(poses)
        out["poses"] = int(len(trajectory))
        out["timestamps"] = stamps is not None
        steps = trajectory[1:, :3, 3] - trajectory[:-1, :3, 3]
        out["path_length_m"] = round(float((steps**2).sum(1).__pow__(0.5).sum()), 1)
        try:
            matched = match_scans(scans, list(range(len(trajectory))))
            out["scans_matched"] = sum(m is not None for m in matched)
        except ValueError as e:
            out["scans_matched"] = 0
            out["matching_problem"] = str(e)
    candidates = [root, root.parent]
    out["gravity_candidates"] = sorted(
        {
            str(p)
            for base in candidates
            for p in base.iterdir()
            if (p.is_dir() and (p.name.lower() in {"oxts", "gravity", "imu"}))
            or (p.is_file() and p.name.lower() in {"gravity.txt", "ups.txt"})
        }
    )
    return out


def slam_odometry(
    scans: str,
    out_dir: str,
    max_range: float = 80.0,
    voxel_size: float | None = None,
    max_frames: int | None = None,
    deskew: bool = False,
    pointcloud_topic: str | None = None,
    imu_topic: str | None = None,
    imu_to_lidar: list[float] | None = None,
) -> dict[str, Any]:
    """LiDAR odometry (the Rust core's; pip install "cloudanalyzer[fast]") for raw scans: a folder
    of KITTI .bin, PCD or PLY named in time order, or a ROS bag (.bag, .mcap, .db3 or a rosbag2
    folder; pip install "cloudanalyzer[ros]"). Writes trajectory.tum (one pose per scan) and
    map.ply to out_dir. A bag's PointCloud2 scans are written to out_dir/scans, and when it has
    a sensor_msgs/Imu topic, each scan's up direction to out_dir/gravity/gravity.txt (in the
    LiDAR frame: imu_to_lidar is the 3x3 rotation, row-major, identity when omitted). Then call
    posegraph_fix with folder=the scans ("scans" in the result), poses=the trajectory,
    gravity=the gravity file if any, and keyframe_spacing about 1 m for 10 Hz scans."""
    from ca.posegraph_fix import odometry

    return odometry(
        scans,
        out_dir,
        max_range=max_range,
        voxel_size=voxel_size,
        max_frames=max_frames,
        deskew=deskew,
        pointcloud_topic=pointcloud_topic,
        imu_topic=imu_topic,
        imu_to_lidar=imu_to_lidar,
    )


def posegraph_fix(
    folder: str,
    out_dir: str | None = None,
    poses: str | None = None,
    keyframe_spacing: float = 0.0,
    gravity: str | None = None,
    remove_dynamic: bool = False,
    truth: str | None = None,
    voxel: float = 0.4,
    map_voxel: float = 0.2,
    find_loops: bool = True,
    calibrate_imu: bool = True,
) -> dict[str, Any]:
    """Fix a SLAM map: find loops with ICP, tie keyframes to IMU gravity (a KITTI OXTS folder
    or a 'frame ux uy uz' file), optimise, optionally leave dynamic points (traffic) out, and
    write the fixed g2o, KITTI/TUM poses and the map (PLY) to out_dir. poses names the poses
    file when it is not in the folder (trajectory.tum from slam_odometry); keyframe_spacing
    keeps one pose every so many metres of it. With truth (ground-truth poses, one per frame)
    the report has the ATE before and after. The IMU's rotation into the LiDAR frame is estimated
    from the drive (calibrate_imu) and reported with how much its up directions disagree."""
    from ca.posegraph_fix import fix_session

    return fix_session(
        folder,
        out_dir,
        poses=poses,
        keyframe_spacing=keyframe_spacing,
        voxel=voxel,
        loops=find_loops,
        gravity=gravity,
        calibrate_gravity=calibrate_imu,
        remove_dynamic=remove_dynamic,
        map_voxel=map_voxel,
        truth=truth,
    )


def posegraph_compare(
    first: str,
    second: str,
    here: int,
    there: int,
    out_dir: str | None = None,
    gravity_first: str | None = None,
    gravity_second: str | None = None,
    reach: float = 50.0,
    min_change: float = 0.3,
) -> dict[str, Any]:
    """Join two drives through the same places and list what changed between them. here and
    there are one node of each (vertex id or frame number) standing at the same spot; the
    report gives the join, the loops, M3C2 statistics and the largest changed objects
    (centroid, size, mean change); out_dir receives both maps and the M3C2 result."""
    from ca.posegraph_fix import compare_sessions

    return compare_sessions(
        first,
        second,
        out_dir,
        here=here,
        there=there,
        gravity_first=gravity_first,
        gravity_second=gravity_second,
        reach=reach,
        min_change=min_change,
    )


_viewers: list[Any] = []


def view_link(paths: list[str]) -> dict[str, Any]:
    """A link that opens results (files, or folders of them: .ply maps, .tum trajectories ...)
    in the CloudAnalyzer web app, for the person to look at: the files are served from this
    machine (127.0.0.1 only) for as long as this server runs. Give the person the link."""
    from ca.web_view import Viewer

    viewer = Viewer(paths).serve_in_background()
    _viewers.append(viewer)
    return {"link": viewer.link, "files": [str(f) for f in viewer.files]}


def cloud_info(path: str) -> dict[str, Any]:
    """A point cloud's size, bounds, centroid and density."""
    from ca.info import get_info

    return get_info(path)


def evaluate_map(candidate: str, reference: str, thresholds: list[float] | None = None) -> dict[str, Any]:
    """A map against a reference map: Chamfer and Hausdorff distances, F1 at thresholds, AUC."""
    from ca.evaluate import evaluate

    return evaluate(candidate, reference, thresholds)


def evaluate_trajectory(estimate: str, reference: str, align_rigid: bool = True) -> dict[str, Any]:
    """A trajectory (TUM or CSV, timestamped) against a reference: ATE, RPE, drift, coverage."""
    from ca.trajectory import evaluate_trajectory as evaluate

    return evaluate(estimate, reference, align_rigid=align_rigid)


def tile_copc(source: str, out_dir: str, grid_size: float, halo: float = 0,
              bounds: list[float] | None = None, origin: list[float] | None = None,
              chunk_size: int = 10_000, resume: bool = False,
              stop_after_nodes: int | None = None, max_node_output_bytes: int = 256 << 20,
              max_fragments_per_node: int = 65_536) -> dict[str, Any]:
    """Create or resume full-density XY COPC tiles with source attributes, halo and disk checkpoints.

    Use a new directory initially; compatible resume verifies source/options and committed hashes.
    Coordinates are in source units. Stop after new nodes for bounded calls; packs require export
    before viewing. Halo copies do not increase core totals; fixed halo is not global-neighbor proof.
    """
    from ca.copc_tiles import tile_copc as run
    return run(source, out_dir, grid_size, halo=halo, bounds=bounds, origin=origin,
               chunk_size=chunk_size, resume=resume, stop_after_nodes=stop_after_nodes,
               max_node_output_bytes=max_node_output_bytes, max_fragments_per_node=max_fragments_per_node)


def export_copc_tile(out_dir: str, i: int, j: int, output: str, include_halo: bool = False) -> dict[str, Any]:
    """Export one committed COPC job tile to a new LAS/LAZ with original schema and CRS.

    Default exports uniquely owned core records. include_halo adds halo/source identity dimensions.
    Existing output is refused. Hashes are checked and bounded frames stream without a global cloud.
    """
    from ca.copc_tiles import export_copc_tile as run
    return run(out_dir, i, j, output, include_halo=include_halo)


TOOLS = [apply_supported_hd_plan, export_mapping_preview, export_mapping_run, inspect_mapping_bundle, continue_mapping_run, start_mapping_run, inspect_mapping_run, advance_mapping_run, start_mapping_job, inspect_mapping_job, propose_mapping_corridors, inspect_mapping_corridors, generate_mapping_geometry, inspect_mapping_geometry, generate_mapping_corridor_lanes, diagnose_mapping_candidate, generate_mapping_candidate, select_mapping_candidate, session_layout, slam_odometry, posegraph_fix, posegraph_compare, build_vector_map, connect_vector_map_junctions, measure_vector_map_signal, measure_vector_map_crosswalk, discover_vector_map_features, edit_vector_map_relations, propose_vector_map_relations, tile_copc, export_copc_tile, view_link, cloud_info, evaluate_map, evaluate_trajectory]


def build_server():
    """The MCP server with CloudAnalyzer's tools (MCP Python SDK 1.x or 2.x)."""
    try:
        from mcp.server.mcpserver import MCPServer as Server
    except ImportError:
        try:
            from mcp.server.fastmcp import FastMCP as Server
        except ImportError as e:
            raise RuntimeError('the MCP server needs the MCP SDK: pip install "cloudanalyzer[mcp]"') from e
    server = Server(name="cloudanalyzer", instructions=INSTRUCTIONS)
    for tool in TOOLS:
        server.tool()(tool)
    return server


def main() -> None:
    build_server().run()
