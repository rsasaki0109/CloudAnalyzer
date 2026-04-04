"""Web-based 3D point cloud viewer using Three.js."""

import json
import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d

from ca.core.web_sampling import reduce_point_cloud_for_web
from ca.core.web_progressive_loading import plan_progressive_loading_for_web
from ca.core.web_trajectory_sampling import reduce_trajectory_for_web
from ca.io import load_point_cloud
from ca.log import logger
from ca.metrics import compute_nn_distance
from ca.trajectory import evaluate_trajectory, load_trajectory

_MAX_TRAJECTORY_DISPLAY_POINTS = 4_000
_MIN_TRAJECTORY_DISPLAY_POINTS = 200
_MAX_PROGRESSIVE_INITIAL_POINTS = 120_000
_MIN_PROGRESSIVE_INITIAL_POINTS = 20_000
_MAX_PROGRESSIVE_CHUNK_POINTS = 60_000
_MIN_PROGRESSIVE_CHUNK_POINTS = 10_000

_VIEWER_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>CloudAnalyzer Viewer</title>
<style>
  body { margin: 0; overflow: hidden; background: #1a1a2e; font-family: sans-serif; }
  canvas { display: block; }
  #info {
    position: absolute; top: 10px; left: 10px; color: #e0e0e0;
    background: rgba(0,0,0,0.7); padding: 12px 16px; border-radius: 8px;
    font-size: 13px; line-height: 1.6; pointer-events: none;
  }
  #controls {
    position: absolute; top: 10px; right: 10px; color: #e0e0e0;
    background: rgba(0,0,0,0.7); padding: 12px 16px; border-radius: 8px;
    font-size: 13px;
  }
  #controls label { display: block; margin: 4px 0; }
  #controls input[type=range] { width: 120px; vertical-align: middle; }
  #controls input[type=checkbox] { vertical-align: middle; }
  #controls select { background: #333; color: #fff; border: 1px solid #555; padding: 2px 4px; }
  #controls button {
    background: #1f2937; color: #f8fafc; border: 1px solid #475569;
    border-radius: 6px; padding: 4px 8px; cursor: pointer;
  }
  #controls button:hover { background: #334155; }
  .control-actions { margin-top: 8px; display: flex; justify-content: flex-end; }
  #thresholdInfo {
    margin-top: 6px; font-size: 12px; line-height: 1.5; color: #cbd5e1;
  }
  #pointInspection,
  #trajectoryInspection {
    margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(203, 213, 225, 0.18);
    font-size: 12px; line-height: 1.5; color: #cbd5e1;
    max-width: 260px;
  }
  #pointInspectionTitle,
  #trajectoryInspectionTitle { color: #f8fafc; font-weight: 700; margin-bottom: 4px; }
  #pointInspectionHint,
  #trajectoryInspectionHint { color: #94a3b8; }
  #pointInspectionBody,
  #trajectoryInspectionBody { margin-top: 4px; }
  #trajectoryTimeline {
    margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(203, 213, 225, 0.18);
    font-size: 12px; line-height: 1.4; color: #cbd5e1;
    max-width: 260px;
  }
  #trajectoryTimelineTitle { color: #f8fafc; font-weight: 700; margin-bottom: 4px; }
  #trajectoryTimelineHint { color: #94a3b8; margin-bottom: 6px; }
  #trajectoryTimelineChart { display: grid; gap: 10px; }
  #progressiveStatus {
    margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(203, 213, 225, 0.18);
    font-size: 12px; line-height: 1.5; color: #cbd5e1;
    max-width: 260px;
  }
  .timeline-series-title { color: #e2e8f0; margin-bottom: 4px; font-weight: 600; }
  .timeline-empty { color: #94a3b8; font-style: italic; }
  .timeline-svg { display: block; width: 100%; height: auto; background: rgba(15, 23, 42, 0.35); border-radius: 8px; }
  .timeline-axis-label { fill: #94a3b8; font-size: 10px; }
  .timeline-series-line { fill: none; stroke-width: 2; }
  .timeline-series-dot { cursor: pointer; }
  #legend {
    position: absolute; left: 10px; bottom: 10px; color: #e0e0e0;
    background: rgba(0,0,0,0.7); padding: 12px 16px; border-radius: 8px;
    font-size: 12px; width: 240px;
  }
  #legendBar {
    height: 12px; border-radius: 999px; margin: 8px 0 6px;
    background: linear-gradient(90deg, #1f3cff 0%, #00d4ff 25%, #8ef000 50%, #ffb000 75%, #ff3300 100%);
  }
  #legendLabels {
    display: flex; justify-content: space-between; gap: 8px; color: #cbd5e1;
  }
</style>
</head>
<body>
<div id="info">Loading...</div>
<div id="controls">
  <label>Point Size <input type="range" id="ptSize" min="0.5" max="10" step="0.5" value="2"></label>
  <label id="refToggleWrap" style="display:none"><input type="checkbox" id="showReference" checked> Reference Overlay</label>
  <label id="refOpacityWrap" style="display:none">Ref Opacity <input type="range" id="refOpacity" min="0.05" max="1" step="0.05" value="0.35"></label>
  <label id="trajectoryToggleWrap" style="display:none"><input type="checkbox" id="showTrajectory" checked> Estimated Trajectory</label>
  <label id="trajectoryReferenceToggleWrap" style="display:none"><input type="checkbox" id="showTrajectoryReference" checked> Reference Trajectory</label>
  <label id="trajectoryMarkerToggleWrap" style="display:none"><input type="checkbox" id="showTrajectoryWorstMarker" checked> Worst ATE Marker</label>
  <label id="trajectorySegmentToggleWrap" style="display:none"><input type="checkbox" id="showTrajectoryWorstSegment" checked> Worst RPE Segment</label>
  <label id="thresholdWrap" style="display:none">Error Threshold <input type="range" id="distThreshold" min="0" max="1" step="0.001" value="0"></label>
  <label>Color <select id="colorMode">
    <option value="heatmap">Heatmap</option>
    <option value="height">Height</option>
    <option value="distance">Distance from center</option>
    <option value="white">White</option>
  </select></label>
  <label>BG <select id="bgColor">
    <option value="#1a1a2e">Dark</option>
    <option value="#ffffff">White</option>
    <option value="#000000">Black</option>
  </select></label>
  <div id="thresholdInfo" style="display:none">
    Threshold: <span id="distThresholdValue">0.0000</span><br>
    Visible source points: <span id="visibleCount">0 / 0 (0.0%)</span>
  </div>
  <div id="progressiveStatus" style="display:none"></div>
  <div id="pointInspection" style="display:none">
    <div id="pointInspectionTitle">Point Inspection</div>
    <div id="pointInspectionHint">Click a point.</div>
    <div id="pointInspectionBody"></div>
  </div>
  <div id="trajectoryInspection" style="display:none">
    <div id="trajectoryInspectionTitle">Trajectory Inspection</div>
    <div id="trajectoryInspectionHint">Click the worst marker or segment.</div>
    <div id="trajectoryInspectionBody"></div>
  </div>
  <div id="trajectoryTimeline" style="display:none">
    <div id="trajectoryTimelineTitle">Trajectory Error Timeline</div>
    <div id="trajectoryTimelineHint">Click a point to focus the viewer.</div>
    <div id="trajectoryTimelineChart"></div>
  </div>
  <div class="control-actions">
    <button type="button" id="resetView">Reset View</button>
  </div>
</div>
<div id="legend" style="display:none">
  <div><b>Distance Legend</b></div>
  <div id="legendBar"></div>
  <div id="legendLabels">
    <span id="legendMin">0.0000</span>
    <span id="legendMean">mean 0.0000</span>
    <span id="legendMax">0.0000</span>
  </div>
</div>

<script type="importmap">
{ "imports": { "three": "https://cdn.jsdelivr.net/npm/three@0.170.0/build/three.module.js",
               "three/addons/": "https://cdn.jsdelivr.net/npm/three@0.170.0/examples/jsm/" }}
</script>
<script type="module">
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';

const scene = new THREE.Scene();
scene.background = new THREE.Color('#1a1a2e');
const camera = new THREE.PerspectiveCamera(60, window.innerWidth / window.innerHeight, 0.1, 10000);
const renderer = new THREE.WebGLRenderer({ antialias: true });
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.setPixelRatio(window.devicePixelRatio);
document.body.appendChild(renderer.domElement);

const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.1;
const raycaster = new THREE.Raycaster();
raycaster.params.Points.threshold = 12;
raycaster.params.Line.threshold = 6;
const pointer = new THREE.Vector2();

let pointCloud;
let referenceCloud;
let trajectoryLine;
let trajectoryReferenceLine;
let trajectoryWorstMarker;
let trajectoryWorstSegment;
let pickedPointMarker;
let pickedReferenceMarker;
let pickedCorrespondenceLine;
let viewerData;
let defaultCameraPosition;
let defaultControlTarget;
let trajectorySelection = null;

function centerFlatPositions(flatPositions, center) {
  if (!flatPositions) {
    return flatPositions;
  }
  for (let i = 0; i < flatPositions.length; i += 3) {
    flatPositions[i] -= center.x;
    flatPositions[i + 1] -= center.y;
    flatPositions[i + 2] -= center.z;
  }
  return flatPositions;
}

function updateReferenceGeometry() {
  if (!referenceCloud) {
    return;
  }
  const referencePositions = window._referenceBasePositions || [];
  referenceCloud.geometry.setAttribute(
    'position',
    new THREE.BufferAttribute(new Float32Array(referencePositions), 3),
  );
  if (referencePositions.length > 0) {
    referenceCloud.geometry.computeBoundingSphere();
  }
}

function updateInfoPanel() {
  if (!viewerData) {
    return;
  }
  const loadedPoints = (window._sourceBasePositions || []).length / 3;
  document.getElementById('info').innerHTML = makeInfo(
    viewerData,
    loadedPoints,
    window._sceneExtent || 1,
  );
}

function updateProgressiveStatus() {
  const statusEl = document.getElementById('progressiveStatus');
  const progressive = viewerData && viewerData.progressive_loading;
  if (!statusEl || !progressive || !progressive.enabled) {
    return;
  }

  const lines = [];
  if (progressive.source) {
    const loaded = ((window._sourceBasePositions || []).length / 3).toLocaleString();
    lines.push(
      `Source stream: ${loaded} / ${progressive.source.total_points.toLocaleString()} (${progressive.source.strategy})`,
    );
  }
  if (progressive.reference) {
    const loaded = ((window._referenceBasePositions || []).length / 3).toLocaleString();
    lines.push(
      `Reference stream: ${loaded} / ${progressive.reference.total_points.toLocaleString()} (${progressive.reference.strategy})`,
    );
  }
  statusEl.innerHTML = lines.join('<br>');
  statusEl.style.display = lines.length > 0 ? 'block' : 'none';
}

async function loadProgressiveStream(streamName, streamMeta, center) {
  if (!streamMeta || !streamMeta.enabled || !streamMeta.chunks || streamMeta.chunks.length === 0) {
    return;
  }

  for (const chunk of streamMeta.chunks) {
    const resp = await fetch(chunk.path);
    const payload = await resp.json();
    const positions = Array.from(payload.positions || []);
    centerFlatPositions(positions, center);

    if (streamName === 'source') {
      window._sourceBasePositions.push(...positions);
      window._positions = window._sourceBasePositions.slice();
      if (payload.distances) {
        if (!window._distances) {
          window._distances = [];
        }
        window._distances.push(...payload.distances);
      }
      updateSourceGeometry();
    } else if (streamName === 'reference') {
      if (!window._referenceBasePositions) {
        window._referenceBasePositions = [];
      }
      window._referenceBasePositions.push(...positions);
      updateReferenceGeometry();
    }

    updateProgressiveStatus();
    updateInfoPanel();
    await new Promise((resolve) => setTimeout(resolve, 0));
  }
}

async function loadProgressiveStreams(center) {
  const progressive = viewerData && viewerData.progressive_loading;
  if (!progressive || !progressive.enabled) {
    return;
  }

  if (progressive.source) {
    await loadProgressiveStream('source', progressive.source, center);
  }
  if (progressive.reference) {
    await loadProgressiveStream('reference', progressive.reference, center);
  }
}

async function loadPoints() {
  const resp = await fetch('data.json');
  const data = await resp.json();
  viewerData = data;
  const center = data.scene_bounds
    ? {
        x: data.scene_bounds.center[0],
        y: data.scene_bounds.center[1],
        z: data.scene_bounds.center[2],
      }
    : { x: 0, y: 0, z: 0 };
  const positions = Array.from(data.positions || []);
  const referencePositions = data.reference_positions ? Array.from(data.reference_positions) : null;
  const trajectory = data.trajectory || null;
  const trajectoryPositions = trajectory && trajectory.estimated_positions
    ? Array.from(trajectory.estimated_positions)
    : null;
  const trajectoryReferencePositions = trajectory && trajectory.reference_positions
    ? Array.from(trajectory.reference_positions)
    : null;
  centerFlatPositions(positions, center);
  centerFlatPositions(referencePositions, center);
  centerFlatPositions(trajectoryPositions, center);
  centerFlatPositions(trajectoryReferencePositions, center);
  const n = positions.length / 3;
  const heatmapOption = document.querySelector('#colorMode option[value="heatmap"]');
  if (!data.distances && heatmapOption) {
    heatmapOption.remove();
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(new Float32Array(positions), 3));
  const cx = center.x;
  const cy = center.y;
  const cz = center.z;
  const extent = data.scene_bounds ? data.scene_bounds.extent : 1.0;
  window._sceneExtent = extent;

  const initialMode = data.distances ? 'heatmap' : 'height';
  document.getElementById('colorMode').value = initialMode;

  const material = new THREE.PointsMaterial({ size: 2, vertexColors: true, sizeAttenuation: true });
  pointCloud = new THREE.Points(geometry, material);
  scene.add(pointCloud);
  if (referencePositions) {
    const referenceGeometry = new THREE.BufferGeometry();
    referenceGeometry.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array(referencePositions), 3),
    );
    const referenceMaterial = new THREE.PointsMaterial({
      size: 1.5,
      color: '#d1d5db',
      transparent: true,
      opacity: 0.35,
      sizeAttenuation: true,
    });
    referenceCloud = new THREE.Points(referenceGeometry, referenceMaterial);
    scene.add(referenceCloud);
    document.getElementById('refToggleWrap').style.display = 'block';
    document.getElementById('refOpacityWrap').style.display = 'block';
  }
  if (trajectoryPositions) {
    const trajectoryGeometry = new THREE.BufferGeometry();
    trajectoryGeometry.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array(trajectoryPositions), 3),
    );
    const trajectoryMaterial = new THREE.LineBasicMaterial({ color: '#f59e0b' });
    trajectoryLine = new THREE.Line(trajectoryGeometry, trajectoryMaterial);
    scene.add(trajectoryLine);
    document.getElementById('trajectoryToggleWrap').style.display = 'block';
  }
  if (trajectoryReferencePositions) {
    const trajectoryReferenceGeometry = new THREE.BufferGeometry();
    trajectoryReferenceGeometry.setAttribute(
      'position',
      new THREE.BufferAttribute(new Float32Array(trajectoryReferencePositions), 3),
    );
    const trajectoryReferenceMaterial = new THREE.LineBasicMaterial({ color: '#2dd4bf' });
    trajectoryReferenceLine = new THREE.Line(trajectoryReferenceGeometry, trajectoryReferenceMaterial);
    scene.add(trajectoryReferenceLine);
    document.getElementById('trajectoryReferenceToggleWrap').style.display = 'block';
  }
  if (trajectory && trajectory.worst_ate_index !== null && trajectoryPositions) {
    const worstIndex = trajectory.worst_ate_index;
    const worstGeometry = new THREE.BufferGeometry();
    worstGeometry.setAttribute(
      'position',
      new THREE.BufferAttribute(
        new Float32Array([
          trajectoryPositions[worstIndex * 3],
          trajectoryPositions[worstIndex * 3 + 1],
          trajectoryPositions[worstIndex * 3 + 2],
        ]),
        3,
      ),
    );
    const worstMaterial = new THREE.PointsMaterial({
      size: 8,
      color: '#ef4444',
      sizeAttenuation: false,
    });
    trajectoryWorstMarker = new THREE.Points(worstGeometry, worstMaterial);
    trajectoryWorstMarker.userData.inspectType = 'worst-ate';
    scene.add(trajectoryWorstMarker);
    document.getElementById('trajectoryMarkerToggleWrap').style.display = 'block';
  }
  if (trajectory && trajectory.worst_rpe_index !== null && trajectoryPositions) {
    const worstIndex = trajectory.worst_rpe_index;
    const segmentGeometry = new THREE.BufferGeometry();
    segmentGeometry.setAttribute(
      'position',
      new THREE.BufferAttribute(
        new Float32Array([
          trajectoryPositions[worstIndex * 3],
          trajectoryPositions[worstIndex * 3 + 1],
          trajectoryPositions[worstIndex * 3 + 2],
          trajectoryPositions[(worstIndex + 1) * 3],
          trajectoryPositions[(worstIndex + 1) * 3 + 1],
          trajectoryPositions[(worstIndex + 1) * 3 + 2],
        ]),
        3,
      ),
    );
    const segmentMaterial = new THREE.LineBasicMaterial({ color: '#f43f5e' });
    trajectoryWorstSegment = new THREE.Line(segmentGeometry, segmentMaterial);
    trajectoryWorstSegment.userData.inspectType = 'worst-rpe';
    scene.add(trajectoryWorstSegment);
    document.getElementById('trajectorySegmentToggleWrap').style.display = 'block';
  }
  if (trajectory && trajectory.mode === 'paired') {
    document.getElementById('trajectoryInspection').style.display = 'block';
    document.getElementById('trajectoryTimeline').style.display = 'block';
  }

  camera.position.set(0, 0, extent * 0.8);
  controls.target.set(0, 0, 0);
  controls.update();
  defaultCameraPosition = camera.position.clone();
  defaultControlTarget = controls.target.clone();

  document.getElementById('info').innerHTML = makeInfo(data, n, extent);

  // Store for recolor
  window._positions = positions;
  window._n = n;
  window._minZ = data.source_z_bounds ? data.source_z_bounds[0] - cz : -extent / 2;
  window._maxZ = data.source_z_bounds ? data.source_z_bounds[1] - cz : extent / 2;
  window._distances = data.distances ? Array.from(data.distances) : null;
  window._distanceStats = data.distance_stats || null;
  window._sourceBasePositions = positions.slice();
  window._referenceBasePositions = referencePositions ? referencePositions.slice() : null;
  window._sourceTotalPoints = data.display_points || n;
  window._centerOffset = { x: cx, y: cy, z: cz };
  window._pickThreshold = Math.max(extent * 0.003, 0.03);

  const pickedPointGeometry = new THREE.SphereGeometry(Math.max(extent * 0.012, 0.03), 20, 12);
  const pickedPointMaterial = new THREE.MeshBasicMaterial({ color: '#fde68a' });
  pickedPointMarker = new THREE.Mesh(pickedPointGeometry, pickedPointMaterial);
  pickedPointMarker.visible = false;
  scene.add(pickedPointMarker);

  const pickedReferenceMaterial = new THREE.MeshBasicMaterial({ color: '#2dd4bf' });
  pickedReferenceMarker = new THREE.Mesh(pickedPointGeometry.clone(), pickedReferenceMaterial);
  pickedReferenceMarker.visible = false;
  scene.add(pickedReferenceMarker);

  const correspondenceGeometry = new THREE.BufferGeometry();
  correspondenceGeometry.setAttribute(
    'position',
    new THREE.BufferAttribute(new Float32Array(6), 3),
  );
  const correspondenceMaterial = new THREE.LineBasicMaterial({ color: '#f8fafc' });
  pickedCorrespondenceLine = new THREE.Line(correspondenceGeometry, correspondenceMaterial);
  pickedCorrespondenceLine.visible = false;
  scene.add(pickedCorrespondenceLine);

  document.getElementById('pointInspection').style.display = 'block';
  clearPointInspection();

  if (data.distances && data.distance_stats) {
    const thresholdInput = document.getElementById('distThreshold');
    const maxDistance = data.distance_stats.max || 0;
    thresholdInput.max = String(maxDistance);
    thresholdInput.step = String(Math.max(maxDistance / 200, 0.0001));
    thresholdInput.value = '0';
    document.getElementById('thresholdWrap').style.display = 'block';
    document.getElementById('thresholdInfo').style.display = 'block';
    document.getElementById('legend').style.display = 'block';
    document.getElementById('legendMin').textContent = data.distance_stats.min.toFixed(4);
    document.getElementById('legendMean').textContent = `mean ${data.distance_stats.mean.toFixed(4)}`;
    document.getElementById('legendMax').textContent = data.distance_stats.max.toFixed(4);
  }

  updateSourceGeometry();
  renderTrajectoryTimeline();
  updateProgressiveStatus();
  updateInfoPanel();
  void loadProgressiveStreams(center);
}

function makeInfo(data, n, extent) {
  let trajectoryInfo = '';
  let progressiveInfo = '';
  if (data.progressive_loading && data.progressive_loading.enabled && data.progressive_loading.source) {
    const loadedPoints = window._sourceBasePositions
      ? (window._sourceBasePositions.length / 3)
      : n;
    progressiveInfo =
      `<br>Loaded now: ${loadedPoints.toLocaleString()} / ${data.progressive_loading.source.total_points.toLocaleString()}` +
      `<br>Progressive strategy: ${data.progressive_loading.source.strategy}`;
  }
  if (data.trajectory) {
    if (data.trajectory.mode === 'paired') {
      trajectoryInfo =
        `<br>Trajectory matched: ${data.trajectory.matching.matched_poses.toLocaleString()} (${(data.trajectory.matching.coverage_ratio * 100).toFixed(1)}%)` +
        `<br>Trajectory displayed: ${data.trajectory.displayed_estimated_pose_count.toLocaleString()} / ${data.trajectory.estimated_pose_count.toLocaleString()}` +
        `<br>Trajectory sampler: ${data.trajectory.sampling.strategy}` +
        `<br>Trajectory align: ${data.trajectory.alignment.mode}` +
        `<br>Trajectory ATE RMSE (display): ${data.trajectory.ate.rmse.toFixed(4)}` +
        `<br>Trajectory RPE RMSE (display): ${data.trajectory.rpe.rmse.toFixed(4)}` +
        `<br>Trajectory worst ATE: ${data.trajectory.error_stats.max.toFixed(4)}` +
        `<br>Trajectory worst RPE: ${data.trajectory.rpe.max.toFixed(4)}`;
    } else {
      trajectoryInfo =
        `<br>Trajectory poses: ${data.trajectory.displayed_estimated_pose_count.toLocaleString()} / ${data.trajectory.estimated_pose_count.toLocaleString()}` +
        `<br>Trajectory sampler: ${data.trajectory.sampling.strategy}` +
        `<br>Trajectory file: ${data.trajectory.estimated_filename}`;
    }
  }
  const displayPoints = (data.display_points ?? n).toLocaleString();
  if (data.viewer_mode === 'heatmap' && data.distance_stats) {
    const original = data.original_points && data.original_points !== data.display_points
      ? `<br>Original source points: ${data.original_points.toLocaleString()}`
      : '';
    const referenceOriginal = data.reference_points && data.reference_points !== data.reference_display_points
      ? ` / ${data.reference_points.toLocaleString()} original`
      : '';
    return `<b>CloudAnalyzer Heatmap Viewer</b><br>` +
      `Source: ${data.source_filename}<br>` +
      `Reference: ${data.target_filename}<br>` +
      `Display points: ${displayPoints}${original}<br>` +
      `Reference overlay: ${(data.reference_display_points ?? 0).toLocaleString()}${referenceOriginal}<br>` +
      `Mean distance: ${data.distance_stats.mean.toFixed(4)}<br>` +
      `Max distance: ${data.distance_stats.max.toFixed(4)}<br>` +
      `Extent: ${extent.toFixed(1)}m` +
      progressiveInfo +
      trajectoryInfo;
  }

  return `<b>CloudAnalyzer Web Viewer</b><br>` +
    `Points: ${displayPoints}<br>` +
    `Extent: ${extent.toFixed(1)}m<br>` +
    `File: ${data.filename}` +
    progressiveInfo +
    trajectoryInfo;
}

function computeColors(pos, n, minZ, maxZ, mode, distances, distanceStats) {
  const colors = new Float32Array(n * 3);
  const rangeZ = maxZ - minZ || 1;
  const distMin = distanceStats ? distanceStats.min : 0;
  const distRange = distanceStats ? (distanceStats.max - distanceStats.min || 1) : 1;
  for (let i=0; i<n; i++) {
    let t;
    if (mode === 'heatmap' && distances) {
      t = (distances[i] - distMin) / distRange;
    } else if (mode === 'height') {
      t = (pos[i*3+2] - minZ) / rangeZ;
    } else if (mode === 'distance') {
      const dx=pos[i*3], dy=pos[i*3+1], dz=pos[i*3+2];
      t = Math.sqrt(dx*dx+dy*dy+dz*dz) / (rangeZ * 0.7);
      t = Math.min(t, 1);
    } else {
      colors[i*3]=1; colors[i*3+1]=1; colors[i*3+2]=1;
      continue;
    }
    // Turbo-ish colormap
    colors[i*3]   = Math.max(0, Math.min(1, 1.5 - Math.abs(t - 0.75) * 4));
    colors[i*3+1] = Math.max(0, Math.min(1, 1.5 - Math.abs(t - 0.5) * 4));
    colors[i*3+2] = Math.max(0, Math.min(1, 1.5 - Math.abs(t - 0.25) * 4));
  }
  return colors;
}

function updateThresholdInfo(threshold, visibleCount) {
  if (!viewerData || !viewerData.distances) {
    return;
  }

  const total = viewerData.display_points ?? visibleCount;
  const ratio = total > 0 ? (visibleCount / total) * 100 : 0;
  document.getElementById('distThresholdValue').textContent = threshold.toFixed(4);
  document.getElementById('visibleCount').textContent =
    `${visibleCount.toLocaleString()} / ${total.toLocaleString()} (${ratio.toFixed(1)}%)`;
}

function clearTrajectoryInspection() {
  const titleEl = document.getElementById('trajectoryInspectionTitle');
  const body = document.getElementById('trajectoryInspectionBody');
  const hint = document.getElementById('trajectoryInspectionHint');
  if (titleEl) {
    titleEl.textContent = 'Trajectory Inspection';
  }
  if (body) {
    body.innerHTML = '';
  }
  if (hint) {
    hint.textContent = 'Click the worst marker or segment.';
  }
}

function defaultPointInspectionHint() {
  return referenceCloud ? 'Click a source or reference point.' : 'Click a point.';
}

function clearPointInspection() {
  const titleEl = document.getElementById('pointInspectionTitle');
  const body = document.getElementById('pointInspectionBody');
  const hint = document.getElementById('pointInspectionHint');
  if (titleEl) {
    titleEl.textContent = 'Point Inspection';
  }
  if (body) {
    body.innerHTML = '';
  }
  if (hint) {
    hint.textContent = defaultPointInspectionHint();
  }
  if (pickedPointMarker) {
    pickedPointMarker.visible = false;
  }
  if (pickedReferenceMarker) {
    pickedReferenceMarker.visible = false;
  }
  if (pickedCorrespondenceLine) {
    pickedCorrespondenceLine.visible = false;
  }
}

function updateCorrespondenceOverlay(sourcePosition, referencePosition) {
  if (pickedPointMarker && sourcePosition && sourcePosition.length >= 3) {
    pickedPointMarker.position.set(sourcePosition[0], sourcePosition[1], sourcePosition[2]);
    pickedPointMarker.visible = true;
  }
  if (pickedReferenceMarker && referencePosition && referencePosition.length >= 3) {
    pickedReferenceMarker.position.set(referencePosition[0], referencePosition[1], referencePosition[2]);
    pickedReferenceMarker.visible = true;
  } else if (pickedReferenceMarker) {
    pickedReferenceMarker.visible = false;
  }
  if (pickedCorrespondenceLine && sourcePosition && referencePosition) {
    const attrs = pickedCorrespondenceLine.geometry.getAttribute('position');
    attrs.setXYZ(0, sourcePosition[0], sourcePosition[1], sourcePosition[2]);
    attrs.setXYZ(1, referencePosition[0], referencePosition[1], referencePosition[2]);
    attrs.needsUpdate = true;
    pickedCorrespondenceLine.visible = true;
  } else if (pickedCorrespondenceLine) {
    pickedCorrespondenceLine.visible = false;
  }
}

function showPointInspection(title, lines, position, referencePosition = null) {
  const titleEl = document.getElementById('pointInspectionTitle');
  const body = document.getElementById('pointInspectionBody');
  const hint = document.getElementById('pointInspectionHint');
  if (titleEl) {
    titleEl.textContent = title;
  }
  if (hint) {
    hint.textContent = '';
  }
  if (body) {
    body.innerHTML = lines.map((line) => `<div>${line}</div>`).join('');
  }
  updateCorrespondenceOverlay(position, referencePosition);
}

function showTrajectoryInspection(title, lines) {
  const titleEl = document.getElementById('trajectoryInspectionTitle');
  const body = document.getElementById('trajectoryInspectionBody');
  const hint = document.getElementById('trajectoryInspectionHint');
  if (titleEl) {
    titleEl.textContent = title;
  }
  if (hint) {
    hint.textContent = '';
  }
  if (body) {
    body.innerHTML = lines.map((line) => `<div>${line}</div>`).join('');
  }
}

function describePointSelection(intersection) {
  if (!intersection || intersection.index === undefined || intersection.index === null) {
    return null;
  }

  const center = window._centerOffset || { x: 0, y: 0, z: 0 };
  const visibleIndex = intersection.index;

  if (pointCloud && intersection.object === pointCloud) {
    const sourcePositions = window._sourceBasePositions || [];
    const sourceDistances = window._distances || null;
    const visibleIndices = window._sourceVisibleIndices || null;
    const referencePositions = window._referenceBasePositions || null;
    const sourceIndex = visibleIndices ? visibleIndices[visibleIndex] : visibleIndex;
    const start = sourceIndex * 3;
    if (start + 2 >= sourcePositions.length) {
      return null;
    }

    const centeredPosition = [
      sourcePositions[start],
      sourcePositions[start + 1],
      sourcePositions[start + 2],
    ];
    const lines = [
      `Display index: ${visibleIndex.toLocaleString()}`,
      `Point index: ${sourceIndex.toLocaleString()}`,
      `X: ${(centeredPosition[0] + center.x).toFixed(4)}`,
      `Y: ${(centeredPosition[1] + center.y).toFixed(4)}`,
      `Z: ${(centeredPosition[2] + center.z).toFixed(4)}`,
    ];
    if (sourceDistances && sourceDistances[sourceIndex] !== undefined) {
      lines.push(`Distance to reference: ${sourceDistances[sourceIndex].toFixed(4)}`);
    }
    let referencePosition = null;
    if (referencePositions && referencePositions.length >= 3) {
      let bestDistanceSq = Infinity;
      for (let i = 0; i < referencePositions.length; i += 3) {
        const dx = referencePositions[i] - centeredPosition[0];
        const dy = referencePositions[i + 1] - centeredPosition[1];
        const dz = referencePositions[i + 2] - centeredPosition[2];
        const distanceSq = dx * dx + dy * dy + dz * dz;
        if (distanceSq < bestDistanceSq) {
          bestDistanceSq = distanceSq;
          referencePosition = [
            referencePositions[i],
            referencePositions[i + 1],
            referencePositions[i + 2],
          ];
        }
      }
      if (referencePosition) {
        lines.push(
          `Nearest displayed reference: ${Math.sqrt(bestDistanceSq).toFixed(4)}`,
        );
      }
    }
    if (
      viewerData &&
      viewerData.original_points &&
      viewerData.display_points &&
      viewerData.original_points !== viewerData.display_points
    ) {
      lines.push('Display cloud: downsampled for browser');
    }
    return {
      title: viewerData && viewerData.viewer_mode === 'heatmap' ? 'Source Point' : 'Cloud Point',
      lines,
      position: centeredPosition,
      referencePosition,
    };
  }

  if (referenceCloud && intersection.object === referenceCloud) {
    const referencePositions = window._referenceBasePositions || [];
    const start = visibleIndex * 3;
    if (start + 2 >= referencePositions.length) {
      return null;
    }

    const centeredPosition = [
      referencePositions[start],
      referencePositions[start + 1],
      referencePositions[start + 2],
    ];
    const lines = [
      `Display index: ${visibleIndex.toLocaleString()}`,
      `X: ${(centeredPosition[0] + center.x).toFixed(4)}`,
      `Y: ${(centeredPosition[1] + center.y).toFixed(4)}`,
      `Z: ${(centeredPosition[2] + center.z).toFixed(4)}`,
    ];
    if (
      viewerData &&
      viewerData.reference_points &&
      viewerData.reference_display_points &&
      viewerData.reference_points !== viewerData.reference_display_points
    ) {
      lines.push('Reference overlay: downsampled for browser');
    }
    return {
      title: 'Reference Point',
      lines,
      position: centeredPosition,
      referencePosition: null,
    };
  }

  return null;
}

function selectPointFeature(intersection) {
  const description = describePointSelection(intersection);
  if (!description) {
    clearPointInspection();
    return;
  }
  trajectorySelection = null;
  renderTrajectoryTimeline();
  clearTrajectoryInspection();
  showPointInspection(
    description.title,
    description.lines,
    description.position,
    description.referencePosition || null,
  );
}

function buildTimelineSeriesSvg(label, selectionType, timestamps, values, color, activeSelection) {
  if (!timestamps || !values || timestamps.length === 0 || values.length === 0) {
    return `<div><div class="timeline-series-title">${label}</div><div class="timeline-empty">No samples</div></div>`;
  }

  const width = 240;
  const height = 92;
  const paddingLeft = 28;
  const paddingRight = 8;
  const paddingTop = 8;
  const paddingBottom = 18;
  const minTime = timestamps[0];
  const maxTime = timestamps[timestamps.length - 1];
  const minValue = Math.min(...values);
  const maxValue = Math.max(...values);
  const timeRange = maxTime - minTime || 1;
  const valueRange = maxValue - minValue || 1;
  const plotWidth = width - paddingLeft - paddingRight;
  const plotHeight = height - paddingTop - paddingBottom;

  const points = values.map((value, index) => {
    const x = paddingLeft + (((timestamps[index] - minTime) / timeRange) * plotWidth);
    const y = paddingTop + plotHeight - (((value - minValue) / valueRange) * plotHeight);
    return { x, y, value, timestamp: timestamps[index], index };
  });
  const polyline = points.map((point) => `${point.x.toFixed(2)},${point.y.toFixed(2)}`).join(' ');
  const dots = points.map((point) => {
    const isSelected =
      activeSelection &&
      activeSelection.type === selectionType &&
      activeSelection.index === point.index;
    const radius = isSelected ? 4.5 : 2.8;
    const stroke = isSelected ? '#f8fafc' : color;
    const strokeWidth = isSelected ? 1.5 : 0.75;
    return (
      `<circle class="timeline-series-dot" ` +
      `cx="${point.x.toFixed(2)}" cy="${point.y.toFixed(2)}" r="${radius}" fill="${color}" ` +
      `stroke="${stroke}" stroke-width="${strokeWidth}" ` +
      `data-trajectory-selection="${selectionType}" data-trajectory-index="${point.index}">` +
      `<title>${label} @ ${point.timestamp.toFixed(3)} = ${point.value.toFixed(4)}</title>` +
      `</circle>`
    );
  }).join('');

  return `
    <div>
      <div class="timeline-series-title">${label}</div>
      <svg class="timeline-svg" viewBox="0 0 ${width} ${height}" role="img" aria-label="${label} timeline">
        <line x1="${paddingLeft}" y1="${height - paddingBottom}" x2="${width - paddingRight}" y2="${height - paddingBottom}" stroke="#334155" stroke-width="1" />
        <line x1="${paddingLeft}" y1="${paddingTop}" x2="${paddingLeft}" y2="${height - paddingBottom}" stroke="#334155" stroke-width="1" />
        <polyline class="timeline-series-line" points="${polyline}" stroke="${color}" />
        ${dots}
        <text class="timeline-axis-label" x="${paddingLeft}" y="${height - 4}">${minTime.toFixed(2)}</text>
        <text class="timeline-axis-label" x="${width - paddingRight}" y="${height - 4}" text-anchor="end">${maxTime.toFixed(2)}</text>
        <text class="timeline-axis-label" x="${paddingLeft - 4}" y="${paddingTop + 10}" text-anchor="end">${maxValue.toFixed(3)}</text>
        <text class="timeline-axis-label" x="${paddingLeft - 4}" y="${height - paddingBottom}" text-anchor="end">${minValue.toFixed(3)}</text>
      </svg>
    </div>
  `;
}

function renderTrajectoryTimeline() {
  const chart = document.getElementById('trajectoryTimelineChart');
  const hint = document.getElementById('trajectoryTimelineHint');
  if (!chart) {
    return;
  }
  if (!viewerData || !viewerData.trajectory || viewerData.trajectory.mode !== 'paired') {
    chart.innerHTML = '';
    return;
  }

  const trajectory = viewerData.trajectory;
  chart.innerHTML = [
    buildTimelineSeriesSvg(
      'ATE',
      'ate',
      trajectory.timestamps || [],
      trajectory.ate_errors || [],
      '#60a5fa',
      trajectorySelection,
    ),
    buildTimelineSeriesSvg(
      'RPE',
      'rpe',
      trajectory.rpe_timestamps || [],
      trajectory.rpe_errors || [],
      '#fb7185',
      trajectorySelection,
    ),
  ].join('');

  chart.querySelectorAll('[data-trajectory-selection]').forEach((node) => {
    node.addEventListener('click', (event) => {
      const target = event.currentTarget;
      if (!(target instanceof Element)) {
        return;
      }
      const selectionType = target.dataset.trajectorySelection;
      const selectionIndex = Number(target.dataset.trajectoryIndex);
      if (selectionType && Number.isFinite(selectionIndex)) {
        selectTrajectoryFeature(selectionType, selectionIndex, true);
      }
    });
  });

  if (hint) {
    hint.textContent = trajectorySelection
      ? 'Timeline selection is synced with the viewer.'
      : 'Click a point to focus the viewer.';
  }
}

function focusCameraOnSelection(selectionPositions) {
  if (!selectionPositions || selectionPositions.length < 3) {
    return;
  }

  let minX = Infinity, minY = Infinity, minZ = Infinity;
  let maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < selectionPositions.length; i += 3) {
    const x = selectionPositions[i];
    const y = selectionPositions[i + 1];
    const z = selectionPositions[i + 2];
    if (x < minX) minX = x;
    if (y < minY) minY = y;
    if (z < minZ) minZ = z;
    if (x > maxX) maxX = x;
    if (y > maxY) maxY = y;
    if (z > maxZ) maxZ = z;
  }

  const target = new THREE.Vector3(
    (minX + maxX) / 2,
    (minY + maxY) / 2,
    (minZ + maxZ) / 2,
  );
  const selectionExtent = Math.max(maxX - minX, maxY - minY, maxZ - minZ);
  const direction = new THREE.Vector3().subVectors(camera.position, controls.target);
  if (direction.lengthSq() < 1e-8) {
    direction.set(0, 0, 1);
  }
  direction.normalize();
  const sceneExtent = window._sceneExtent || 1;
  const focusDistance = Math.max(selectionExtent * 4, sceneExtent * 0.18, 0.75);

  camera.position.copy(target.clone().add(direction.multiplyScalar(focusDistance)));
  controls.target.copy(target);
  controls.update();
}

function focusTrajectorySelection(type, index) {
  if (!viewerData || !viewerData.trajectory || viewerData.trajectory.mode !== 'paired') {
    return false;
  }
  const trajectory = viewerData.trajectory;
  const estimatedPositions = trajectory.estimated_positions || [];
  if (type === 'ate') {
    const start = index * 3;
    if (start + 2 >= estimatedPositions.length) {
      return false;
    }
    focusCameraOnSelection(estimatedPositions.slice(start, start + 3));
    return true;
  }
  if (type === 'rpe') {
    const start = index * 3;
    if (start + 5 >= estimatedPositions.length) {
      return false;
    }
    focusCameraOnSelection(estimatedPositions.slice(start, start + 6));
    return true;
  }
  return false;
}

function describeTrajectorySelection(type, index) {
  if (!viewerData || !viewerData.trajectory || viewerData.trajectory.mode !== 'paired') {
    return null;
  }
  const trajectory = viewerData.trajectory;
  if (type === 'ate' && trajectory.timestamps && trajectory.ate_errors) {
    const timestamp = trajectory.timestamps[index];
    const error = trajectory.ate_errors[index];
    if (timestamp === undefined || error === undefined) {
      return null;
    }
    const lines = [
      `Timestamp: ${timestamp.toFixed(3)}`,
      `Position error: ${error.toFixed(4)}`,
    ];
    if (trajectory.worst_ate_sample && trajectory.worst_ate_index === index) {
      lines.push(`Time delta: ${trajectory.worst_ate_sample.time_delta.toFixed(4)}`);
    }
    return {
      title: index === trajectory.worst_ate_index ? 'Worst ATE Pose' : 'ATE Sample',
      lines,
    };
  }
  if (type === 'rpe' && trajectory.rpe_timestamps && trajectory.rpe_errors && trajectory.timestamps) {
    const timestamp = trajectory.rpe_timestamps[index];
    const error = trajectory.rpe_errors[index];
    const startTimestamp = trajectory.timestamps[index];
    const endTimestamp = trajectory.timestamps[index + 1];
    if (
      timestamp === undefined ||
      error === undefined ||
      startTimestamp === undefined ||
      endTimestamp === undefined
    ) {
      return null;
    }
    return {
      title: index === trajectory.worst_rpe_index ? 'Worst RPE Segment' : 'RPE Segment',
      lines: [
        `Center: ${timestamp.toFixed(3)}`,
        `Start: ${startTimestamp.toFixed(3)}`,
        `End: ${endTimestamp.toFixed(3)}`,
        `Translation error: ${error.toFixed(4)}`,
      ],
    };
  }
  return null;
}

function selectTrajectoryFeature(type, index, focusCamera = false) {
  const description = describeTrajectorySelection(type, index);
  if (!description) {
    trajectorySelection = null;
    renderTrajectoryTimeline();
    clearTrajectoryInspection();
    return;
  }
  trajectorySelection = { type, index };
  if (focusCamera) {
    const focused = focusTrajectorySelection(type, index);
    if (focused) {
      description.lines.push('Camera: focused on selected feature');
    }
  }
  showTrajectoryInspection(description.title, description.lines);
  renderTrajectoryTimeline();
}

function resetView() {
  if (defaultCameraPosition && defaultControlTarget) {
    camera.position.copy(defaultCameraPosition);
    controls.target.copy(defaultControlTarget);
    controls.update();
  }
  trajectorySelection = null;
  renderTrajectoryTimeline();
  clearTrajectoryInspection();
  clearPointInspection();
}

function onViewerClick(event) {
  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);

  if (viewerData && viewerData.trajectory && viewerData.trajectory.mode === 'paired') {
    const trajectoryTargets = [];
    if (trajectoryWorstMarker && trajectoryWorstMarker.visible) {
      trajectoryTargets.push(trajectoryWorstMarker);
    }
    if (trajectoryWorstSegment && trajectoryWorstSegment.visible) {
      trajectoryTargets.push(trajectoryWorstSegment);
    }

    if (trajectoryTargets.length > 0) {
      const trajectoryIntersections = raycaster.intersectObjects(trajectoryTargets, false);
      if (trajectoryIntersections.length > 0) {
        const object = trajectoryIntersections[0].object;
        clearPointInspection();
        if (object.userData.inspectType === 'worst-ate' && viewerData.trajectory.worst_ate_sample) {
          selectTrajectoryFeature('ate', viewerData.trajectory.worst_ate_index, true);
          return;
        }
        if (object.userData.inspectType === 'worst-rpe' && viewerData.trajectory.worst_rpe_segment) {
          selectTrajectoryFeature('rpe', viewerData.trajectory.worst_rpe_index, true);
          return;
        }
      }
    }
  }

  const pointTargets = [];
  if (pointCloud && pointCloud.visible) {
    pointTargets.push(pointCloud);
  }
  if (referenceCloud && referenceCloud.visible) {
    pointTargets.push(referenceCloud);
  }

  if (pointTargets.length > 0) {
    const previousPointThreshold = raycaster.params.Points.threshold;
    raycaster.params.Points.threshold = window._pickThreshold || previousPointThreshold;
    const pointIntersections = raycaster.intersectObjects(pointTargets, false);
    raycaster.params.Points.threshold = previousPointThreshold;
    if (pointIntersections.length > 0) {
      selectPointFeature(pointIntersections[0]);
      return;
    }
  }

  trajectorySelection = null;
  renderTrajectoryTimeline();
  clearTrajectoryInspection();
  clearPointInspection();
}

function updateSourceGeometry() {
  if (!pointCloud) return;

  const basePositions = window._sourceBasePositions || window._positions;
  const baseDistances = window._distances;
  const threshold = baseDistances ? parseFloat(document.getElementById('distThreshold').value) : 0;
  const mode = document.getElementById('colorMode').value;

  let positions = basePositions;
  let distances = baseDistances;
  let visibleCount = basePositions.length / 3;
  let visibleIndices = null;

  if (baseDistances && threshold > 0) {
    let kept = 0;
    for (let i=0; i<baseDistances.length; i++) {
      if (baseDistances[i] >= threshold) kept++;
    }

    const filteredPositions = new Float32Array(kept * 3);
    const filteredDistances = new Float32Array(kept);
    const filteredIndices = new Uint32Array(kept);
    let out = 0;
    for (let i=0; i<baseDistances.length; i++) {
      if (baseDistances[i] < threshold) continue;
      filteredPositions[out*3] = basePositions[i*3];
      filteredPositions[out*3+1] = basePositions[i*3+1];
      filteredPositions[out*3+2] = basePositions[i*3+2];
      filteredDistances[out] = baseDistances[i];
      filteredIndices[out] = i;
      out++;
    }

    positions = filteredPositions;
    distances = filteredDistances;
    visibleCount = kept;
    visibleIndices = filteredIndices;
  }

  const colors = computeColors(
    positions,
    visibleCount,
    window._minZ,
    window._maxZ,
    mode,
    distances,
    window._distanceStats,
  );

  pointCloud.geometry.setAttribute(
    'position',
    new THREE.BufferAttribute(
      positions instanceof Float32Array ? positions : new Float32Array(positions),
      3,
    ),
  );
  pointCloud.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  window._sourceVisibleIndices = visibleIndices;
  if (visibleCount > 0) {
    pointCloud.geometry.computeBoundingSphere();
  }
  updateThresholdInfo(threshold, visibleCount);
  clearPointInspection();
}

loadPoints();

// Controls
document.getElementById('ptSize').addEventListener('input', e => {
  const size = parseFloat(e.target.value);
  if (pointCloud) pointCloud.material.size = size;
  if (referenceCloud) referenceCloud.material.size = Math.max(0.5, size * 0.75);
});
document.getElementById('colorMode').addEventListener('change', e => {
  if (!pointCloud) return;
  updateSourceGeometry();
});
document.getElementById('showReference').addEventListener('change', e => {
  if (referenceCloud) {
    referenceCloud.visible = e.target.checked;
  }
});
document.getElementById('showTrajectory').addEventListener('change', e => {
  if (trajectoryLine) {
    trajectoryLine.visible = e.target.checked;
  }
});
document.getElementById('showTrajectoryReference').addEventListener('change', e => {
  if (trajectoryReferenceLine) {
    trajectoryReferenceLine.visible = e.target.checked;
  }
});
document.getElementById('showTrajectoryWorstMarker').addEventListener('change', e => {
  if (trajectoryWorstMarker) {
    trajectoryWorstMarker.visible = e.target.checked;
  }
});
document.getElementById('showTrajectoryWorstSegment').addEventListener('change', e => {
  if (trajectoryWorstSegment) {
    trajectoryWorstSegment.visible = e.target.checked;
  }
});
document.getElementById('refOpacity').addEventListener('input', e => {
  if (referenceCloud) {
    referenceCloud.material.opacity = parseFloat(e.target.value);
  }
});
document.getElementById('distThreshold').addEventListener('input', () => {
  updateSourceGeometry();
});
document.getElementById('bgColor').addEventListener('change', e => {
  scene.background = new THREE.Color(e.target.value);
});
document.getElementById('resetView').addEventListener('click', resetView);
renderer.domElement.addEventListener('click', onViewerClick);

window.addEventListener('resize', () => {
  camera.aspect = window.innerWidth / window.innerHeight;
  camera.updateProjectionMatrix();
  renderer.setSize(window.innerWidth, window.innerHeight);
});

function animate() {
  requestAnimationFrame(animate);
  controls.update();
  renderer.render(scene, camera);
}
animate();
</script>
</body>
</html>"""


def _progressive_initial_budget(max_points: int, display_points: int) -> int:
    """Choose a bounded initial payload size for browser loading."""
    if display_points <= 0:
        return 0
    budget = max(
        _MIN_PROGRESSIVE_INITIAL_POINTS,
        min(max_points // 12, _MAX_PROGRESSIVE_INITIAL_POINTS),
    )
    return min(display_points, max(1, budget))


def _progressive_chunk_budget(max_points: int, display_points: int) -> int:
    """Choose a bounded deferred chunk size."""
    if display_points <= 0:
        return 0
    budget = max(
        _MIN_PROGRESSIVE_CHUNK_POINTS,
        min(max_points // 16, _MAX_PROGRESSIVE_CHUNK_POINTS),
    )
    return min(display_points, max(1, budget))


def _prepare_progressive_loading_payload(
    positions: np.ndarray,
    max_points: int,
    label: str,
    distances: np.ndarray | None = None,
    stream_name: str = "source",
) -> tuple[np.ndarray, np.ndarray | None, dict, dict[str, str]]:
    """Plan a small initial payload plus deferred chunk responses."""

    display_points = int(positions.shape[0])
    if display_points == 0:
        return (
            positions,
            distances,
            {
                "enabled": False,
                "strategy": None,
                "design": None,
                "initial_points": 0,
                "total_points": 0,
                "chunk_count": 0,
                "chunks": [],
            },
            {},
        )

    initial_points = _progressive_initial_budget(max_points, display_points)
    if display_points <= initial_points:
        return (
            positions,
            distances,
            {
                "enabled": False,
                "strategy": None,
                "design": None,
                "initial_points": display_points,
                "total_points": display_points,
                "chunk_count": 0,
                "chunks": [],
            },
            {},
        )

    plan = plan_progressive_loading_for_web(
        positions=positions,
        initial_points=initial_points,
        chunk_points=_progressive_chunk_budget(max_points, display_points),
        distances=distances,
        label=label,
    )
    chunk_payloads: dict[str, str] = {}
    chunk_descriptors = []
    for chunk_index, chunk in enumerate(plan.chunks):
        path = f"chunks/{stream_name}/{chunk_index}.json"
        chunk_payloads[path] = json.dumps(
            {
                "positions": chunk.positions.flatten().tolist(),
                "distances": (
                    chunk.distances.tolist() if chunk.distances is not None else None
                ),
            }
        )
        chunk_descriptors.append(
            {
                "path": path,
                "points": chunk.point_count,
            }
        )

    return (
        plan.initial_positions,
        plan.initial_distances,
        {
            "enabled": True,
            "strategy": plan.strategy,
            "design": plan.design,
            "initial_points": plan.initial_points,
            "total_points": plan.total_displayed_points,
            "chunk_count": len(plan.chunks),
            "chunks": chunk_descriptors,
        },
        chunk_payloads,
    )


def _scene_bounds(*arrays: np.ndarray) -> dict[str, list[float] | float]:
    """Compute full-scene bounds from already prepared display arrays."""

    valid_arrays = [array for array in arrays if array.size > 0]
    if not valid_arrays:
        return {
            "center": [0.0, 0.0, 0.0],
            "extent": 1.0,
        }
    stacked = np.vstack(valid_arrays)
    mins = np.min(stacked, axis=0)
    maxs = np.max(stacked, axis=0)
    center = (mins + maxs) / 2.0
    extent = float(np.max(maxs - mins))
    return {
        "center": center.tolist(),
        "extent": extent if extent > 0 else 1.0,
    }


def _make_handler(html: str, data_json: str, chunk_payloads: dict[str, str] | None = None):
    """Create a custom HTTP handler serving the viewer and data."""
    payloads = chunk_payloads or {}

    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            path = self.path.split('?', 1)[0]
            normalized = path.lstrip('/')
            if path == '/' or normalized == 'index.html':
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                self.wfile.write(html.encode())
            elif normalized == 'data.json':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(data_json.encode())
            elif normalized in payloads:
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(payloads[normalized].encode())
            else:
                self.send_error(404)

        def log_message(self, format, *args):
            pass  # Suppress default logging

    return Handler


def _downsample_for_web(
    pcd: o3d.geometry.PointCloud,
    max_points: int,
    label: str,
) -> o3d.geometry.PointCloud:
    """Reduce a point cloud for browser display via the stable core interface."""
    result = reduce_point_cloud_for_web(pcd, max_points=max_points, label=label)
    if result.reduced_points < result.original_points:
        logger.info(
            "Reduced %s to %d points for web display using %s",
            label,
            result.reduced_points,
            result.strategy,
        )
    return result.point_cloud


def _trajectory_display_budget(max_points: int, pose_count: int) -> int:
    """Pick a browser-safe trajectory budget without tying it too tightly to cloud density."""
    if pose_count <= 0:
        return 0
    return min(
        pose_count,
        max(_MIN_TRAJECTORY_DISPLAY_POINTS, min(max_points, _MAX_TRAJECTORY_DISPLAY_POINTS)),
    )


def _downsample_trajectory_for_web(
    positions: np.ndarray,
    timestamps: np.ndarray,
    max_points: int,
    label: str,
    preserve_indices: tuple[int, ...] = (),
):
    """Reduce trajectory overlays for browser display via the stable core interface."""
    result = reduce_trajectory_for_web(
        positions=positions,
        timestamps=timestamps,
        max_points=_trajectory_display_budget(max_points, positions.shape[0]),
        label=label,
        preserve_indices=preserve_indices,
    )
    if result.reduced_points < result.original_points:
        logger.info(
            "Reduced %s to %d poses for web display using %s",
            label,
            result.reduced_points,
            result.strategy,
        )
    return result


def _summarize_error_series(values: np.ndarray) -> dict[str, float]:
    """Summarize an error series for viewer metadata."""
    if values.size == 0:
        return {"rmse": 0.0, "max": 0.0, "mean": 0.0, "min": 0.0}
    return {
        "rmse": float(np.sqrt(np.mean(np.square(values)))),
        "max": float(np.max(values)),
        "mean": float(np.mean(values)),
        "min": float(np.min(values)),
    }


def _prepare_viewer_bundle(
    paths: list[str],
    max_points: int = 2_000_000,
    heatmap: bool = False,
    trajectory_path: str | None = None,
    trajectory_reference_path: str | None = None,
    trajectory_max_time_delta: float = 0.05,
    trajectory_align_origin: bool = False,
    trajectory_align_rigid: bool = False,
) -> tuple[dict, dict[str, str]]:
    """Prepare browser payload plus deferred chunk payloads."""
    if trajectory_reference_path is not None and trajectory_path is None:
        raise ValueError("--trajectory-reference requires --trajectory")
    if trajectory_path is None and (trajectory_align_origin or trajectory_align_rigid):
        raise ValueError("trajectory alignment options require --trajectory")
    if trajectory_path is not None and trajectory_reference_path is None and (
        trajectory_align_origin or trajectory_align_rigid
    ):
        raise ValueError(
            "trajectory alignment options require --trajectory-reference"
        )

    trajectory_data = None
    if trajectory_path is not None:
        trajectory_data = _prepare_trajectory_viewer_data(
            trajectory_path,
            trajectory_reference_path=trajectory_reference_path,
            max_points=max_points,
            max_time_delta=trajectory_max_time_delta,
            align_origin=trajectory_align_origin,
            align_rigid=trajectory_align_rigid,
        )
    trajectory_positions = (
        np.asarray(trajectory_data["estimated_positions"], dtype=float).reshape(-1, 3)
        if trajectory_data and trajectory_data.get("estimated_positions")
        else np.zeros((0, 3), dtype=float)
    )
    trajectory_reference_positions = (
        np.asarray(trajectory_data["reference_positions"], dtype=float).reshape(-1, 3)
        if trajectory_data and trajectory_data.get("reference_positions")
        else np.zeros((0, 3), dtype=float)
    )

    if heatmap:
        if len(paths) != 2:
            raise ValueError("--heatmap requires exactly 2 point cloud files")

        source_path, target_path = paths
        source = load_point_cloud(source_path)
        target = load_point_cloud(target_path)
        logger.info("Loaded %d source points from %s", len(source.points), source_path)
        logger.info("Loaded %d reference points from %s", len(target.points), target_path)

        display_budget = max(1, max_points // 2)
        display_source = _downsample_for_web(source, display_budget, "source cloud")
        display_target = _downsample_for_web(target, display_budget, "reference cloud")
        source_positions = np.asarray(display_source.points, dtype=float)
        target_positions = np.asarray(display_target.points, dtype=float)
        distances = compute_nn_distance(display_source, target)
        (
            initial_source_positions,
            initial_source_distances,
            source_progressive,
            source_chunk_payloads,
        ) = _prepare_progressive_loading_payload(
            positions=source_positions,
            max_points=max_points,
            label="source cloud",
            distances=distances,
            stream_name="source",
        )
        (
            initial_reference_positions,
            _,
            reference_progressive,
            reference_chunk_payloads,
        ) = _prepare_progressive_loading_payload(
            positions=target_positions,
            max_points=max_points,
            label="reference cloud",
            stream_name="reference",
        )
        chunk_payloads = {**source_chunk_payloads, **reference_chunk_payloads}
        progressive_enabled = bool(
            source_progressive["enabled"] or reference_progressive["enabled"]
        )

        data = {
            "positions": initial_source_positions.flatten().tolist(),
            "reference_positions": initial_reference_positions.flatten().tolist(),
            "filename": Path(source_path).name,
            "viewer_mode": "heatmap",
            "source_filename": Path(source_path).name,
            "target_filename": Path(target_path).name,
            "display_points": int(source_positions.shape[0]),
            "initial_display_points": int(initial_source_positions.shape[0]),
            "original_points": len(source.points),
            "reference_points": len(target.points),
            "reference_display_points": int(target_positions.shape[0]),
            "reference_initial_display_points": int(initial_reference_positions.shape[0]),
            "distances": (
                initial_source_distances.tolist() if initial_source_distances is not None else None
            ),
            "distance_stats": {
                "mean": float(np.mean(distances)),
                "max": float(np.max(distances)),
                "min": float(np.min(distances)),
            },
            "source_z_bounds": [
                float(np.min(source_positions[:, 2])) if source_positions.size else 0.0,
                float(np.max(source_positions[:, 2])) if source_positions.size else 0.0,
            ],
            "scene_bounds": _scene_bounds(
                source_positions,
                target_positions,
                trajectory_positions,
                trajectory_reference_positions,
            ),
            "progressive_loading": {
                "enabled": progressive_enabled,
                "source": source_progressive if source_progressive["enabled"] else None,
                "reference": (
                    reference_progressive if reference_progressive["enabled"] else None
                ),
            },
        }
        if trajectory_data is not None:
            data["trajectory"] = trajectory_data
        return data, chunk_payloads

    merged = o3d.geometry.PointCloud()
    for path in paths:
        merged += load_point_cloud(path)

    total = len(merged.points)
    logger.info("Loaded %d points from %d file(s)", total, len(paths))
    display_cloud = _downsample_for_web(merged, max_points, "merged cloud")
    display_positions = np.asarray(display_cloud.points, dtype=float)
    (
        initial_positions,
        _,
        source_progressive,
        source_chunk_payloads,
    ) = _prepare_progressive_loading_payload(
        positions=display_positions,
        max_points=max_points,
        label="merged cloud",
        stream_name="source",
    )

    data = {
        "positions": initial_positions.flatten().tolist(),
        "filename": ", ".join(Path(p).name for p in paths),
        "viewer_mode": "standard",
        "display_points": int(display_positions.shape[0]),
        "initial_display_points": int(initial_positions.shape[0]),
        "original_points": total,
        "source_z_bounds": [
            float(np.min(display_positions[:, 2])) if display_positions.size else 0.0,
            float(np.max(display_positions[:, 2])) if display_positions.size else 0.0,
        ],
        "scene_bounds": _scene_bounds(
            display_positions,
            trajectory_positions,
            trajectory_reference_positions,
        ),
        "progressive_loading": {
            "enabled": bool(source_progressive["enabled"]),
            "source": source_progressive if source_progressive["enabled"] else None,
            "reference": None,
        },
    }
    if trajectory_data is not None:
        data["trajectory"] = trajectory_data
    return data, source_chunk_payloads


def _prepare_viewer_data(
    paths: list[str],
    max_points: int = 2_000_000,
    heatmap: bool = False,
    trajectory_path: str | None = None,
    trajectory_reference_path: str | None = None,
    trajectory_max_time_delta: float = 0.05,
    trajectory_align_origin: bool = False,
    trajectory_align_rigid: bool = False,
) -> dict:
    """Prepare browser payload for standard or heatmap web viewing."""
    data, _ = _prepare_viewer_bundle(
        paths,
        max_points=max_points,
        heatmap=heatmap,
        trajectory_path=trajectory_path,
        trajectory_reference_path=trajectory_reference_path,
        trajectory_max_time_delta=trajectory_max_time_delta,
        trajectory_align_origin=trajectory_align_origin,
        trajectory_align_rigid=trajectory_align_rigid,
    )
    return data


def _prepare_trajectory_viewer_data(
    trajectory_path: str,
    trajectory_reference_path: str | None = None,
    max_points: int = 2_000_000,
    max_time_delta: float = 0.05,
    align_origin: bool = False,
    align_rigid: bool = False,
) -> dict:
    """Prepare trajectory overlay payload for the web viewer."""
    if trajectory_reference_path is None:
        trajectory = load_trajectory(trajectory_path)
        positions = np.asarray(trajectory["positions"], dtype=float)
        timestamps = np.asarray(trajectory["timestamps"], dtype=float)
        reduced = _downsample_trajectory_for_web(
            positions=positions,
            timestamps=timestamps,
            max_points=max_points,
            label=f"trajectory {Path(trajectory_path).name}",
        )
        return {
            "mode": "single",
            "estimated_filename": Path(trajectory_path).name,
            "estimated_positions": reduced.positions.flatten().tolist(),
            "estimated_pose_count": int(positions.shape[0]),
            "displayed_estimated_pose_count": int(reduced.reduced_points),
            "reference_positions": None,
            "reference_pose_count": None,
            "displayed_reference_pose_count": None,
            "timestamps": (
                reduced.timestamps.tolist() if reduced.timestamps is not None else None
            ),
            "ate_errors": None,
            "rpe_timestamps": None,
            "rpe_errors": None,
            "worst_ate_index": None,
            "worst_ate_sample": None,
            "worst_rpe_index": None,
            "worst_rpe_segment": None,
            "sampling": {
                "strategy": reduced.strategy,
                "design": reduced.design,
                "display_budget": _trajectory_display_budget(max_points, positions.shape[0]),
                "reduction_ratio": reduced.reduction_ratio,
            },
        }

    result = evaluate_trajectory(
        trajectory_path,
        trajectory_reference_path,
        max_time_delta=max_time_delta,
        align_origin=align_origin,
        align_rigid=align_rigid,
    )
    matched = result["matched_trajectory"]
    estimated_positions = np.asarray(matched["estimated_positions"], dtype=float)
    reference_positions = np.asarray(matched["reference_positions"], dtype=float)
    timestamps = np.asarray(matched["timestamps"], dtype=float)
    full_ate_errors = np.asarray(matched["ate_errors"], dtype=float)
    full_rpe_errors = np.asarray(result["error_series"]["rpe_translation"], dtype=float)
    worst_ate_source_index = int(np.argmax(full_ate_errors)) if full_ate_errors.size > 0 else None
    worst_rpe_source_index = int(np.argmax(full_rpe_errors)) if full_rpe_errors.size > 0 else None

    preserve_indices = tuple(
        index
        for index in (
            worst_ate_source_index,
            worst_rpe_source_index,
            (
                worst_rpe_source_index + 1
                if worst_rpe_source_index is not None
                else None
            ),
        )
        if index is not None
    )
    reduced = _downsample_trajectory_for_web(
        positions=estimated_positions,
        timestamps=timestamps,
        max_points=max_points,
        label=f"matched trajectory {Path(trajectory_path).name}",
        preserve_indices=preserve_indices,
    )
    kept_indices = reduced.kept_indices
    displayed_reference_positions = reference_positions[kept_indices]
    displayed_timestamps = (
        reduced.timestamps
        if reduced.timestamps is not None
        else timestamps[kept_indices]
    )
    ate_errors = full_ate_errors[kept_indices]
    if reduced.positions.shape[0] >= 2:
        rpe_errors = np.linalg.norm(
            np.diff(reduced.positions, axis=0) - np.diff(displayed_reference_positions, axis=0),
            axis=1,
        )
        rpe_timestamps = (
            (displayed_timestamps[:-1] + displayed_timestamps[1:]) / 2.0
        )
    else:
        rpe_errors = np.zeros(0, dtype=float)
        rpe_timestamps = np.zeros(0, dtype=float)

    ate_summary = _summarize_error_series(ate_errors)
    rpe_summary = _summarize_error_series(rpe_errors)
    kept_lookup = {int(index): pos for pos, index in enumerate(kept_indices.tolist())}
    worst_ate_index = (
        kept_lookup.get(worst_ate_source_index)
        if worst_ate_source_index is not None
        else None
    )
    worst_rpe_index = int(np.argmax(rpe_errors)) if rpe_errors.size > 0 else None
    return {
        "mode": "paired",
        "estimated_filename": Path(trajectory_path).name,
        "reference_filename": Path(trajectory_reference_path).name,
        "estimated_positions": reduced.positions.flatten().tolist(),
        "reference_positions": displayed_reference_positions.flatten().tolist(),
        "estimated_pose_count": int(estimated_positions.shape[0]),
        "reference_pose_count": int(reference_positions.shape[0]),
        "displayed_estimated_pose_count": int(reduced.reduced_points),
        "displayed_reference_pose_count": int(displayed_reference_positions.shape[0]),
        "timestamps": displayed_timestamps.tolist(),
        "alignment": result["alignment"],
        "matching": {
            "matched_poses": result["matching"]["matched_poses"],
            "coverage_ratio": result["matching"]["coverage_ratio"],
        },
        "ate": {"rmse": ate_summary["rmse"], "max": ate_summary["max"]},
        "rpe": {"rmse": rpe_summary["rmse"], "max": rpe_summary["max"]},
        "drift": {
            "endpoint": result["drift"]["endpoint"],
        },
        "ate_errors": ate_errors.tolist(),
        "rpe_timestamps": rpe_timestamps.tolist(),
        "rpe_errors": rpe_errors.tolist(),
        "error_stats": ate_summary,
        "worst_ate_index": worst_ate_index,
        "worst_ate_sample": (
            result["worst_ate_samples"][0] if result["worst_ate_samples"] else None
        ),
        "worst_rpe_index": worst_rpe_index,
        "worst_rpe_segment": (
            {
                "start_timestamp": float(displayed_timestamps[worst_rpe_index]),
                "end_timestamp": float(displayed_timestamps[worst_rpe_index + 1]),
                "translation_error": float(rpe_errors[worst_rpe_index]),
            }
            if worst_rpe_index is not None
            else None
        ),
        "sampling": {
            "strategy": reduced.strategy,
            "design": reduced.design,
            "display_budget": _trajectory_display_budget(max_points, estimated_positions.shape[0]),
            "reduction_ratio": reduced.reduction_ratio,
            "preserve_indices": list(preserve_indices),
        },
    }


def serve(
    paths: list[str],
    port: int = 8080,
    max_points: int = 2_000_000,
    open_browser: bool = True,
    heatmap: bool = False,
    trajectory_path: str | None = None,
    trajectory_reference_path: str | None = None,
    trajectory_max_time_delta: float = 0.05,
    trajectory_align_origin: bool = False,
    trajectory_align_rigid: bool = False,
) -> None:
    """Start a web viewer for point cloud(s).

    Args:
        paths: Point cloud file paths.
        port: HTTP port.
        max_points: Max points to display (downsample if larger).
        open_browser: Auto-open browser.
        heatmap: If True, color the first point cloud by distance to the second.
        trajectory_path: Optional estimated trajectory to overlay.
        trajectory_reference_path: Optional reference trajectory for aligned overlay.
        trajectory_max_time_delta: Matching tolerance for paired trajectory overlays.
        trajectory_align_origin: Apply origin alignment to trajectory overlay.
        trajectory_align_rigid: Apply rigid alignment to trajectory overlay.
    """
    data, chunk_payloads = _prepare_viewer_bundle(
        paths,
        max_points=max_points,
        heatmap=heatmap,
        trajectory_path=trajectory_path,
        trajectory_reference_path=trajectory_reference_path,
        trajectory_max_time_delta=trajectory_max_time_delta,
        trajectory_align_origin=trajectory_align_origin,
        trajectory_align_rigid=trajectory_align_rigid,
    )
    data_json = json.dumps(data)

    handler = _make_handler(_VIEWER_HTML, data_json, chunk_payloads)
    server = HTTPServer(('0.0.0.0', port), handler)

    url = f"http://localhost:{port}"
    logger.info("Serving at %s", url)
    print(f"CloudAnalyzer Web Viewer: {url}")
    print("Press Ctrl+C to stop.")

    if open_browser:
        threading.Timer(0.5, lambda: webbrowser.open(url)).start()

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
        server.server_close()


def export_static_bundle(
    paths: list[str],
    output_dir: str,
    max_points: int = 2_000_000,
    heatmap: bool = False,
    trajectory_path: str | None = None,
    trajectory_reference_path: str | None = None,
    trajectory_max_time_delta: float = 0.05,
    trajectory_align_origin: bool = False,
    trajectory_align_rigid: bool = False,
) -> dict:
    """Write a static browser bundle that can be served from GitHub Pages or any static host."""

    data, chunk_payloads = _prepare_viewer_bundle(
        paths,
        max_points=max_points,
        heatmap=heatmap,
        trajectory_path=trajectory_path,
        trajectory_reference_path=trajectory_reference_path,
        trajectory_max_time_delta=trajectory_max_time_delta,
        trajectory_align_origin=trajectory_align_origin,
        trajectory_align_rigid=trajectory_align_rigid,
    )

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)
    (root / "index.html").write_text(_VIEWER_HTML)
    (root / "data.json").write_text(json.dumps(data, indent=2))

    for relative_path, payload in chunk_payloads.items():
        destination = root / relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(payload)

    exported_files = 2 + len(chunk_payloads)
    return {
        "output_dir": str(root),
        "index_html": str(root / "index.html"),
        "data_json": str(root / "data.json"),
        "chunk_count": len(chunk_payloads),
        "exported_files": exported_files,
        "viewer_mode": data["viewer_mode"],
        "display_points": data["display_points"],
    }
