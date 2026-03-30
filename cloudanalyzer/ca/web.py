"""Web-based 3D point cloud viewer using Three.js."""

import json
import threading
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path
from typing import Optional

import numpy as np
import open3d as o3d

from ca.io import load_point_cloud
from ca.log import logger
from ca.metrics import compute_nn_distance
from ca.trajectory import evaluate_trajectory, load_trajectory

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
  #trajectoryInspection {
    margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(203, 213, 225, 0.18);
    font-size: 12px; line-height: 1.5; color: #cbd5e1;
    max-width: 260px;
  }
  #trajectoryInspectionTitle { color: #f8fafc; font-weight: 700; margin-bottom: 4px; }
  #trajectoryInspectionHint { color: #94a3b8; }
  #trajectoryInspectionBody { margin-top: 4px; }
  #trajectoryTimeline {
    margin-top: 10px; padding-top: 10px; border-top: 1px solid rgba(203, 213, 225, 0.18);
    font-size: 12px; line-height: 1.4; color: #cbd5e1;
    max-width: 260px;
  }
  #trajectoryTimelineTitle { color: #f8fafc; font-weight: 700; margin-bottom: 4px; }
  #trajectoryTimelineHint { color: #94a3b8; margin-bottom: 6px; }
  #trajectoryTimelineChart { display: grid; gap: 10px; }
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
let viewerData;
let defaultCameraPosition;
let defaultControlTarget;
let trajectorySelection = null;

async function loadPoints() {
  const resp = await fetch('/data.json');
  const data = await resp.json();
  viewerData = data;
  const positions = new Float32Array(data.positions);
  const referencePositions = data.reference_positions ? new Float32Array(data.reference_positions) : null;
  const trajectory = data.trajectory || null;
  const trajectoryPositions = trajectory && trajectory.estimated_positions
    ? new Float32Array(trajectory.estimated_positions)
    : null;
  const trajectoryReferencePositions = trajectory && trajectory.reference_positions
    ? new Float32Array(trajectory.reference_positions)
    : null;
  const n = positions.length / 3;
  const heatmapOption = document.querySelector('#colorMode option[value="heatmap"]');
  if (!data.distances && heatmapOption) {
    heatmapOption.remove();
  }

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

  // Compute bounds
  let minX=Infinity,minY=Infinity,minZ=Infinity,maxX=-Infinity,maxY=-Infinity,maxZ=-Infinity;
  for (let i=0; i<n; i++) {
    const x=positions[i*3], y=positions[i*3+1], z=positions[i*3+2];
    if(x<minX)minX=x; if(y<minY)minY=y; if(z<minZ)minZ=z;
    if(x>maxX)maxX=x; if(y>maxY)maxY=y; if(z>maxZ)maxZ=z;
  }
  if (referencePositions) {
    for (let i=0; i<referencePositions.length / 3; i++) {
      const x=referencePositions[i*3], y=referencePositions[i*3+1], z=referencePositions[i*3+2];
      if(x<minX)minX=x; if(y<minY)minY=y; if(z<minZ)minZ=z;
      if(x>maxX)maxX=x; if(y>maxY)maxY=y; if(z>maxZ)maxZ=z;
    }
  }
  if (trajectoryPositions) {
    for (let i=0; i<trajectoryPositions.length / 3; i++) {
      const x=trajectoryPositions[i*3], y=trajectoryPositions[i*3+1], z=trajectoryPositions[i*3+2];
      if(x<minX)minX=x; if(y<minY)minY=y; if(z<minZ)minZ=z;
      if(x>maxX)maxX=x; if(y>maxY)maxY=y; if(z>maxZ)maxZ=z;
    }
  }
  if (trajectoryReferencePositions) {
    for (let i=0; i<trajectoryReferencePositions.length / 3; i++) {
      const x=trajectoryReferencePositions[i*3], y=trajectoryReferencePositions[i*3+1], z=trajectoryReferencePositions[i*3+2];
      if(x<minX)minX=x; if(y<minY)minY=y; if(z<minZ)minZ=z;
      if(x>maxX)maxX=x; if(y>maxY)maxY=y; if(z>maxZ)maxZ=z;
    }
  }
  const cx=(minX+maxX)/2, cy=(minY+maxY)/2, cz=(minZ+maxZ)/2;
  const extent = Math.max(maxX-minX, maxY-minY, maxZ-minZ);
  window._sceneExtent = extent;

  // Center points
  for (let i=0; i<n; i++) {
    positions[i*3] -= cx;
    positions[i*3+1] -= cy;
    positions[i*3+2] -= cz;
  }
  geometry.attributes.position.needsUpdate = true;
  if (referencePositions) {
    for (let i=0; i<referencePositions.length / 3; i++) {
      referencePositions[i*3] -= cx;
      referencePositions[i*3+1] -= cy;
      referencePositions[i*3+2] -= cz;
    }
  }
  if (trajectoryPositions) {
    for (let i=0; i<trajectoryPositions.length / 3; i++) {
      trajectoryPositions[i*3] -= cx;
      trajectoryPositions[i*3+1] -= cy;
      trajectoryPositions[i*3+2] -= cz;
    }
  }
  if (trajectoryReferencePositions) {
    for (let i=0; i<trajectoryReferencePositions.length / 3; i++) {
      trajectoryReferencePositions[i*3] -= cx;
      trajectoryReferencePositions[i*3+1] -= cy;
      trajectoryReferencePositions[i*3+2] -= cz;
    }
  }

  const initialMode = data.distances ? 'heatmap' : 'height';
  document.getElementById('colorMode').value = initialMode;

  const material = new THREE.PointsMaterial({ size: 2, vertexColors: true, sizeAttenuation: true });
  pointCloud = new THREE.Points(geometry, material);
  scene.add(pointCloud);
  if (referencePositions) {
    const referenceGeometry = new THREE.BufferGeometry();
    referenceGeometry.setAttribute('position', new THREE.BufferAttribute(referencePositions, 3));
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
    trajectoryGeometry.setAttribute('position', new THREE.BufferAttribute(trajectoryPositions, 3));
    const trajectoryMaterial = new THREE.LineBasicMaterial({ color: '#f59e0b' });
    trajectoryLine = new THREE.Line(trajectoryGeometry, trajectoryMaterial);
    scene.add(trajectoryLine);
    document.getElementById('trajectoryToggleWrap').style.display = 'block';
  }
  if (trajectoryReferencePositions) {
    const trajectoryReferenceGeometry = new THREE.BufferGeometry();
    trajectoryReferenceGeometry.setAttribute('position', new THREE.BufferAttribute(trajectoryReferencePositions, 3));
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
  window._minZ = minZ-cz;
  window._maxZ = maxZ-cz;
  window._distances = data.distances || null;
  window._distanceStats = data.distance_stats || null;
  window._sourceBasePositions = positions.slice();
  window._sourceTotalPoints = n;

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
}

function makeInfo(data, n, extent) {
  let trajectoryInfo = '';
  if (data.trajectory) {
    if (data.trajectory.mode === 'paired') {
      trajectoryInfo =
        `<br>Trajectory matched: ${data.trajectory.matching.matched_poses.toLocaleString()} (${(data.trajectory.matching.coverage_ratio * 100).toFixed(1)}%)` +
        `<br>Trajectory align: ${data.trajectory.alignment.mode}` +
        `<br>Trajectory ATE RMSE: ${data.trajectory.ate.rmse.toFixed(4)}` +
        `<br>Trajectory RPE RMSE: ${data.trajectory.rpe.rmse.toFixed(4)}` +
        `<br>Trajectory worst ATE: ${data.trajectory.error_stats.max.toFixed(4)}` +
        `<br>Trajectory worst RPE: ${data.trajectory.rpe.max.toFixed(4)}`;
    } else {
      trajectoryInfo =
        `<br>Trajectory poses: ${data.trajectory.estimated_pose_count.toLocaleString()}` +
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
      trajectoryInfo;
  }

  return `<b>CloudAnalyzer Web Viewer</b><br>` +
    `Points: ${displayPoints}<br>` +
    `Extent: ${extent.toFixed(1)}m<br>` +
    `File: ${data.filename}` +
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
      if (!(target instanceof HTMLElement)) {
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
}

function onViewerClick(event) {
  if (!viewerData || !viewerData.trajectory || viewerData.trajectory.mode !== 'paired') {
    return;
  }

  const rect = renderer.domElement.getBoundingClientRect();
  pointer.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
  pointer.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;
  raycaster.setFromCamera(pointer, camera);

  const pickTargets = [];
  if (trajectoryWorstMarker && trajectoryWorstMarker.visible) {
    pickTargets.push(trajectoryWorstMarker);
  }
  if (trajectoryWorstSegment && trajectoryWorstSegment.visible) {
    pickTargets.push(trajectoryWorstSegment);
  }
  if (pickTargets.length === 0) {
    trajectorySelection = null;
    renderTrajectoryTimeline();
    clearTrajectoryInspection();
    return;
  }

  const intersections = raycaster.intersectObjects(pickTargets, false);
  if (intersections.length === 0) {
    trajectorySelection = null;
    renderTrajectoryTimeline();
    clearTrajectoryInspection();
    return;
  }

  const object = intersections[0].object;
  if (object.userData.inspectType === 'worst-ate' && viewerData.trajectory.worst_ate_sample) {
    selectTrajectoryFeature('ate', viewerData.trajectory.worst_ate_index, true);
    return;
  }
  if (object.userData.inspectType === 'worst-rpe' && viewerData.trajectory.worst_rpe_segment) {
    selectTrajectoryFeature('rpe', viewerData.trajectory.worst_rpe_index, true);
    return;
  }
  trajectorySelection = null;
  renderTrajectoryTimeline();
  clearTrajectoryInspection();
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

  if (baseDistances && threshold > 0) {
    let kept = 0;
    for (let i=0; i<baseDistances.length; i++) {
      if (baseDistances[i] >= threshold) kept++;
    }

    const filteredPositions = new Float32Array(kept * 3);
    const filteredDistances = new Float32Array(kept);
    let out = 0;
    for (let i=0; i<baseDistances.length; i++) {
      if (baseDistances[i] < threshold) continue;
      filteredPositions[out*3] = basePositions[i*3];
      filteredPositions[out*3+1] = basePositions[i*3+1];
      filteredPositions[out*3+2] = basePositions[i*3+2];
      filteredDistances[out] = baseDistances[i];
      out++;
    }

    positions = filteredPositions;
    distances = filteredDistances;
    visibleCount = kept;
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

  pointCloud.geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  pointCloud.geometry.setAttribute('color', new THREE.BufferAttribute(colors, 3));
  if (visibleCount > 0) {
    pointCloud.geometry.computeBoundingSphere();
  }
  updateThresholdInfo(threshold, visibleCount);
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


def _make_handler(html: str, data_json: str):
    """Create a custom HTTP handler serving the viewer and data."""
    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == '/' or self.path == '/index.html':
                self.send_response(200)
                self.send_header('Content-Type', 'text/html')
                self.end_headers()
                self.wfile.write(html.encode())
            elif self.path == '/data.json':
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.end_headers()
                self.wfile.write(data_json.encode())
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
    """Downsample a point cloud for browser display if needed."""
    if len(pcd.points) <= max_points:
        return pcd

    reduced = pcd
    voxel = 0.01
    while len(reduced.points) > max_points:
        reduced = reduced.voxel_down_sample(voxel)
        voxel *= 1.5

    logger.info("Downsampled %s to %d points for web display", label, len(reduced.points))
    return reduced


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
            max_time_delta=trajectory_max_time_delta,
            align_origin=trajectory_align_origin,
            align_rigid=trajectory_align_rigid,
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
        distances = compute_nn_distance(display_source, target)

        data = {
            "positions": np.asarray(display_source.points).flatten().tolist(),
            "reference_positions": np.asarray(display_target.points).flatten().tolist(),
            "filename": Path(source_path).name,
            "viewer_mode": "heatmap",
            "source_filename": Path(source_path).name,
            "target_filename": Path(target_path).name,
            "display_points": len(display_source.points),
            "original_points": len(source.points),
            "reference_points": len(target.points),
            "reference_display_points": len(display_target.points),
            "distances": distances.tolist(),
            "distance_stats": {
                "mean": float(np.mean(distances)),
                "max": float(np.max(distances)),
                "min": float(np.min(distances)),
            },
        }
        if trajectory_data is not None:
            data["trajectory"] = trajectory_data
        return data

    merged = o3d.geometry.PointCloud()
    for path in paths:
        merged += load_point_cloud(path)

    total = len(merged.points)
    logger.info("Loaded %d points from %d file(s)", total, len(paths))
    display_cloud = _downsample_for_web(merged, max_points, "merged cloud")

    data = {
        "positions": np.asarray(display_cloud.points).flatten().tolist(),
        "filename": ", ".join(Path(p).name for p in paths),
        "viewer_mode": "standard",
        "display_points": len(display_cloud.points),
        "original_points": total,
    }
    if trajectory_data is not None:
        data["trajectory"] = trajectory_data
    return data


def _prepare_trajectory_viewer_data(
    trajectory_path: str,
    trajectory_reference_path: str | None = None,
    max_time_delta: float = 0.05,
    align_origin: bool = False,
    align_rigid: bool = False,
) -> dict:
    """Prepare trajectory overlay payload for the web viewer."""
    if trajectory_reference_path is None:
        trajectory = load_trajectory(trajectory_path)
        positions = np.asarray(trajectory["positions"], dtype=float)
        return {
            "mode": "single",
            "estimated_filename": Path(trajectory_path).name,
            "estimated_positions": positions.flatten().tolist(),
            "estimated_pose_count": int(positions.shape[0]),
            "reference_positions": None,
            "reference_pose_count": None,
            "timestamps": None,
            "ate_errors": None,
            "rpe_timestamps": None,
            "rpe_errors": None,
            "worst_ate_index": None,
            "worst_ate_sample": None,
            "worst_rpe_index": None,
            "worst_rpe_segment": None,
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
    ate_errors = np.asarray(matched["ate_errors"], dtype=float)
    rpe_errors = np.asarray(result["error_series"]["rpe_translation"], dtype=float)
    worst_ate_index = int(np.argmax(ate_errors)) if ate_errors.size > 0 else None
    worst_rpe_index = int(np.argmax(rpe_errors)) if rpe_errors.size > 0 else None
    return {
        "mode": "paired",
        "estimated_filename": Path(trajectory_path).name,
        "reference_filename": Path(trajectory_reference_path).name,
        "estimated_positions": estimated_positions.flatten().tolist(),
        "reference_positions": reference_positions.flatten().tolist(),
        "estimated_pose_count": int(estimated_positions.shape[0]),
        "reference_pose_count": int(reference_positions.shape[0]),
        "timestamps": matched["timestamps"],
        "alignment": result["alignment"],
        "matching": {
            "matched_poses": result["matching"]["matched_poses"],
            "coverage_ratio": result["matching"]["coverage_ratio"],
        },
        "ate": {
            "rmse": result["ate"]["rmse"],
            "max": result["ate"]["max"],
        },
        "rpe": {
            "rmse": result["rpe_translation"]["rmse"],
            "max": result["rpe_translation"]["max"],
        },
        "drift": {
            "endpoint": result["drift"]["endpoint"],
        },
        "ate_errors": ate_errors.tolist(),
        "rpe_timestamps": result["error_series"]["rpe_timestamps"],
        "rpe_errors": rpe_errors.tolist(),
        "error_stats": {
            "mean": float(np.mean(ate_errors)),
            "max": float(np.max(ate_errors)),
            "min": float(np.min(ate_errors)),
        },
        "worst_ate_index": worst_ate_index,
        "worst_ate_sample": (
            result["worst_ate_samples"][0] if result["worst_ate_samples"] else None
        ),
        "worst_rpe_index": worst_rpe_index,
        "worst_rpe_segment": (
            result["worst_rpe_segments"][0] if result["worst_rpe_segments"] else None
        ),
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
    data = _prepare_viewer_data(
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

    handler = _make_handler(_VIEWER_HTML, data_json)
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
