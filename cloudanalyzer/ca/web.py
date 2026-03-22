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
  #controls select { background: #333; color: #fff; border: 1px solid #555; padding: 2px 4px; }
</style>
</head>
<body>
<div id="info">Loading...</div>
<div id="controls">
  <label>Point Size <input type="range" id="ptSize" min="0.5" max="10" step="0.5" value="2"></label>
  <label>Color <select id="colorMode">
    <option value="height">Height</option>
    <option value="distance">Distance from center</option>
    <option value="white">White</option>
  </select></label>
  <label>BG <select id="bgColor">
    <option value="#1a1a2e">Dark</option>
    <option value="#ffffff">White</option>
    <option value="#000000">Black</option>
  </select></label>
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

let pointCloud;

async function loadPoints() {
  const resp = await fetch('/data.json');
  const data = await resp.json();
  const positions = new Float32Array(data.positions);
  const n = positions.length / 3;

  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.BufferAttribute(positions, 3));

  // Compute bounds
  let minX=Infinity,minY=Infinity,minZ=Infinity,maxX=-Infinity,maxY=-Infinity,maxZ=-Infinity;
  for (let i=0; i<n; i++) {
    const x=positions[i*3], y=positions[i*3+1], z=positions[i*3+2];
    if(x<minX)minX=x; if(y<minY)minY=y; if(z<minZ)minZ=z;
    if(x>maxX)maxX=x; if(y>maxY)maxY=y; if(z>maxZ)maxZ=z;
  }
  const cx=(minX+maxX)/2, cy=(minY+maxY)/2, cz=(minZ+maxZ)/2;
  const extent = Math.max(maxX-minX, maxY-minY, maxZ-minZ);

  // Center points
  for (let i=0; i<n; i++) {
    positions[i*3] -= cx;
    positions[i*3+1] -= cy;
    positions[i*3+2] -= cz;
  }
  geometry.attributes.position.needsUpdate = true;

  // Color by height
  applyColor(geometry, positions, n, minZ-cz, maxZ-cz, 'height');

  const material = new THREE.PointsMaterial({ size: 2, vertexColors: true, sizeAttenuation: true });
  pointCloud = new THREE.Points(geometry, material);
  scene.add(pointCloud);

  camera.position.set(0, 0, extent * 0.8);
  controls.target.set(0, 0, 0);
  controls.update();

  document.getElementById('info').innerHTML =
    `<b>CloudAnalyzer Web Viewer</b><br>` +
    `Points: ${n.toLocaleString()}<br>` +
    `Extent: ${extent.toFixed(1)}m<br>` +
    `File: ${data.filename}`;

  // Store for recolor
  window._positions = positions;
  window._n = n;
  window._minZ = minZ-cz;
  window._maxZ = maxZ-cz;
}

function applyColor(geom, pos, n, minZ, maxZ, mode) {
  const colors = new Float32Array(n * 3);
  const rangeZ = maxZ - minZ || 1;
  for (let i=0; i<n; i++) {
    let t;
    if (mode === 'height') {
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
  geom.setAttribute('color', new THREE.BufferAttribute(colors, 3));
}

loadPoints();

// Controls
document.getElementById('ptSize').addEventListener('input', e => {
  if (pointCloud) pointCloud.material.size = parseFloat(e.target.value);
});
document.getElementById('colorMode').addEventListener('change', e => {
  if (!pointCloud) return;
  applyColor(pointCloud.geometry, window._positions, window._n, window._minZ, window._maxZ, e.target.value);
  pointCloud.geometry.attributes.color.needsUpdate = true;
});
document.getElementById('bgColor').addEventListener('change', e => {
  scene.background = new THREE.Color(e.target.value);
});

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


def serve(
    paths: list[str],
    port: int = 8080,
    max_points: int = 2_000_000,
    open_browser: bool = True,
) -> None:
    """Start a web viewer for point cloud(s).

    Args:
        paths: Point cloud file paths.
        port: HTTP port.
        max_points: Max points to display (downsample if larger).
        open_browser: Auto-open browser.
    """
    # Load and merge
    merged = o3d.geometry.PointCloud()
    for path in paths:
        pcd = load_point_cloud(path)
        merged += pcd

    total = len(merged.points)
    logger.info("Loaded %d points from %d file(s)", total, len(paths))

    # Downsample if needed
    if total > max_points:
        ratio = max_points / total
        voxel = 0.01
        while len(merged.points) > max_points:
            merged = merged.voxel_down_sample(voxel)
            voxel *= 1.5
        logger.info("Downsampled to %d points for web display", len(merged.points))

    points = np.asarray(merged.points).flatten().tolist()
    data = {
        "positions": points,
        "filename": ", ".join(Path(p).name for p in paths),
    }
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
