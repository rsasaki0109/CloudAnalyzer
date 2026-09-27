// three.js viewport: level-of-detail point clouds, Z-up orbit camera.

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { EdlPass } from "./edl";
import { type LodNode, selectNodes } from "./lod";

interface LodCloud {
  id: number;
  nodes: LodNode[];
  /** Interleaved xyz / rgb in octree order; node objects are views into these. */
  positions: Float32Array;
  colors: Uint8Array;
  material: THREE.PointsMaterial;
  group: THREE.Group;
  /** Node index -> drawable, created on first use. */
  objects: Map<number, THREE.Points>;
  visible: boolean;
}

export interface PickHit {
  cloudId: number;
  /** Point index in the cloud's octree order. */
  index: number;
  /** Position in render (shifted) coordinates. */
  position: THREE.Vector3;
}

export class Viewer {
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  private readonly renderer: THREE.WebGLRenderer;
  private readonly controls: OrbitControls;
  private readonly clouds = new Map<number, LodCloud>();
  private readonly meshes = new Map<number, THREE.Mesh>();
  private pointSize = 2;
  private pointBudget = 3_000_000;
  private needsRender = true;
  private needsLod = true;
  /** Called with the number of points drawn after each LOD update. */
  onDrawn: (points: number) => void = () => {};
  /** Called for a click (or tap) that was not a camera drag. */
  onClick: (clientX: number, clientY: number) => void = () => {};
  /** Called for a double click or double tap, after both clicks. */
  onDoubleClick: (clientX: number, clientY: number) => void = () => {};
  /** Called after every rendered frame, e.g. to move HTML overlays. */
  onAfterRender: () => void = () => {};
  private readonly annotations = new THREE.Group();
  /** Clipping box in render coordinates, or null when clipping is off. */
  private clip: THREE.Box3 | null = null;
  private readonly clipPlanes = Array.from({ length: 6 }, () => new THREE.Plane());
  private readonly clipHelper = new THREE.Box3Helper(new THREE.Box3(), 0xffd54f);
  private readonly edl = new EdlPass();
  private edlEnabled = true;

  constructor(private readonly container: HTMLElement) {
    THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
    this.renderer = new THREE.WebGLRenderer({ antialias: false });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    this.renderer.localClippingEnabled = true;
    container.appendChild(this.renderer.domElement);

    this.camera = new THREE.PerspectiveCamera(50, 1, 0.01, 1e6);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(10, -10, 10);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.screenSpacePanning = true;
    this.controls.addEventListener("change", () => this.requestRender(true));

    // Soft ambient light plus a headlight that follows the camera, so meshes
    // are always lit from the viewing direction.
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x445566, 1.2));
    const headlight = new THREE.DirectionalLight(0xffffff, 1.6);
    headlight.position.set(0, 0, 1);
    this.camera.add(headlight);
    this.scene.add(this.camera);

    const axes = new THREE.AxesHelper(1);
    axes.name = "axes";
    this.scene.add(axes);

    this.clipHelper.visible = false;
    this.scene.add(this.clipHelper);
    this.annotations.renderOrder = 1;
    this.scene.add(this.annotations);

    // A click is a press and release of a single pointer without moving
    // the camera; a second finger (pinch, two-finger pan) cancels it.
    let down: { x: number; y: number; time: number } | null = null;
    let last: { x: number; y: number; time: number } | null = null;
    const active = new Set<number>();
    const canvas = this.renderer.domElement;
    canvas.addEventListener("pointerdown", (e) => {
      active.add(e.pointerId);
      down =
        e.button === 0 && active.size === 1 ? { x: e.clientX, y: e.clientY, time: performance.now() } : null;
    });
    const release = (e: PointerEvent) => {
      active.delete(e.pointerId);
    };
    canvas.addEventListener("pointercancel", (e) => {
      release(e);
      down = null;
    });
    canvas.addEventListener("pointerup", (e) => {
      release(e);
      if (!down || e.button !== 0) return;
      const now = performance.now();
      // Fingers wobble more than a mouse.
      const slop = e.pointerType === "touch" ? 12 : 5;
      const moved = Math.hypot(e.clientX - down.x, e.clientY - down.y);
      const held = now - down.time;
      down = null;
      if (moved >= slop || held > 600) return;
      this.onClick(e.clientX, e.clientY);
      if (last && now - last.time < 350 && Math.hypot(e.clientX - last.x, e.clientY - last.y) < 3 * slop) {
        last = null;
        this.onDoubleClick(e.clientX, e.clientY);
      } else {
        last = { x: e.clientX, y: e.clientY, time: now };
      }
    });

    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
    this.loop();
  }

  setBackground(color: string): void {
    this.scene.background = new THREE.Color(color);
    this.requestRender();
  }

  /** Schedule a redraw; `lod` also re-selects which octree nodes to draw. */
  requestRender(lod = false): void {
    this.needsRender = true;
    this.needsLod ||= lod;
  }

  private resize(): void {
    const { clientWidth: w, clientHeight: h } = this.container;
    if (w === 0 || h === 0) return;
    this.renderer.setSize(w, h, false);
    const size = this.renderer.getDrawingBufferSize(new THREE.Vector2());
    this.edl.setSize(size.x, size.y);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.requestRender(true);
  }

  private loop = (): void => {
    requestAnimationFrame(this.loop);
    // Damping keeps emitting "change" while the camera settles.
    this.controls.update();
    if (!this.needsRender) return;
    this.needsRender = false;
    if (this.needsLod) {
      this.needsLod = false;
      this.updateLod();
    }
    if (this.edlEnabled) this.edl.render(this.renderer, this.scene, this.camera);
    else this.renderer.render(this.scene, this.camera);
    this.onAfterRender();
  };

  private updateLod(): void {
    this.camera.updateMatrixWorld();
    const sources = [...this.clouds.values()].filter((c) => c.visible);
    const selection = selectNodes(
      sources,
      this.camera,
      this.renderer.domElement.height,
      this.pointBudget,
      1,
      this.clip,
    );
    for (const cloud of this.clouds.values()) {
      const wanted = new Set(selection.nodes.get(cloud.id) ?? []);
      for (const [index, object] of cloud.objects) {
        object.visible = wanted.has(index);
      }
      for (const index of wanted) {
        if (!cloud.objects.has(index)) this.createNode(cloud, index);
      }
      this.evict(cloud, wanted);
    }
    this.onDrawn(selection.points);
  }

  private createNode(cloud: LodCloud, index: number): void {
    const { start, count } = cloud.nodes[index];
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute(
      "position",
      new THREE.BufferAttribute(cloud.positions.subarray(start * 3, (start + count) * 3), 3),
    );
    geometry.setAttribute(
      "color",
      new THREE.BufferAttribute(cloud.colors.subarray(start * 4, (start + count) * 4), 4, true),
    );
    geometry.boundingBox = cloud.nodes[index].box.clone();
    geometry.boundingSphere = geometry.boundingBox.getBoundingSphere(new THREE.Sphere());
    const points = new THREE.Points(geometry, cloud.material);
    cloud.objects.set(index, points);
    cloud.group.add(points);
  }

  /** Free GPU buffers of hidden nodes once a cloud holds much more than the budget. */
  private evict(cloud: LodCloud, wanted: Set<number>): void {
    let resident = 0;
    for (const index of cloud.objects.keys()) resident += cloud.nodes[index].count;
    if (resident <= 2 * this.pointBudget) return;
    for (const [index, object] of cloud.objects) {
      if (wanted.has(index)) continue;
      cloud.group.remove(object);
      object.geometry.dispose();
      cloud.objects.delete(index);
      resident -= cloud.nodes[index].count;
      if (resident <= this.pointBudget) break;
    }
  }

  add(id: number, positions: Float32Array, colors: Uint8Array, nodes: LodNode[]): void {
    const material = new THREE.PointsMaterial({
      size: this.pointSize,
      sizeAttenuation: false,
      vertexColors: true,
      // Points with alpha 0 (e.g. a hidden class) are discarded.
      alphaTest: 0.5,
      clippingPlanes: this.clip ? this.clipPlanes : null,
    });
    const group = new THREE.Group();
    this.scene.add(group);
    this.clouds.set(id, {
      id,
      nodes,
      positions,
      colors,
      material,
      group,
      objects: new Map(),
      visible: true,
    });
    this.requestRender(true);
  }

  /** Add a triangle mesh (positions in render coordinates) with a solid color. */
  addMesh(id: number, positions: Float32Array, indices: Uint32Array, color: [number, number, number]): void {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setIndex(new THREE.BufferAttribute(indices, 1));
    geometry.computeVertexNormals();
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();
    const material = new THREE.MeshStandardMaterial({
      color: new THREE.Color(...color.map((c) => c / 255)),
      roughness: 0.85,
      metalness: 0,
      side: THREE.DoubleSide,
      flatShading: true,
      // Push the surface back a little so points lying on it stay visible.
      polygonOffset: true,
      polygonOffsetFactor: 1,
      polygonOffsetUnits: 1,
      clippingPlanes: this.clip ? this.clipPlanes : null,
    });
    const mesh = new THREE.Mesh(geometry, material);
    this.meshes.set(id, mesh);
    this.scene.add(mesh);
    this.requestRender();
  }

  setMeshColor(id: number, color: [number, number, number]): void {
    const mesh = this.meshes.get(id);
    if (!mesh) return;
    (mesh.material as THREE.MeshStandardMaterial).color.setRGB(...(color.map((c) => c / 255) as [number, number, number]));
    this.requestRender();
  }

  remove(id: number): void {
    const mesh = this.meshes.get(id);
    if (mesh) {
      this.scene.remove(mesh);
      mesh.geometry.dispose();
      (mesh.material as THREE.Material).dispose();
      this.meshes.delete(id);
      this.requestRender();
      return;
    }
    const cloud = this.clouds.get(id);
    if (!cloud) return;
    this.scene.remove(cloud.group);
    for (const object of cloud.objects.values()) object.geometry.dispose();
    cloud.material.dispose();
    this.clouds.delete(id);
    this.requestRender(true);
  }

  /** Replace a cloud's colors (interleaved rgba in octree order; alpha 0 hides a point). */
  setColors(id: number, colors: Uint8Array): void {
    const cloud = this.clouds.get(id);
    if (!cloud) return;
    cloud.colors = colors;
    for (const [index, object] of cloud.objects) {
      const { start, count } = cloud.nodes[index];
      object.geometry.setAttribute(
        "color",
        new THREE.BufferAttribute(colors.subarray(start * 4, (start + count) * 4), 4, true),
      );
    }
    this.requestRender();
  }

  setVisible(id: number, visible: boolean): void {
    const mesh = this.meshes.get(id);
    if (mesh) {
      mesh.visible = visible;
      this.requestRender();
      return;
    }
    const cloud = this.clouds.get(id);
    if (!cloud) return;
    cloud.visible = visible;
    cloud.group.visible = visible;
    this.requestRender(true);
  }

  setPointSize(size: number): void {
    this.pointSize = size;
    for (const cloud of this.clouds.values()) cloud.material.size = size;
    this.requestRender();
  }

  /** Eye-Dome Lighting on/off and strength (1 is the default look). */
  setEdl(enabled: boolean, strength = 1): void {
    this.edlEnabled = enabled;
    this.edl.strength = strength;
    this.requestRender();
  }

  setPointBudget(points: number): void {
    this.pointBudget = points;
    this.requestRender(true);
  }

  /**
   * The front-most drawn point within `radiusPx` CSS pixels of a screen
   * position, or null. Only nodes currently drawn are searched, so the
   * result is always a point the user can see.
   */
  pick(clientX: number, clientY: number, radiusPx = 6): PickHit | null {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1,
    );
    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(ndc, this.camera);
    const { origin, direction } = raycaster.ray;
    // Allowed distance from the ray grows linearly with depth.
    const pixelsPerUnit = rect.height / (2 * Math.tan(THREE.MathUtils.degToRad(this.camera.fov) / 2));
    const slope = radiusPx / pixelsPerUnit;
    const slope2 = slope * slope;
    const box = new THREE.Box3();
    // Collect every drawn point inside the pick cone, then take the one
    // closest to the cursor among those at (nearly) the front-most depth.
    const candidates: { cloud: LodCloud; index: number; depth: number; perp2: number }[] = [];
    const tmp = new THREE.Vector3();
    for (const cloud of this.clouds.values()) {
      if (!cloud.visible) continue;
      for (const [index, object] of cloud.objects) {
        if (!object.visible) continue;
        const node = cloud.nodes[index];
        const reach = node.box.distanceToPoint(origin) + node.box.max.distanceTo(node.box.min);
        box.copy(node.box).expandByScalar(slope * reach);
        if (!raycaster.ray.intersectsBox(box)) continue;
        const p = cloud.positions;
        for (let i = node.start, end = node.start + node.count; i < end; i++) {
          const vx = p[i * 3] - origin.x;
          const vy = p[i * 3 + 1] - origin.y;
          const vz = p[i * 3 + 2] - origin.z;
          const depth = vx * direction.x + vy * direction.y + vz * direction.z;
          if (depth <= this.camera.near) continue;
          const perp2 = vx * vx + vy * vy + vz * vz - depth * depth;
          if (perp2 > slope2 * depth * depth) continue;
          if (cloud.colors[i * 4 + 3] === 0) continue; // hidden by a filter
          if (this.clip && !this.clip.containsPoint(tmp.set(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]))) continue;
          candidates.push({ cloud, index: i, depth, perp2 });
        }
      }
    }
    if (candidates.length === 0) return null;
    let front = Number.POSITIVE_INFINITY;
    for (const c of candidates) front = Math.min(front, c.depth);
    let best = candidates[0];
    let bestScore = Number.POSITIVE_INFINITY;
    for (const c of candidates) {
      if (c.depth > front * 1.01) continue;
      // Compare angular offsets so near and far candidates are judged alike.
      const score = c.perp2 / (c.depth * c.depth);
      if (score < bestScore) {
        bestScore = score;
        best = c;
      }
    }
    const p = best.cloud.positions;
    return {
      cloudId: best.cloud.id,
      index: best.index,
      position: new THREE.Vector3(p[best.index * 3], p[best.index * 3 + 1], p[best.index * 3 + 2]),
    };
  }

  /** Replace the markers and line segments drawn on top of the clouds. */
  setAnnotations(
    markers: { position: THREE.Vector3; color: string }[],
    segments: [THREE.Vector3, THREE.Vector3][],
  ): void {
    for (const child of [...this.annotations.children]) {
      this.annotations.remove(child);
      const object = child as THREE.Points | THREE.LineSegments;
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
    }
    if (markers.length) {
      const geometry = new THREE.BufferGeometry().setFromPoints(markers.map((m) => m.position));
      const colors = markers.flatMap((m) => new THREE.Color(m.color).toArray());
      geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
      const material = new THREE.PointsMaterial({
        size: 11,
        sizeAttenuation: false,
        vertexColors: true,
        depthTest: false,
        transparent: true,
      });
      const points = new THREE.Points(geometry, material);
      points.renderOrder = 2;
      this.annotations.add(points);
    }
    if (segments.length) {
      const geometry = new THREE.BufferGeometry().setFromPoints(segments.flat());
      const material = new THREE.LineBasicMaterial({ color: 0x4fc3f7, depthTest: false, transparent: true });
      const lines = new THREE.LineSegments(geometry, material);
      lines.renderOrder = 1;
      this.annotations.add(lines);
    }
    this.requestRender();
  }

  /** CSS pixel position of a render-space point inside the viewport, or null when behind the camera. */
  project(position: THREE.Vector3): { x: number; y: number } | null {
    const v = position.clone().project(this.camera);
    if (v.z < -1 || v.z > 1) return null;
    const { clientWidth: w, clientHeight: h } = this.renderer.domElement;
    return { x: ((v.x + 1) / 2) * w, y: ((1 - v.y) / 2) * h };
  }

  /**
   * Show only what lies inside `box` (render coordinates), or everything
   * when `box` is null. Drawn with GPU clipping planes; octree nodes outside
   * the box are skipped so the point budget goes to what remains.
   */
  setClipBox(box: THREE.Box3 | null): void {
    this.clip = box ? box.clone() : null;
    if (box) {
      const normals = [
        [1, 0, 0],
        [-1, 0, 0],
        [0, 1, 0],
        [0, -1, 0],
        [0, 0, 1],
        [0, 0, -1],
      ];
      normals.forEach(([x, y, z], i) => {
        const normal = new THREE.Vector3(x, y, z);
        // A plane keeps the side its normal points to: n.p + c >= 0.
        const corner = i % 2 === 0 ? box.min : box.max;
        this.clipPlanes[i].setFromNormalAndCoplanarPoint(normal, corner);
      });
      this.clipHelper.box.copy(box);
    }
    this.clipHelper.visible = !!box;
    const planes = box ? this.clipPlanes : null;
    for (const cloud of this.clouds.values()) {
      cloud.material.clippingPlanes = planes;
      cloud.material.needsUpdate = true;
    }
    for (const mesh of this.meshes.values()) {
      const material = mesh.material as THREE.Material;
      material.clippingPlanes = planes;
      material.needsUpdate = true;
    }
    this.requestRender(true);
  }

  /** Bounding box of all visible clouds and meshes (render coordinates). */
  contentBounds(): THREE.Box3 {
    const box = new THREE.Box3();
    for (const cloud of this.clouds.values()) {
      if (cloud.visible && cloud.nodes[0]) box.union(tightBox(cloud));
    }
    for (const mesh of this.meshes.values()) {
      if (mesh.visible && mesh.geometry.boundingBox) box.union(mesh.geometry.boundingBox);
    }
    return box;
  }

  /** Pan so that `point` is at the centre of the view and the orbit pivot. */
  centerOn(point: THREE.Vector3): void {
    const offset = point.clone().sub(this.controls.target);
    this.controls.target.add(offset);
    this.camera.position.add(offset);
    this.controls.update();
    this.requestRender(true);
  }

  /** Camera position and orbit target (render coordinates). */
  getCamera(): { position: THREE.Vector3; target: THREE.Vector3 } {
    return { position: this.camera.position.clone(), target: this.controls.target.clone() };
  }

  /** Place the camera at `position` looking at (and orbiting) `target`. */
  setCamera(position: THREE.Vector3, target: THREE.Vector3): void {
    this.controls.target.copy(target);
    this.camera.position.copy(position);
    const distance = Math.max(position.distanceTo(target), 1e-6);
    const bounds = this.contentBounds();
    const radius = bounds.isEmpty() ? distance : bounds.getBoundingSphere(new THREE.Sphere()).radius;
    this.camera.near = distance / 1000;
    this.camera.far = (distance + 2 * radius) * 10;
    this.camera.updateProjectionMatrix();
    this.scene.getObjectByName("axes")?.scale.setScalar(Math.max(radius, 1e-3) * 0.2);
    this.controls.update();
    this.requestRender(true);
  }

  /** Frame all visible clouds, keeping the current viewing direction. */
  fit(): void {
    const box = this.clip ? this.clip.clone() : this.contentBounds();
    if (box.isEmpty()) return;
    const sphere = box.getBoundingSphere(new THREE.Sphere());
    const radius = Math.max(sphere.radius, 1e-3);
    const direction = this.camera.position.clone().sub(this.controls.target).normalize();
    if (direction.lengthSq() === 0) direction.set(1, -1, 1).normalize();
    const distance = radius / Math.sin(THREE.MathUtils.degToRad(this.camera.fov / 2));
    this.controls.target.copy(sphere.center);
    this.camera.position.copy(sphere.center).addScaledVector(direction, distance);
    this.camera.near = distance / 1000;
    this.camera.far = distance * 100;
    this.camera.updateProjectionMatrix();
    const axes = this.scene.getObjectByName("axes");
    axes?.scale.setScalar(radius * 0.2);
    this.controls.update();
    this.requestRender(true);
  }

  /** Look along a principal axis at the current target. */
  view(direction: THREE.Vector3Like): void {
    const distance = this.camera.position.distanceTo(this.controls.target);
    const dir = new THREE.Vector3().copy(direction).normalize();
    this.camera.position.copy(this.controls.target).addScaledVector(dir, distance);
    this.controls.update();
    this.requestRender(true);
  }
}

const tightBoxes = new WeakMap<LodCloud, THREE.Box3>();

/** Bounding box of the actual points (the octree root is a padded cube). */
function tightBox(cloud: LodCloud): THREE.Box3 {
  let box = tightBoxes.get(cloud);
  if (!box) {
    box = new THREE.Box3().setFromArray(cloud.positions);
    tightBoxes.set(cloud, box);
  }
  return box;
}
