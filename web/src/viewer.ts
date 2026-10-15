// three.js viewport: level-of-detail point clouds, Z-up orbit camera.

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
import { TransformControls } from "three/examples/jsm/controls/TransformControls.js";
import { Line2 } from "three/examples/jsm/lines/Line2.js";
import { LineGeometry } from "three/examples/jsm/lines/LineGeometry.js";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";
import { EdlPass } from "./edl";
import { type LodNode, selectNodes } from "./lod";

interface LodCloud {
  id: number;
  /** Octree nodes, boxes in render coordinates. */
  nodes: LodNode[];
  /** Where `positions` are relative to, in render coordinates (the group's position). */
  offset: THREE.Vector3;
  /** Interleaved xyz / rgb in octree order; node objects are views into these. */
  positions: Float32Array;
  colors: Uint8Array;
  material: THREE.PointsMaterial;
  /** Adaptive size: a material per point size (world units), made on demand. */
  sized: Map<number, THREE.PointsMaterial>;
  group: THREE.Group;
  /** Node index -> drawable, created on first use. */
  objects: Map<number, THREE.Points>;
  visible: boolean;
  /** The file behind a thinned cloud, drawn at full density near the camera (see `setDetail`). */
  detail: Detail | null;
}

/** A chunk of the file behind a thinned cloud. */
interface DetailPart {
  /** Box of all of its points, render coordinates. */
  box: THREE.Box3;
  count: number;
  /** Rough distance between neighbouring points at full density. */
  spacing: number;
  /**
   * Drawables once the points arrived (see `addDetail`), one per slice of
   * the chunk, so a strip-shaped chunk is drawn and counted only where it is
   * in view; with what the app needs to recolor them.
   */
  pieces: DetailPiece[] | null;
  data: unknown;
  /** Frame it was last drawn in, for least-recently-used eviction. */
  used: number;
}

interface DetailPiece {
  object: THREE.Points;
  /** Render coordinates. */
  box: THREE.Box3;
  /** Its points in the chunk's arrays. */
  start: number;
  count: number;
}

interface Detail {
  parts: DetailPart[];
  /** The loaded cloud keeps 1 in `keepEvery` points of the file. */
  keepEvery: number;
  /** Off when the colors come from data that only the loaded points have (e.g. distances). */
  active: boolean;
}

/** A full-density chunk is wanted once the loaded points would be this many pixels apart. */
const DETAIL_PX = 1.5;

export interface PickHit {
  cloudId: number;
  /** Point index in the cloud's octree order. */
  index: number;
  /** Position in render (shifted) coordinates. */
  position: THREE.Vector3;
}

export class Viewer {
  readonly scene = new THREE.Scene();
  /**
   * Drawn after the scene, without EDL (which darkens sparse points next to
   * background): the pose graph's thinned scans. With EDL on, it is drawn
   * over the clouds rather than hidden behind them.
   */
  readonly overlay = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  private readonly renderer: THREE.WebGLRenderer;
  private readonly controls: OrbitControls;
  private readonly clouds = new Map<number, LodCloud>();
  private readonly meshes = new Map<number, THREE.Mesh>();
  /** Polylines (e.g. trajectories), by their owner's id. */
  private readonly lines = new Map<number, Line2>();
  /** Draw points as discs instead of squares. */
  private roundPoints = false;
  /** Point materials of other modules that follow the point shape. */
  private readonly pointMaterials = new Set<THREE.Material>();
  /** Wide-line materials whose pixel width needs the drawing size. */
  private readonly lineMaterials = new Set<LineMaterial>();
  private pointSize = 2;
  private cloudBrightness = 1;
  /** Fixed: every point `pointSize` pixels. Adaptive: as large as the local point spacing, in world units. */
  private sizeMode: "fixed" | "adaptive" = "fixed";
  private pointBudget = 3_000_000;
  private needsRender = true;
  private needsLod = true;
  /** Called after each LOD update with the points drawn and how many of them are full-density chunks. */
  onDrawn: (points: number, detail: { chunks: number; points: number }) => void = () => {};
  /**
   * Called after each LOD update with the full-density chunks wanted but not
   * yet given with {@link addDetail}, most needed first; replaces earlier lists.
   */
  onDetailWanted: (wanted: { id: number; chunk: number }[]) => void = () => {};
  private fullDetail = true;
  private frame = 0;
  /** Called for a click (or tap) that was not a camera drag. */
  onClick: (clientX: number, clientY: number) => void = () => {};
  /** Called for a double click or double tap, after both clicks. */
  onDoubleClick: (clientX: number, clientY: number) => void = () => {};
  onToolPointerDown: (x: number, y: number) => boolean = () => false;
  onToolPointerMove: (x: number, y: number) => void = () => {};
  onToolPointerUp: (x: number, y: number) => void = () => {};
  onToolPointerCancel: () => void = () => {};
  private toolDrag: { pointer: number; controlsEnabled: boolean } | null = null;
  /** Called after every rendered frame, e.g. to move HTML overlays. */
  onAfterRender: () => void = () => {};
  /** Independent overlay listeners alongside the picking annotations. */
  readonly afterRenderListeners = new Set<() => void>();
  private readonly annotations = new THREE.Group();
  private readonly profileGroup = new THREE.Group();
  /** Clipping box in render coordinates, or null when clipping is off. */
  private clip: THREE.Box3 | null = null;
  private readonly clipPlanes = Array.from({ length: 6 }, () => new THREE.Plane());
  private readonly clipHelper = new THREE.Box3Helper(new THREE.Box3(), 0xffd54f);
  private readonly edl = new EdlPass();
  private edlEnabled = true;
  /** Gizmo moving one cloud or mesh around its centre, while active. */
  private gizmo: { id: number; controls: TransformControls; pivot: THREE.Object3D; center: THREE.Vector3 } | null =
    null;
  /** Called while the gizmo moves an object. */
  onGizmoChange: () => void = () => {};

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
    this.profileGroup.renderOrder = 1;
    this.scene.add(this.profileGroup);

    // A click is a press and release of a single pointer without moving
    // the camera; a second finger (pinch, two-finger pan) cancels it.
    let down: { x: number; y: number; time: number } | null = null;
    let last: { x: number; y: number; time: number } | null = null;
    const active = new Set<number>();
    const canvas = this.renderer.domElement;
    // Capture before OrbitControls sees the press. A tool takes ownership only
    // when it actually hits a handle; empty space retains camera navigation.
    canvas.addEventListener("pointerdown", (e) => {
      if (this.toolDrag || e.button !== 0 || !e.isPrimary || !this.onToolPointerDown(e.clientX, e.clientY)) return;
      down = last = null;
      this.toolDrag = { pointer: e.pointerId, controlsEnabled: this.controls.enabled };
      this.controls.enabled = false;
      canvas.setPointerCapture(e.pointerId);
      e.preventDefault();
      e.stopImmediatePropagation();
    }, true);
    canvas.addEventListener("pointermove", (e) => {
      if (this.toolDrag?.pointer !== e.pointerId) return;
      e.preventDefault();
      e.stopImmediatePropagation();
      this.onToolPointerMove(e.clientX, e.clientY);
    }, true);
    canvas.addEventListener("pointerup", (e) => {
      if (this.toolDrag?.pointer !== e.pointerId) return;
      e.preventDefault();
      e.stopImmediatePropagation();
      this.releaseToolDrag();
      this.onToolPointerUp(e.clientX, e.clientY);
    }, true);
    canvas.addEventListener("pointercancel", (e) => {
      if (this.toolDrag?.pointer !== e.pointerId) return;
      e.stopImmediatePropagation();
      this.cancelToolDrag();
    }, true);
    canvas.addEventListener("lostpointercapture", () => this.cancelToolDrag());
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

  private releaseToolDrag(): void {
    const drag = this.toolDrag;
    if (!drag) return;
    this.toolDrag = null;
    this.controls.enabled = drag.controlsEnabled;
    const canvas = this.renderer.domElement;
    if (canvas.hasPointerCapture(drag.pointer)) canvas.releasePointerCapture(drag.pointer);
  }

  /** Restore camera interaction and discard a tool's unfinished preview. */
  cancelToolDrag(): void {
    if (!this.toolDrag) return;
    this.releaseToolDrag();
    this.onToolPointerCancel();
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
    for (const material of this.lineMaterials) material.resolution.set(size.x, size.y);
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
    this.draw();
    this.onAfterRender();
    for (const listener of this.afterRenderListeners) listener();
  };

  private draw(): void {
    if (this.useEdl()) this.edl.render(this.renderer, this.scene, this.camera);
    else this.renderer.render(this.scene, this.camera);
    if (this.overlay.children.length === 0) return;
    const autoClear = this.renderer.autoClear;
    this.renderer.autoClear = false;
    this.renderer.render(this.overlay, this.camera);
    this.renderer.autoClear = autoClear;
  }

  private updateLod(): void {
    this.camera.updateMatrixWorld();
    const sources = [...this.clouds.values()]
      .filter((c) => c.visible)
      .map((c) => (this.gizmo?.id === c.id ? { ...c, matrix: this.gizmoMatrix()! } : c));
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
    const detail = this.updateDetail(this.pointBudget - selection.points);
    for (const cloud of this.clouds.values()) this.assignSizes(cloud, new Set(selection.nodes.get(cloud.id) ?? []));
    this.onDrawn(selection.points + detail.points, detail);
  }

  /**
   * Show full-density chunks of thinned clouds: those in view where the
   * loaded points would be more than {@link DETAIL_PX} apart on screen,
   * coarsest first, within what the octree LOD left of the point budget.
   * Chunks not loaded yet are asked for with {@link onDetailWanted}.
   */
  private updateDetail(budget: number): { chunks: number; points: number } {
    this.frame++;
    const camera = this.camera;
    const frustum = new THREE.Frustum().setFromProjectionMatrix(
      new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse),
    );
    const pixelsPerUnit =
      this.renderer.domElement.height / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2));
    const inView = (box: THREE.Box3) => frustum.intersectsBox(box) && (!this.clip || this.clip.intersectsBox(box));
    const candidates: { cloud: LodCloud; part: DetailPart; index: number; px: number; matrix: THREE.Matrix4 | null }[] = [];
    for (const cloud of this.clouds.values()) {
      const detail = cloud.detail;
      if (!detail || !detail.active || !cloud.visible || !this.fullDetail) continue;
      const matrix = this.gizmo?.id === cloud.id ? this.gizmoMatrix() : null;
      const thinned = Math.sqrt(detail.keepEvery);
      detail.parts.forEach((part, index) => {
        const box = matrix ? part.box.clone().applyMatrix4(matrix) : part.box;
        if (!inView(box)) return;
        const px = (part.spacing * thinned * pixelsPerUnit) / Math.max(box.distanceToPoint(camera.position), camera.near);
        if (px >= DETAIL_PX) candidates.push({ cloud, part, index, px, matrix });
      });
    }
    candidates.sort((a, b) => b.px - a.px);
    const shown = new Set<DetailPiece>();
    const wanted: { id: number; chunk: number }[] = [];
    let reserved = 0;
    const drawn = { chunks: 0, points: 0 };
    for (const { cloud, part, index, matrix } of candidates) {
      // A chunk still loading keeps room for all of its points; a loaded one
      // counts only its slices in view.
      const pieces = part.pieces?.filter((p) => inView(matrix ? p.box.clone().applyMatrix4(matrix) : p.box));
      const points = pieces ? pieces.reduce((sum, p) => sum + p.count, 0) : part.count;
      if (reserved + points > budget) break;
      reserved += points;
      if (!pieces) {
        wanted.push({ id: cloud.id, chunk: index });
        continue;
      }
      for (const piece of pieces) shown.add(piece);
      part.used = this.frame;
      drawn.chunks++;
      drawn.points += points;
    }
    let resident = 0;
    const idle: { cloud: LodCloud; part: DetailPart }[] = [];
    for (const cloud of this.clouds.values()) {
      for (const part of cloud.detail?.parts ?? []) {
        if (!part.pieces) continue;
        for (const piece of part.pieces) piece.object.visible = shown.has(piece);
        resident += part.count;
        if (part.used !== this.frame) idle.push({ cloud, part });
      }
    }
    // Free the least recently drawn chunks once much more than the budget is held.
    if (resident > 2 * this.pointBudget) {
      idle.sort((a, b) => a.part.used - b.part.used);
      for (const { cloud, part } of idle) {
        if (resident <= this.pointBudget) break;
        this.freeDetail(cloud, part);
        resident -= part.count;
      }
    }
    this.onDetailWanted(wanted);
    return drawn;
  }

  /**
   * Chunks of the file behind a thinned cloud that loaded 1 in `keepEvery`
   * points: their boxes (render coordinates) and point counts. The viewer
   * asks for the ones it wants (see {@link onDetailWanted}).
   */
  setDetail(id: number, chunks: { box: THREE.Box3; count: number }[], keepEvery: number): void {
    const cloud = this.clouds.get(id);
    if (!cloud) return;
    this.dropDetail(cloud);
    const parts = chunks.map(({ box, count }) => {
      // Airborne data is 2.5D: spread the points over the two longest sides.
      const [a, b, c] = box.getSize(new THREE.Vector3()).toArray().sort((x, y) => y - x);
      const area = b > 0 ? a * b : a * Math.max(c, a / Math.max(1, count));
      return { box, count, spacing: Math.sqrt(area / Math.max(1, count)), pieces: null, data: null, used: 0 };
    });
    cloud.detail = { parts, keepEvery, active: true };
    this.requestRender(true);
  }

  /** Turn a cloud's full-density chunks on or off (e.g. while colored by data only its loaded points have). */
  setDetailActive(id: number, active: boolean): void {
    const detail = this.clouds.get(id)?.detail;
    if (!detail || detail.active === active) return;
    detail.active = active;
    this.requestRender(true);
  }

  /** Full-density chunks on or off for every cloud. */
  setFullDetail(enabled: boolean): void {
    this.fullDetail = enabled;
    this.requestRender(true);
  }

  /**
   * The points of a chunk asked for with {@link onDetailWanted}: xyz relative
   * to the cloud's offset, rgba colors and its slices (`count` and box
   * relative to the offset, 7 numbers each, see `DetailChunk.pieces`);
   * `data` comes back to {@link recolorDetail}.
   */
  addDetail(id: number, chunk: number, positions: Float32Array, colors: Uint8Array, slices: Float64Array, data: unknown): void {
    const cloud = this.clouds.get(id);
    const part = cloud?.detail?.parts[chunk];
    if (!cloud || !part || part.pieces) return;
    part.pieces = [];
    for (let s = 0, start = 0; s < slices.length; s += 7) {
      const count = slices[s];
      const local = new THREE.Box3().setFromArray(slices.subarray(s + 1, s + 7));
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.BufferAttribute(positions.subarray(start * 3, (start + count) * 3), 3));
      geometry.setAttribute("color", new THREE.BufferAttribute(colors.subarray(start * 4, (start + count) * 4), 4, true));
      geometry.boundingBox = local.clone();
      geometry.boundingSphere = local.getBoundingSphere(new THREE.Sphere());
      const object = new THREE.Points(geometry, cloud.material);
      object.visible = false;
      cloud.group.add(object);
      part.pieces.push({ object, box: local.translate(cloud.offset), start, count });
      start += count;
    }
    part.data = data;
    part.used = this.frame;
    this.requestRender(true);
  }

  /** New rgba colors for every full-density chunk held for a cloud, from the `data` given with each. */
  recolorDetail(id: number, colorize: (data: unknown) => Uint8Array): void {
    for (const part of this.clouds.get(id)?.detail?.parts ?? []) {
      if (!part.pieces) continue;
      const colors = colorize(part.data);
      for (const { object, start, count } of part.pieces) {
        object.geometry.setAttribute("color", new THREE.BufferAttribute(colors.subarray(start * 4, (start + count) * 4), 4, true));
      }
    }
    this.requestRender();
  }

  private freeDetail(cloud: LodCloud, part: DetailPart): void {
    for (const { object } of part.pieces ?? []) {
      cloud.group.remove(object);
      object.geometry.dispose();
    }
    part.pieces = null;
    part.data = null;
  }

  /** Buffers held by cached drawables as well as the current scene. */
  memoryData(): { data: unknown[]; geometries: THREE.BufferGeometry[] } {
    const data: unknown[] = [], geometries: THREE.BufferGeometry[] = [];
    for (const cloud of this.clouds.values()) {
      data.push(cloud.positions, cloud.colors, cloud.nodes);
      for (const object of cloud.objects.values()) geometries.push(object.geometry);
      for (const part of cloud.detail?.parts ?? []) {
        data.push(part.data);
        for (const piece of part.pieces ?? []) geometries.push(piece.object.geometry);
      }
    }
    for (const object of [...this.meshes.values(), ...this.lines.values()]) geometries.push(object.geometry);
    return {data, geometries};
  }

  /** Drop cached geometry and full-density chunks that are not currently drawn. */
  releaseUnusedDetails(): void {
    for (const cloud of this.clouds.values()) {
      for (const [index, object] of cloud.objects) if (!cloud.visible || !object.visible) {
        cloud.group.remove(object); object.geometry.dispose(); cloud.objects.delete(index);
      }
      for (const part of cloud.detail?.parts ?? []) if (!cloud.visible || !cloud.detail?.active || part.used !== this.frame) this.freeDetail(cloud, part);
    }
    this.requestRender();
  }

  private dropDetail(cloud: LodCloud): void {
    for (const part of cloud.detail?.parts ?? []) this.freeDetail(cloud, part);
    cloud.detail = null;
  }

  /**
   * In adaptive mode, give each drawn node points as wide as the finest
   * spacing drawn in its subtree, so near surfaces close up and coarse
   * ancestors do not paint over finer detail.
   */
  private assignSizes(cloud: LodCloud, wanted: Set<number>): void {
    const parts = cloud.detail?.parts ?? [];
    if (this.sizeMode === "fixed") {
      for (const object of cloud.objects.values()) object.material = cloud.material;
      for (const part of parts) for (const { object } of part.pieces ?? []) object.material = cloud.material;
      return;
    }
    // Spacings rounded to half powers of two, so chunks share a few materials.
    for (const part of parts) {
      const material = this.sizedMaterial(cloud, 2 ** (Math.round(Math.log2(part.spacing) * 2) / 2));
      for (const { object } of part.pieces ?? []) object.material = material;
    }
    const finest = (index: number): number => {
      const node = cloud.nodes[index];
      // The lattice spacing, or wider where a node (e.g. a leaf) holds fewer
      // points than its lattice has cells (points mostly lie on surfaces),
      // capped so lone outliers do not become huge squares.
      const edge = node.box.max.x - node.box.min.x;
      let spacing = Math.min(4 * node.spacing, Math.max(node.spacing, edge / Math.sqrt(Math.max(1, node.count))));
      for (const child of node.children) {
        if (wanted.has(child)) spacing = Math.min(spacing, finest(child));
      }
      const object = cloud.objects.get(index);
      if (object) object.material = this.sizedMaterial(cloud, spacing);
      return spacing;
    };
    if (cloud.nodes.length && wanted.has(0)) finest(0);
  }

  private sizedMaterial(cloud: LodCloud, spacing: number): THREE.PointsMaterial {
    let material = cloud.sized.get(spacing);
    if (!material) {
      material = this.shaped(roundable(cloud.material.clone()));
      material.sizeAttenuation = true;
      cloud.sized.set(spacing, material);
    }
    material.size = spacing * this.pointSize * 0.5;
    return material;
  }

  /** Every point material of a cloud (the fixed one and the adaptive ones). */
  private materials(cloud: LodCloud): THREE.PointsMaterial[] {
    return [cloud.material, ...cloud.sized.values()];
  }

  /** Dim point-cloud context for map review without changing colors or visibility. */
  setCloudBrightness(value: number): void {
    if (!Number.isFinite(value)) return;
    this.cloudBrightness = Math.min(1, Math.max(0, value));
    for (const cloud of this.clouds.values()) for (const material of this.materials(cloud)) {
      material.color.setScalar(this.cloudBrightness);
    }
    this.requestRender();
  }

  /**
   * EDL shades depth steps between neighbouring pixels; adaptive points are
   * wide squares with a step at every edge, which it would turn dark.
   */
  private useEdl(): boolean {
    return this.edlEnabled && this.sizeMode === "fixed";
  }

  setPointSizeMode(mode: "fixed" | "adaptive"): void {
    this.sizeMode = mode;
    this.requestRender(true);
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
    geometry.boundingBox = cloud.nodes[index].box.clone().translate(cloud.offset.clone().negate());
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

  /**
   * Add a cloud whose `positions` are relative to `offset` (render
   * coordinates). The offset goes into the object's float64 matrix rather
   * than the float32 vertices, so a far-away cloud stays precise.
   */
  add(id: number, positions: Float32Array, colors: Uint8Array, nodes: LodNode[], offset: THREE.Vector3): void {
    const material = this.shaped(
      roundable(
        new THREE.PointsMaterial({
          color: new THREE.Color().setScalar(this.cloudBrightness),
          size: this.pointSize,
          sizeAttenuation: false,
          vertexColors: true,
          // Points with alpha 0 (e.g. a hidden class) are discarded.
          alphaTest: 0.5,
          clippingPlanes: this.clip ? this.clipPlanes : null,
        }),
      ),
    );
    const group = new THREE.Group();
    group.position.copy(offset);
    this.scene.add(group);
    this.clouds.set(id, {
      id,
      nodes,
      offset: offset.clone(),
      positions,
      colors,
      material,
      sized: new Map(),
      group,
      objects: new Map(),
      visible: true,
      detail: null,
    });
    this.requestRender(true);
  }

  /** Add a triangle mesh with a solid color; `positions` are relative to `offset`, as in {@link add}. */
  addMesh(
    id: number,
    positions: Float32Array,
    indices: Uint32Array,
    color: [number, number, number],
    offset: THREE.Vector3,
  ): void {
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
    mesh.position.copy(offset);
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
    if (this.gizmo?.id === id) this.endGizmo();
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
    this.dropDetail(cloud);
    for (const material of this.materials(cloud)) material.dispose();
    this.clouds.delete(id);
    this.requestRender(true);
  }

  /** The object drawing a cloud or mesh. */
  private object(id: number): THREE.Object3D | undefined {
    return this.meshes.get(id) ?? this.clouds.get(id)?.group;
  }

  /**
   * Show a translate or rotate gizmo on a cloud or mesh, pivoting around
   * `center` (render coordinates). Moving it only changes how the object is
   * drawn; read the motion with {@link gizmoMatrix}.
   */
  startGizmo(id: number, center: THREE.Vector3, mode: "translate" | "rotate"): void {
    this.endGizmo();
    const object = this.object(id);
    if (!object) return;
    const pivot = new THREE.Object3D();
    pivot.position.copy(center);
    this.scene.add(pivot);
    const controls = new TransformControls(this.camera, this.renderer.domElement);
    controls.setMode(mode);
    controls.attach(pivot);
    controls.addEventListener("dragging-changed", (e) => {
      this.controls.enabled = !e.value;
    });
    controls.addEventListener("objectChange", () => {
      // The object sits at its offset (see `add`); the motion comes on top.
      object.matrixAutoUpdate = false;
      object.matrix.copy(this.gizmoMatrix()!).multiply(new THREE.Matrix4().makeTranslation(object.position));
      object.matrixWorldNeedsUpdate = true;
      this.onGizmoChange();
      this.requestRender(true);
    });
    controls.addEventListener("change", () => this.requestRender());
    this.scene.add(controls.getHelper());
    this.gizmo = { id, controls, pivot, center: center.clone() };
    this.requestRender();
  }

  setGizmoMode(mode: "translate" | "rotate"): void {
    this.gizmo?.controls.setMode(mode);
  }

  /**
   * A move / rotate gizmo on any object of the overlay (e.g. a pose graph
   * node): `onChange` while it is dragged, `onRelease` when a drag ends.
   * Returns a function that removes it.
   */
  attachGizmo(
    target: THREE.Object3D,
    mode: "translate" | "rotate",
    onChange: () => void,
    onRelease: () => void,
  ): { setMode(mode: "translate" | "rotate"): void; detach(): void } {
    const controls = new TransformControls(this.camera, this.renderer.domElement);
    controls.setMode(mode);
    controls.attach(target);
    controls.addEventListener("dragging-changed", (e) => {
      this.controls.enabled = !e.value;
      if (!e.value) onRelease();
    });
    controls.addEventListener("objectChange", () => {
      onChange();
      this.requestRender();
    });
    controls.addEventListener("change", () => this.requestRender());
    const helper = controls.getHelper();
    this.overlay.add(helper);
    this.requestRender();
    return {
      setMode: (m) => controls.setMode(m),
      detach: () => {
        this.overlay.remove(helper);
        controls.detach();
        controls.dispose();
        this.controls.enabled = true;
        this.requestRender();
      },
    };
  }

  /** The gizmo's motion so far, in render coordinates, or null without a gizmo. */
  gizmoMatrix(): THREE.Matrix4 | null {
    if (!this.gizmo) return null;
    const { pivot, center } = this.gizmo;
    pivot.updateMatrix();
    return pivot.matrix.clone().multiply(new THREE.Matrix4().makeTranslation(-center.x, -center.y, -center.z));
  }

  /** Remove the gizmo and draw its object where it was. */
  endGizmo(): void {
    if (!this.gizmo) return;
    const { id, controls, pivot } = this.gizmo;
    this.gizmo = null;
    this.scene.remove(controls.getHelper(), pivot);
    controls.detach();
    controls.dispose();
    this.controls.enabled = true;
    const object = this.object(id);
    if (object) {
      object.matrix.identity();
      object.matrixAutoUpdate = true;
      object.matrixWorldNeedsUpdate = true;
    }
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
    for (const cloud of this.clouds.values()) {
      cloud.material.size = size;
      for (const [spacing, material] of cloud.sized) material.size = spacing * size * 0.5;
    }
    this.requestRender();
  }

  /** `material` with the current point shape. */
  private shaped<M extends THREE.Material>(material: M): M {
    material.defines = { ...material.defines };
    if (this.roundPoints) material.defines.ROUND_POINTS = "";
    else delete material.defines.ROUND_POINTS;
    material.needsUpdate = true;
    return material;
  }

  /** Draw every point as a disc (like iridescence) or as a square. */
  setRoundPoints(round: boolean): void {
    this.roundPoints = round;
    for (const cloud of this.clouds.values()) for (const material of this.materials(cloud)) this.shaped(material);
    for (const material of this.pointMaterials) this.shaped(material);
    this.requestRender();
  }

  /**
   * Let another module's point material follow the point shape. A built-in
   * material is made {@link roundable}; a shader material must test
   * `ROUND_POINTS` itself.
   */
  registerPointMaterial(material: THREE.Material): void {
    if (!(material instanceof THREE.ShaderMaterial)) roundable(material);
    this.pointMaterials.add(this.shaped(material));
  }

  /** A wide-line material (width in pixels) that follows the drawing size. */
  lineMaterial(parameters: ConstructorParameters<typeof LineMaterial>[0]): LineMaterial {
    const material = new LineMaterial(parameters);
    const size = this.renderer.getDrawingBufferSize(new THREE.Vector2());
    material.resolution.set(size.x, size.y);
    this.lineMaterials.add(material);
    return material;
  }

  /** Stop resizing a line material made by {@link lineMaterial}, and free it. */
  disposeLineMaterial(material: LineMaterial): void {
    this.lineMaterials.delete(material);
    material.dispose();
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
      // The eye relative to the cloud's positions.
      const eye = origin.clone().sub(cloud.offset);
      for (const [index, object] of cloud.objects) {
        if (!object.visible) continue;
        const node = cloud.nodes[index];
        const reach = node.box.distanceToPoint(origin) + node.box.max.distanceTo(node.box.min);
        box.copy(node.box).expandByScalar(slope * reach);
        if (!raycaster.ray.intersectsBox(box)) continue;
        const p = cloud.positions;
        for (let i = node.start, end = node.start + node.count; i < end; i++) {
          const vx = p[i * 3] - eye.x;
          const vy = p[i * 3 + 1] - eye.y;
          const vz = p[i * 3 + 2] - eye.z;
          const depth = vx * direction.x + vy * direction.y + vz * direction.z;
          if (depth <= this.camera.near) continue;
          const perp2 = vx * vx + vy * vy + vz * vz - depth * depth;
          if (perp2 > slope2 * depth * depth) continue;
          if (cloud.colors[i * 4 + 3] === 0) continue; // hidden by a filter
          if (this.clip && !this.clip.containsPoint(tmp.set(p[i * 3], p[i * 3 + 1], p[i * 3 + 2]).add(cloud.offset))) {
            continue;
          }
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
      position: new THREE.Vector3(p[best.index * 3], p[best.index * 3 + 1], p[best.index * 3 + 2]).add(best.cloud.offset),
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
      const { geometry, origin } = localGeometry(markers.map((m) => m.position));
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
      points.position.copy(origin);
      points.renderOrder = 2;
      this.annotations.add(points);
    }
    if (segments.length) {
      const { geometry, origin } = localGeometry(segments.flat());
      const material = new THREE.LineBasicMaterial({ color: 0x4fc3f7, depthTest: false, transparent: true });
      const lines = new THREE.LineSegments(geometry, material);
      lines.position.copy(origin);
      lines.renderOrder = 1;
      this.annotations.add(lines);
    }
    this.requestRender();
  }

  /**
   * Add or replace a polyline (interleaved xyz in render coordinates), in
   * one color or, with `colors` (interleaved rgba), one per vertex.
   */
  setLine(id: number, positions: Float64Array, color: string, colors?: Uint8Array): void {
    this.removeLine(id);
    // Stored relative to the first vertex, like `localGeometry`.
    const origin = new THREE.Vector3().fromArray(positions);
    const local = new Float32Array(positions.length);
    for (let i = 0; i < positions.length; i++) local[i] = positions[i] - origin.getComponent(i % 3);
    const geometry = new LineGeometry();
    geometry.setPositions(local);
    if (colors) {
      const rgb = new Float32Array((colors.length / 4) * 3);
      for (let i = 0; i < colors.length / 4; i++) {
        for (let c = 0; c < 3; c++) rgb[i * 3 + c] = colors[i * 4 + c] / 255;
      }
      geometry.setColors(rgb);
    }
    geometry.computeBoundingBox();
    geometry.computeBoundingSphere();
    // A 2-pixel wide line (like iridescence's trajectories). Without depth
    // writes EDL does not outline (and so blacken) it; drawn after the
    // clouds, it is still hidden by points in front.
    const material = this.lineMaterial({
      color: colors ? 0xffffff : new THREE.Color(color).getHex(),
      vertexColors: !!colors,
      linewidth: 2,
      depthWrite: false,
    });
    const line = new Line2(geometry, material);
    line.position.copy(origin);
    line.renderOrder = 1;
    this.lines.set(id, line);
    this.scene.add(line);
    this.requestRender();
  }

  removeLine(id: number): void {
    const line = this.lines.get(id);
    if (!line) return;
    this.scene.remove(line);
    line.geometry.dispose();
    this.disposeLineMaterial(line.material);
    this.lines.delete(id);
    this.requestRender();
  }

  setLineVisible(id: number, visible: boolean): void {
    const line = this.lines.get(id);
    if (!line) return;
    line.visible = visible;
    this.requestRender();
  }

  /** Where the ray under a CSS pixel meets the horizontal plane at height `z`. */
  groundPoint(clientX: number, clientY: number, z: number): THREE.Vector3 | null {
    const rect = this.renderer.domElement.getBoundingClientRect();
    const ndc = new THREE.Vector2(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1,
    );
    const raycaster = new THREE.Raycaster();
    raycaster.setFromCamera(ndc, this.camera);
    return raycaster.ray.intersectPlane(new THREE.Plane(new THREE.Vector3(0, 0, 1), -z), new THREE.Vector3());
  }

  /** Draw a profile polyline and the outline of its band (render coordinates). */
  setProfile(vertices: THREE.Vector3[], halfWidth: number): void {
    for (const child of [...this.profileGroup.children]) {
      this.profileGroup.remove(child);
      const object = child as THREE.Line;
      object.geometry.dispose();
      (object.material as THREE.Material).dispose();
    }
    const material = () =>
      new THREE.LineBasicMaterial({ color: 0xffb74d, depthTest: false, transparent: true });
    // Everything relative to the first vertex, like `localGeometry`.
    this.profileGroup.position.copy(vertices[0] ?? new THREE.Vector3());
    const local = (points: THREE.Vector3[]) =>
      new THREE.BufferGeometry().setFromPoints(points.map((p) => p.clone().sub(this.profileGroup.position)));
    if (vertices.length >= 2) {
      this.profileGroup.add(new THREE.Line(local(vertices), material()));
      // Band edges: each segment offset sideways, plus the two ends.
      const edges: THREE.Vector3[] = [];
      for (let i = 1; i < vertices.length; i++) {
        const [a, b] = [vertices[i - 1], vertices[i]];
        const side = new THREE.Vector3(a.y - b.y, b.x - a.x, 0);
        if (side.lengthSq() === 0) continue;
        side.setLength(halfWidth);
        for (const s of [1, -1]) edges.push(a.clone().addScaledVector(side, s), b.clone().addScaledVector(side, s));
        if (i === 1) edges.push(a.clone().add(side), a.clone().sub(side));
        if (i === vertices.length - 1) edges.push(b.clone().add(side), b.clone().sub(side));
      }
      const band = new THREE.LineSegments(local(edges), material());
      (band.material as THREE.LineBasicMaterial).opacity = 0.5;
      this.profileGroup.add(band);
    }
    if (vertices.length) {
      const dots = new THREE.Points(
        local(vertices),
        new THREE.PointsMaterial({ color: 0xffb74d, size: 8, sizeAttenuation: false, depthTest: false, transparent: true }),
      );
      this.profileGroup.add(dots);
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
      for (const material of this.materials(cloud)) {
        material.clippingPlanes = planes;
        material.needsUpdate = true;
      }
    }
    for (const mesh of this.meshes.values()) {
      const material = mesh.material as THREE.Material;
      material.clippingPlanes = planes;
      material.needsUpdate = true;
    }
    this.requestRender(true);
  }

  /** Bounding box of all visible clouds, meshes, lines and overlay objects (render coordinates). */
  contentBounds(): THREE.Box3 {
    const box = new THREE.Box3();
    for (const cloud of this.clouds.values()) {
      if (cloud.visible && cloud.nodes[0]) box.union(tightBox(cloud));
    }
    for (const object of [...this.meshes.values(), ...this.lines.values()]) {
      if (object.visible && object.geometry.boundingBox) {
        box.union(object.geometry.boundingBox.clone().translate(object.position));
      }
    }
    // The overlay too (e.g. a pose graph's scans), from its geometries' bounds.
    if (this.overlay.children.some((c) => c.visible)) {
      this.overlay.updateMatrixWorld(true);
      box.union(new THREE.Box3().setFromObject(this.overlay));
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

  /** The current view rendered into a new 2D canvas (device pixels). */
  snapshot(): HTMLCanvasElement {
    // Draw now and copy straight away, while the WebGL buffer still holds it.
    this.draw();
    const source = this.renderer.domElement;
    const out = document.createElement("canvas");
    out.width = source.width;
    out.height = source.height;
    out.getContext("2d")!.drawImage(source, 0, 0);
    return out;
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
    this.frameBox(this.clip ? this.clip.clone() : this.contentBounds());
  }

  /** Frame `box` (render space), keeping the current viewing direction. */
  frameBox(box: THREE.Box3): void {
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
    box = new THREE.Box3().setFromArray(cloud.positions).translate(cloud.offset);
    tightBoxes.set(cloud, box);
  }
  return box;
}

/**
 * Geometry for points in render coordinates, stored relative to the first
 * one (returned as `origin`, where to place the object): float32 vertices
 * would round render coordinates far from the global shift, e.g. on a UTM
 * cloud opened after a local one.
 */
function localGeometry(points: THREE.Vector3[]): { geometry: THREE.BufferGeometry; origin: THREE.Vector3 } {
  const origin = points[0]?.clone() ?? new THREE.Vector3();
  const geometry = new THREE.BufferGeometry().setFromPoints(points.map((p) => p.clone().sub(origin)));
  return { geometry, origin };
}

/**
 * Let a built-in points material draw discs when `ROUND_POINTS` is defined
 * (see `Viewer.setRoundPoints`): fragments outside the inscribed circle of
 * each point's square are discarded.
 */
export function roundable<M extends THREE.Material>(material: M): M {
  material.onBeforeCompile = (shader) => {
    shader.fragmentShader = shader.fragmentShader.replace(
      "#include <clipping_planes_fragment>",
      "#include <clipping_planes_fragment>\n#ifdef ROUND_POINTS\nif (length(gl_PointCoord - 0.5) > 0.5) discard;\n#endif",
    );
  };
  return material;
}
