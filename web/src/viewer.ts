// three.js viewport: level-of-detail point clouds, Z-up orbit camera.

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";
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

export class Viewer {
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  private readonly renderer: THREE.WebGLRenderer;
  private readonly controls: OrbitControls;
  private readonly clouds = new Map<number, LodCloud>();
  private pointSize = 2;
  private pointBudget = 3_000_000;
  private needsRender = true;
  private needsLod = true;
  /** Called with the number of points drawn after each LOD update. */
  onDrawn: (points: number) => void = () => {};

  constructor(private readonly container: HTMLElement) {
    THREE.Object3D.DEFAULT_UP.set(0, 0, 1);
    this.renderer = new THREE.WebGLRenderer({ antialias: false });
    this.renderer.setPixelRatio(window.devicePixelRatio);
    container.appendChild(this.renderer.domElement);

    this.camera = new THREE.PerspectiveCamera(50, 1, 0.01, 1e6);
    this.camera.up.set(0, 0, 1);
    this.camera.position.set(10, -10, 10);

    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;
    this.controls.screenSpacePanning = true;
    this.controls.addEventListener("change", () => this.requestRender(true));

    const axes = new THREE.AxesHelper(1);
    axes.name = "axes";
    this.scene.add(axes);

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
    this.renderer.render(this.scene, this.camera);
  };

  private updateLod(): void {
    this.camera.updateMatrixWorld();
    const sources = [...this.clouds.values()].filter((c) => c.visible);
    const selection = selectNodes(
      sources,
      this.camera,
      this.renderer.domElement.height,
      this.pointBudget,
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
      new THREE.BufferAttribute(cloud.colors.subarray(start * 3, (start + count) * 3), 3, true),
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

  remove(id: number): void {
    const cloud = this.clouds.get(id);
    if (!cloud) return;
    this.scene.remove(cloud.group);
    for (const object of cloud.objects.values()) object.geometry.dispose();
    cloud.material.dispose();
    this.clouds.delete(id);
    this.requestRender(true);
  }

  /** Replace a cloud's colors (interleaved rgb in octree order). */
  setColors(id: number, colors: Uint8Array): void {
    const cloud = this.clouds.get(id);
    if (!cloud) return;
    cloud.colors = colors;
    for (const [index, object] of cloud.objects) {
      const { start, count } = cloud.nodes[index];
      object.geometry.setAttribute(
        "color",
        new THREE.BufferAttribute(colors.subarray(start * 3, (start + count) * 3), 3, true),
      );
    }
    this.requestRender();
  }

  setVisible(id: number, visible: boolean): void {
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

  setPointBudget(points: number): void {
    this.pointBudget = points;
    this.requestRender(true);
  }

  /** Frame all visible clouds, keeping the current viewing direction. */
  fit(): void {
    const box = new THREE.Box3();
    for (const cloud of this.clouds.values()) {
      if (cloud.visible && cloud.nodes[0]) box.union(tightBox(cloud));
    }
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
