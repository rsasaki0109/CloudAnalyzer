// three.js viewport: one THREE.Points per cloud, Z-up orbit camera.

import * as THREE from "three";
import { OrbitControls } from "three/examples/jsm/controls/OrbitControls.js";

export class Viewer {
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  private readonly renderer: THREE.WebGLRenderer;
  private readonly controls: OrbitControls;
  private readonly objects = new Map<number, THREE.Points>();
  private pointSize = 2;
  private needsRender = true;

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
    this.controls.addEventListener("change", () => this.requestRender());

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

  requestRender(): void {
    this.needsRender = true;
  }

  private resize(): void {
    const { clientWidth: w, clientHeight: h } = this.container;
    if (w === 0 || h === 0) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
    this.requestRender();
  }

  private loop = (): void => {
    requestAnimationFrame(this.loop);
    // Damping keeps emitting "change" while the camera settles.
    this.controls.update();
    if (!this.needsRender) return;
    this.needsRender = false;
    this.renderer.render(this.scene, this.camera);
  };

  add(id: number, positions: Float32Array, colors: Uint8Array): void {
    const geometry = new THREE.BufferGeometry();
    geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3, true));
    geometry.computeBoundingSphere();
    const material = new THREE.PointsMaterial({
      size: this.pointSize,
      sizeAttenuation: false,
      vertexColors: true,
    });
    const points = new THREE.Points(geometry, material);
    this.objects.set(id, points);
    this.scene.add(points);
    this.requestRender();
  }

  remove(id: number): void {
    const points = this.objects.get(id);
    if (!points) return;
    this.scene.remove(points);
    points.geometry.dispose();
    (points.material as THREE.Material).dispose();
    this.objects.delete(id);
    this.requestRender();
  }

  setColors(id: number, colors: Uint8Array): void {
    const points = this.objects.get(id);
    if (!points) return;
    points.geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3, true));
    this.requestRender();
  }

  setVisible(id: number, visible: boolean): void {
    const points = this.objects.get(id);
    if (points) points.visible = visible;
    this.requestRender();
  }

  setPointSize(size: number): void {
    this.pointSize = size;
    for (const points of this.objects.values()) {
      (points.material as THREE.PointsMaterial).size = size;
    }
    this.requestRender();
  }

  /** Frame all visible clouds, keeping the current viewing direction. */
  fit(): void {
    const box = new THREE.Box3();
    for (const points of this.objects.values()) {
      if (points.visible && points.geometry.boundingSphere) {
        box.expandByObject(points);
      }
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
    this.requestRender();
  }

  /** Look along a principal axis at the current target. */
  view(direction: THREE.Vector3Like): void {
    const distance = this.camera.position.distanceTo(this.controls.target);
    const dir = new THREE.Vector3().copy(direction).normalize();
    this.camera.position.copy(this.controls.target).addScaledVector(dir, distance);
    this.controls.update();
    this.requestRender();
  }
}
