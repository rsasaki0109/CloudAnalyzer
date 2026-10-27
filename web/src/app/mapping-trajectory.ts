/** Transient overlay; coordinates never modify the delivered map or native poses. */
import * as THREE from 'three';
import type { MappingTrajectory } from '../mapping-trajectory';
import { viewer, toRender } from './state';
import { $ } from './dom';
const toggle = $<HTMLInputElement>('mapping-trajectory-show');
const fit = $<HTMLButtonElement>('mapping-trajectory-fit');
let trajectory: MappingTrajectory | null = null;
let group: THREE.Group | null = null;
function clearOverlay(): void {
  if (group) {
    viewer.overlay.remove(group);
    for (const child of group.children) {
      const drawable = child as THREE.Line | THREE.Points;
      drawable.geometry.dispose();
      const materials = Array.isArray(drawable.material) ? drawable.material : [drawable.material];
      materials.forEach(material => material.dispose());
    }
    group = null;
  }
  viewer.requestRender();
}
export function setMappingTrajectory(value: MappingTrajectory | null, reason?: string): void {
  clearOverlay(); trajectory = value; toggle.checked = false;
  toggle.disabled = fit.disabled = !value || !!value.unavailable;
  $('mapping-trajectory-status').textContent = reason ?? value?.unavailable ?? (value
    ? `${value.positions.length.toLocaleString()} of ${value.count.toLocaleString()} poses shown when enabled. Cyan: saved trajectory; green: start; orange: end. Display only, in the package's original frame; no alignment or accuracy verification.`
    : 'Reopen the original generated-map package to display its trajectory.');
}
function show(): void {
  clearOverlay();
  if (!trajectory || trajectory.unavailable || !toggle.checked) return;
  const positions = trajectory.positions, origin = positions[0];
  group = new THREE.Group(); group.name = 'mapping-review-trajectory';
  group.position.copy(toRender(origin));
  const relative = (p: number[]) => p.map((v, i) => v - origin[i]);
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions.flatMap(relative), 3));
  group.add(new THREE.Line(geometry, new THREE.LineBasicMaterial({color:0x36d8ee, depthTest:false, depthWrite:false})));
  for (const [point, color] of [[positions[0], 0x62ef91], [positions.at(-1)!, 0xffa044]] as const) {
    const marker = new THREE.BufferGeometry(); marker.setAttribute('position', new THREE.Float32BufferAttribute(relative(point), 3));
    group.add(new THREE.Points(marker, new THREE.PointsMaterial({color, size:9, sizeAttenuation:false, depthTest:false, depthWrite:false})));
  }
  viewer.overlay.add(group); viewer.requestRender();
}
toggle.onchange = show;
fit.onclick = () => {
  if (!trajectory?.bounds || trajectory.unavailable) return;
  toggle.checked = true; show();
  const b = trajectory.bounds;
  viewer.frameBox(new THREE.Box3(toRender([b[0],b[1],b[2]]), toRender([b[3],b[4],b[5]])));
};
