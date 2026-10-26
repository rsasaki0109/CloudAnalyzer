import * as THREE from 'three';
import type { LocatedInterval } from '../mapping-locations';
import { viewer, toRender } from './state';
import { $ } from './dom';
let group: THREE.Group | null=null;
export function clearMappingLocation(): void {
  if(group){viewer.overlay.remove(group);for(const item of group.children){const line=item as THREE.Line;line.geometry.dispose();(line.material as THREE.Material).dispose();}group=null;}
  $('mapping-location-clear').hidden=true;$('mapping-location-status').textContent='';viewer.requestRender();
}
export function showMappingLocation(interval:LocatedInterval,label:string):void {
  clearMappingLocation();const origin=interval.left[0];group=new THREE.Group();group.name='mapping-source-interval';group.position.copy(toRender(origin));
  const bounds=new THREE.Box3();
  for(const points of [interval.left,interval.right]){
    const geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(points.flatMap(p=>p.map((v,i)=>v-origin[i])),3));
    group.add(new THREE.Line(geometry,new THREE.LineBasicMaterial({color:0xffbf47,depthTest:false,depthWrite:false})));
    for(const p of points)bounds.expandByPoint(toRender(p));
  }
  viewer.overlay.add(group);viewer.frameBox(bounds);
  $('mapping-location-status').textContent=`${label}. Amber: original source boundaries, with linearly interpolated interval endpoints. This is a source hypothesis, not a validated road lane.`;
  $('mapping-location-clear').hidden=false;viewer.requestRender();
}
$('mapping-location-clear').onclick=clearMappingLocation;
