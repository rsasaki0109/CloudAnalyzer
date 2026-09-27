// Level-of-detail selection over the nested octrees built by ca-core.
//
// Each node holds a uniform subsample of its cube; drawing a node plus all of
// its ancestors shows the cloud at that node's density. Nodes are refined in
// order of on-screen point spacing until the point budget is spent.

import * as THREE from "three";

const NODE_STRIDE = 15;

export interface LodNode {
  start: number;
  count: number;
  /** Cube in render (shifted) coordinates. */
  box: THREE.Box3;
  /** Approximate distance between neighbouring points of this node. */
  spacing: number;
  children: number[];
}

/** Parse the flat node table from `Cloud.lodNodes()`. */
export function parseNodes(flat: Float64Array, grid: number, shift: [number, number, number]): LodNode[] {
  const nodes: LodNode[] = [];
  for (let i = 0; i < flat.length; i += NODE_STRIDE) {
    const size = flat[i + 5];
    const min = new THREE.Vector3(flat[i + 2] - shift[0], flat[i + 3] - shift[1], flat[i + 4] - shift[2]);
    const children: number[] = [];
    for (let c = 0; c < 8; c++) {
      const child = flat[i + 7 + c];
      if (child >= 0) children.push(child);
    }
    nodes.push({
      start: flat[i],
      count: flat[i + 1],
      box: new THREE.Box3(min, min.clone().addScalar(size)),
      spacing: size / grid,
      children,
    });
  }
  return nodes;
}

export interface LodSource {
  id: number;
  nodes: LodNode[];
}

export interface Selection {
  /** Selected node indices per cloud id. */
  nodes: Map<number, number[]>;
  points: number;
}

/**
 * Choose the nodes to draw: breadth-first by largest on-screen spacing,
 * skipping nodes outside the frustum, until `budget` points are used or the
 * spacing drops below `minSpacingPx`.
 */
export function selectNodes(
  sources: LodSource[],
  camera: THREE.PerspectiveCamera,
  viewportHeight: number,
  budget: number,
  minSpacingPx = 1,
): Selection {
  const frustum = new THREE.Frustum().setFromProjectionMatrix(
    new THREE.Matrix4().multiplyMatrices(camera.projectionMatrix, camera.matrixWorldInverse),
  );
  const pixelsPerUnit = viewportHeight / (2 * Math.tan(THREE.MathUtils.degToRad(camera.fov) / 2));
  const eye = camera.position;
  const spacingPx = (node: LodNode) =>
    (node.spacing * pixelsPerUnit) / Math.max(node.box.distanceToPoint(eye), camera.near);

  const heap = new MaxHeap<{ source: LodSource; node: number }>();
  for (const source of sources) {
    const root = source.nodes[0];
    if (root && frustum.intersectsBox(root.box)) {
      // Roots always go first so every visible cloud shows at least a coarse view.
      heap.push({ source, node: 0 }, Number.POSITIVE_INFINITY);
    }
  }
  const selected = new Map<number, number[]>();
  let points = 0;
  while (heap.size > 0) {
    const { source, node } = heap.pop()!;
    const n = source.nodes[node];
    if (points + n.count > budget && points > 0) break;
    points += n.count;
    let list = selected.get(source.id);
    if (!list) selected.set(source.id, (list = []));
    list.push(node);
    if (spacingPx(n) < minSpacingPx) continue;
    for (const child of n.children) {
      const c = source.nodes[child];
      if (frustum.intersectsBox(c.box)) heap.push({ source, node: child }, spacingPx(c));
    }
  }
  return { nodes: selected, points };
}

class MaxHeap<T> {
  private items: { value: T; key: number }[] = [];

  get size(): number {
    return this.items.length;
  }

  push(value: T, key: number): void {
    const items = this.items;
    items.push({ value, key });
    let i = items.length - 1;
    while (i > 0) {
      const parent = (i - 1) >> 1;
      if (items[parent].key >= items[i].key) break;
      [items[parent], items[i]] = [items[i], items[parent]];
      i = parent;
    }
  }

  pop(): T | undefined {
    const items = this.items;
    const top = items[0];
    const last = items.pop();
    if (items.length > 0 && last) {
      items[0] = last;
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        const r = l + 1;
        let m = i;
        if (l < items.length && items[l].key > items[m].key) m = l;
        if (r < items.length && items[r].key > items[m].key) m = r;
        if (m === i) break;
        [items[m], items[i]] = [items[i], items[m]];
        i = m;
      }
    }
    return top?.value;
  }
}
