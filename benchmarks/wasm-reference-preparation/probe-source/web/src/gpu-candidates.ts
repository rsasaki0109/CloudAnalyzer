
function __caMark(phase: string, id?: number): void {
  const trace = (globalThis as any).__caPreparationTrace;
  if (!trace) return;
  if (trace.events.length >= 256) { trace.omitted++; return; }
  trace.events.push({ phase, at: performance.now(), ...(id === undefined ? {} : { id }) });
}
function __caScope(): number | undefined {
  const trace = (globalThis as any).__caPreparationTrace;
  return trace ? trace.nextScope++ : undefined;
}
function __caPreparationSnapshot() {
  const trace = (globalThis as any).__caPreparationTrace;
  return trace ? { events: trace.events.slice(), omitted: trace.omitted } : undefined;
}
/** Rounded GPU nearest-neighbour candidates. Callers must verify them in f64. */
const SHADER = (stackCapacity: 20 | 32) => `
override workgroupSize: u32 = 64u;
override inheritedBounds: bool = false;
@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
// start, end, right child, axis (255 = leaf), split coordinate bits.
@group(0) @binding(1) var<storage, read> nodes: array<u32>;
@group(0) @binding(2) var<storage, read> queries: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> candidates: array<u32>;
// Lazily uploaded rounded subtree boxes, six components per node.
@group(0) @binding(4) var<storage, read> boxes: array<f32>;
// Optional stackless traversal shares the ordinary point/node buffers.
@group(0) @binding(5) var<storage, read> parents: array<u32>;

// A near-leaf seed is only a hint. The f64 verifier scans that leaf and
// searches every ancestral sibling whose exact bounds cannot exclude it.
fn leafSeed(q: vec3<f32>) -> u32 {
    var node = 0u;
    for (var depth = 0u; depth < 32u; depth += 1u) {
        let base = node * 5u;
        let axis = nodes[base + 3u];
        if (axis == 255u) {
            var best = 3.402823e38f;
            var slot = 0xffffffffu;
            let end = min(nodes[base + 1u], nodes[base] + 4096u);
            for (var i = nodes[base]; i < end; i += 1u) {
                let delta = points[i].xyz - q;
                let d = dot(delta, delta);
                if (d < best) { best = d; slot = i; }
            }
            return slot;
        }
        node = select(nodes[base + 2u], node + 1u,
            q[axis] < bitcast<f32>(nodes[base + 4u]));
    }
    return 0xffffffffu;
}

@compute @workgroup_size(workgroupSize)
fn leafNearest(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&queries)) { return; }
    candidates[id.x] = leafSeed(queries[id.x].xyz);
}

@compute @workgroup_size(workgroupSize)
fn nearest(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&queries)) { return; }
    let q = queries[id.x].xyz;
    var best = 3.402823e38f;
    var slot = 0xffffffffu;
    var stack: array<u32, ${stackCapacity}>;
    var lower: array<f32, ${stackCapacity}>;
    var pending = 1u;
    stack[0] = 0u;
    lower[0] = 0.0;
    // Bound a dispatch even on degenerate data. An unfinished search still
    // provides only a hint, which the authoritative CPU search checks.
    var visits = 0u;
    loop {
        if (pending == 0u || visits >= 4096u) { break; }
        pending -= 1u;
        let node = stack[pending];
        let parentLower = lower[pending];
        if (parentLower > best) { continue; }
        visits += 1u;
        let base = node * 5u;
        let axis = nodes[base + 3u];
        if (axis == 255u) {
            let end = min(nodes[base + 1u], nodes[base] + 4096u);
            for (var i = nodes[base]; i < end; i += 1u) {
                let delta = points[i].xyz - q;
                let d = dot(delta, delta);
                if (d < best) { best = d; slot = i; }
            }
        } else {
            let delta = q[axis] - bitcast<f32>(nodes[base + 4u]);
            let near = select(nodes[base + 2u], node + 1u, delta < 0.0);
            let far = select(node + 1u, nodes[base + 2u], delta < 0.0);
            if (pending + 2u > ${stackCapacity}u) { break; }
            stack[pending] = far;
            // Retain ancestor exclusions after a closer near-leaf hit.
            lower[pending] = select(delta * delta, max(parentLower, delta * delta), inheritedBounds);
            pending += 1u;
            stack[pending] = near;
            lower[pending] = select(0.0, parentLower, inheritedBounds);
            pending += 1u;
        }
    }
    candidates[id.x] = slot;
}

// Separate entry keeps box loads and the fifth binding out of ordinary searches.
@compute @workgroup_size(workgroupSize)
fn nearestBox(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&queries)) { return; }
    let q = queries[id.x].xyz;
    var best = 3.402823e38f;
    var slot = 0xffffffffu;
    var stack: array<u32, 32>;
    var lower: array<f32, 32>;
    var pending = 1u;
    stack[0] = 0u;
    lower[0] = 0.0;
    // Bound a dispatch even on degenerate data. An unfinished search still
    // provides only a hint, which the authoritative CPU search checks.
    var visits = 0u;
    loop {
        if (pending == 0u || visits >= 4096u) { break; }
        pending -= 1u;
        let node = stack[pending];
        let parentLower = lower[pending];
        if (parentLower > best) { continue; }
        let boundsBase = node * 6u;
        let lo = vec3<f32>(boxes[boundsBase], boxes[boundsBase + 1u], boxes[boundsBase + 2u]);
        let hi = vec3<f32>(boxes[boundsBase + 3u], boxes[boundsBase + 4u], boxes[boundsBase + 5u]);
        let gap = max(max(lo - q, q - hi), vec3<f32>(0.0));
        if (dot(gap, gap) > best) { continue; }
        visits += 1u;
        let base = node * 5u;
        let axis = nodes[base + 3u];
        if (axis == 255u) {
            let end = min(nodes[base + 1u], nodes[base] + 4096u);
            for (var i = nodes[base]; i < end; i += 1u) {
                let delta = points[i].xyz - q;
                let d = dot(delta, delta);
                if (d < best) { best = d; slot = i; }
            }
        } else {
            let delta = q[axis] - bitcast<f32>(nodes[base + 4u]);
            let near = select(nodes[base + 2u], node + 1u, delta < 0.0);
            let far = select(node + 1u, nodes[base + 2u], delta < 0.0);
            if (pending + 2u > 32u) { break; }
            stack[pending] = far;
            // Retain ancestor exclusions after a closer near-leaf hit.
            lower[pending] = select(delta * delta, max(parentLower, delta * delta), inheritedBounds);
            pending += 1u;
            stack[pending] = near;
            lower[pending] = select(0.0, parentLower, inheritedBounds);
            pending += 1u;
        }
    }
    candidates[id.x] = slot;
}

@compute @workgroup_size(workgroupSize)
fn nearestParent(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&queries)) { return; }
    let q = queries[id.x].xyz;
    let missing = 0xffffffffu;
    var best = 3.402823e38f;
    var slot = missing;
    var node = 0u;
    var previous = missing;
    var visits = 0u;
    var steps = 0u;
    loop {
        if (node == missing || steps >= 12288u || visits >= 4096u) { break; }
        steps += 1u;
        let base = node * 5u;
        let axis = nodes[base + 3u];
        let parent = parents[node];
        var next = parent;
        if (previous == parent) {
            visits += 1u;
            if (axis == 255u) {
                let end = min(nodes[base + 1u], nodes[base] + 4096u);
                for (var i = nodes[base]; i < end; i += 1u) {
                    let delta = points[i].xyz - q;
                    let d = dot(delta, delta);
                    if (d < best) { best = d; slot = i; }
                }
            } else {
                let delta = q[axis] - bitcast<f32>(nodes[base + 4u]);
                next = select(nodes[base + 2u], node + 1u, delta < 0.0);
            }
        } else if (axis != 255u) {
            let delta = q[axis] - bitcast<f32>(nodes[base + 4u]);
            let near = select(nodes[base + 2u], node + 1u, delta < 0.0);
            let far = select(node + 1u, nodes[base + 2u], delta < 0.0);
            if (previous == near && delta * delta <= best) { next = far; }
        }
        previous = node;
        node = next;
    }
    candidates[id.x] = slot;
}`;

const PACKED_SHADER = `
override workgroupSize: u32 = 64u;
override inheritedBounds: bool = false;
@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
// Top two bits encode axis (3 = leaf); low 30 bits encode right child/start.
@group(0) @binding(1) var<storage, read> nodes: array<vec2<u32>>;
@group(0) @binding(2) var<storage, read> queries: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> candidates: array<u32>;
@compute @workgroup_size(workgroupSize)
fn nearest(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= arrayLength(&queries)) { return; }
    let q = queries[id.x].xyz;
    var best = 3.402823e38f;
    var slot = 0xffffffffu;
    var stack: array<u32, 32>;
    var lower: array<f32, 32>;
    var pending = 1u;
    stack[0] = 0u; lower[0] = 0.0;
    var visits = 0u;
    loop {
        if (pending == 0u || visits >= 4096u) { break; }
        pending -= 1u;
        let node = stack[pending];
        let parentLower = lower[pending];
        if (parentLower > best) { continue; }
        visits += 1u;
        let entry = nodes[node];
        let axis = entry.x >> 30u;
        let data = entry.x & 0x3fffffffu;
        if (axis == 3u) {
            let end = min(entry.y, data + 4096u);
            for (var i = data; i < end; i += 1u) {
                let delta = points[i].xyz - q;
                let d = dot(delta, delta);
                if (d < best) { best = d; slot = i; }
            }
        } else {
            let delta = q[axis] - bitcast<f32>(entry.y);
            let near = select(data, node + 1u, delta < 0.0);
            let far = select(node + 1u, data, delta < 0.0);
            if (pending + 2u > 32u) { break; }
            stack[pending] = far;
            lower[pending] = select(delta * delta, max(parentLower, delta * delta), inheritedBounds);
            pending += 1u;
            stack[pending] = near;
            lower[pending] = select(0.0, parentLower, inheritedBounds);
            pending += 1u;
        }
    }
    candidates[id.x] = slot;
}`;

/** Preserve every GPU-used field; connectivity validation uses temporary bytes only. */
export function packGpuNodePairs(nodes: Uint32Array, pointCount: number): Uint32Array<ArrayBuffer> {
  if (!nodes.length || nodes.length % 5 || !Number.isSafeInteger(pointCount) || pointCount < 1 || pointCount > 2_000_000 ||
    nodes[0] !== 0 || nodes[1] !== pointCount) throw new Error("Invalid packed GPU node source dimensions");
  const count = nodes.length / 5;
  if (count > pointCount * 2 - 1 || count >= 0x40000000) throw new Error("Packed GPU node count exceeded");
  const packed = new Uint32Array(count * 2), seen = new Uint8Array(count);
  seen[0] = 1;
  for (let node = 0; node < count; node++) {
    const base = node * 5, start = nodes[base], end = nodes[base + 1], axis = nodes[base + 3];
    if (!seen[node] || start >= end || end > pointCount) throw new Error("Invalid packed GPU node range");
    const leaf = axis === 255, right = nodes[base + 2], left = node + 1;
    if (!leaf) {
      if (axis > 2 || left >= count || right <= left || right >= count || seen[left] || seen[right] ||
        nodes[left * 5] !== start || nodes[left * 5 + 1] !== nodes[right * 5] || nodes[right * 5 + 1] !== end)
        throw new Error("Invalid packed GPU child ranges");
      seen[left] = 1; seen[right] = 1;
    }
    packed[node * 2] = ((leaf ? 3 : axis) << 30) | (leaf ? start : right);
    packed[node * 2 + 1] = nodes[base + (leaf ? 1 : 4)];
  }
  return packed;
}

/** Boxes enclose the rounded points used by the GPU; exact distances stay in WASM. */
export function packGpuNodeBoxes(points: Float32Array, nodes: Uint32Array): Float32Array<ArrayBuffer> {
  if (!points.length || points.length % 4 || !nodes.length || nodes.length % 5)
    throw new Error("Invalid GPU box source dimensions");
  const count = nodes.length / 5, pointCount = points.length / 4;
  if (nodes[0] !== 0 || nodes[1] !== pointCount) throw new Error("Incomplete GPU box root range");
  const bounds = new Float32Array(count * 6);
  for (let node = count - 1; node >= 0; node--) {
    const base = node * 5, box = node * 6;
    const start = nodes[base], end = nodes[base + 1], axis = nodes[base + 3];
    if (start >= end || end > pointCount) throw new Error("Invalid GPU box node range");
    if (axis === 255) {
      for (let a = 0; a < 3; a++) {
        let lo = Infinity, hi = -Infinity;
        for (let i = start; i < end; i++) {
          const v = points[i * 4 + a];
          // Nonfinite rounded coordinates provide no exclusion on this axis.
          if (!Number.isFinite(v)) { lo = -Infinity; hi = Infinity; break; }
          lo = Math.min(lo, v); hi = Math.max(hi, v);
        }
        bounds[box + a] = lo; bounds[box + 3 + a] = hi;
      }
    } else {
      const left = node + 1, right = nodes[base + 2];
      if (axis > 2 || left >= count || right <= left || right >= count ||
        nodes[left * 5] !== start || nodes[left * 5 + 1] !== nodes[right * 5] || nodes[right * 5 + 1] !== end)
        throw new Error("Invalid GPU box child ranges");
      for (let a = 0; a < 3; a++) {
        bounds[box + a] = Math.min(bounds[left * 6 + a], bounds[right * 6 + a]);
        bounds[box + 3 + a] = Math.max(bounds[left * 6 + 3 + a], bounds[right * 6 + 3 + a]);
      }
    }
  }
  return bounds;
}

/** One parent per preorder node; root uses the missing-candidate sentinel. */
export function packGpuParentLinks(nodes: Uint32Array, pointCount: number): Uint32Array<ArrayBuffer> {
  if (!nodes.length || nodes.length % 5 || !Number.isSafeInteger(pointCount) || pointCount < 1 ||
    nodes[0] !== 0 || nodes[1] !== pointCount) throw new Error("Invalid GPU parent source dimensions");
  const count = nodes.length / 5, missing = 0xffffffff;
  const parents = new Uint32Array(count).fill(missing);
  for (let node = 0; node < count; node++) {
    const base = node * 5, start = nodes[base], end = nodes[base + 1], axis = nodes[base + 3];
    if (start >= end || end > pointCount) throw new Error("Invalid GPU parent node range");
    if (axis === 255) continue;
    const left = node + 1, right = nodes[base + 2];
    if (axis > 2 || left >= count || right <= left || right >= count ||
      parents[left] !== missing || parents[right] !== missing ||
      nodes[left * 5] !== start || nodes[left * 5 + 1] !== nodes[right * 5] || nodes[right * 5 + 1] !== end)
      throw new Error("Invalid GPU parent child ranges");
    parents[left] = node; parents[right] = node;
  }
  if (parents.subarray(1).some(parent => parent === missing)) throw new Error("Disconnected GPU parent nodes");
  return parents;
}

export function validateGpuWorkgroupSize(size: number): void {
  if (![32, 64, 128, 256].includes(size))
    throw new Error("WebGPU C2C workgroup size must be 32, 64, 128 or 256");
}

/** Device/pipeline shared by references; its error scopes and queue are serial. */
export class GpuCandidateDevice {
  private pipelines = new Map<string, GPUComputePipeline>();
  private preparing = new Map<string, Promise<GPUComputePipeline>>();
  private pending = Promise.resolve();
  private active = 0;
  private lost = false;
  get batchSize(): number { return this.batchLimit(64); }

  get usable(): boolean { return !this.lost; }
  get timestampQueries(): boolean { return this.device.features.has("timestamp-query"); }

  private constructor(readonly device: GPUDevice) {
    void device.lost.then(() => { this.lost = true; });
  }

  batchLimit(size: number): number {
    validateGpuWorkgroupSize(size);
    const limits = this.device.limits;
    if (size > limits.maxComputeInvocationsPerWorkgroup || size > limits.maxComputeWorkgroupSizeX)
      throw new Error("WebGPU workgroup size exceeds device limits");
    return Math.min(262_144, limits.maxComputeWorkgroupsPerDimension * size,
      Math.floor(limits.maxStorageBufferBindingSize / 16), Math.floor(limits.maxBufferSize / 16));
  }

  static async create(requirePhysical = false, inheritedBounds = false, timestamps = false): Promise<GpuCandidateDevice> {
    if (!globalThis.navigator?.gpu) throw new Error("WebGPU is unavailable in this browser");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No WebGPU adapter is available");
    if (requirePhysical && adapter.info.isFallbackAdapter) throw new Error("A physical WebGPU adapter is required");
    const runtime = new GpuCandidateDevice(await adapter.requestDevice(timestamps && adapter.features.has("timestamp-query")
      ? { requiredFeatures: ["timestamp-query"] } : undefined));
    try {
      await runtime.run(() => runtime.preparePipeline(false, 64, inheritedBounds));
      return runtime;
    } catch (error) {
      runtime.dispose();
      throw error;
    }
  }

  getPipeline(leafOnly: boolean, size: number, inheritedBounds = false, boxBounds = false, parentTraversal = false, compactStack = false, packedNodes = false): GPUComputePipeline | undefined {
    packedNodes &&= !leafOnly && !boxBounds && !parentTraversal;
    compactStack &&= !leafOnly && !boxBounds && !parentTraversal && !packedNodes;
    return this.pipelines.get(`${leafOnly}:${size}:${!leafOnly && !parentTraversal && inheritedBounds}:${!leafOnly && boxBounds}:${!leafOnly && parentTraversal}:${compactStack}:${packedNodes}`);
  }

  /** Called inside the serialized device scope; optional variants compile once. */
  preparePipeline(leafOnly: boolean, size: number, inheritedBounds = false, boxBounds = false, parentTraversal = false, compactStack = false, packedNodes = false): Promise<GPUComputePipeline> {
    this.batchLimit(size);
    packedNodes &&= !leafOnly && !boxBounds && !parentTraversal;
    compactStack &&= !leafOnly && !boxBounds && !parentTraversal && !packedNodes;
    if (!leafOnly && boxBounds && parentTraversal) throw new Error("Choose one GPU traversal variant");
    const key = `${leafOnly}:${size}:${!leafOnly && !parentTraversal && inheritedBounds}:${!leafOnly && boxBounds}:${!leafOnly && parentTraversal}:${compactStack}:${packedNodes}`;
    let ready = this.preparing.get(key);
    if (!ready) {
      const ready = this.device.createComputePipelineAsync({ layout: "auto", label: `c2c:${leafOnly ? "leaf" : parentTraversal ? "parents" : boxBounds ? "boxes" : packedNodes ? "packed" : compactStack ? "stack20" : "stack32"}:${size}`, compute: {
        module: this.device.createShaderModule({ code: packedNodes ? PACKED_SHADER : SHADER(compactStack ? 20 : 32) }),
        entryPoint: leafOnly ? "leafNearest" : parentTraversal ? "nearestParent" : boxBounds ? "nearestBox" : "nearest",
        constants: { workgroupSize: size, ...(leafOnly || parentTraversal ? {} : { inheritedBounds: Number(inheritedBounds) }) },
      } }).then(pipeline => { this.pipelines.set(key, pipeline); return pipeline; });
      this.preparing.set(key, ready);
      void ready.catch(() => { if (this.preparing.get(key) === ready) this.preparing.delete(key); });
      return ready;
    }
    return ready;
  }

  run<T>(work: () => Promise<T>): Promise<T> {
    // Start immediately when idle so next-batch dispatch can overlap CPU
    // verification. Serialize across references, not just within each one:
    // error scopes belong to the device and readback may still be in flight.
    const result = this.active++ === 0 ? this.checked(work)
      : this.pending.then(() => this.checked(work));
    this.pending = result.then(() => { this.active--; }, () => { this.active--; });
    return result;
  }

  private async checked<T>(work: () => Promise<T>): Promise<T> {
    if (this.lost) throw new Error("WebGPU device was lost");
    const __scope = __caScope(); __caMark("device-scope-enter", __scope);
    this.device.pushErrorScope("out-of-memory");
    this.device.pushErrorScope("internal");
    this.device.pushErrorScope("validation");
    try {
      return await work();
    } finally {
      // Pop all scopes even when an allocation, pipeline or map fails.
      __caMark("scope-pop-start", __scope);
      const errors = await Promise.all([
        this.device.popErrorScope(), this.device.popErrorScope(), this.device.popErrorScope(),
      ]);
      __caMark("scope-pop-end", __scope);
      const error = errors.find(Boolean);
      if (error) throw new Error(`WebGPU: ${error.message}`);
      if (this.lost) throw new Error("WebGPU device was lost");
    }
  }

  dispose(): void {
    this.lost = true;
    this.device.destroy();
  }
}

interface BatchBuffers {
  capacity: number; input: GPUBuffer; output: GPUBuffer; readback: GPUBuffer;
  count?: number; pipeline?: GPUComputePipeline; bindings?: GPUBindGroup;
}
interface BatchTimestamps { queries: GPUQuerySet; resolve: GPUBuffer; readback: GPUBuffer }
interface BatchSlot { work?: BatchBuffers; timestamps?: BatchTimestamps; pending: Promise<void>; active: number }

/** Reference buffers and reusable batches on either a shared or owned device. */
export class GpuCandidateSearch {
  private buffers: GPUBuffer[] = [];
  private querySets: GPUQuerySet[] = [];
  private points!: GPUBuffer;
  private nodeBuffers = new Map<boolean, GPUBuffer>();
  private nodePreparing = new Map<boolean, Promise<void>>();
  private originalNodeBytes = 0;
  private boxBuffer?: GPUBuffer;
  private boxPreparing?: Promise<void>;
  private boxPrepared = false;
  private parentBuffer?: GPUBuffer;
  private parentPreparing?: Promise<void>;
  private parentPrepared = false;
  private slots: BatchSlot[] = [{ pending: Promise.resolve(), active: 0 }, { pending: Promise.resolve(), active: 0 }];
  private nextSlot = 0;
  private disposed = false;

  get usable(): boolean { return !this.disposed && this.runtime.usable; }
  get batchSize(): number { return this.runtime.batchSize; }
  get timestampQueries(): boolean { return this.runtime.timestampQueries; }
  get boxBoundsReady(): boolean { return this.boxPrepared; }
  get boxBoundsBytes(): number { return this.boxBuffer?.size ?? 0; }
  get parentTraversalReady(): boolean { return this.parentPrepared; }
  get parentTraversalBytes(): number { return this.parentBuffer?.size ?? 0; }
  get nodeCount(): number { return this.originalNodeBytes / 20; }
  get retainedNodeBytes(): number { return this.nodeBytes(false) + this.nodeBytes(true); }
  nodeBytes(packed = false): number { return this.nodeBuffers.get(packed)?.size ?? 0; }
  nodeRepresentationReady(packed = false): boolean { return this.nodeBuffers.has(packed); }
  batchLimit(size: number): number { return this.runtime.batchLimit(size); }
  private get device(): GPUDevice { return this.runtime.device; }

  private constructor(private runtime: GpuCandidateDevice, private ownsDevice: boolean) {}

  static async create(points: Float32Array, nodes: Uint32Array, runtime?: GpuCandidateDevice, inheritedBounds = false, timestamps = false, packedNodes = false): Promise<GpuCandidateSearch> {
    const search = new GpuCandidateSearch(runtime ?? await GpuCandidateDevice.create(false, inheritedBounds, timestamps), !runtime);
    try {
      search.originalNodeBytes = nodes.byteLength;
      __caMark("reference-upload-queued");
      await search.runtime.run(async () => {
        __caMark("reference-upload-enter");
        search.points = search.upload(points, "points");
        __caMark("node-encode-start");
        const data = packedNodes ? packGpuNodePairs(nodes, points.length / 4) : nodes;
        __caMark("node-encode-end");
        search.nodeBuffers.set(packedNodes, search.upload(data, "nodes"));
        __caMark("reference-upload-work-end");
      });
      __caMark("reference-upload-ready");
      return search;
    } catch (error) {
      search.dispose();
      throw error;
    }
  }

  private buffer(size: number, usage: GPUBufferUsageFlags): GPUBuffer {
    if (size <= 0 || size > this.device.limits.maxBufferSize ||
      ((usage & GPUBufferUsage.STORAGE) && size > this.device.limits.maxStorageBufferBindingSize)) {
      throw new Error("WebGPU buffer limit exceeded");
    }
    const buffer = this.device.createBuffer({ size, usage });
    this.buffers.push(buffer);
    return buffer;
  }

  /** Lazily add a node representation; points and batch slots remain shared. */
  prepareNodes(packed: boolean, source: () => Uint32Array): Promise<void> {
    if (!this.usable) return Promise.reject(new Error("WebGPU reference was disposed or its device was lost"));
    if (this.nodeRepresentationReady(packed)) return Promise.resolve();
    let ready = this.nodePreparing.get(packed);
    if (!ready) {
      let buffer: GPUBuffer | undefined;
      ready = this.runtime.run(async () => {
        if (!this.usable) throw new Error("WebGPU reference was disposed or its device was lost");
        const nodes = source();
        if (nodes.byteLength !== this.originalNodeBytes) throw new Error("GPU node source does not match the reference dimensions");
        const data = packed ? packGpuNodePairs(nodes, this.points.size / 16) : nodes;
        buffer = this.buffer(data.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        this.device.queue.writeBuffer(buffer, 0, data as Uint32Array<ArrayBuffer>);
      }).then(() => {
        if (!this.usable) throw new Error("WebGPU reference was disposed or its device was lost");
        this.nodeBuffers.set(packed, buffer!);
      }).catch(error => {
        if (buffer) {
          buffer.destroy(); const i = this.buffers.indexOf(buffer); if (i >= 0) this.buffers.splice(i, 1);
        }
        this.nodePreparing.delete(packed);
        throw error;
      });
      this.nodePreparing.set(packed, ready);
    }
    return ready;
  }

  /** Export/pack/upload only once, sharing ordinary point/node buffers. */
  prepareBoxBounds(source: () => { points: Float32Array; nodes: Uint32Array }): Promise<void> {
    if (!this.usable) return Promise.reject(new Error("WebGPU reference was disposed or its device was lost"));
    if (this.boxPrepared) return Promise.resolve();
    if (!this.boxPreparing) {
      this.boxPreparing = this.runtime.run(async () => {
        if (!this.usable) throw new Error("WebGPU reference was disposed or its device was lost");
        const { points, nodes } = source();
        if (points.byteLength !== this.points.size || nodes.byteLength !== this.originalNodeBytes)
          throw new Error("GPU box source does not match the reference dimensions");
        const bounds = packGpuNodeBoxes(points, nodes);
        this.boxBuffer = this.buffer(bounds.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        this.device.queue.writeBuffer(this.boxBuffer, 0, bounds);
      }).then(() => { this.boxPrepared = true; }).catch(error => {
        if (this.boxBuffer) {
          this.boxBuffer.destroy(); this.buffers.splice(this.buffers.indexOf(this.boxBuffer), 1);
        }
        this.boxBuffer = undefined; this.boxPreparing = undefined;
        throw error;
      });
    }
    return this.boxPreparing;
  }

  /** Lazy immutable parents, with shared preparation and retry after failure. */
  prepareParentTraversal(source: () => Uint32Array): Promise<void> {
    if (!this.usable) return Promise.reject(new Error("WebGPU reference was disposed or its device was lost"));
    if (this.parentPrepared) return Promise.resolve();
    if (!this.parentPreparing) {
      this.parentPreparing = this.runtime.run(async () => {
        if (!this.usable) throw new Error("WebGPU reference was disposed or its device was lost");
        const nodes = source();
        if (nodes.byteLength !== this.originalNodeBytes) throw new Error("GPU parent source does not match the reference dimensions");
        const parents = packGpuParentLinks(nodes, this.points.size / 16);
        this.parentBuffer = this.buffer(parents.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        this.device.queue.writeBuffer(this.parentBuffer, 0, parents);
      }).then(() => { this.parentPrepared = true; }).catch(error => {
        if (this.parentBuffer) {
          this.parentBuffer.destroy(); this.buffers.splice(this.buffers.indexOf(this.parentBuffer), 1);
        }
        this.parentBuffer = undefined; this.parentPreparing = undefined;
        throw error;
      });
    }
    return this.parentPreparing;
  }

  private upload(data: Float32Array | Uint32Array, label = "reference"): GPUBuffer {
    __caMark(label + "-allocate-start");
    const buffer = this.buffer(data.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    __caMark(label + "-allocate-end");
    __caMark(label + "-write-start");
    this.device.queue.writeBuffer(buffer, 0, data as Float32Array<ArrayBuffer> | Uint32Array<ArrayBuffer>);
    __caMark(label + "-write-end");
    return buffer;
  }

  find(queries: Float32Array, leafOnly = false, workgroupSize = 64,
    bufferedReadback = false, inheritedBounds = false, timing?: import("./gpu-diagnostics").GpuBatchTiming, boxBounds = false, parentTraversal = false, compactStack = false, packedNodes = false): Promise<Uint32Array<ArrayBuffer>> {
    const slot = this.slots[bufferedReadback ? this.nextSlot++ % 2 : 0];
    // A slot stays leased through mapping/copy/unmap, including failed scopes.
    // The ordinary path keeps its historical single-buffer serialization.
    const execute = async () => {
      if (!bufferedReadback) {
        return this.runtime.run(async () => {
          const batch = await this.submitBatch(slot, queries, leafOnly, workgroupSize, inheritedBounds, timing, boxBounds, parentTraversal, compactStack, packedNodes);
          const result = await batch.result;
          if (!result.ok) throw result.error;
          return result.candidates;
        });
      }
      let batch: Awaited<ReturnType<GpuCandidateSearch["submitBatch"]>> | undefined;
      try {
        // Return a wrapper, not the readback promise: close this submission's
        // device scopes before another slot submits while mapping is in flight.
        await this.runtime.run(async () => {
          batch = await this.submitBatch(slot, queries, leafOnly, workgroupSize, inheritedBounds, timing, boxBounds, parentTraversal, compactStack, packedNodes);
        });
        const result = await batch!.result;
        if (!result.ok) throw result.error;
        return result.candidates;
      } finally {
        // A validation error can arrive before mapping settles. Drain it before
        // releasing the slot; the result promise handles map rejection at once.
        if (batch) await batch.result;
      }
    };
    // Keep idle-slot submission synchronous so it precedes main-worker WASM.
    const result = slot.active++ === 0 ? execute() : slot.pending.then(execute);
    slot.pending = result.then(() => { slot.active--; }, () => { slot.active--; });
    return result;
  }

  private async submitBatch(slot: BatchSlot, queries: Float32Array, leafOnly: boolean, workgroupSize: number, inheritedBounds: boolean,
    timing: import("./gpu-diagnostics").GpuBatchTiming | undefined, boxBounds: boolean, parentTraversal: boolean, compactStack: boolean, packedNodes: boolean) {
    const batchSize = this.batchLimit(workgroupSize);
    const count = queries.length / 4;
    if (!Number.isInteger(count) || count < 1 || count > batchSize) throw new Error("Invalid WebGPU query batch");
    if (!this.usable) throw new Error("WebGPU reference was disposed or its device was lost");
    // A separate entry point keeps the large DFS stacks out of the hint
    // kernel. Ready pipelines dispatch synchronously for CPU/GPU overlap.
    if (!leafOnly && boxBounds && parentTraversal) throw new Error("Choose one GPU traversal variant");
    if (parentTraversal && !leafOnly && !this.parentPrepared) throw new Error("GPU parent links are not prepared");
    if (boxBounds && !leafOnly && !this.boxPrepared) throw new Error("GPU box bounds are not prepared");
    packedNodes &&= !leafOnly && !boxBounds && !parentTraversal;
    const nodes = this.nodeBuffers.get(packedNodes);
    if (!nodes) throw new Error("GPU node representation is not prepared");
    const pipeline = this.runtime.getPipeline(leafOnly, workgroupSize, inheritedBounds, boxBounds, parentTraversal, compactStack, packedNodes) ??
      await this.runtime.preparePipeline(leafOnly, workgroupSize, inheritedBounds, boxBounds, parentTraversal, compactStack, packedNodes);
    if (!this.usable) throw new Error("WebGPU reference was disposed or its device was lost");
    if (!slot.work || slot.work.capacity < count) {
      // Grow only this idle slot; the other readback may still be mapped.
      if (slot.work) for (const buffer of [slot.work.input, slot.work.output, slot.work.readback]) {
        buffer.destroy();
        this.buffers.splice(this.buffers.indexOf(buffer), 1);
      }
      slot.work = undefined;
      const capacity = Math.min(batchSize, 2 ** Math.ceil(Math.log2(count)));
      const allocated: GPUBuffer[] = [];
      const allocate = (size: number, usage: GPUBufferUsageFlags) => {
        const buffer = this.buffer(size, usage);
        allocated.push(buffer);
        return buffer;
      };
      try {
        slot.work = { capacity,
          input: allocate(capacity * 16, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST),
          output: allocate(capacity * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC),
          readback: allocate(capacity * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST),
        };
      } catch (error) {
        for (const buffer of allocated) {
          buffer.destroy(); this.buffers.splice(this.buffers.indexOf(buffer), 1);
        }
        throw error;
      }
    }
    const work = slot.work;
    const { input, output, readback } = work;
    const uploadStarted = timing ? performance.now() : 0;
    this.device.queue.writeBuffer(input, 0, queries as Float32Array<ArrayBuffer>);
    if (timing) timing.uploadCallMillis = performance.now() - uploadStarted;
    if (work.count !== count || work.pipeline !== pipeline) {
      // Bind only the live range so arrayLength excludes retained capacity
      // and the final partial workgroup cannot access stale query values.
      work.bindings = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
        entries: (boxBounds && !leafOnly ? [this.points, nodes, input, output, this.boxBuffer!] : [this.points, nodes, input, output]).map((buffer, binding) => ({ binding,
          resource: { buffer, size: binding === 2 ? count * 16 : binding === 3 ? count * 4 : buffer.size } })).concat(
            parentTraversal && !leafOnly ? [{ binding: 5, resource: { buffer: this.parentBuffer!, size: this.parentBuffer!.size } }] : []),
      });
      work.count = count;
      work.pipeline = pipeline;
    }
    if (timing && this.timestampQueries && !slot.timestamps) {
      const queries = this.device.createQuerySet({ type: "timestamp", count: 2 });
      const allocated: GPUBuffer[] = [];
      try {
        const resolve = this.buffer(16, GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC);
        allocated.push(resolve);
        const readback = this.buffer(16, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
        allocated.push(readback);
        slot.timestamps = { queries, resolve, readback };
        this.querySets.push(queries);
      } catch (error) {
        queries.destroy();
        for (const buffer of allocated) { buffer.destroy(); this.buffers.splice(this.buffers.indexOf(buffer), 1); }
        throw error;
      }
    }
    const timestamps = timing ? slot.timestamps : undefined;
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass(timestamps ? { timestampWrites: {
      querySet: timestamps.queries, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1,
    } } : undefined);
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, work.bindings!);
    pass.dispatchWorkgroups(Math.ceil(count / workgroupSize));
    pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, count * 4);
    if (timestamps) {
      encoder.resolveQuerySet(timestamps.queries, 0, 2, timestamps.resolve, 0);
      encoder.copyBufferToBuffer(timestamps.resolve, 0, timestamps.readback, 0, 16);
    }
    this.device.queue.submit([encoder.finish()]);
    const readbackStarted = timing ? performance.now() : 0;
    const candidates = (async () => {
      await readback.mapAsync(GPUMapMode.READ, 0, count * 4);
      try { return new Uint32Array(readback.getMappedRange(0, count * 4).slice(0)); }
      finally { readback.unmap(); }
    })();
    const timestampResult = timestamps ? (async () => {
      await timestamps.readback.mapAsync(GPUMapMode.READ, 0, 16);
      try {
        const values = new BigUint64Array(timestamps.readback.getMappedRange(0, 16));
        const elapsed = values[1] - values[0];
        if (elapsed < 0n || elapsed > BigInt(Number.MAX_SAFE_INTEGER)) throw new Error("Invalid GPU timestamp interval");
        return Number(elapsed) / 1_000_000;
      } finally { timestamps.readback.unmap(); }
    })() : undefined;
    // Both maps must settle before releasing/reusing the slot, including when
    // either rejects. A timestamp failure must not leave a candidate map alive.
    const mapped = timestampResult ? Promise.allSettled([candidates, timestampResult]).then(results => {
      const [candidate, timestamp] = results;
      if (candidate.status === "rejected") throw candidate.reason;
      if (timestamp.status === "rejected") throw timestamp.reason;
      timing!.shaderMillis = timestamp.value;
      return candidate.value;
    }) : candidates;
    const settled = mapped.then(candidates => ({ ok: true as const, candidates }), error => ({ ok: false as const, error }));
    const result = timing ? settled.then(result => { timing.readbackMillis = performance.now() - readbackStarted; return result; }) : settled;
    return { result };
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const buffer of this.buffers.splice(0)) buffer.destroy();
    for (const queries of this.querySets.splice(0)) queries.destroy();
    for (const slot of this.slots) { slot.work = undefined; slot.timestamps = undefined; }
    this.boxBuffer = undefined; this.boxPrepared = false;
    this.parentBuffer = undefined; this.parentPrepared = false;
    this.nodeBuffers.clear(); this.nodePreparing.clear();
    if (this.ownsDevice) this.runtime.dispose();
  }
}
