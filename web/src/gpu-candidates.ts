/** Rounded GPU nearest-neighbour candidates. Callers must verify them in f64. */
const SHADER = `
override workgroupSize: u32 = 64u;
override inheritedBounds: bool = false;
@group(0) @binding(0) var<storage, read> points: array<vec4<f32>>;
// start, end, right child, axis (255 = leaf), split coordinate bits.
@group(0) @binding(1) var<storage, read> nodes: array<u32>;
@group(0) @binding(2) var<storage, read> queries: array<vec4<f32>>;
@group(0) @binding(3) var<storage, read_write> candidates: array<u32>;

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
}`;

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

  static async create(requirePhysical = false, inheritedBounds = false): Promise<GpuCandidateDevice> {
    if (!globalThis.navigator?.gpu) throw new Error("WebGPU is unavailable in this browser");
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("No WebGPU adapter is available");
    if (requirePhysical && adapter.info.isFallbackAdapter) throw new Error("A physical WebGPU adapter is required");
    const runtime = new GpuCandidateDevice(await adapter.requestDevice());
    try {
      await runtime.run(() => runtime.preparePipeline(false, 64, inheritedBounds));
      return runtime;
    } catch (error) {
      runtime.dispose();
      throw error;
    }
  }

  getPipeline(leafOnly: boolean, size: number, inheritedBounds = false): GPUComputePipeline | undefined {
    return this.pipelines.get(`${leafOnly}:${size}:${!leafOnly && inheritedBounds}`);
  }

  /** Called inside the serialized device scope; optional variants compile once. */
  preparePipeline(leafOnly: boolean, size: number, inheritedBounds = false): Promise<GPUComputePipeline> {
    this.batchLimit(size);
    const key = `${leafOnly}:${size}:${!leafOnly && inheritedBounds}`;
    let ready = this.preparing.get(key);
    if (!ready) {
      const ready = this.device.createComputePipelineAsync({ layout: "auto", compute: {
        module: this.device.createShaderModule({ code: SHADER }),
        entryPoint: leafOnly ? "leafNearest" : "nearest", constants: { workgroupSize: size, ...(leafOnly ? {} : { inheritedBounds: Number(inheritedBounds) }) },
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
    this.device.pushErrorScope("out-of-memory");
    this.device.pushErrorScope("internal");
    this.device.pushErrorScope("validation");
    try {
      return await work();
    } finally {
      // Pop all scopes even when an allocation, pipeline or map fails.
      const errors = await Promise.all([
        this.device.popErrorScope(), this.device.popErrorScope(), this.device.popErrorScope(),
      ]);
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
interface BatchSlot { work?: BatchBuffers; pending: Promise<void>; active: number }

/** Reference buffers and reusable batches on either a shared or owned device. */
export class GpuCandidateSearch {
  private buffers: GPUBuffer[] = [];
  private points!: GPUBuffer;
  private nodes!: GPUBuffer;
  private slots: BatchSlot[] = [{ pending: Promise.resolve(), active: 0 }, { pending: Promise.resolve(), active: 0 }];
  private nextSlot = 0;
  private disposed = false;

  get usable(): boolean { return !this.disposed && this.runtime.usable; }
  get batchSize(): number { return this.runtime.batchSize; }
  batchLimit(size: number): number { return this.runtime.batchLimit(size); }
  private get device(): GPUDevice { return this.runtime.device; }

  private constructor(private runtime: GpuCandidateDevice, private ownsDevice: boolean) {}

  static async create(points: Float32Array, nodes: Uint32Array, runtime?: GpuCandidateDevice, inheritedBounds = false): Promise<GpuCandidateSearch> {
    const search = new GpuCandidateSearch(runtime ?? await GpuCandidateDevice.create(false, inheritedBounds), !runtime);
    try {
      await search.runtime.run(async () => {
        search.points = search.upload(points);
        search.nodes = search.upload(nodes);
      });
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

  private upload(data: Float32Array | Uint32Array): GPUBuffer {
    const buffer = this.buffer(data.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
    this.device.queue.writeBuffer(buffer, 0, data as Float32Array<ArrayBuffer> | Uint32Array<ArrayBuffer>);
    return buffer;
  }

  find(queries: Float32Array, leafOnly = false, workgroupSize = 64,
    bufferedReadback = false, inheritedBounds = false): Promise<Uint32Array<ArrayBuffer>> {
    const slot = this.slots[bufferedReadback ? this.nextSlot++ % 2 : 0];
    // A slot stays leased through mapping/copy/unmap, including failed scopes.
    // The ordinary path keeps its historical single-buffer serialization.
    const execute = async () => {
      if (!bufferedReadback) {
        return this.runtime.run(async () => {
          const batch = await this.submitBatch(slot, queries, leafOnly, workgroupSize, inheritedBounds);
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
          batch = await this.submitBatch(slot, queries, leafOnly, workgroupSize, inheritedBounds);
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

  private async submitBatch(slot: BatchSlot, queries: Float32Array, leafOnly: boolean, workgroupSize: number, inheritedBounds: boolean) {
    const batchSize = this.batchLimit(workgroupSize);
    const count = queries.length / 4;
    if (!Number.isInteger(count) || count < 1 || count > batchSize) throw new Error("Invalid WebGPU query batch");
    if (!this.usable) throw new Error("WebGPU reference was disposed or its device was lost");
    // A separate entry point keeps the large DFS stacks out of the hint
    // kernel. Ready pipelines dispatch synchronously for CPU/GPU overlap.
    const pipeline = this.runtime.getPipeline(leafOnly, workgroupSize, inheritedBounds) ??
      await this.runtime.preparePipeline(leafOnly, workgroupSize, inheritedBounds);
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
    this.device.queue.writeBuffer(input, 0, queries as Float32Array<ArrayBuffer>);
    if (work.count !== count || work.pipeline !== pipeline) {
      // Bind only the live range so arrayLength excludes retained capacity
      // and the final partial workgroup cannot access stale query values.
      work.bindings = this.device.createBindGroup({ layout: pipeline.getBindGroupLayout(0),
        entries: [this.points, this.nodes, input, output].map((buffer, binding) => ({ binding,
          resource: { buffer, size: binding === 2 ? count * 16 : binding === 3 ? count * 4 : buffer.size } })),
      });
      work.count = count;
      work.pipeline = pipeline;
    }
    const encoder = this.device.createCommandEncoder();
    const pass = encoder.beginComputePass();
    pass.setPipeline(pipeline);
    pass.setBindGroup(0, work.bindings!);
    pass.dispatchWorkgroups(Math.ceil(count / workgroupSize));
    pass.end();
    encoder.copyBufferToBuffer(output, 0, readback, 0, count * 4);
    this.device.queue.submit([encoder.finish()]);
    const result = (async () => {
      await readback.mapAsync(GPUMapMode.READ, 0, count * 4);
      try { return new Uint32Array(readback.getMappedRange(0, count * 4).slice(0)); }
      finally { readback.unmap(); }
    })().then(candidates => ({ ok: true as const, candidates }), error => ({ ok: false as const, error }));
    return { result };
  }

  dispose(): void {
    if (this.disposed) return;
    this.disposed = true;
    for (const buffer of this.buffers.splice(0)) buffer.destroy();
    for (const slot of this.slots) slot.work = undefined;
    if (this.ownsDevice) this.runtime.dispose();
  }
}
