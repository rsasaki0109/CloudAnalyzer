import { GpuCandidateDevice } from "./gpu-candidates";

// Bounded f32 search only supplies untrusted point slots. Exact f64 CPU
// traversal chooses every final neighbour and preserves historical tie order.
const SHADER = `
struct Parameters { first: u32, count: u32, k: u32, unused: u32 }
@group(0) @binding(0) var<storage, read> nodes: array<u32>;
@group(0) @binding(1) var<storage, read> points: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> queries: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: Parameters;
@group(0) @binding(4) var<storage, read_write> hints: array<u32>;
@compute @workgroup_size(64)
fn candidates(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.count) { return; }
    let q = queries[params.first + id.x].xyz;
    var distances: array<f32, 32>;
    var slots: array<u32, 32>;
    for (var j = 0u; j < params.k; j += 1u) {
        distances[j] = 3.402823e38f; slots[j] = 0xffffffffu;
    }
    var pending: array<u32, 32>;
    var lower: array<f32, 32>;
    var size = 1u; pending[0] = 0u; lower[0] = 0.0;
    // A work cap also bounds pathological ties/rounding. Incomplete rows use
    // sentinels; the verifier then falls back to the original CPU query.
    for (var visits = 0u; visits < 256u && size > 0u; visits += 1u) {
        size -= 1u;
        if (lower[size] > distances[params.k - 1u]) { continue; }
        let node = pending[size]; let base = node * 5u;
        if (base + 4u >= arrayLength(&nodes)) { break; }
        let axis = nodes[base + 3u];
        if (axis == 255u) {
            let end = min(nodes[base + 1u], arrayLength(&points));
            for (var slot = nodes[base]; slot < end; slot += 1u) {
                let delta = points[slot].xyz - q;
                let d = dot(delta, delta);
                if (!(d < distances[params.k - 1u])) { continue; }
                var at = params.k - 1u;
                loop {
                    if (at == 0u || !(d < distances[at - 1u])) { break; }
                    distances[at] = distances[at - 1u]; slots[at] = slots[at - 1u]; at -= 1u;
                }
                distances[at] = d; slots[at] = slot;
            }
        } else {
            if (axis > 2u || size + 2u > 32u) { break; }
            let diff = q[axis] - bitcast<f32>(nodes[base + 4u]);
            let near = select(nodes[base + 2u], node + 1u, diff < 0.0);
            let far = select(node + 1u, nodes[base + 2u], diff < 0.0);
            pending[size] = far; lower[size] = diff * diff;
            pending[size + 1u] = near; lower[size + 1u] = 0.0; size += 2u;
        }
    }
    for (var j = 0u; j < params.k; j += 1u) { hints[id.x * params.k + j] = slots[j]; }
}`;

const pipelines = new WeakMap<GPUDevice, Promise<GPUComputePipeline>>();

/** Immutable per-part tree/queries with serialized grow-only result buffers. */
export class GpuKnnCandidates {
  private buffers: GPUBuffer[] = [];
  private inputs: GPUBuffer[] = [];
  private output?: GPUBuffer;
  private readback?: GPUBuffer;
  private capacity = 0;
  private disposed = false;
  private constructor(private runtime: GpuCandidateDevice, readonly count: number, private pipeline: GPUComputePipeline) {}

  static async create(runtime: GpuCandidateDevice, nodes: Uint32Array, queries: Float32Array, points: Float32Array): Promise<GpuKnnCandidates> {
    if (!nodes.length || nodes.length % 5 || !queries.length || queries.length % 4 || points.length !== queries.length)
      throw new Error("Invalid GPU nearest-k buffers");
    let part: GpuKnnCandidates | undefined;
    try {
      return await runtime.run(async () => {
        let ready = pipelines.get(runtime.device);
        if (!ready) {
          ready = runtime.device.createComputePipelineAsync({ layout: "auto", compute: {
            module: runtime.device.createShaderModule({ code: SHADER }), entryPoint: "candidates",
          } });
          pipelines.set(runtime.device, ready);
          void ready.catch(() => { if (pipelines.get(runtime.device) === ready) pipelines.delete(runtime.device); });
        }
        part = new GpuKnnCandidates(runtime, queries.length / 4, await ready);
        for (const data of [nodes, points, queries]) {
          const buffer = part.buffer(data.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
          runtime.device.queue.writeBuffer(buffer, 0, data as Uint32Array<ArrayBuffer> | Float32Array<ArrayBuffer>);
          part.inputs.push(buffer);
        }
        part.inputs.push(part.buffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST));
        return part;
      });
    } catch (error) { part?.dispose(); throw error; }
  }

  private buffer(size: number, usage: GPUBufferUsageFlags): GPUBuffer {
    const limits = this.runtime.device.limits;
    if (size < 1 || size > limits.maxBufferSize || (usage & GPUBufferUsage.STORAGE) && size > limits.maxStorageBufferBindingSize)
      throw new Error("GPU nearest-k buffer exceeds device limits");
    const buffer = this.runtime.device.createBuffer({ size, usage }); this.buffers.push(buffer); return buffer;
  }

  get bytes(): number { return this.buffers.reduce((n, b) => n + b.size, 0); }

  async find(k: number): Promise<{ hints: Uint32Array<ArrayBuffer>; batches: number; completeRows: number }> {
    if (!Number.isInteger(k) || k < 1 || k > 32) throw new Error("GPU nearest-k k must be in 1..32");
    const hints = new Uint32Array(this.count * k);
    const limits = this.runtime.device.limits;
    const limit = Math.min(16_384, this.runtime.batchLimit(64), Math.floor(Math.min(limits.maxBufferSize, limits.maxStorageBufferBindingSize) / (32 * 4)));
    if (limit < 1) throw new Error("GPU nearest-k output exceeds device limits");
    let batches = 0, completeRows = 0;
    for (let first = 0; first < this.count; first += limit) {
      const count = Math.min(limit, this.count - first), bytes = count * k * 4;
      const row = await this.runtime.run(async () => {
        if (this.disposed) throw new Error("GPU nearest-k buffers were retired");
        if (this.capacity < bytes) {
          for (const buffer of [this.output, this.readback]) if (buffer) {
            buffer.destroy(); this.buffers.splice(this.buffers.indexOf(buffer), 1);
          }
          this.output = this.readback = undefined; this.capacity = 0;
          // Reserve the largest supported row, so changing k needs no reallocation.
          this.output = this.buffer(limit * 32 * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
          this.readback = this.buffer(limit * 32 * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
          this.capacity = limit * 32 * 4;
        }
        const device = this.runtime.device;
        device.queue.writeBuffer(this.inputs[3], 0, new Uint32Array([first, count, k, 0]));
        const bindings = device.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries:
          [...this.inputs, this.output!].map((buffer, binding) => ({ binding, resource: { buffer, size: binding === 4 ? bytes : buffer.size } })) });
        const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
        pass.setPipeline(this.pipeline); pass.setBindGroup(0, bindings); pass.dispatchWorkgroups(Math.ceil(count / 64)); pass.end();
        encoder.copyBufferToBuffer(this.output!, 0, this.readback!, 0, bytes); device.queue.submit([encoder.finish()]);
        await this.readback!.mapAsync(GPUMapMode.READ, 0, bytes);
        try { return new Uint32Array(this.readback!.getMappedRange(0, bytes).slice(0)); }
        finally { this.readback!.unmap(); }
      });
      hints.set(row, first * k); batches++;
      for (let j = k - 1; j < row.length; j += k) if (row[j] !== 0xffffffff) completeRows++;
    }
    return { hints, batches, completeRows };
  }

  dispose(): void { this.disposed = true; for (const buffer of this.buffers.splice(0)) buffer.destroy(); }
}
