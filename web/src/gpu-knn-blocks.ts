import { GpuCandidateDevice } from "./gpu-candidates";

// This selects only an untrusted block, never final neighbours or distances.
const SHADER = `
struct Parameters { first: u32, count: u32, k: u32, unused: u32 }
@group(0) @binding(0) var<storage, read> nodes: array<u32>;
@group(0) @binding(1) var<storage, read> queries: array<vec4<f32>>;
@group(0) @binding(2) var<uniform> params: Parameters;
@group(0) @binding(3) var<storage, read_write> hints: array<u32>;
@compute @workgroup_size(64)
fn block(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= params.count) { return; }
    let q = queries[params.first + id.x].xyz;
    var node = 0u;
    for (var depth = 0u; depth < 32u; depth += 1u) {
        let base = node * 5u;
        if (base + 4u >= arrayLength(&nodes)) { break; }
        let axis = nodes[base + 3u];
        if (axis == 255u) { hints[id.x] = node; return; }
        let right = nodes[base + 2u];
        let leftBase = (node + 1u) * 5u;
        let rightBase = right * 5u;
        if (axis > 2u || leftBase + 4u >= arrayLength(&nodes) || rightBase + 4u >= arrayLength(&nodes)) { break; }
        let children = min(nodes[leftBase + 1u] - nodes[leftBase], nodes[rightBase + 1u] - nodes[rightBase]);
        if (children < params.k) { hints[id.x] = node; return; }
        node = select(right, node + 1u, q[axis] < bitcast<f32>(nodes[base + 4u]));
    }
    hints[id.x] = 0xffffffffu;
}`;

let shared: Promise<GpuCandidateDevice> | undefined;
const pipelines = new WeakMap<GPUDevice, Promise<GPUComputePipeline>>();
export async function knnGpuDevice(): Promise<GpuCandidateDevice> {
  const old = shared && await shared;
  if (old && !old.usable) { old.dispose(); shared = undefined; }
  if (!shared) {
    const ready = GpuCandidateDevice.create(true);
    shared = ready;
    void ready.catch(() => { if (shared === ready) shared = undefined; });
  }
  return shared;
}

/** Immutable per-part queries/splits and one reusable serialized readback. */
export class GpuKnnBlocks {
  private buffers: GPUBuffer[] = [];
  private nodes!: GPUBuffer;
  private queries!: GPUBuffer;
  private params!: GPUBuffer;
  private output?: GPUBuffer;
  private readback?: GPUBuffer;
  private capacity = 0;
  private disposed = false;
  private constructor(private runtime: GpuCandidateDevice, readonly count: number, private pipeline: GPUComputePipeline) {}

  static async create(runtime: GpuCandidateDevice, nodes: Uint32Array, queries: Float32Array): Promise<GpuKnnBlocks> {
    if (!nodes.length || nodes.length % 5 || !queries.length || queries.length % 4) throw new Error("Invalid GPU neighbourhood buffers");
    let part: GpuKnnBlocks | undefined;
    try {
      return await runtime.run(async () => {
        let ready = pipelines.get(runtime.device);
        if (!ready) {
          ready = runtime.device.createComputePipelineAsync({ layout: "auto", compute: {
            module: runtime.device.createShaderModule({ code: SHADER }), entryPoint: "block",
          } });
          pipelines.set(runtime.device, ready);
          void ready.catch(() => { if (pipelines.get(runtime.device) === ready) pipelines.delete(runtime.device); });
        }
        part = new GpuKnnBlocks(runtime, queries.length / 4, await ready);
        part.nodes = part.buffer(nodes.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        part.queries = part.buffer(queries.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST);
        part.params = part.buffer(16, GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST);
        runtime.device.queue.writeBuffer(part.nodes, 0, nodes as Uint32Array<ArrayBuffer>);
        runtime.device.queue.writeBuffer(part.queries, 0, queries as Float32Array<ArrayBuffer>);
        return part;
      });
    } catch (error) { part?.dispose(); throw error; }
  }

  private buffer(size: number, usage: GPUBufferUsageFlags): GPUBuffer {
    const limits = this.runtime.device.limits;
    if (size < 1 || size > limits.maxBufferSize || (usage & GPUBufferUsage.STORAGE) && size > limits.maxStorageBufferBindingSize)
      throw new Error("GPU neighbourhood buffer exceeds device limits");
    const buffer = this.runtime.device.createBuffer({ size, usage });
    this.buffers.push(buffer); return buffer;
  }

  get bytes(): number { return this.buffers.reduce((n, b) => n + b.size, 0); }

  async find(k: number): Promise<{ hints: Uint32Array<ArrayBuffer>; batches: number }> {
    if (!Number.isInteger(k) || k < 1 || k > 32) throw new Error("GPU neighbourhood k must be in 1..32");
    const hints = new Uint32Array(this.count);
    const limit = Math.min(65_536, this.runtime.batchLimit(64));
    let batches = 0;
    for (let first = 0; first < this.count; first += limit) {
      const count = Math.min(limit, this.count - first);
      const row = await this.runtime.run(async () => {
        if (this.disposed) throw new Error("GPU neighbourhood buffers were retired");
        if (this.capacity < count) {
          for (const buffer of [this.output, this.readback]) if (buffer) {
            buffer.destroy(); this.buffers.splice(this.buffers.indexOf(buffer), 1);
          }
          this.output = this.readback = undefined; this.capacity = 0;
          const capacity = Math.min(limit, 2 ** Math.ceil(Math.log2(count)));
          this.output = this.buffer(capacity * 4, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC);
          this.readback = this.buffer(capacity * 4, GPUBufferUsage.MAP_READ | GPUBufferUsage.COPY_DST);
          this.capacity = capacity;
        }
        const device = this.runtime.device;
        device.queue.writeBuffer(this.params, 0, new Uint32Array([first, count, k, 0]));
        const bindings = device.createBindGroup({ layout: this.pipeline.getBindGroupLayout(0), entries:
          [this.nodes, this.queries, this.params, this.output!].map((buffer, binding) => ({ binding,
            resource: { buffer, size: binding === 3 ? count * 4 : buffer.size } })) });
        const encoder = device.createCommandEncoder(), pass = encoder.beginComputePass();
        pass.setPipeline(this.pipeline); pass.setBindGroup(0, bindings); pass.dispatchWorkgroups(Math.ceil(count / 64)); pass.end();
        encoder.copyBufferToBuffer(this.output!, 0, this.readback!, 0, count * 4); device.queue.submit([encoder.finish()]);
        await this.readback!.mapAsync(GPUMapMode.READ, 0, count * 4);
        try { return new Uint32Array(this.readback!.getMappedRange(0, count * 4).slice(0)); }
        finally { this.readback!.unmap(); }
      });
      hints.set(row, first); batches++;
    }
    return { hints, batches };
  }

  dispose(): void {
    this.disposed = true;
    for (const buffer of this.buffers.splice(0)) buffer.destroy();
  }
}
