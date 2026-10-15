import { GpuCandidateDevice } from "./gpu-candidates";

const SHADER = `
struct Parameters { count: u32, unused0: u32, unused1: u32, unused2: u32 }
@group(0) @binding(0) var<storage, read> nodes: array<u32>;
@group(0) @binding(1) var<storage, read> triangles: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read> queries: array<vec4<f32>>;
@group(0) @binding(3) var<uniform> params: Parameters;
@group(0) @binding(4) var<storage, read_write> candidates: array<u32>;
fn lower(node: u32, q: vec3<f32>) -> f32 {
    let b = node * 8u;
    let lo = vec3<f32>(bitcast<f32>(nodes[b]),bitcast<f32>(nodes[b+1u]),bitcast<f32>(nodes[b+2u]));
    let hi = vec3<f32>(bitcast<f32>(nodes[b+3u]),bitcast<f32>(nodes[b+4u]),bitcast<f32>(nodes[b+5u]));
    let gap = max(max(lo-q,q-hi),vec3<f32>(0.0)); return dot(gap,gap);
}
fn closest(p: vec3<f32>, a: vec3<f32>, b: vec3<f32>, c: vec3<f32>) -> vec3<f32> {
    let ab=b-a; let ac=c-a; let ap=p-a; let d1=dot(ab,ap); let d2=dot(ac,ap);
    if(d1<=0.0 && d2<=0.0){return a;}
    let bp=p-b; let d3=dot(ab,bp); let d4=dot(ac,bp);
    if(d3>=0.0 && d4<=d3){return b;}
    let vc=d1*d4-d3*d2;
    if(vc<=0.0 && d1>=0.0 && d3<=0.0){return a+ab*(d1/(d1-d3));}
    let cp=p-c; let d5=dot(ab,cp); let d6=dot(ac,cp);
    if(d6>=0.0 && d5<=d6){return c;}
    let vb=d5*d2-d1*d6;
    if(vb<=0.0 && d2>=0.0 && d6<=0.0){return a+ac*(d2/(d2-d6));}
    let va=d3*d6-d5*d4;
    if(va<=0.0 && d4-d3>=0.0 && d5-d6>=0.0){let w=(d4-d3)/((d4-d3)+(d5-d6));return b+(c-b)*w;}
    let denom=1.0/(va+vb+vc); return a+(ab*(vb*denom)+ac*(vc*denom));
}
@compute @workgroup_size(64)
fn nearest(@builtin(global_invocation_id) id: vec3<u32>) {
    if(id.x>=params.count){return;}
    let q=queries[id.x].xyz; var best=3.402823e38f; var slot=0xffffffffu;
    var stack: array<u32,32>; var size=1u; stack[0]=0u;
    for(var visits=0u;visits<4096u && size>0u;visits+=1u){
        size-=1u; let node=stack[size]; let b=node*8u;
        if(b+7u>=arrayLength(&nodes)){break;}
        if(lower(node,q)>=best){continue;}
        let start=nodes[b+6u]; let count=nodes[b+7u];
        if(count>0u){
            let end=min(start+count,arrayLength(&triangles)/3u);
            for(var t=start;t<end;t+=1u){
                let hit=closest(q,triangles[t*3u].xyz,triangles[t*3u+1u].xyz,triangles[t*3u+2u].xyz);
                let delta=q-hit; let d=dot(delta,delta); if(d<best){best=d;slot=t;}
            }
        }else{
            let left=node+1u; let right=start;
            if(size+2u>32u || right*8u+7u>=arrayLength(&nodes) || left*8u+7u>=arrayLength(&nodes)){break;}
            let dl=lower(left,q); let dr=lower(right,q);
            stack[size]=select(left,right,dl<=dr);stack[size+1u]=select(right,left,dl<=dr);size+=2u;
        }
    }
    candidates[id.x]=slot;
}`;

const BUDGET = 256 * 1024 * 1024;
let allocated = 0;
let shared: Promise<GpuCandidateDevice> | undefined;
const pipelines = new WeakMap<GPUDevice, Promise<GPUComputePipeline>>();
export async function meshGpuDevice(): Promise<GpuCandidateDevice> {
  const old = shared && await shared;
  if (old && !old.usable) { old.dispose(); shared = undefined; }
  if (!shared) {
    const ready = GpuCandidateDevice.create(true); shared = ready;
    void ready.catch(() => { if (shared === ready) shared = undefined; });
  }
  return shared;
}

/** Rounded geometry supplies only one untrusted triangle slot per query. */
export class GpuMeshReference {
  private buffers: GPUBuffer[] = [];
  private sources: GPUBuffer[] = [];
  private params!: GPUBuffer;
  private queries?: GPUBuffer;
  private output?: GPUBuffer;
  private readback?: GPUBuffer;
  private capacity = 0;
  private disposed = false;
  private constructor(private runtime: GpuCandidateDevice, private origin: Float64Array, private pipeline: GPUComputePipeline) {}

  static async create(runtime: GpuCandidateDevice, origin: Float64Array, nodes: Uint32Array, triangles: Float32Array): Promise<GpuMeshReference> {
    if (origin.length !== 3 || !origin.every(Number.isFinite) || !nodes.length || nodes.length % 8 || !triangles.length || triangles.length % 12)
      throw new Error("Invalid GPU mesh buffers");
    let ref: GpuMeshReference | undefined;
    try {
      return await runtime.run(async () => {
        let ready = pipelines.get(runtime.device);
        if (!ready) {
          ready = runtime.device.createComputePipelineAsync({ layout: "auto", compute: {
            module: runtime.device.createShaderModule({ code: SHADER }), entryPoint: "nearest",
          } }); pipelines.set(runtime.device, ready);
          void ready.catch(() => { if (pipelines.get(runtime.device) === ready) pipelines.delete(runtime.device); });
        }
        ref = new GpuMeshReference(runtime, origin, await ready);
        for (const data of [nodes,triangles]) {
          const b = ref.buffer(data.byteLength,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);
          runtime.device.queue.writeBuffer(b,0,data as Uint32Array<ArrayBuffer>|Float32Array<ArrayBuffer>); ref.sources.push(b);
        }
        ref.params = ref.buffer(16,GPUBufferUsage.UNIFORM|GPUBufferUsage.COPY_DST);
        return ref;
      });
    } catch (error) { ref?.dispose(); throw error; }
  }

  private buffer(size: number, usage: GPUBufferUsageFlags): GPUBuffer {
    const limits = this.runtime.device.limits;
    if (size < 1 || size > limits.maxBufferSize || (usage & GPUBufferUsage.STORAGE) && size > limits.maxStorageBufferBindingSize)
      throw new Error("GPU mesh buffer exceeds device limits");
    if (allocated + size > BUDGET) throw new Error("GPU mesh buffer budget exceeded");
    const b = this.runtime.device.createBuffer({size,usage}); this.buffers.push(b); allocated += size; return b;
  }
  private destroy(b: GPUBuffer) { b.destroy(); allocated -= b.size; this.buffers.splice(this.buffers.indexOf(b),1); }
  get bytes(): number { return this.buffers.reduce((n,b)=>n+b.size,0); }

  async find(points: Float64Array): Promise<{ hints: Uint32Array<ArrayBuffer>; batches: number; completeRows: number }> {
    if (points.length % 3) throw new Error("Mesh queries must be XYZ triples");
    const total = points.length/3, hints = new Uint32Array(total), limit = Math.min(65_536,this.runtime.batchLimit(64));
    let batches = 0, completeRows = 0;
    for (let first = 0; first < total; first += limit) {
      const count = Math.min(limit,total-first), packed = new Float32Array(count*4);
      for(let i=0;i<count;i++)for(let a=0;a<3;a++)packed[i*4+a]=points[(first+i)*3+a]-this.origin[a];
      const row = await this.runtime.run(async () => {
        if (this.disposed) throw new Error("GPU mesh reference was retired");
        if (this.capacity < count) {
          for (const b of [this.queries,this.output,this.readback]) if(b)this.destroy(b);
          this.queries=this.output=this.readback=undefined;this.capacity=0;
          const capacity=Math.min(limit,2**Math.ceil(Math.log2(count)));
          this.queries=this.buffer(capacity*16,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_DST);
          this.output=this.buffer(capacity*4,GPUBufferUsage.STORAGE|GPUBufferUsage.COPY_SRC);
          this.readback=this.buffer(capacity*4,GPUBufferUsage.MAP_READ|GPUBufferUsage.COPY_DST);this.capacity=capacity;
        }
        const d=this.runtime.device;d.queue.writeBuffer(this.queries!,0,packed);d.queue.writeBuffer(this.params,0,new Uint32Array([count,0,0,0]));
        const bindings=d.createBindGroup({layout:this.pipeline.getBindGroupLayout(0),entries:[...this.sources,this.queries!,this.params,this.output!].map((buffer,binding)=>({binding,resource:{buffer,size:binding===2?count*16:binding===4?count*4:buffer.size}}))});
        const encoder=d.createCommandEncoder(),pass=encoder.beginComputePass();pass.setPipeline(this.pipeline);pass.setBindGroup(0,bindings);pass.dispatchWorkgroups(Math.ceil(count/64));pass.end();
        encoder.copyBufferToBuffer(this.output!,0,this.readback!,0,count*4);d.queue.submit([encoder.finish()]);
        await this.readback!.mapAsync(GPUMapMode.READ,0,count*4);
        try{return new Uint32Array(this.readback!.getMappedRange(0,count*4).slice(0));}finally{this.readback!.unmap();}
      });
      hints.set(row,first);batches++;for(const slot of row)if(slot!==0xffffffff)completeRows++;
    }
    return {hints,batches,completeRows};
  }

  dispose(): void { this.disposed=true;for(const b of [...this.buffers])this.destroy(b); }
}
