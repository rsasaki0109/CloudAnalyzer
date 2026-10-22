// Measurement-only transforms. Production sources and shaders are never edited.
export const PREPARATION_EVENT_LIMIT = 256;
export const preparationPrelude = `
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
`;
export const preparationTargets = ['web/src/gpu-candidates.ts', 'web/src/gpu-c2c.ts', 'web/src/gpu-verifiers.ts'];

export function instrumentPreparationSource(name, original) {
  if (!preparationTargets.includes(name)) throw new Error(`Unexpected preparation target: ${name}`);
  let source = original;
  const replace = (before, after) => {
    const key = before.startsWith(' ') ? '\n' + before : before;
    const value = before.startsWith(' ') ? '\n' + after : after;
    if (source.split(key).length !== 2) throw new Error(`Preparation target changed: ${name}: ${before.slice(0, 80)}`);
    source = source.replace(key, value);
  };
  if (name.endsWith('gpu-candidates.ts')) {
    replace(`    if (this.lost) throw new Error("WebGPU device was lost");
    this.device.pushErrorScope("out-of-memory");`, `    if (this.lost) throw new Error("WebGPU device was lost");
    const __scope = __caScope(); __caMark("device-scope-enter", __scope);
    this.device.pushErrorScope("out-of-memory");`);
    replace(`      const errors = await Promise.all([
        this.device.popErrorScope(), this.device.popErrorScope(), this.device.popErrorScope(),
      ]);`, `      __caMark("scope-pop-start", __scope);
      const errors = await Promise.all([
        this.device.popErrorScope(), this.device.popErrorScope(), this.device.popErrorScope(),
      ]);
      __caMark("scope-pop-end", __scope);`);
    replace(`      await search.runtime.run(async () => {
        search.points = search.upload(points);
        search.nodeBuffers.set(packedNodes, search.upload(packedNodes ? packGpuNodePairs(nodes, points.length / 4) : nodes));
      });`, `      __caMark("reference-upload-queued");
      await search.runtime.run(async () => {
        __caMark("reference-upload-enter");
        search.points = search.upload(points, "points");
        __caMark("node-encode-start");
        const data = packedNodes ? packGpuNodePairs(nodes, points.length / 4) : nodes;
        __caMark("node-encode-end");
        search.nodeBuffers.set(packedNodes, search.upload(data, "nodes"));
        __caMark("reference-upload-work-end");
      });
      __caMark("reference-upload-ready");`);
    replace(`  private upload(data: Float32Array | Uint32Array): GPUBuffer {
    const buffer = this.buffer(data.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, this.mappedReference);
    try {
      if (this.mappedReference) {
        try {
          new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
        } finally { buffer.unmap(); }
      } else {
        this.device.queue.writeBuffer(buffer, 0, data as Float32Array<ArrayBuffer> | Uint32Array<ArrayBuffer>);
      }
      return buffer;
    } catch (error) {
      // Upload can fail before the caller receives the allocated buffer.
      buffer.destroy();
      const i = this.buffers.indexOf(buffer); if (i >= 0) this.buffers.splice(i, 1);
      throw error;
    }
  }`, `  private upload(data: Float32Array | Uint32Array, label = "reference"): GPUBuffer {
    __caMark(label + "-allocate-start");
    const buffer = this.buffer(data.byteLength, GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST, this.mappedReference);
    __caMark(label + "-allocate-end");
    __caMark(label + "-write-start");
    try {
      if (this.mappedReference) {
        try {
          new Uint8Array(buffer.getMappedRange()).set(new Uint8Array(data.buffer, data.byteOffset, data.byteLength));
        } finally { buffer.unmap(); }
      } else {
        this.device.queue.writeBuffer(buffer, 0, data as Float32Array<ArrayBuffer> | Uint32Array<ArrayBuffer>);
      }
      __caMark(label + "-write-end");
      return buffer;
    } catch (error) {
      // Upload can fail before the caller receives the allocated buffer.
      buffer.destroy();
      const i = this.buffers.indexOf(buffer); if (i >= 0) this.buffers.splice(i, 1);
      throw error;
    }
  }`);
  } else if (name.endsWith('gpu-c2c.ts')) {
    replace('  const start = performance.now();', '  __caMark("c2c-start");\n  const start = performance.now();');
    replace(`    if (current) retire(current);
    current = { cloud: reference, index: new GpuC2cReference(reference), users: 0,
      retired: false, disposed: false, reuseDevice, searches: new Map() };`, `    __caMark("reference-retire-start"); if (current) retire(current); __caMark("reference-retire-end");
    __caMark("tree-start");
    current = { cloud: reference, index: new GpuC2cReference(reference), users: 0,
      retired: false, disposed: false, reuseDevice, searches: new Map() };
    __caMark("tree-end");`);
    replace('    job = new GpuC2cJob(compared, entry.index);', '    __caMark("query-snapshot-start"); job = new GpuC2cJob(compared, entry.index); __caMark("query-snapshot-end");');
    replace('}).finally(() => { verifyPrepareMillis = performance.now() - started; });', '}).finally(() => { verifyPrepareMillis = performance.now() - started; __caMark("verifiers-ready"); });');
    replace('      const packed = entry.index.candidateBuffers();', '      __caMark("candidate-export-start"); const packed = entry.index.candidateBuffers(); __caMark("candidate-export-end");');
    replace('        const points = packed.takePoints(), nodes = packed.takeNodes();', '        __caMark("points-copy-start"); const points = packed.takePoints(); __caMark("points-copy-end");\n        __caMark("nodes-copy-start"); const nodes = packed.takeNodes(); __caMark("nodes-copy-end");');
    replace('        gpu = await search!.ready;', '        __caMark("node-readiness-start"); gpu = await search!.ready;');
    replace('      } finally { nodePrepareMillis = performance.now() - nodeStarted; }', '        __caMark("node-readiness-end");\n      } finally { nodePrepareMillis = performance.now() - nodeStarted; }');
    replace('      const queries = job!.queries(first, count);', '      if (first === 0) __caMark("first-query-export-start");\n      const queries = job!.queries(first, count);\n      if (first === 0) __caMark("first-query-export-end");');
    replace('      const result = gpu!.find(queries,', '      if (first === 0) __caMark("first-dispatch");\n      const result = gpu!.find(queries,');
    replace('    return { distances, compute: { backend:', '    __caMark("c2c-return");\n    return { distances, compute: { preparationProbe: __caPreparationSnapshot(), backend:');
  } else {
    replace('        reference ??= this.snapshot ? index.snapshot() : index.xyz();', '        if (!reference) { __caMark("verifier-snapshot-start"); reference = this.snapshot ? index.snapshot() : index.xyz(); __caMark("verifier-snapshot-end"); }');
    replace('        const snapshot = reference.slice();', '        __caMark("verifier-lane-copy-start", lane); const snapshot = reference.slice(); __caMark("verifier-lane-copy-end", lane);');
  }
  return preparationPrelude + source;
}
