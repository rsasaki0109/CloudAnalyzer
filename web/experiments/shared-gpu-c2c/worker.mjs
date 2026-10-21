// Standalone exact verifier. Each query job owns coordinates; the reference stays cached.
let module, wasm, snapshot, ready;
let threads = 0, snapshotBytes = 0, queryBytes = 0, referenceGeneration = 0, queryGeneration = 0;
const memory = () => ({ committedBytes: wasm?.memory.buffer.byteLength ?? 0,
  sharedMemory: !!wasm && typeof SharedArrayBuffer === 'function' && wasm.memory.buffer instanceof SharedArrayBuffer,
  snapshotBytes, queryBytes, referenceGeneration, queryGeneration });

self.onmessage = async ({ data: { seq, req } }) => {
  try {
    let value, transfer = [];
    if (req.kind === 'init') {
      if (ready) throw new Error('Already initialized');
      if (!['serial', 'threads'].includes(req.build)) throw new Error('Unknown build');
      if (req.build === 'threads' && (!self.crossOriginIsolated || typeof SharedArrayBuffer !== 'function'))
        throw new Error('Shared threads require cross-origin isolation');
      if (!Number.isInteger(req.threads) || req.threads < 1 || req.threads > 8) throw new Error('Threads must be 1..8');
      ready = (async () => {
        module = await import(`./pkg-${req.build}/ca_wasm_shared_gpu_c2c.js`);
        wasm = await module.default();
        if (req.build === 'threads') { await module.initThreadPool(req.threads); threads = req.threads; }
        return { threads, isolated: self.crossOriginIsolated, indices: Array.from(module.threadIndices()), ...memory() };
      })();
      value = await ready;
    } else {
      if (!ready) throw new Error('Initialize the worker first');
      await ready;
      if (req.kind === 'prepare') {
        if (!(req.snapshot instanceof Uint8Array) || !(req.queries instanceof Float64Array)) throw new Error('Invalid snapshot arrays');
        const next = new module.SharedGpuC2c(req.snapshot, req.queries);
        snapshot?.free(); snapshot = next;
        snapshotBytes = req.snapshot.byteLength; queryBytes = req.queries.byteLength;
        referenceGeneration++; queryGeneration++;
        value = memory();
      } else if (req.kind === 'queries') {
        if (!snapshot) throw new Error('No prepared snapshot');
        if (!(req.queries instanceof Float64Array)) throw new Error('Invalid query array');
        snapshot.setQueries(req.queries);
        queryBytes = req.queries.byteLength; queryGeneration++;
        value = memory();
      } else if (req.kind === 'release-queries') {
        if (!snapshot) throw new Error('No prepared snapshot');
        snapshot.releaseQueries(); queryBytes = 0;
        value = memory();
      } else if (req.kind === 'release') {
        snapshot?.free(); snapshot = undefined; snapshotBytes = 0; queryBytes = 0; value = memory();
      } else if (req.kind === 'verify') {
        if (!snapshot) throw new Error('No prepared snapshot');
        if (!queryBytes) throw new Error('No prepared queries');
        if (!Number.isSafeInteger(req.first) || req.first < 0 || req.first > 2_000_000 ||
            !(req.candidates instanceof Uint32Array) ||
            !['parallel', 'bounds', 'simd'].every(key => typeof req[key] === 'boolean')) throw new Error('Invalid verification request');
        const started = performance.now();
        const result = snapshot.verify(req.first, req.candidates, req.bounds, req.simd, req.parallel);
        try {
          const distances = result.takeDistances();
          value = { distances, certified: result.certified, millis: performance.now() - started,
            threads: req.parallel ? threads : 0, ...memory() };
          transfer = [distances.buffer];
        } finally { result.free(); }
      } else throw new Error('Unknown request');
    }
    self.postMessage({ seq, ok: true, value }, { transfer });
  } catch (error) {
    self.postMessage({ seq, ok: false, error: error instanceof Error ? error.message : String(error) });
  }
};
