// Only loaded by the standalone experiment, never by the default application.
let module, wasm, snapshot, ready;
let threads = 0;
const memory = () => ({
  committedBytes: wasm?.memory.buffer.byteLength ?? 0,
  sharedMemory: !!wasm && typeof SharedArrayBuffer === 'function' && wasm.memory.buffer instanceof SharedArrayBuffer,
  retainedBytes: snapshot?.retainedBytes ?? 0,
});

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
        module = await import(`./pkg-${req.build}/ca_wasm_shared_c2c.js`);
        wasm = await module.default();
        if (req.build === 'threads') {
          await module.initThreadPool(req.threads);
          threads = req.threads;
        }
        return { threads, isolated: self.crossOriginIsolated,
          indices: Array.from(module.threadIndices()), ...memory() };
      })();
      value = await ready;
    } else {
      if (!ready) throw new Error('Initialize the worker first');
      await ready;
      if (req.kind === 'prepare') {
        // Replace only after successful validation and independent ownership.
        const next = new module.SharedC2c(req.reference, req.queries);
        snapshot?.free(); snapshot = next;
        value = memory();
      } else if (req.kind === 'release') {
        snapshot?.free(); snapshot = undefined; value = memory();
      } else {
        if (!snapshot) throw new Error('No prepared snapshot');
        if (typeof req.parallel !== 'boolean') throw new Error('Choose serial or parallel computation');
        const start = performance.now();
        if (req.kind === 'compute') {
          const result = snapshot.compute(req.parallel);
          try {
            const distances = result.takeDistances(), s = result.summary();
            value = { distances, stats: { count: s[0], min: s[1], max: s[2], mean: s[3], rms: s[4], stdDev: s[5], median: s[6] },
              millis: performance.now() - start, threads: req.parallel ? threads : 0, ...memory() };
            transfer = [distances.buffer];
          } finally { result.free(); }
        } else if (req.kind === 'exact') {
          value = snapshot.exact(req.parallel); transfer = [value.buffer];
        } else throw new Error('Unknown request');
      }
    }
    self.postMessage({ seq, ok: true, value }, { transfer });
  } catch (error) {
    self.postMessage({ seq, ok: false, error: error instanceof Error ? error.message : String(error) });
  }
};
