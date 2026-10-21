import { build } from 'vite';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { resolve, relative } from 'node:path';
import { fileURLToPath } from 'node:url';
const repository = fileURLToPath(new URL('../../', import.meta.url));
const experiment = resolve(repository, 'web/experiments/shared-gpu-c2c');
const output = resolve(experiment, 'pkg-driver');
if (existsSync(output)) throw new Error('Driver output already exists');
await build({ configFile: false, root: resolve(repository, 'web'), publicDir: false, base: './',
  worker: { format: 'es' }, build: { outDir: output, target: 'es2022', modulePreload: false,
    rollupOptions: { input: resolve(experiment, 'driver.ts'), output: { entryFileNames: 'driver.js' } } } });
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const inputs = ['web/experiments/shared-gpu-c2c/driver.ts', 'web/scripts/build-shared-gpu-driver.mjs',
  'web/package.json', 'web/package-lock.json', 'web/src/gpu-verifiers.ts', 'web/src/gpu-candidates.ts',
  'web/src/gpu-diagnostics.ts', 'web/src/pool.ts', 'web/src/c2c-worker.ts', 'web/src/protocol.ts', 'web/src/wasm/ca_wasm.js',
  'web/src/wasm/ca_wasm_bg.wasm'];
const artifacts = {};
function collect(directory) {
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const path = resolve(directory, entry.name);
    if (entry.isDirectory()) collect(path);
    else artifacts[relative(output, path).replaceAll('\\', '/')] = hash(readFileSync(path));
  }
}
collect(output);
writeFileSync(resolve(output, 'build-info.json'), JSON.stringify({ schemaVersion: 1,
  normalizedInputs: Object.fromEntries(inputs.map(name => [name, hash(name.endsWith('.wasm')
    ? readFileSync(resolve(repository, name)) : readFileSync(resolve(repository, name), 'utf8').replaceAll('\r\n', '\n'))])),
  artifacts }, null, 2) + '\n', { flag: 'wx' });
