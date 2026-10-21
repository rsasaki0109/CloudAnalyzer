// Rebuild production JS/WGSL while reusing an exactly identified unchanged WASM build.
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
const args = process.argv.slice(2);
if (args.length !== 4 || args[0] !== '--output' || args[2] !== '--wasm-build-info' || existsSync(args[1]))
  throw new Error('Supply --output NEW-build.json --wasm-build-info prior-build-or-manifest.json');
const root = fileURLToPath(new URL('../../', import.meta.url)), web = resolve(root, 'web');
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const priorBytes = readFileSync(args[3]), input = JSON.parse(priorBytes), prior = input.finalBuild ?? input;
const files = ['rust/Cargo.toml', 'rust/Cargo.lock', 'rust/.cargo/config.toml',
  'rust/crates/ca-core/Cargo.toml', 'rust/crates/ca-wasm/Cargo.toml',
  'web/package.json', 'web/package-lock.json', 'web/tsconfig.json', 'web/vite.config.ts',
  'web/scripts/build-query-export.mjs', 'web/scripts/build-gpu-frontend.mjs'];
function collect(directory, accept, output) {
  for (const entry of readdirSync(resolve(root, directory), { withFileTypes: true })) {
    const name = `${directory}/${entry.name}`;
    if (entry.isDirectory()) collect(name, accept, output);
    else if (accept(name)) output.push(name);
  }
}
for (const directory of ['rust/crates/ca-core/src', 'rust/crates/ca-wasm/src']) collect(directory, n => n.endsWith('.rs'), files);
collect('web/src', n => !n.startsWith('web/src/wasm/') && /\.(ts|css)$/.test(n), files);
const sources = () => Object.fromEntries(files.sort().map(n => [n, hash(readFileSync(resolve(root, n), 'utf8').replaceAll('\r\n', '\n'))]));
const before = sources(), rustNames = files.filter(n => n.startsWith('rust/')).sort();
if (JSON.stringify(rustNames) !== JSON.stringify(Object.keys(prior.normalizedInputs).filter(n => n.startsWith('rust/')).sort()))
  throw new Error('Rust source set changed; rebuild WASM');
for (const name of rustNames) if (before[name] !== prior.normalizedInputs[name]) throw new Error(`Rust source changed; rebuild WASM: ${name}`);
const wasmNames = ['ca_wasm.js', 'ca_wasm.d.ts', 'ca_wasm_bg.wasm', 'ca_wasm_bg.wasm.d.ts'].map(n => `web/src/wasm/${n}`);
const wasmArtifacts = Object.fromEntries(wasmNames.map(n => [n, prior.artifacts[n]]));
function verifyWasm() {
  for (const [name, expected] of Object.entries(wasmArtifacts)) if (!expected || hash(readFileSync(resolve(root, name))) !== expected)
    throw new Error(`WASM/glue artifact changed: ${name}`);
}
verifyWasm();
const commands = [['node_modules/typescript/bin/tsc', '--noEmit'], ['node_modules/vite/bin/vite.js', 'build']];
for (const command of commands) {
  const result = spawnSync(process.execPath, command, { cwd: web, stdio: 'inherit', windowsHide: true });
  if (result.error || result.status !== 0) throw result.error ?? new Error(`Frontend command failed: ${command}`);
}
verifyWasm();
if (JSON.stringify(before) !== JSON.stringify(sources()) || hash(readFileSync(args[3])) !== hash(priorBytes))
  throw new Error('Build sources or prior metadata changed');
const artifacts = [];
collect('web/dist/assets', () => true, artifacts);
collect('web/src/wasm', n => /\.(wasm|js|ts)$/.test(n), artifacts);
const metadata = { schemaVersion: 1, compiler: prior.compiler, wasmPackVersion: prior.wasmPackVersion,
  wasmPackArgs: prior.wasmPackArgs, rustflags: prior.rustflags, normalizedInputs: before,
  artifacts: Object.fromEntries(artifacts.sort().map(n => [n, hash(readFileSync(resolve(root, n)))])),
  wasmReuse: { priorMetadataSha256: hash(priorBytes), unchangedRustInputs: Object.fromEntries(rustNames.map(n => [n, before[n]])),
    unchangedArtifacts: wasmArtifacts, frontendNodeVersion: process.version, frontendCommands: commands,
    note: 'Only unchanged Rust/WASM inputs and all four generated package artifacts authorize reuse. JS/WGSL is rebuilt and all current inputs are recorded.' } };
writeFileSync(args[1], JSON.stringify(metadata, null, 2) + '\n', { flag: 'wx' });
console.log(`Recorded ${files.length} sources and ${artifacts.length} artifacts; reused ${wasmNames.length} verified WASM artifacts`);
