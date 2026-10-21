// Standard production WASM/Vite build, with immutable source/artifact provenance.
import { spawnSync } from 'node:child_process';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, readdirSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

if (process.argv.length !== 4 || process.argv[2] !== '--output' || existsSync(process.argv[3]))
  throw new Error('Supply --output NEW-build.json');
const root = fileURLToPath(new URL('../../', import.meta.url)), web = resolve(root, 'web');
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const files = ['rust/Cargo.toml', 'rust/Cargo.lock', 'rust/.cargo/config.toml',
  'rust/crates/ca-core/Cargo.toml', 'rust/crates/ca-wasm/Cargo.toml',
  'web/package.json', 'web/package-lock.json', 'web/tsconfig.json', 'web/vite.config.ts',
  'web/scripts/build-query-export.mjs'];
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
const before = sources(), env = { ...process.env, RUSTUP_TOOLCHAIN: 'stable', RUSTFLAGS: '-C target-feature=+simd128' };
function run(command, args, capture = false) {
  const result = spawnSync(command, args, { cwd: web, env, windowsHide: true,
    encoding: 'utf8', stdio: capture ? 'pipe' : 'inherit' });
  if (result.error || result.status !== 0) throw result.error ?? new Error(`${command} failed: status=${result.status}, signal=${result.signal}, ${result.stderr ?? "see build log"}`);
  return result.stdout?.trim();
}
const wasmPack = process.env.CA_WASM_PACK || 'wasm-pack';
const compiler = run(process.env.CA_RUSTC || 'rustc', ['-vV'], true), wasmPackVersion = run(wasmPack, ['--version'], true);
const wasmPackArgs = ['build', '../rust/crates/ca-wasm', '--target', 'web', '--release', '--out-dir', '../../../web/src/wasm', '--no-pack', '--locked'];
if (process.env.CA_WASM_PACK_NO_INSTALL === '1') wasmPackArgs.splice(wasmPackArgs.length - 1, 0, '--mode', 'no-install');
run(wasmPack, wasmPackArgs);
run(process.execPath, ['node_modules/typescript/bin/tsc', '--noEmit']);
run(process.execPath, ['node_modules/vite/bin/vite.js', 'build']);
if (JSON.stringify(before) !== JSON.stringify(sources())) throw new Error('Source changed while building');
const artifacts = [];
collect('web/dist/assets', () => true, artifacts);
collect('web/src/wasm', n => /\.(wasm|js|ts)$/.test(n), artifacts);
const metadata = { schemaVersion: 1, compiler, wasmPackVersion, wasmPackArgs, rustflags: env.RUSTFLAGS,
  normalizedInputs: before, artifacts: Object.fromEntries(artifacts.sort().map(n => [n, hash(readFileSync(resolve(root, n)))])) };
writeFileSync(process.argv[3], JSON.stringify(metadata, null, 2) + '\n', { flag: 'wx' });
console.log(`Recorded ${files.length} sources and ${artifacts.length} artifacts`);
