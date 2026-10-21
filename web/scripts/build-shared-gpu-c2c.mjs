// Standalone build: leaves the production package, Rust flags and Vite untouched.
import { spawnSync } from 'node:child_process';
import { existsSync, mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { resolve, relative } from 'node:path';
import { fileURLToPath } from 'node:url';

const args = process.argv.slice(2);
if (args.length !== 1 || !['serial', 'threads'].includes(args[0]))
  throw new Error('Usage: node scripts/build-shared-gpu-c2c.mjs serial|threads');
const mode = args[0];
const crate = fileURLToPath(new URL('../experiments/shared-gpu-c2c/', import.meta.url));
const output = resolve(crate, `pkg-${mode}`);
if (existsSync(output)) throw new Error(`Build output already exists: ${output}`);
const nightly = 'nightly-2026-10-20';
const flags = mode === 'threads' ? [] : ['-C', 'target-feature=+simd128'];
if (mode === 'threads') flags.push(
  '-C', 'target-feature=+simd128,+atomics,+bulk-memory,+mutable-globals',
  ...['--shared-memory', '--import-memory', '--max-memory=1073741824',
    '--export=__wasm_init_tls', '--export=__tls_size', '--export=__tls_align', '--export=__tls_base']
    .flatMap(value => ['-C', `link-arg=${value}`]),
);
const env = { ...process.env, RUSTFLAGS: flags.join(' '), RUSTUP_TOOLCHAIN: mode === 'threads' ? nightly : 'stable' };
const target = 'wasm32-unknown-unknown';
const build = ['build', '--manifest-path', resolve(crate, 'Cargo.toml'), '--target', target, '--release', '--locked'];
if (mode === 'threads') build.push('--features', 'threads', '-Zbuild-std=panic_abort,std');
function run(command, args, capture = false) {
  const result = spawnSync(command, args, { env, stdio: capture ? 'pipe' : 'inherit', encoding: 'utf8', windowsHide: true });
  if (result.error) throw result.error;
  if (result.status !== 0) throw new Error(`${command} failed (${result.status}): ${result.stderr ?? ''}`);
  return result.stdout?.trim();
}
const bindgen = process.env.CA_WASM_BINDGEN || 'wasm-bindgen';
const bindgenVersion = run(bindgen, ['--version'], true);
if (bindgenVersion !== 'wasm-bindgen 0.2.129') throw new Error('Need wasm-bindgen-cli 0.2.129');
run(process.env.CA_CARGO || 'cargo', build);
mkdirSync(output);
const compiled = resolve(env.CARGO_TARGET_DIR || resolve(crate, 'target'), target, 'release/ca_wasm_shared_gpu_c2c.wasm');
run(bindgen, [compiled, '--target', 'web', '--out-dir', output, '--out-name', 'ca_wasm_shared_gpu_c2c']);
const repository = resolve(crate, '../../..');
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const normalized = path => hash(readFileSync(path, 'utf8').replaceAll('\r\n', '\n'));
const inputs = [resolve(crate, 'Cargo.toml'), resolve(crate, 'Cargo.lock'),
  fileURLToPath(import.meta.url), resolve(repository, 'rust/crates/ca-core/Cargo.toml'), resolve(repository, 'rust/Cargo.toml'), resolve(repository, 'rust/Cargo.lock')];
function collect(directory) {
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const path = resolve(directory, entry.name);
    if (entry.isDirectory()) collect(path);
    else if (entry.name.endsWith('.rs')) inputs.push(path);
  }
}
collect(resolve(crate, 'src'));
collect(resolve(repository, 'rust/crates/ca-core/src'));
const compiler = run('rustc', ['-vV'], true);
const artifacts = [];
function collectArtifacts(directory) {
  for (const entry of readdirSync(directory, { withFileTypes: true })) {
    const path = resolve(directory, entry.name);
    if (entry.isDirectory()) collectArtifacts(path);
    else if (/\.(wasm|js|ts)$/.test(entry.name)) artifacts.push(path);
  }
}
collectArtifacts(output);
writeFileSync(resolve(output, 'build-info.json'), JSON.stringify({
  schemaVersion: 1, mode, createdUtc: new Date().toISOString(), compiler, bindgenVersion,
  rustflags: env.RUSTFLAGS, cargoArgs: build,
  inputs: Object.fromEntries(inputs.sort().map(path => [relative(repository, path).replaceAll('\\', '/'), hash(readFileSync(path))])),
  normalizedInputs: Object.fromEntries(inputs.map(path => [relative(repository, path).replaceAll('\\', '/'), normalized(path)])),
  artifacts: Object.fromEntries(artifacts.sort().map(path => [relative(output, path).replaceAll('\\', '/'), hash(readFileSync(path))])),
}, null, 2) + '\n', { flag: 'wx' });
