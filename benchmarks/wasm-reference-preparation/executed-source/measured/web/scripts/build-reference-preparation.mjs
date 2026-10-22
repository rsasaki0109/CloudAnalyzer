// Build an isolated host-phase probe with the same production worker and WASM.
import { build } from 'vite';
import { createHash } from 'node:crypto';
import { existsSync, readFileSync, readdirSync, writeFileSync, mkdirSync } from 'node:fs';
import { resolve, relative, sep } from 'node:path';
import { fileURLToPath } from 'node:url';
import { spawnSync } from 'node:child_process';
import { preparationTargets, instrumentPreparationSource } from './gpu-reference-preparation-probe.mjs';
const args = process.argv.slice(2);
if (args.length !== 6 || args[0] !== '--output' || args[2] !== '--out-dir' || args[4] !== '--wasm-build-info') throw new Error('Supply --output NEW.json --out-dir NEW-directory --wasm-build-info prior-metadata.json');
const root = resolve(fileURLToPath(new URL('../../', import.meta.url))), web = resolve(root, 'web'), out = resolve(args[3]), metadataFile = resolve(args[1]);
if (!out.startsWith(root + sep) || existsSync(out) || existsSync(metadataFile) || preparationTargets.some(n => out.startsWith(resolve(root, n)))) throw new Error('Output must be a new directory inside the workspace');
const hash = b => createHash('sha256').update(b).digest('hex'), normalized = n => readFileSync(resolve(root, n), 'utf8').replaceAll('\r\n','\n');
const priorBytes = readFileSync(args[5]), priorDoc = JSON.parse(priorBytes), prior = priorDoc.finalBuild ?? priorDoc;
const files = ['rust/Cargo.toml','rust/Cargo.lock','rust/.cargo/config.toml','rust/crates/ca-core/Cargo.toml','rust/crates/ca-wasm/Cargo.toml','web/package.json','web/package-lock.json','web/tsconfig.json','web/vite.config.ts','web/scripts/build-query-export.mjs','web/scripts/build-gpu-frontend.mjs'];
function inputsIn(dir, accept) { for (const e of readdirSync(resolve(root,dir),{withFileTypes:true})) { const n=dir+'/'+e.name; if(e.isDirectory())inputsIn(n,accept);else if(accept(n))files.push(n); } }
for(const dir of ['rust/crates/ca-core/src','rust/crates/ca-wasm/src'])inputsIn(dir,n=>n.endsWith('.rs'));
inputsIn('web/src',n=>!n.startsWith('web/src/wasm/')&&/\.(ts|css)$/.test(n));
const inputs=Object.fromEntries(files.sort().map(n=>[n,hash(normalized(n))]));
const rustNames=files.filter(n=>n.startsWith('rust/')).sort();
if(JSON.stringify(rustNames)!==JSON.stringify(Object.keys(prior.normalizedInputs).filter(n=>n.startsWith('rust/')).sort())||rustNames.some(n=>inputs[n]!==prior.normalizedInputs[n]))throw new Error('Rust sources changed; rebuild WASM');
for (const n of ['web/scripts/build-reference-preparation.mjs','web/scripts/gpu-reference-preparation-probe.mjs']) inputs[n] = hash(normalized(n));
const wasmNames = ['ca_wasm.js','ca_wasm.d.ts','ca_wasm_bg.wasm','ca_wasm_bg.wasm.d.ts'].map(n => 'web/src/wasm/' + n);
const wasm = Object.fromEntries(wasmNames.map(n => [n, prior.artifacts[n]]));
function verify() {
  for (const [n,h] of Object.entries(inputs)) if (hash(normalized(n)) !== h) throw new Error('Source changed: '+n);
  for (const [n,h] of Object.entries(wasm)) if (!h || hash(readFileSync(resolve(root,n))) !== h) throw new Error('WASM/glue changed: '+n);
  if (hash(readFileSync(args[5])) !== hash(priorBytes)) throw new Error('Prior metadata changed');
}
verify();
const check = spawnSync(process.execPath, ['node_modules/typescript/bin/tsc','--noEmit'], { cwd:web, stdio:'inherit', windowsHide:true });
if (check.error || check.status !== 0) throw check.error ?? new Error('Production typecheck failed');
const transformed = Object.fromEntries(preparationTargets.map(n => [n, instrumentPreparationSource(n,normalized(n))]));
const contexts = new Set();
const plugin = context => ({ name:'bounded-reference-preparation-probe', enforce:'pre', transform(code,id) {
  const n = relative(root,id.split('?')[0]).split(sep).join('/');
  if (!Object.hasOwn(transformed,n)) return;
  if (hash(code.replaceAll('\r\n','\n')) !== inputs[n]) throw new Error('Unexpected Vite preparation input: '+n);
  contexts.add(context+':'+n); return { code:transformed[n], map:null };
} });
await build({ root:web, configFile:resolve(web,'vite.config.ts'), plugins:[plugin('main')], worker:{plugins:()=>[plugin('worker')]}, build:{outDir:out,emptyOutDir:false,copyPublicDir:false} });
for (const n of preparationTargets) if (!contexts.has('worker:'+n)) throw new Error('Worker instrumentation was not applied: '+n);
verify();
const artifacts = {};
function collect(dir) { for (const e of readdirSync(dir,{withFileTypes:true})) { const p=resolve(dir,e.name); if(e.isDirectory())collect(p);else artifacts[relative(root,p).split(sep).join('/')]=hash(readFileSync(p)); } }
collect(out);
const sourceOut = resolve(out,'instrumented-source'); mkdirSync(sourceOut);
const instrumentedSources = {};
for (const [n,code] of Object.entries(transformed)) { const name=n.split('/').at(-1), p=resolve(sourceOut,name); writeFileSync(p,code,{flag:'wx'}); instrumentedSources[n]={ originalSha256:inputs[n],sha256:hash(code),source:relative(root,p).split(sep).join('/') }; artifacts[relative(root,p).split(sep).join('/')]=hash(readFileSync(p)); }
const record = { schemaVersion:1, purpose:'measurement-only-host-phase-build', outputDirectory:relative(root,out).split(sep).join('/'), frontendNodeVersion:process.version,
 normalizedInputs:inputs, artifacts, wasmReuse:{priorMetadataSha256:hash(priorBytes),unchangedRustInputs:Object.fromEntries(Object.entries(inputs).filter(([n])=>n.startsWith('rust/'))),unchangedArtifacts:wasm}, instrumentedSources,transformContexts:[...contexts].sort(),
 note:'Unmodified production sources pass typecheck. Three identified transformations add bounded optional host-clock phase events only; every shader and exact verification operation remain unchanged. No GPU timestamps or adapter identity added. Instrumented build is separate from ordinary web/dist.' };
writeFileSync(metadataFile,JSON.stringify(record,null,2)+'\n',{flag:'wx'}); console.log(JSON.stringify({output:args[1],sources:Object.keys(inputs).length,artifacts:Object.keys(artifacts).length,contexts:record.transformContexts}));
