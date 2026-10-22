// Two matched stable-WASM builds in one isolated source directory.
import {spawnSync,execFileSync} from 'node:child_process';
import {createHash} from 'node:crypto';
import {existsSync,mkdirSync,readFileSync,writeFileSync,readdirSync,lstatSync} from 'node:fs';
import {resolve,relative,sep} from 'node:path';
import {fileURLToPath} from 'node:url';
import {collectionSource,transformCollectionSource} from './kdtree-collection-source.mjs';

const args=process.argv.slice(2);
if(args.length!==4||args[0]!=='--output'||args[2]!=='--out-dir'||existsSync(args[1])||existsSync(args[3]))throw new Error('Supply --output NEW-build.json --out-dir NEW-workspace-directory');
const root=resolve(fileURLToPath(new URL('../../',import.meta.url))),output=resolve(args[1]),directory=resolve(args[3]);
if(!directory.startsWith(root+sep)||!output.startsWith(root+sep))throw new Error('Build outputs must stay in the workspace');
const hash=b=>createHash('sha256').update(b).digest('hex'),normal=p=>hash(readFileSync(p,'utf8').replaceAll('\r\n','\n'));
const names=execFileSync('git',['ls-files','-z','rust'],{cwd:root,encoding:'utf8',windowsHide:true}).split('\0').filter(Boolean).sort();
if(!names.length||names.some(n=>!n.startsWith('rust/')||n.split('/').includes('..')||!lstatSync(resolve(root,n)).isFile()))throw new Error('Unexpected Rust workspace files');
const rawInputs=Object.fromEntries(names.map(n=>[n,hash(readFileSync(resolve(root,n)))]));
const compiledNames=['rust/Cargo.toml','rust/Cargo.lock','rust/.cargo/config.toml','rust/crates/ca-core/Cargo.toml','rust/crates/ca-wasm/Cargo.toml'];
function collect(folder){for(const e of readdirSync(resolve(root,folder),{withFileTypes:true})){const n=folder+'/'+e.name;if(e.isDirectory())collect(n);else if(n.endsWith('.rs'))compiledNames.push(n);}}
collect('rust/crates/ca-core/src');collect('rust/crates/ca-wasm/src');compiledNames.sort();
const rustInputs=Object.fromEntries(compiledNames.map(n=>[n,normal(resolve(root,n))]));
const helperNames=['web/scripts/build-kdtree-collection.mjs','web/scripts/kdtree-collection-source.mjs'];
const helpers=Object.fromEntries(helperNames.map(n=>[n,normal(resolve(root,n))]));
const target=resolve(process.env.CARGO_TARGET_DIR||resolve(root,'rust/target'));
if(!target.startsWith(root+sep))throw new Error('Cargo target must stay in the workspace');
const env={...process.env,RUSTUP_TOOLCHAIN:'stable',RUSTFLAGS:'-C target-feature=+simd128',CARGO_TARGET_DIR:target,CARGO_NET_OFFLINE:'true'};
function run(command,argv,capture=false){const r=spawnSync(command,argv,{cwd:root,env,encoding:'utf8',stdio:capture?'pipe':'inherit',windowsHide:true});if(r.error||r.status!==0)throw r.error??new Error(`${command} failed (${r.status}): ${r.stderr??''}`);return r.stdout?.trim();}
const compiler=run(process.env.CA_RUSTC||'rustc',['-vV'],true),wasmPack=process.env.CA_WASM_PACK||'wasm-pack',wasmPackVersion=run(wasmPack,['--version'],true);
mkdirSync(directory);
const workspace=resolve(directory,'source');
for(const n of names){const p=resolve(workspace,n);mkdirSync(resolve(p,'..'),{recursive:true});writeFileSync(p,readFileSync(resolve(root,n)),{flag:'wx'});}
const original=readFileSync(resolve(root,collectionSource),'utf8');
const metadata={schemaVersion:1,purpose:'isolated-kdtree-collection-build',compiler,wasmPackVersion,rustflags:env.RUSTFLAGS,offline:true,sourceDirectory:relative(root,workspace).replaceAll('\\','/'),cargoTargetDirectory:relative(root,target).replaceAll('\\','/'),rustInputs,helpers,variants:{},copiedWorkspaceFileCount:names.length,
 note:'Both variants compile the complete unchanged production Rust workspace in the same isolated directory and stable/SIMD configuration. Only the final point/index collection in KdTree::new differs. Production sources, generated package and frontend are untouched. Dependency installation and source reuse are not substituted for compilation.'};
for(const mode of ['baseline','unzip']){
 const source=transformCollectionSource(original,mode);writeFileSync(resolve(workspace,collectionSource),source);
 const archive=resolve(directory,mode+'-kdtree.rs');writeFileSync(archive,source,{flag:'wx'});
 const packageDirectory=resolve(directory,'pkg-'+mode),argv=['build',resolve(workspace,'rust/crates/ca-wasm'),'--target','web','--release','--out-dir',packageDirectory,'--no-pack','--locked'];
 if(process.env.CA_WASM_PACK_NO_INSTALL==='1')argv.splice(argv.length-1,0,'--mode','no-install');
 console.log('Building isolated collection mode '+mode);run(wasmPack,argv);
 const inputs={...rustInputs,[collectionSource]:hash(source)};
 for(const[n,v]of Object.entries(inputs))if(normal(resolve(workspace,n))!==v)throw new Error('Isolated source changed: '+n);
 const artifacts=Object.fromEntries(['ca_wasm.js','ca_wasm.d.ts','ca_wasm_bg.wasm','ca_wasm_bg.wasm.d.ts'].map(n=>{const p=resolve(packageDirectory,n);return[relative(root,p).replaceAll('\\','/'),hash(readFileSync(p))];}));
 metadata.variants[mode]={mode,normalizedRustInputs:inputs,sourceSha256:hash(source),sourceArchive:relative(root,archive).replaceAll('\\','/'),packageDirectory:relative(root,packageDirectory).replaceAll('\\','/'),wasmPackArgs:argv,artifacts};
}
for(const[n,v]of Object.entries(rawInputs))if(hash(readFileSync(resolve(root,n)))!==v)throw new Error('Production workspace changed: '+n);
for(const[n,v]of Object.entries(helpers))if(normal(resolve(root,n))!==v)throw new Error('Build helper changed: '+n);
metadata.productionWorkspaceUnchanged=true;writeFileSync(output,JSON.stringify(metadata,null,2)+'\n',{flag:'wx'});
console.log(JSON.stringify({output,sources:compiledNames.length,variants:Object.keys(metadata.variants),productionWorkspaceUnchanged:true}));
