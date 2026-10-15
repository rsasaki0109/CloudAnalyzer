import assert from 'node:assert/strict';
import { test } from 'node:test';
import { writeProjectSnapshot, readProjectSnapshot } from '../src/project-snapshot.ts';
import { indexReviewZip, readReviewMember, writeReviewZip, REVIEW_LIMIT, WORKSPACE_LIMIT, ZIP_CHECK_CHUNK } from '../src/review-zip.ts';
import { referenceFile } from '../src/source-reference.ts';
import { crc32 } from 'node:zlib';

const signal=()=>new AbortController().signal;
async function fixture() {
  const file=new File([Buffer.from('canonical point records')],'workspace-0.ply');
  const project={app:'CloudAnalyzer Project',version:1,session:{clouds:[{name:'Filtered survey',source:await referenceFile(file),transforms:[],loadMaxPoints:0,displayPreview:true}]}};
  return {file,project};
}
async function entries(zip) {
  const directory=await indexReviewZip(zip,signal());
  const result=[];
  for(const [name,member] of directory) result.push([name,new Blob([await readReviewMember(zip,member,signal())])]);
  return result;
}
const asFile=blob=>new File([blob],'project.cloudanalyzer.zip');

test('snapshot retains native file bytes, display aliases and preview provenance',async()=>{
  const {file,project}=await fixture(), zip=asFile(await writeProjectSnapshot(project,[file],signal()));
  const result=await readProjectSnapshot(zip,signal());
  assert.equal(result[0].name,'project.cloudanalyzer.json');
  assert.deepEqual(JSON.parse(await result[0].text()),project);
  assert.equal(result[1].name,file.name);assert.equal(await result[1].text(),await file.text());
});
test('any changed point bytes reject a snapshot before workspace loading',async()=>{
  const {file,project}=await fixture(), zip=asFile(await writeProjectSnapshot(project,[file],signal()));
  const files=await entries(zip);files.find(([name])=>name.startsWith('clouds/'))[1]=new Blob(['changed point records!!!']);
  await assert.rejects(readProjectSnapshot(asFile(await writeReviewZip(files,signal())),signal()),/identity|hash differs/);
});
test('native snapshot coordinates cannot have an additional saved transform or loading cap',async()=>{
  const {file,project}=await fixture();
  project.session.clouds[0].transforms=[[1,0,0,5,0,1,0,0,0,0,1,0,0,0,0,1]];
  await assert.rejects(writeProjectSnapshot(project,[file],signal()),/identity\/frame/);
  project.session.clouds[0].transforms=[];project.session.clouds[0].loadMaxPoints=100;
  await assert.rejects(writeProjectSnapshot(project,[file],signal()),/identity\/frame/);
});
test('missing, unlisted and reassigned sources cannot satisfy the snapshot project',async()=>{
  const {file,project}=await fixture(), zip=asFile(await writeProjectSnapshot(project,[file],signal()));
  const files=await entries(zip);
  await assert.rejects(readProjectSnapshot(asFile(await writeReviewZip(files.filter(([name])=>!name.startsWith('clouds/')),signal())),signal()),/members differ/);
  await assert.rejects(readProjectSnapshot(asFile(await writeReviewZip([...files,['clouds/extra.ply',file]],signal())),signal()),/members differ/);
  const wrong=new File(['another source'],file.name);
  await assert.rejects(writeProjectSnapshot(project,[wrong],signal()),/identity\/frame/);
});
test('ZIP construction rejects oversized/duplicate content and supports cancellation',async()=>{
  await assert.rejects(writeReviewZip([['big',new Blob([new Uint8Array(REVIEW_LIMIT+1)])]],signal()),/64 MiB/);
  await assert.rejects(writeReviewZip([['same',new Blob()],['same',new Blob()]],signal()),/limit/);
  const controller=new AbortController();controller.abort();
  await assert.rejects(writeReviewZip([['small',new Blob(['x'])]],controller.signal),{name:'AbortError'});
});

test('complete snapshot keeps graph scans with equal basenames and distinct fingerprints', async()=>{
  const {file,project}=await fixture();
  const one=new File(['first scan'],'000001.ply'), two=new File(['second scan'],'000001.ply'), graph=new File(['poses'],'poses.kitti');
  project.poseGraph={sources:[{graph:await referenceFile(graph),scans:[await referenceFile(one)]},{graph:null,scans:[await referenceFile(two)]}]};
  const zip=asFile(await writeProjectSnapshot(project,[file],signal(),[graph,one,two]));
  const restored=await readProjectSnapshot(zip,signal());
  assert.deepEqual(restored.slice(2).map(f=>f.name),['poses.kitti','000001.ply','000001.ply']);
  assert.deepEqual(await Promise.all(restored.slice(2).map(f=>f.text())),['poses','first scan','second scan']);
});
test('missing, swapped and unreferenced original inputs reject complete snapshots',async()=>{
  const {file,project}=await fixture(), original=new File(['scan'],'000000.ply'), graph=new File(['poses'],'poses.g2o');
  project.poseGraph={sources:[{graph:await referenceFile(graph),scans:[await referenceFile(original)]}]};
  await assert.rejects(writeProjectSnapshot(project,[file],signal(),[graph]),/missing verified/);
  await assert.rejects(writeProjectSnapshot(project,[file],signal(),[graph,new File(['bad!'],original.name)]),/missing verified/);
  await assert.rejects(writeProjectSnapshot(project,[file],signal(),[graph,original,new File(['extra'],'unexpected.ply')]),/unreferenced/);
});
test('workspace member allowance does not relax generated review ZIP limits',async()=>{
  const files=Array.from({length:130},(_,i)=>[`assets/${i}`,new Blob()]);
  await assert.rejects(writeReviewZip(files,signal()),/member limit/);
  const zip=asFile(await writeReviewZip([['manifest.json',new Blob(['{}'])],...files],signal(),2048));
  await assert.rejects(indexReviewZip(zip,signal()),/excessive/);
  assert.equal((await indexReviewZip(zip,signal(),2048)).size,131);
  await assert.rejects(indexReviewZip(zip,signal(),99999),/Invalid ZIP member limit/);
});
test('original review archive bytes are separate from edited project geometry',async()=>{
  const {file,project}=await fixture(), archive=new File(['original immutable review archive'],'review.zip');
  project.reviewArchive=await referenceFile(archive);
  const zip=asFile(await writeProjectSnapshot(project,[file],signal(),[archive]));
  const restored=await readProjectSnapshot(zip,signal());
  assert.equal(restored.at(-1).name,archive.name);assert.equal(await restored.at(-1).text(),await archive.text());
  const table=await entries(zip), changed=table.find(([p])=>p.startsWith('assets/'));
  changed[1]=new Blob(['changed original review archive!!!']);
  await assert.rejects(readProjectSnapshot(asFile(await writeReviewZip(table,signal(),2048)),signal()),/identity|hash differs/);
});

test('metadata-only v3 workspaces retain external graph and review references',async()=>{
  const {file,project}=await fixture(),graph=new File(['original graph'],'poses.g2o'),archive=new File(['original review'],'review.zip');
  project.poseGraph={sources:[{graph:await referenceFile(graph),scans:[]}]};
  project.reviewArchive=await referenceFile(archive);
  const zip=asFile(await writeProjectSnapshot(project,[file],signal()));
  const restored=await readProjectSnapshot(zip,signal());
  assert.equal(restored.length,2);
  assert.deepEqual(JSON.parse(await restored[0].text()).poseGraph,project.poseGraph);
  const table=await entries(zip),entry=table.find(([p])=>p==='manifest.json'),manifest=JSON.parse(await entry[1].text());
  assert.equal(manifest.pose_graph_sources_included,false);assert.equal(manifest.review_archive_included,false);
  manifest.pose_graph_sources_included=true;entry[1]=new Blob([JSON.stringify(manifest)]);
  await assert.rejects(readProjectSnapshot(asFile(await writeReviewZip(table,signal())),signal()),/provenance/);
});

test('large workspace retains more than 64 MiB without whole-member arrayBuffer reads',async()=>{
  const chunk=new Blob([new Uint8Array(8*1024*1024).fill(73)]);
  const file=new File([...Array(8).fill(chunk),new Uint8Array(17).fill(91)],'large.ply');
  const source=await referenceFile(file),project={app:'CloudAnalyzer Project',version:1,session:{clouds:[{name:'Large survey',source,transforms:[],loadMaxPoints:0}]}};
  const original=Blob.prototype.arrayBuffer,reads=[];
  Blob.prototype.arrayBuffer=function(){reads.push(this.size);return original.call(this);};
  try {
    const zip=asFile(await writeProjectSnapshot(project,[file],signal()));
    assert(zip.size>REVIEW_LIMIT);
    await assert.rejects(indexReviewZip(zip,signal()),/browser limits/);
    const directory=await indexReviewZip(zip,signal(),2048,WORKSPACE_LIMIT);
    const member=directory.get('clouds/000.ply');assert.equal(member.bytes,file.size);
    const restored=await readProjectSnapshot(zip,signal());
    assert.deepEqual(await referenceFile(restored[1]),source);
    assert(reads.length>1 && Math.max(...reads)<=8*1024*1024,JSON.stringify(reads));
    const localOffset=member.offset-30-Buffer.byteLength(member.path);
    const cloudHeader=new DataView(await zip.slice(localOffset,localOffset+30).arrayBuffer());
    let expected=0;
    for(let offset=0;offset<file.size;offset+=8*1024*1024)
      expected=crc32(new Uint8Array(await file.slice(offset,offset+8*1024*1024).arrayBuffer()),expected);
    assert.equal(cloudHeader.getUint32(14,true),expected);
    // The old browser-save budget remains explicitly available to its caller.
    await assert.rejects(writeProjectSnapshot(project,[file],signal(),[],REVIEW_LIMIT),/64 MiB/);
  } finally {Blob.prototype.arrayBuffer=original;}
});

test('ZIP checksum reads at most one MiB and agrees with independent CRC32',async()=>{
  const blob=new Blob([new Uint8Array(2*ZIP_CHECK_CHUNK+39).fill(255)]);
  const original=Blob.prototype.arrayBuffer,reads=[];
  Blob.prototype.arrayBuffer=function(){reads.push(this.size);return original.call(this);};
  let zip;
  try {zip=await writeReviewZip([['manifest.json',blob]],signal());}
  finally {Blob.prototype.arrayBuffer=original;}
  assert.equal(Math.max(...reads),ZIP_CHECK_CHUNK);
  assert.equal(new DataView(await zip.slice(0,30).arrayBuffer()).getUint32(14,true),crc32(new Uint8Array(await blob.arrayBuffer())));
});

test('legacy SHA-256 workspace versions still read with their original identity rules',async()=>{
  for(const version of [1,2]) {
    const {file,project}=await fixture(),graph=new File(['original graph'],'poses.g2o');
    const assets=version===2?[graph]:[];
    if(version===2)project.poseGraph={sources:[{graph:await referenceFile(graph),scans:[]}]};
    const table=await entries(asFile(await writeProjectSnapshot(project,[file],signal(),assets)));
    const manifestEntry=table.find(([name])=>name==='manifest.json'),manifest=JSON.parse(await manifestEntry[1].text());
    manifest.schema=`cloudanalyzer.project_snapshot.v${version}`;
    for(const d of manifest.files){const blob=table.find(([name])=>name===d.path)[1];d.sha256=Buffer.from(await crypto.subtle.digest('SHA-256',await blob.arrayBuffer())).toString('hex');delete d.identity;}
    manifestEntry[1]=new Blob([JSON.stringify(manifest)]);
    const restored=await readProjectSnapshot(asFile(await writeReviewZip(table,signal())),signal());
    assert.equal(await restored[1].text(),await file.text());
    if(version===2)assert.equal(await restored[2].text(),await graph.text());
  }
});

test('input directory names do not replace original scan basenames in the archive',async()=>{
  const {file,project}=await fixture(),scan=new File(['scan records'],'000000.ply');
  Object.defineProperty(scan,'webkitRelativePath',{value:'session-a/000000.ply'});
  project.poseGraph={sources:[{graph:null,scans:[await referenceFile(scan)]}]};
  const restored=await readProjectSnapshot(asFile(await writeProjectSnapshot(project,[file],signal(),[scan])),signal());
  assert.equal(restored[2].name,'000000.ply');assert.equal(await restored[2].text(),'scan records');
});

test('cancellation between ZIP checksum chunks publishes no archive',async()=>{
  const controller=new AbortController(),blob=new Blob([new Uint8Array(3*ZIP_CHECK_CHUNK)]);
  const original=Blob.prototype.arrayBuffer;let reads=0;
  Blob.prototype.arrayBuffer=async function(){const result=await original.call(this);if(++reads===1)controller.abort();return result;};
  try {await assert.rejects(writeReviewZip([['manifest.json',blob]],controller.signal),{name:'AbortError'});assert.equal(reads,1);}
  finally {Blob.prototype.arrayBuffer=original;}
});

test('manual snapshots refuse content above 256 MiB before hashing or allocating records',async()=>{
  const chunk=new Blob([new Uint8Array(1024*1024)]),file=new File([...Array(257).fill(chunk)],'oversized.ply');
  const project={session:{clouds:[{}]}};
  await assert.rejects(writeProjectSnapshot(project,[file],signal()),/256 MiB/);
  await assert.rejects(writeReviewZip([['manifest.json',file]],signal(),129,WORKSPACE_LIMIT),/256 MiB/);
});

test('oversized ZIP directory is rejected before reading the directory buffer',async()=>{
  const end=new Uint8Array(22),e=new DataView(end.buffer),size=11*1024*1024;
  e.setUint32(0,0x06054b50,true);e.setUint16(8,1,true);e.setUint16(10,1,true);e.setUint32(12,size,true);e.setUint32(16,0,true);
  const file=new File([new Blob([new Uint8Array(size)]),end],'malformed.zip');
  const original=Blob.prototype.arrayBuffer,reads=[];
  Blob.prototype.arrayBuffer=function(){reads.push(this.size);return original.call(this);};
  try {await assert.rejects(indexReviewZip(file,signal(),2048,WORKSPACE_LIMIT),/directory/);assert(Math.max(...reads)<=65557);}
  finally {Blob.prototype.arrayBuffer=original;}
});
