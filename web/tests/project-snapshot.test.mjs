import assert from 'node:assert/strict';
import { test } from 'node:test';
import { writeProjectSnapshot, readProjectSnapshot } from '../src/project-snapshot.ts';
import { indexReviewZip, readReviewMember, writeReviewZip, REVIEW_LIMIT } from '../src/review-zip.ts';
import { referenceFile } from '../src/source-reference.ts';

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
