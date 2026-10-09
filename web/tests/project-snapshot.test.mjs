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
