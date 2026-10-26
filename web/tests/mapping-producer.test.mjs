import assert from 'node:assert/strict';
import {test} from 'node:test';
import {readMappingReview} from '../src/mapping-review.ts';
import {packageFixture,addGlimEvidence,addPassingGlimPreflight,addMatchedGlimConfiguration} from './mapping-review-fixture.mjs';
const read=bytes=>readMappingReview(new File([bytes],'review.zip'),new AbortController().signal);
test('linked producer summary uses all verified identities without retaining raw reports',async()=>{
  const result=await read(packageFixture(d=>addGlimEvidence(d)));
  assert.match(result.producer.join('\n'),/Declared producer: GLIM/);
  assert.match(result.producer.join('\n'),/source trajectory hashes match/);
  assert.equal(result.members.size,3);
});
for(const [name,change] of [
  ['failed',r=>r.status='process_failed'],['boolean exit',r=>r.exit_code=false],
  ['changed config',r=>r.configuration_copy_matches=false],['bad image',r=>r.image_id='mutable:latest'],
  ['different trajectory',r=>r.trajectory={...r.trajectory,sha256:'b'.repeat(64)}],
  ['missing check',r=>delete r.trajectory_check],
])test(`does not claim linked execution for ${name}`,async()=>{
  const result=await read(packageFixture(d=>addGlimEvidence(d,{mutateRun:change})));
  assert.match(result.producer.join('\n'),/No verified execution link/);
  assert.doesNotMatch(result.producer.join('\n'),/Recorded GLIM image/);
});
test('older packages show unavailable producer information',async()=>{
  assert.match((await read(packageFixture())).producer.join('\n'),/unavailable/);
});

for (const [label,bytes] of [['malformed',Buffer.from('{invalid')],['oversized',Buffer.alloc(4*1024*1024+1,32)]])
test(`${label} optional producer JSON does not prevent opening maps`,async()=>{
  const {createHash}=await import('node:crypto');
  const result=await read(packageFixture(d=>{
    addGlimEvidence(d);
    const path=d.manifest.roles.slam_import;
    d.entries.find(([name])=>name===path)[1]=bytes;
    Object.assign(d.manifest.files.find(f=>f.path===path),{bytes:bytes.length,sha256:createHash('sha256').update(bytes).digest('hex')});
  }));
  assert.match(result.producer.join('\n'),/unavailable/);
  assert.doesNotMatch(result.producer.join('\n'),/Recorded GLIM image/);
  assert.equal(result.members.size,3);
});


test('shows passed preflight only with the exact inventoried input and saved policy',async()=>{
  const result=await read(packageFixture(d=>addGlimEvidence(d,{mutateRun:addPassingGlimPreflight,addMatchedGlimConfiguration})));
  assert.match(result.producer.join('\n'),/recorded basic checks passed/);
  assert.match(result.producer.join('\n'),/IMU gap limit: 50 ms; maximum observed: 10 ms/);
});
for(const [name,change] of [
  ['hash mismatch',r=>r.input_preflight.input.sha256='c'.repeat(64)],
  ['failed',r=>r.input_preflight.status='blocked'],
  ['issues',r=>r.input_preflight.issues=['unavailable IMU']],
  ['gap count',r=>r.input_preflight.imu_gaps.count=1],
  ['excessive gap',r=>r.input_preflight.imu_gaps.max_observed_ns=60_000_000],
  ['bad policy',r=>r.input_preflight.imu_gaps.limit_ns=-1],
  ['different schema',r=>r.input_preflight.schema='other'],
  ['multiple bags',r=>r.inputs.bag.push(r.inputs.bag[0])],
  ['contradiction',r=>r.input_preflight_status='not_requested'],
])test(`preflight ${name} cannot show a pass`,async()=>{
  const result=await read(packageFixture(d=>addGlimEvidence(d,{mutateRun:r=>{addPassingGlimPreflight(r);change(r);}})));
  assert.match(result.producer.join('\n'),/no diagnostic pass is shown/);
  assert.doesNotMatch(result.producer.join('\n'),/recorded basic checks passed/);
});
test('distinguishes old reports from explicitly unrequested preflight',async()=>{
  const old=await read(packageFixture(d=>addGlimEvidence(d)));
  assert.match(old.producer.join('\n'),/not recorded/);
  const skipped=await read(packageFixture(d=>addGlimEvidence(d,{mutateRun:r=>{r.input_preflight_status='not_requested';r.input_preflight=null;}})));
  assert.match(skipped.producer.join('\n'),/not performed/);
});


test('configuration topic summary requires inventoried settings and matching inspected topics',async()=>{
  const result=await read(packageFixture(d=>addGlimEvidence(d,{mutateRun:addMatchedGlimConfiguration})));
  assert.match(result.producer.join('\n'),/Configuration topics: recorded match/);
  assert.match(result.producer.join('\n'),/Calibration, scaling, offsets and module compatibility remain unverified/);
});
for(const [name,change] of [
  ['point topic',r=>r.config_topics.points_topic='/other'],
  ['IMU topic',r=>r.config_topics.imu_topic='/other'],
  ['hash',r=>r.config_topics.files[1].sha256='f'.repeat(64)],
  ['size',r=>r.config_topics.files[1].bytes++],
  ['path',r=>r.config_topics.files[1].path='/elsewhere/config_ros.json'],
  ['missing file',r=>r.config_topics.files.pop()],
  ['missing inventory',r=>delete r.inputs.configuration],
  ['duplicate inventory path',r=>r.inputs.configuration.push(r.inputs.configuration[0])],
  ['unsupported schema',r=>r.config_topics.schema='other'],
  ['failed status',r=>r.config_topics.status='blocked'],
  ['malformed inventory',r=>r.inputs.configuration.push(null)],
])test(`configuration ${name} cannot show a match or prevent map loading`,async()=>{
  const result=await read(packageFixture(d=>addGlimEvidence(d,{mutateRun:r=>{addMatchedGlimConfiguration(r);change(r);}})));
  assert.match(result.producer.join('\n'),/no configuration match is shown/);
  assert.doesNotMatch(result.producer.join('\n'),/Configuration topics: recorded match/);
  assert.match(result.producer.join('\n'),/recorded basic checks passed/);
  assert.equal(result.members.size,3);
});
test('configuration match is suppressed when input preflight is inconsistent',async()=>{
  const result=await read(packageFixture(d=>addGlimEvidence(d,{mutateRun:r=>{addMatchedGlimConfiguration(r);r.input_preflight.input.sha256='c'.repeat(64);}})));
  assert.doesNotMatch(result.producer.join('\n'),/Configuration topics: recorded match/);
});
test('older successful diagnostics without configuration evidence remain usable',async()=>{
  const result=await read(packageFixture(d=>addGlimEvidence(d,{mutateRun:addPassingGlimPreflight})));
  assert.match(result.producer.join('\n'),/Configuration topic match: not recorded/);
});
