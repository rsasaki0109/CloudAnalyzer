import assert from 'node:assert/strict';
import test from 'node:test';
import { changeCandidates } from '../src/change-review.ts';
import { newLaneTask, laneTaskFile, parseLaneTaskFile, setLaneTaskDecision, laneTaskCounts } from '../src/survey-lane-tasks.ts';
const source = () => ({ analyzedAt: '2026-10-24T00:00:00.000Z', context: { graph: 'survey' }, identity: { algorithm: 'sha256-ply-graph-v1', digest: 'a'.repeat(64) }, staleReason: null });
const candidate = () => changeCandidates(new Float64Array([8,.5,.5,.5,0,0,0,1,1,1,.5]))[0];
const lane = () => ({ id: 101, left: [[-1,2,0],[2,2,0]], right: [[-1,-1,0],[2,-1,0]] });
const overlap = () => ({ xyMargin: 0, zMargin: 1, checked: 1, invalid: 0, limited: 0, total: 1 });
const task = (id = 1) => newLaneTask(id, source(), candidate(), lane(), overlap(), 'b'.repeat(64));

test('follow-up records own immutable source snapshots, independent of source judgment and map changes', () => {
  const review = source(), point = candidate(), road = lane(), settings = overlap();
  const record = newLaneTask(1, review, point, road, settings, null);
  review.context.graph = 'changed'; point.bounds[0] = -100; point.note = 'different'; road.left[0][2] = 9; settings.zMargin = 10;
  assert.equal(record.comparison.graph, 'survey'); assert.equal(record.candidate.bounds[0], 0);
  assert.equal(record.candidate.note, ''); assert.equal(record.lane.left[0][2], 0); assert.equal(record.overlap.zMargin, 1);
  assert.equal(record.mapSha256, null); assert.equal(record.status, 'needs-review');
  const before = structuredClone(record.candidate);
  setLaneTaskDecision(record, 'update-needed', '要確認\n<img src=x onerror=alert(1)>');
  assert.deepEqual(record.candidate, before);
  const parsed = parseLaneTaskFile(laneTaskFile([record]));
  assert.deepEqual(parsed, [record]); assert.equal(parsed[0].note, record.note);
  parsed[0].lane.left[0][0] = -9; assert.equal(record.lane.left[0][0], -1);
});

test('historical states and invalid edits fail atomically without changing the recorded decision', () => {
  const record = task(); setLaneTaskDecision(record, 'deferred', 'Keep');
  for (const [status, note] of [['invented','x'],['update-needed','x'.repeat(2001)],['update-needed','\0']]) {
    const before = structuredClone(record); assert.throws(() => setLaneTaskDecision(record, status, note)); assert.deepEqual(record, before);
  }
  record.staleReason = 'Map replaced'; const before = structuredClone(record);
  assert.throws(() => setLaneTaskDecision(record, 'no-update', 'different'), /Historical/); assert.deepEqual(record, before);
  const stale = source(); stale.staleReason = 'old'; assert.throws(() => newLaneTask(2, stale, candidate(), lane(), overlap(), null));
});

test('task files reject reused IDs, impossible overlap, inconsistent counts and malformed source evidence', () => {
  for (const mutate of [v => v.counts['needs-review']++, v => v.tasks.push(structuredClone(v.tasks[0])),
    v => v.tasks[0].candidate.bounds.reverse(), v => v.tasks[0].candidate.centroid[0] = 99,
    v => v.tasks[0].candidate.note = 'untimestamped', v => v.tasks[0].lane.left[0][2] = NaN,
    v => { for (const curve of [v.tasks[0].lane.left,v.tasks[0].lane.right]) for (const p of curve) p[0] += 100; },
    v => v.tasks[0].mapSha256 = 'not-a-hash', v => v.tasks[0].overlap.checked = 2,
    v => v.tasks[0].overlap.zMargin = -1, v => v.tasks[0].createdAt = 'today',
    v => v.tasks[0].updatedAt = '2025-01-01T00:00:00.000Z', v => v.tasks[0].note = '\0']) {
    const file = laneTaskFile([task()]); mutate(file); assert.throws(() => parseLaneTaskFile(file));
  }
  const large = laneTaskFile([task()]); large.tasks[0].lane.left = Array.from({length:100001},()=>[0,2,0]);
  assert.throws(() => parseLaneTaskFile(large));
  assert.throws(() => parseLaneTaskFile(laneTaskFile(Array.from({length:1001},(_,i)=>task(i+1)))));
});

test('empty and mixed-status historical exports retain complete independent totals', () => {
  assert.deepEqual(parseLaneTaskFile(laneTaskFile([])), []);
  const records = [task(1),task(2),task(3),task(4)];
  setLaneTaskDecision(records[1], 'update-needed', ''); setLaneTaskDecision(records[2], 'no-update', ''); setLaneTaskDecision(records[3], 'deferred', '');
  records[2].staleReason = 'Map changed';
  assert.deepEqual(laneTaskCounts(records), {'needs-review':1,'update-needed':1,'no-update':1,deferred:1});
  assert.deepEqual(parseLaneTaskFile(laneTaskFile(records)), records);
});

test('core snapshots survive round trips, preserve zero and unavailable, and read legacy absence as unknown', () => {
  const evidence = {protocol:'xy-lane-z-v1',status:'complete',total:8,supported:0,heightMarginM:1,reason:null};
  const record = newLaneTask(1,source(),candidate(),lane(),overlap(),null,evidence);
  evidence.supported=8;
  assert.equal(record.coreSupport.supported,0);
  assert.deepEqual(parseLaneTaskFile(laneTaskFile([record])),[record]);
  const unknown=structuredClone(record);unknown.coreSupport={...record.coreSupport,status:'unavailable',supported:null,reason:'Core/triangle budget exceeded'};
  assert.deepEqual(parseLaneTaskFile(laneTaskFile([unknown])),[unknown]);
  const legacy=laneTaskFile([structuredClone(record)]);delete legacy.tasks[0].coreSupport;
  assert.equal(parseLaneTaskFile(legacy)[0].coreSupport,null);
  record.staleReason='Map replaced';setLaneTaskDecision(unknown,'deferred','Keep evidence');
  assert.equal(record.coreSupport.supported,0);assert.equal(unknown.coreSupport.supported,null);
});

test('core evidence rejects inconsistent counts, settings, status, reason and protocol', () => {
  const evidence=()=>({protocol:'xy-lane-z-v1',status:'complete',total:8,supported:3,heightMarginM:1,reason:null});
  for(const mutate of [e=>e.total=7,e=>e.supported=9,e=>e.supported=-1,e=>e.supported=.5,e=>e.supported=null,
    e=>e.heightMarginM=2,e=>e.protocol='invented',e=>e.status='pending',e=>e.reason='unexpected',
    e=>{e.status='unavailable';e.reason='missing';},e=>{e.status='unavailable';e.supported=null;e.reason='';},
    e=>{e.status='unavailable';e.supported=null;e.reason='\0';}]) {
    const record=newLaneTask(1,source(),candidate(),lane(),overlap(),null,evidence());
    mutate(record.coreSupport);assert.throws(()=>parseLaneTaskFile(laneTaskFile([record])));
  }
});
