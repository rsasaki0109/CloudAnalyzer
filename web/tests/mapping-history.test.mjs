import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { parseMappingHistory } from '../src/mapping-history.ts';
const fixture = () => ({schema:'cloudanalyzer.mapping_run.v1',history:[
  {sequence:1,action:{type:'inspect',candidate_ids:[2,3]},reason:'Inspect source',status:'inspected'},
  {sequence:2,action:{type:'draft',decisions:[{candidate_id:2,action:'defer',from_m:1,to_m:4,reason:'Insufficient observations'}]},reason:'Keep unresolved extent visible',status:'audited_draft'},
]});

test('real NCLT saved actions retain order, reasons and included/deferred source intervals', () => {
  const value = JSON.parse(readFileSync(new URL('../../benchmarks/vector-map/nclt-motion-pairs/april/run.json', import.meta.url)));
  const parsed = parseMappingHistory(value);
  assert.equal(parsed.unavailable, null);
  assert.deepEqual(parsed.steps.map(s => s.action), ['inspect','inspect','draft','finish']);
  assert.equal(parsed.steps[2].decisionsTotal, 18);
  assert.equal(parsed.steps[2].decisions[0].action, 'defer');
  assert.equal(parsed.steps[2].reason, value.history[2].reason);
});
test('missing, unsupported and malformed history cannot appear as successful or empty generation', () => {
  for (const value of [{}, null, {...fixture(),history:null}, {...fixture(),history:Array(129).fill(fixture().history[0])}]) {
    assert.ok(parseMappingHistory(value).unavailable);
  }
  for (const mutate of [v=>v.history[1].sequence=1, v=>v.history[0].reason='x'.repeat(4001), v=>v.history[1].action.decisions[0].to_m=NaN]) {
    const v=fixture();mutate(v);assert.ok(parseMappingHistory(v).unavailable);
  }
  assert.deepEqual(parseMappingHistory({schema:'cloudanalyzer.mapping_run.v1',history:[]}),{steps:[],unavailable:null});
});
test('display caps disclose omitted intervals and keep markup as plain data', () => {
  const v=fixture();v.history[0].reason='<img src=x onerror=alert(1)>';
  v.history[1].action.decisions=Array(257).fill(v.history[1].action.decisions[0]);
  const result=parseMappingHistory(v);
  assert.equal(result.steps[1].decisions.length,256);
  assert.equal(result.steps[1].decisionsTotal,257);
  assert.equal(result.steps[0].reason,v.history[0].reason);
});
