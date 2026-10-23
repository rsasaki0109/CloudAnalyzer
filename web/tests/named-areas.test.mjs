import { test } from 'node:test';
import assert from 'node:assert/strict';
import { parseNamedAreas, parseAreaFile, captureNamedAreas, replaceNamedAreas, areaVolumesCsv, MAX_NAMED_AREAS } from '../src/named-areas.ts';

const areas = [{name:'工区 A',bounds:[1000,2000,1001,2001]},{name:'East',bounds:[1001,2000,1002,2001]}];
test('named area parsing normalizes labels and copies bounds without accepting missing or nonfinite coordinates',()=>{
  const source=structuredClone(areas), parsed=parseAreaFile({schema:'cloudanalyzer.named_review_areas.v1',areas:source});
  source[0].bounds[0]=0;assert.equal(parsed[0].bounds[0],1000);
  assert.equal(parseNamedAreas([{name:'  A  ',bounds:[0,0,1,1]}])[0].name,'A');
  for(const value of [null,[{name:'',bounds:[0,0,1,1]}],[{name:'A',bounds:null}],[{name:'A',bounds:[0,0,0,1]}],
    [{name:'A',bounds:[0,0,Infinity,1]}],[{name:'a\nb',bounds:[0,0,1,1]}],
    [{name:'A',bounds:[0,0,1,1]},{name:' A ',bounds:[1,1,2,2]}],
    Array.from({length:MAX_NAMED_AREAS+1},(_,i)=>({name:String(i),bounds:[0,0,1,1]}))])assert.throws(()=>parseNamedAreas(value));
  assert.throws(()=>parseAreaFile({schema:'other',areas}));
});
test('invalid replacement leaves the current library intact and captures are independent',()=>{
  replaceNamedAreas(areas);const saved=captureNamedAreas();saved[0].name='changed';
  assert.throws(()=>replaceNamedAreas([{name:'broken',bounds:null}]));assert.deepEqual(captureNamedAreas(),areas);
  replaceNamedAreas([]);
});
test('area CSV retains independent signed rows, escaped labels and unavailable quantities for no comparison',()=>{
  const base={...areas[0],added:2,removed:1,addedArea:1,removedArea:1,matchedCells:3,totalCells:4,outsideRegionCells:2};
  const csv=areaVolumesCsv([{...base,name:'A,"west"'},{...base,name:'=formula',matchedCells:0,totalCells:0,added:0,removed:0}], 'm');
  const rows=csv.trim().split('\n');assert.equal(rows.length,3);
  assert.ok(rows[1].startsWith('"A,""west""",1000,2000,1001,2001,"m",true,2,1,1,1,1,3,1,2'));
  assert.ok(rows[2].startsWith('"\'=formula"'));assert.ok(rows[2].includes(',false,,,,,,0,0,2'));
  assert.equal(csv.includes('total'),false);
});
