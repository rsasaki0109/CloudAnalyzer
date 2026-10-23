import { test } from 'node:test';
import assert from 'node:assert/strict';
import { volumeVisitReport } from '../src/volume-visit-report.ts';
const input={id:7,name:'baseline<img src=x onerror=alert(1)>',kind:'cloud',loadedPoints:2,filePoints:100,keepEvery:50,preview:true,copcLevels:2};
const row={name:'Area </pre><script>alert(1)</script>',bounds:[0,0,2,1],added:3,removed:2,net:1,addedArea:1,removedArea:1,matchedCells:2,totalCells:3,uncomparedCells:1,outsideRegionCells:4,comparisonAvailable:true,visit:'A & B',visitIndex:1};
const base={schema:'cloudanalyzer.area_volume_visits.v1',baseline:input,visits:[{...input,id:8,name:'A & B',visitIndex:1}],ordering:{policy:'User order; no dates inferred',cloudIds:[8]},settings:{cell:1,height:'min',fillEmpty:false,coordinateUnit:'ft'},sharedGrid:{min:[-.5,-.5],nx:3,ny:1,cell:1,origin:'Lower-left boundary'},support:{mode:'independent'},scope:'2.5D loaded-data scope',rows:[row]};
const embedded=html=>JSON.parse(html.match(/<pre id="comparison-json">([\s\S]*?)<\/pre>/)[1].replaceAll('&lt;','<').replaceAll('&gt;','>').replaceAll('&quot;','"').replaceAll('&#39;',"'").replaceAll('&amp;','&'));
test('standalone report preserves snapshot, escapes hostile labels and records loaded scope and independent-support limits',()=>{
 const original=structuredClone(base), html=volumeVisitReport(base,'2026-10-24T00:00:00.000Z');
 assert.deepEqual(embedded(html),base);assert.deepEqual(base,original);
 assert.ok(html.includes('Fill (ft³)'));assert.ok(html.includes('baseline&lt;img'));assert.ok(html.includes('&lt;/pre&gt;&lt;script&gt;'));
 assert.ok(!/<script|<img|<iframe/i.test(html));assert.ok(html.includes("default-src 'none'"));
 assert.ok(html.includes('Each baseline/visit pair may have different support'));assert.ok(html.includes('Not computed.'));
 assert.ok(html.includes('<td>2</td><td>100</td><td>50</td><td>true</td><td>2</td>'));
 assert.ok(html.includes('No combined total'));assert.ok(html.includes('export time, not acquisition time'));
});
test('common support report separates direct rows, unavailable quantities and exact companion-mask provenance',()=>{
 const snapshot={...base,rows:[{...row,matchedCells:0,uncomparedCells:3,comparisonAvailable:false,added:0,removed:0,net:0,addedArea:0,removedArea:0}],support:{mode:'common_observed',commonGridCells:0,commonGridArea:0,mask:{file:'common-observed-cells.bin',sha256:'a'.repeat(64),byteLength:3,encoding:'uint8 row-major'}},intervals:{scope:'Direct adjacent comparison',rows:[{...row,intervalIndex:1,fromVisitIndex:1,toVisitIndex:2,fromId:8,toId:9,fromVisit:'A',toVisit:'B'}]}};
 const html=volumeVisitReport(snapshot);
 assert.deepEqual(embedded(html),snapshot);
 const baseline=html.match(/<table id="baseline-volumes">([\s\S]*?)<\/table>/)[1];
 assert.equal((baseline.match(/<td>Unavailable<\/td>/g)||[]).length,5);
 assert.ok(html.includes('<td>1. A → 2. B</td>'));assert.ok(html.includes('<td>3</td><td>2</td><td>1</td>'));
 assert.ok(html.includes('SHA-256: '+'a'.repeat(64)));assert.ok(html.includes('mask is not embedded'));
 assert.ok(html.includes('Same comparison fields as Save visit summaries JSON'));
});
test('constant reference and interpolated estimates are explicit instead of implying observed baseline coverage',()=>{
 const html=volumeVisitReport({...base,baseline:{constantHeight:2},settings:{...base.settings,fillEmpty:true}});
 assert.ok(html.includes('Constant height 2'));assert.ok(html.includes('Plane'));
 assert.ok(html.includes('enabled (estimated heights may be included)'));
 assert.ok(html.includes('constant baseline is a plane, not an observed survey footprint'));
});
