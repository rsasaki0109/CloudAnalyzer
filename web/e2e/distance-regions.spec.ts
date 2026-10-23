import { expect, test, type Page, type Download } from '@playwright/test';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
const patches=(lifted:boolean)=>Array.from({length:27},(_,i)=>{
  const patch=Math.floor(i/9),j=i%9;return [patch*10+(j%3)*.1,Math.floor(j/3)*.1,lifted?[0,.5,1][patch]:0];
});
function ply(points:number[][]){
  const header=Buffer.from(`ply\nformat binary_little_endian 1.0\nelement vertex ${points.length}\nproperty float x\nproperty float y\nproperty float z\nend_header\n`),body=Buffer.alloc(points.length*12);
  points.forEach((p,i)=>p.forEach((v,a)=>body.writeFloatLE(v,i*12+a*4)));return Buffer.concat([header,body]);
}
const input=(name:string,buffer:Buffer)=>({name,mimeType:'application/octet-stream',buffer});
async function text(d:Download){const chunks:Buffer[]=[];for await(const c of await d.createReadStream())chunks.push(c);return Buffer.concat(chunks).toString();}
async function json(page:Page){const d=page.waitForEvent('download');await page.locator('#regions-json').click();return JSON.parse(await text(await d));}
async function cellsJson(page:Page){const d=page.waitForEvent('download');await page.locator('#regions-cells-json').click();return JSON.parse(await text(await d));}
async function bothJson(page:Page){const d=page.waitForEvent('download');await page.locator('#regions-both-json').click();return JSON.parse(await text(await d));}
async function trackDistances(page:Page){await page.evaluate(()=>{
  const original=Worker.prototype.postMessage;(window as any).distanceCalls=0;
  Worker.prototype.postMessage=function(message:any,...rest:any[]){if(message?.req?.kind==='c2c')(window as any).distanceCalls++;return (original as any).call(this,message,...rest);};
});}
async function setup(page:Page){
  await page.goto('/');await trackDistances(page);
  await page.locator('#file-input').setInputFiles([input('truth.ply',ply(patches(false))),input('map.ply',ply(patches(true)))]);
  await expect(page.locator('#cloud-list > li')).toHaveCount(2);await expect(page.locator('#task')).toBeHidden();
  await page.locator('#coverage-source').selectOption({label:'map.ply'});await page.locator('#coverage-reference').selectOption({label:'truth.ply'});
  await page.getByText('Locate distance discrepancies',{exact:true}).click();
  await page.locator('#regions-cell').fill('2');await page.locator('#regions-min').fill('5');await page.locator('#regions-threshold').fill('.3');
}
async function run(page:Page){await page.locator('#regions-run').click();await expect(page.locator('#status')).toContainText('Distance regions ready');}
test('two known discrepancy patches rank and focus, export exact metrics and reuse distances for regrouping',async({page},info)=>{
  const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));await setup(page);await run(page);
  await expect(page.locator('#regions-list > li')).toHaveCount(2);
  await expect(page.locator('#regions-list > li').first()).toContainText('9 points · max 1');
  await page.locator('#regions-list button').first().click();const result=await json(page);
  expect(result.exceeding).toBe(18);expect(result.measured).toBe(27);expect(result.regions[0].maxDistance).toBe(1);
  expect(result.regions[1].meanDistance).toBe(.5);expect(result.regions[0].bounds[2]).toBe(1);expect(result.reusedDistances).toBe(false);
  const csv=page.waitForEvent('download');await page.locator('#regions-csv').click();expect((await text(await csv)).trim().split('\n')).toHaveLength(3);
  await page.screenshot({path:info.outputPath('distance-regions.png')});
  await page.locator('#regions-cell').fill('25');await expect(page.locator('#regions-result')).toBeHidden();await run(page);
  const merged=await json(page);expect(merged.regions).toHaveLength(1);expect(merged.regions[0].points).toBe(18);expect(merged.regions[0].meanDistance).toBe(.75);
  expect(merged.reusedDistances).toBe(true);expect(await page.evaluate(()=>(window as any).distanceCalls)).toBe(1);
  await page.locator('#regions-min').fill('20');await run(page);const small=await json(page);expect(small.smallRegionPoints).toBe(18);expect(small.exceeding).toBe(18);
  await page.locator('#regions-threshold').fill('2');await run(page);expect((await json(page)).exceeding).toBe(0);
  await page.locator('#regions-cell').fill('3');const report=page.waitForEvent('download');await page.locator('#report-json').click();
  expect(JSON.parse(await text(await report)).sections['distance-regions']).toBeUndefined();expect(errors).toEqual([]);
});
test('removing a reference invalidates diagnosis even when another cloud has the same name',async({page})=>{
  await setup(page);await run(page);const first=await json(page);
  await page.locator('#file-input').setInputFiles(input('truth.ply',ply(patches(false))));await expect(page.locator('#cloud-list > li')).toHaveCount(3);
  await page.locator('#cloud-list > li').first().locator('button[title="Remove"]').click();await expect(page.locator('#regions-result')).toBeHidden();
  await page.locator('#coverage-reference').selectOption({label:'truth.ply'});await run(page);const second=await json(page);
  expect(second.reference.id).not.toBe(first.reference.id);expect(second.reusedDistances).toBe(false);
  expect(await page.evaluate(()=>(window as any).distanceCalls)).toBe(2);
});
test('cancellation during distance computation discards diagnosis after the active call completes',async({page})=>{
  await setup(page);await page.evaluate(()=>{
    const original=Worker.prototype.postMessage;
    Worker.prototype.postMessage=function(message:any,...rest:any[]){if(message?.req?.kind==='c2c')document.getElementById('task-cancel')!.click();return (original as any).call(this,message,...rest);};
  });
  await page.locator('#regions-run').click();await expect(page.locator('#status')).toContainText('Cancelled');
  await expect(page.locator('#regions-result')).toBeHidden();await expect(page.locator('#c2c-result')).toBeHidden();await expect(page.locator('#cloud-list > li')).toHaveCount(2);
});
test('cancelling grouping or changing settings in flight publishes no stale region report',async({page})=>{
  await setup(page);await run(page);
  await page.evaluate(()=>{
    const original=Worker.prototype.postMessage;
    Worker.prototype.postMessage=function(message:any,...rest:any[]){
      if(message?.distances&&message?.positions)document.getElementById('task-cancel')!.click();
      return (original as any).call(this,message,...rest);
    };
  });
  await page.locator('#regions-run').click();await expect(page.locator('#status')).toContainText('Cancelled');await expect(page.locator('#regions-result')).toBeHidden();
  const report=page.waitForEvent('download');await page.locator('#report-json').click();expect(JSON.parse(await text(await report)).sections['distance-regions']).toBeUndefined();
});
test('settings changed while the grouping worker runs invalidate its result',async({page})=>{
  await setup(page);await page.evaluate(()=>{
    const original=Worker.prototype.postMessage;
    Worker.prototype.postMessage=function(message:any,...rest:any[]){
      if(message?.distances&&message?.positions){const el=document.getElementById('regions-cell') as HTMLInputElement;el.value='3';el.dispatchEvent(new Event('input'));}
      return (original as any).call(this,message,...rest);
    };
  });
  await page.locator('#regions-run').click();await expect(page.locator('#status')).toContainText('inputs/settings changed');
  await expect(page.locator('#regions-result')).toBeHidden();await expect(page.locator('#cloud-list > li')).toHaveCount(2);
});

test('a broad connected region drills into exact cells, focuses and exports without another distance computation',async({page})=>{
  await setup(page);await page.locator('#regions-cell').fill('25');await run(page);
  await page.locator('#regions-fine-cell').fill('1');await page.getByRole('button',{name:'Inspect cells',exact:true}).click();
  await expect(page.locator('#status')).toContainText('Region cell inspection ready');
  await expect(page.locator('#regions-cells-list > li')).toHaveCount(2);
  const r=await cellsJson(page);expect(r.regionPoints).toBe(18);expect(r.cells.map((c:any)=>c.points)).toEqual([9,9]);
  expect(r.cells.map((c:any)=>c.meanDistance)).toEqual([1,.5]);expect(r.parentRegion.points).toBe(18);expect(r.reusedDistances).toBe(true);
  expect(r.unlistedPoints).toBe(0);expect(r.cells[0].bounds[3]-r.cells[0].bounds[0]).toBeLessThan(1);
  await page.locator('#regions-cells-list button').first().click();
  const csv=page.waitForEvent('download');await page.locator('#regions-cells-csv').click();expect((await text(await csv)).trim().split('\n')).toHaveLength(3);
  const report=page.waitForEvent('download');await page.locator('#report-json').click();
  expect(JSON.parse(await text(await report)).sections['distance-region-cells'].metrics.points.value).toBe(18);
  await page.locator('#regions-fine-cell').fill('.1');await expect(page.locator('#regions-cells')).toBeHidden();
  await expect(page.locator('#regions-result')).toBeVisible();await page.getByRole('button',{name:'Inspect cells',exact:true}).click();
  await expect(page.locator('#status')).toContainText('Region cell inspection ready');
  expect((await cellsJson(page)).occupiedCells).toBeGreaterThan(2);expect(await page.evaluate(()=>(window as any).distanceCalls)).toBe(1);
  await page.locator('#coverage-source').selectOption({label:'truth.ply'});await expect(page.locator('#regions-result')).toBeHidden();await expect(page.locator('#regions-cells')).toBeHidden();
  const stale=page.waitForEvent('download');await page.locator('#report-json').click();expect(JSON.parse(await text(await stale)).sections['distance-region-cells']).toBeUndefined();
});

test('cell inspection validates size and discards cancelled or stale worker results',async({page})=>{
  await setup(page);await run(page);await page.locator('#regions-fine-cell').fill('3');
  await page.getByRole('button',{name:'Inspect cells',exact:true}).first().click();await expect(page.locator('#status')).toContainText('no larger');
  await expect(page.locator('#regions-cells')).toBeHidden();await page.locator('#regions-fine-cell').fill('.1');
  await page.evaluate(()=>{
    const original=Worker.prototype.postMessage;(window as any).inspectionAction='cancel';
    Worker.prototype.postMessage=function(message:any,...rest:any[]){
      if(message?.inspection){
        if((window as any).inspectionAction==='cancel')document.getElementById('task-cancel')!.click();
        else {const el=document.getElementById('regions-fine-cell') as HTMLInputElement;el.value='.2';el.dispatchEvent(new Event('input'));}
      }
      return (original as any).call(this,message,...rest);
    };
  });
  await page.getByRole('button',{name:'Inspect cells',exact:true}).first().click();await expect(page.locator('#status')).toContainText('Cancelled');
  await expect(page.locator('#regions-cells')).toBeHidden();await expect(page.locator('#regions-result')).toBeVisible();
  await page.evaluate(()=>(window as any).inspectionAction='change');
  await page.getByRole('button',{name:'Inspect cells',exact:true}).first().click();await expect(page.locator('#status')).toContainText('inputs/settings changed');
  await expect(page.locator('#regions-cells')).toBeHidden();expect(await page.evaluate(()=>(window as any).distanceCalls)).toBe(1);
});

test('bidirectional regions expose asymmetric missing coverage, switch inspection direction and reuse both snapshots',async({page})=>{
  const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));await page.goto('/');await trackDistances(page);
  await page.locator('#file-input').setInputFiles([input('partial.ply',ply(patches(false).slice(0,18))),input('full.ply',ply(patches(false)))]);
  await expect(page.locator('#cloud-list > li')).toHaveCount(2);await expect(page.locator('#task')).toBeHidden();
  await page.locator('#coverage-source').selectOption({label:'full.ply'});await page.locator('#coverage-reference').selectOption({label:'partial.ply'});
  await page.getByText('Locate distance discrepancies',{exact:true}).click();await page.locator('#regions-cell').fill('2');await page.locator('#regions-min').fill('1');
  await page.locator('#regions-both').click();await expect(page.locator('#status')).toContainText('ready in both directions');
  const r=await bothJson(page);expect(r.forward.points).toBe(27);expect(r.reverse.points).toBe(18);
  expect(r.forward.exceeding).toBe(9);expect(r.reverse.exceeding).toBe(0);expect(r.forward.exceedingFractionOfMeasured).toBe(1/3);expect(r.reverse.exceedingFractionOfMeasured).toBe(0);
  expect(r.forward.reusedDistances).toBe(false);expect(r.reverse.reusedDistances).toBe(false);
  await expect(page.locator('#regions-both-summary')).toContainText('33.33%');
  await page.getByRole('button',{name:'Inspect cells',exact:true}).click();await expect(page.locator('#status')).toContainText('cell inspection ready');
  expect((await cellsJson(page)).source.name).toBe('full.ply');
  await page.locator('#regions-direction').selectOption('reverse');await expect(page.locator('#regions-cells')).toBeHidden();await expect(page.locator('#regions-list > li')).toHaveCount(0);
  const reverse=await json(page);expect(reverse.source.name).toBe('partial.ply');expect(reverse.reference.name).toBe('full.ply');expect(reverse.exceeding).toBe(0);
  const report=page.waitForEvent('download');await page.locator('#report-json').click();const qa=JSON.parse(await text(await report));
  expect(qa.sections['bidirectional-regions'].metrics.forwardExceeding.value).toBe(9);expect(qa.sections['bidirectional-regions'].metrics.reverseExceeding.value).toBe(0);
  expect(qa.sections['distance-region-cells']).toBeUndefined();
  await page.locator('#regions-threshold').fill('.4');await expect(page.locator('#regions-result')).toBeHidden();
  await page.locator('#regions-both').click();await expect(page.locator('#status')).toContainText('ready in both directions');const reused=await bothJson(page);
  expect(reused.forward.reusedDistances).toBe(true);expect(reused.reverse.reusedDistances).toBe(true);
  await run(page);await expect(page.locator('#regions-directions')).toBeHidden();
  expect(await page.evaluate(()=>(window as any).distanceCalls)).toBe(2);expect(errors).toEqual([]);
});

test('bidirectional comparison reuses an existing forward diagnosis and inspects reverse cells with correct source identity',async({page})=>{
  await setup(page);await run(page);await page.locator('#regions-both').click();await expect(page.locator('#status')).toContainText('ready in both directions');
  const r=await bothJson(page);expect(r.forward.reusedDistances).toBe(true);expect(r.reverse.reusedDistances).toBe(false);
  await page.locator('#regions-direction').selectOption('reverse');await page.getByRole('button',{name:'Inspect cells',exact:true}).first().click();
  await expect(page.locator('#status')).toContainText('cell inspection ready');const cells=await cellsJson(page);
  expect(cells.source.name).toBe('truth.ply');expect(cells.reference.name).toBe('map.ply');
  await page.locator('#regions-direction').selectOption('forward');await expect(page.locator('#regions-cells')).toBeHidden();
  expect((await json(page)).source.name).toBe('map.ply');expect(await page.evaluate(()=>(window as any).distanceCalls)).toBe(2);
  await page.locator('#file-input').setInputFiles(input('truth.ply',ply(patches(false))));await expect(page.locator('#cloud-list > li')).toHaveCount(3);
  await page.locator('#cloud-list > li').first().locator('button[title="Remove"]').click();await expect(page.locator('#regions-result')).toBeHidden();
  const report=page.waitForEvent('download');await page.locator('#report-json').click();expect(JSON.parse(await text(await report)).sections['bidirectional-regions']).toBeUndefined();
  await page.locator('#coverage-reference').selectOption({label:'truth.ply'});await page.locator('#regions-both').click();await expect(page.locator('#status')).toContainText('ready in both directions');
  const replaced=await bothJson(page);expect(replaced.forward.reference.id).not.toBe(r.forward.reference.id);
  expect(replaced.forward.reusedDistances).toBe(false);expect(replaced.reverse.reusedDistances).toBe(false);expect(await page.evaluate(()=>(window as any).distanceCalls)).toBe(4);
});

for(const action of ['cancel','settings'] as const)test(`bidirectional ${action} during reverse computation publishes neither partial result`,async({page})=>{
  await setup(page);await page.evaluate(action=>{
    const original=Worker.prototype.postMessage;let calls=0;
    Worker.prototype.postMessage=function(message:any,...rest:any[]){
      if(message?.req?.kind==='c2c'&&++calls===2){
        if(action==='cancel')document.getElementById('task-cancel')!.click();
        else {const el=document.getElementById('regions-threshold') as HTMLInputElement;el.value='.4';el.dispatchEvent(new Event('input'));}
      }
      return (original as any).call(this,message,...rest);
    };
  },action);
  await page.locator('#regions-both').click();await expect(page.locator('#status')).toContainText(action==='cancel'?'Cancelled':'inputs changed');
  await expect(page.locator('#regions-result')).toBeHidden();await expect(page.locator('#regions-directions')).toBeHidden();
  const report=page.waitForEvent('download');await page.locator('#report-json').click();const qa=JSON.parse(await text(await report));
  expect(qa.sections['bidirectional-regions']).toBeUndefined();expect(qa.sections['distance-regions']).toBeUndefined();
});
test('actual existing survey subsets produce bounded directed regions without modifying sources',async({page},info)=>{
  const a=process.env.REGIONS_SOURCE,b=process.env.REGIONS_REFERENCE;test.skip(!a||!b,'Optional existing local surveys; no download in CI');test.setTimeout(180000);
  const rawA=await readFile(a!),rawB=await readFile(b!),sha=(v:Buffer)=>createHash('sha256').update(v).digest('hex');
  const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));await page.setViewportSize({width:1600,height:1000});await page.goto('/');await trackDistances(page);
  await page.locator('#file-input').setInputFiles([a!,b!]);await expect(page.locator('#cloud-list > li')).toHaveCount(2,{timeout:120000});await expect(page.locator('#task')).toBeHidden();
  await page.locator('#coverage-source').selectOption({label:'compared.ply'});await page.locator('#coverage-reference').selectOption({label:'reference.ply'});
  await page.getByText('Locate distance discrepancies',{exact:true}).click();await page.locator('#regions-run').click();
  await expect(page.locator('#status')).toContainText('Distance regions ready',{timeout:120000});const result=await json(page);
  expect(result.measured+result.unmeasured+result.invalidPositions).toBe(result.points);
  expect(result.regions.reduce((n:number,r:any)=>n+r.points,0)+result.smallRegionPoints+result.unlistedEligiblePoints).toBe(result.exceeding);
  expect(result.regions.length).toBeLessThanOrEqual(50);expect(result.source.keepEvery).toBe(1);expect(result.reference.keepEvery).toBe(1);
  expect(sha(await readFile(a!))).toBe(sha(rawA));expect(sha(await readFile(b!))).toBe(sha(rawB));expect(errors).toEqual([]);
  await page.getByRole('button',{name:'Top',exact:true}).click();await page.locator('#regions-run').evaluate(el=>el.scrollIntoView({block:'start'}));
  await page.screenshot({path:info.outputPath('actual-distance-regions.png')});
  await writeFile(info.outputPath('regions-receipt.json'),JSON.stringify({source_sha256:sha(rawA),reference_sha256:sha(rawB),result,
    source_files_unchanged:true,browser_errors:errors,registration_defect_accuracy_measured:false},null,2));
  await page.getByRole('button',{name:'Inspect cells',exact:true}).first().click();
  await expect(page.locator('#status')).toContainText('Region cell inspection ready',{timeout:30000});const cells=await cellsJson(page);
  expect(cells.regionPoints).toBe(result.regions[0].points);
  expect(cells.cells.reduce((n:number,c:any)=>n+c.points,0)+cells.unlistedPoints).toBe(cells.regionPoints);
  for(const c of cells.cells){expect(c.bounds[3]-c.bounds[0]).toBeLessThanOrEqual(1);expect(c.bounds[4]-c.bounds[1]).toBeLessThanOrEqual(1);}
  await page.locator('#regions-cells-list button').first().click();await page.locator('#regions-cells-summary').evaluate(el=>el.scrollIntoView({block:'start'}));
  await page.screenshot({path:info.outputPath('actual-region-cells.png')});
  expect(sha(await readFile(a!))).toBe(sha(rawA));expect(sha(await readFile(b!))).toBe(sha(rawB));expect(errors).toEqual([]);
  await writeFile(info.outputPath('cells-receipt.json'),JSON.stringify({source_sha256:sha(rawA),reference_sha256:sha(rawB),result:cells,
    source_files_unchanged:true,browser_errors:errors,registration_defect_accuracy_measured:false},null,2));
  await page.locator('#regions-both').click();await expect(page.locator('#status')).toContainText('ready in both directions',{timeout:120000});const both=await bothJson(page);
  expect(both.forward.reusedDistances).toBe(true);expect(both.reverse.reusedDistances).toBe(false);expect(both.forward.exceeding).toBe(result.exceeding);
  expect(await page.evaluate(()=>(window as any).distanceCalls)).toBe(2);
  for(const r of [both.forward,both.reverse]){
    expect(r.measured+r.unmeasured+r.invalidPositions).toBe(r.points);
    expect(r.regions.reduce((n:number,c:any)=>n+c.points,0)+r.smallRegionPoints+r.unlistedEligiblePoints).toBe(r.exceeding);
    expect(r.exceedingFractionOfMeasured).toBe(r.exceeding/r.measured);
  }
  await page.locator('#regions-direction').selectOption('reverse');expect((await json(page)).source.name).toBe('reference.ply');
  await page.getByRole('button',{name:'Fit',exact:true}).click();await page.getByRole('button',{name:'Top',exact:true}).click();
  await page.locator('#regions-both-summary').evaluate(el=>el.scrollIntoView({block:'start'}));await page.screenshot({path:info.outputPath('actual-bidirectional-regions.png')});
  expect(sha(await readFile(a!))).toBe(sha(rawA));expect(sha(await readFile(b!))).toBe(sha(rawB));expect(errors).toEqual([]);
  await writeFile(info.outputPath('both-receipt.json'),JSON.stringify({source_sha256:sha(rawA),reference_sha256:sha(rawB),result:both,
    source_files_unchanged:true,browser_errors:errors,registration_defect_accuracy_measured:false},null,2));
});
