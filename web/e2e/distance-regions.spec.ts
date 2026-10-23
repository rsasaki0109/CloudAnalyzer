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
async function setup(page:Page){
  await page.goto('/');await page.evaluate(()=>{
    const original=Worker.prototype.postMessage;(window as any).distanceCalls=0;
    Worker.prototype.postMessage=function(message:any,...rest:any[]){if(message?.req?.kind==='c2c')(window as any).distanceCalls++;return (original as any).call(this,message,...rest);};
  });
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
test('actual existing survey subsets produce bounded directed regions without modifying sources',async({page},info)=>{
  const a=process.env.REGIONS_SOURCE,b=process.env.REGIONS_REFERENCE;test.skip(!a||!b,'Optional existing local surveys; no download in CI');test.setTimeout(180000);
  const rawA=await readFile(a!),rawB=await readFile(b!),sha=(v:Buffer)=>createHash('sha256').update(v).digest('hex');
  const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));await page.setViewportSize({width:1600,height:1000});await page.goto('/');
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
});
