import { expect, test, type Page, type Download } from '@playwright/test';
import { writeFile } from 'node:fs/promises';

function ply(points:number[][]){
  const header=Buffer.from(`ply\nformat binary_little_endian 1.0\nelement vertex ${points.length}\nproperty float x\nproperty float y\nproperty float z\nend_header\n`),body=Buffer.alloc(points.length*12);
  points.forEach((p,i)=>p.forEach((v,a)=>body.writeFloatLE(v,i*12+a*4)));return Buffer.concat([header,body]);
}
async function text(d:Download){const chunks:Buffer[]=[];for await(const c of await d.createReadStream())chunks.push(c);return Buffer.concat(chunks).toString();}
async function volumeJson(page:Page){const d=page.waitForEvent('download');await page.locator('#volume-json').click();return JSON.parse(await text(await d));}
async function report(page:Page){const d=page.waitForEvent('download');await page.locator('#report-json').click();return JSON.parse(await text(await d));}
async function setup(page:Page){
  await page.goto('/');
  const before=Array.from({length:6},(_,i)=>[1000+i%3,2000+Math.floor(i/3),10]);
  const after=before.slice(0,5).map((p,i)=>[p[0],p[1],p[2]+[2,-1,0,0,0][i]]);
  await page.locator('#file-input').setInputFiles([
    {name:'before-ground.ply',mimeType:'application/octet-stream',buffer:ply(before)},
    {name:'after-ground.ply',mimeType:'application/octet-stream',buffer:ply(after)},
  ]);
  await expect(page.locator('#cloud-list > li')).toHaveCount(2);await expect(page.locator('#task')).toBeHidden();
  await page.locator('#coverage-source').selectOption({label:'after-ground.ply'});
  await page.locator('#coverage-reference').selectOption({label:'before-ground.ply'});
  await page.locator('#volume-options > summary').click();await page.locator('#volume-fill').check();
  await page.locator('#coverage-volume').click();await expect(page.locator('#volume-fill')).not.toBeChecked();
  await page.locator('#volume-cell').fill('1');
  await page.evaluate(()=>{
    const original=Worker.prototype.postMessage;(window as any).volumeCalls=0;(window as any).unpublishedRemovals=0;
    Worker.prototype.postMessage=function(message:any,...rest:any[]){
      if(message?.req?.kind==='volume')(window as any).volumeCalls++;
      if(message?.req?.kind==='remove')(window as any).unpublishedRemovals++;
      return (original as any).call(this,message,...rest);
    };
  });
}
async function run(page:Page){await page.locator('#volume-run').click();await expect(page.locator('#status')).toContainText('Volume: fill');await expect(page.locator('#task')).toBeHidden();}

test('ground surveys export known cut/fill, signed cell differences, missing coverage and exact roles',async({page},info)=>{
  const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));await setup(page);
  await expect(page.locator('#volume-before')).toHaveValue(await page.locator('#coverage-reference').inputValue());
  await expect(page.locator('#volume-after')).toHaveValue(await page.locator('#coverage-source').inputValue());
  await run(page);await expect(page.locator('#status')).toContainText('fill 2 m³, cut 1 m³, net 1 m³');
  await expect(page.locator('#volume-stats')).toContainText(/Uncompared cells\s*1/);await expect(page.locator('#cloud-list > li')).toHaveCount(3);
  await expect(page.locator('#colorbar-title')).toContainText('Height difference');
  const result=await volumeJson(page);
  expect(result.before.name).toBe('before-ground.ply');expect(result.after.name).toBe('after-ground.ply');
  expect(result.before.loadedPoints).toBe(6);expect(result.after.loadedPoints).toBe(5);
  expect(result.settings).toEqual({cell:1,height:'mean',fillEmpty:false,coordinateUnit:'m'});
  expect(result.result).toEqual({added:2,removed:1,net:1,addedArea:1,removedArea:1,matchedCells:5,totalCells:6,uncomparedCells:1,cell:1});
  const csvEvent=page.waitForEvent('download');await page.locator('#volume-csv').click();const csv=await text(await csvEvent);
  const rows=csv.trim().split('\n');expect(rows).toHaveLength(6);
  const values=rows.slice(1).map(r=>r.split(',').map(Number));
  expect(values.find(r=>r[1]===1000&&r[2]===2000)?.slice(3)).toEqual([12,2]);
  expect(values.find(r=>r[1]===1001&&r[2]===2000)?.slice(3)).toEqual([9,-1]);
  const qa=await report(page);expect(qa.sections.volume.metrics.fill.value).toBe(2);expect(qa.sections.volume.metrics.uncompared.value).toBe(1);
  expect(qa.sections.volume.notes.join(' ')).toContain('interpolation disabled');
  await expect(page.locator('#volume-before')).toHaveValue(String(result.before.id));await expect(page.locator('#volume-after')).toHaveValue(String(result.after.id));
  await page.locator('#volume-isolate').click();
  await expect(page.locator('#cloud-list > li').nth(0).locator('input[title="Show / hide"]')).not.toBeChecked();
  await expect(page.locator('#cloud-list > li').nth(1).locator('input[title="Show / hide"]')).not.toBeChecked();
  await expect(page.locator('#cloud-list > li').nth(2).locator('input[title="Show / hide"]')).toBeChecked();
  await expect(page.locator('#volume-exports')).toBeVisible();expect((await report(page)).sections.volume.metrics.net.value).toBe(1);
  await page.locator('#volume-result').scrollIntoViewIfNeeded();await page.screenshot({path:info.outputPath('surface-volume.png')});
  await writeFile(info.outputPath('volume-receipt.json'),JSON.stringify({fixture:'Synthetic aligned ground cells; known fill 2, cut 1; not a survey accuracy benchmark',result,csv,errors},null,2));
  expect(errors).toEqual([]);
});

test('surface-volume settings and removed inputs clear stale exports and QA; grid budget fails before computation',async({page})=>{
  await setup(page);await run(page);
  await page.locator('#volume-unit').selectOption('ft');await expect(page.locator('#volume-exports')).toBeHidden();
  expect((await report(page)).sections.volume).toBeUndefined();await run(page);
  await expect(page.locator('#status')).toContainText('fill 2 ft³, cut 1 ft³, net 1 ft³');expect((await volumeJson(page)).result.net).toBe(1);
  await page.locator('#volume-cell').fill('0.000001');await expect(page.locator('#volume-result')).toBeHidden();
  await page.locator('#volume-run').click();await expect(page.locator('#status')).toContainText('1,000,000 cells');
  expect(await page.evaluate(()=>(window as any).volumeCalls)).toBe(2);
  await page.locator('#volume-cell').fill('1');await run(page);
  await page.locator('#cloud-list > li').first().locator('button[title="Remove"]').click();
  await expect(page.locator('#volume-result')).toBeHidden();expect((await report(page)).sections.volume).toBeUndefined();
  await page.locator('#volume-before').selectOption('constant');await page.locator('#volume-before-z').fill('');
  await page.locator('#volume-run').click();await expect(page.locator('#status')).toContainText('finite constant');
});

test('cancelled or changed in-flight volume requests release unpublished clouds and never publish stale metrics',async({page})=>{
  const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));await setup(page);
  await page.evaluate(()=>{
    const original=Worker.prototype.postMessage;(window as any).volumeAction='cancel';
    Worker.prototype.postMessage=function(message:any,...rest:any[]){
      if(message?.req?.kind==='volume'){
        if((window as any).volumeAction==='cancel')document.getElementById('task-cancel')!.click();
        else if((window as any).volumeAction==='change'){const el=document.getElementById('volume-cell') as HTMLInputElement;el.value='2';el.dispatchEvent(new Event('input'));}
      }
      return (original as any).call(this,message,...rest);
    };
  });
  await page.locator('#volume-run').click();await expect(page.locator('#status')).toContainText('Cancelled');await expect(page.locator('#task')).toBeHidden();
  await expect(page.locator('#cloud-list > li')).toHaveCount(2);await expect(page.locator('#volume-exports')).toBeHidden();
  expect((await report(page)).sections.volume).toBeUndefined();
  await page.evaluate(()=>(window as any).volumeAction='change');await page.locator('#volume-run').click();
  await expect(page.locator('#status')).toContainText('inputs/settings changed');await expect(page.locator('#task')).toBeHidden();
  await expect(page.locator('#cloud-list > li')).toHaveCount(2);expect(await page.evaluate(()=>(window as any).unpublishedRemovals)).toBe(2);
  await page.evaluate(()=>(window as any).volumeAction='none');await page.locator('#volume-cell').fill('1');await run(page);
  expect((await volumeJson(page)).settings.cell).toBe(1);expect(errors).toEqual([]);
});

async function area(page:Page,bounds:number[]) {
  await page.locator('#volume-area-options').evaluate((el:HTMLDetailsElement)=>{el.open=true;});
  await page.locator('#volume-area-enabled').check();
  for(const [i,name] of ['min-x','min-y','max-x','max-y'].entries()) await page.locator(`#volume-area-${name}`).fill(String(bounds[i]));
}
test('world XY volume area selects closed cell centers, exports exact scoped cells and distinguishes missing from outside',async({page})=>{
  await setup(page);await area(page,[1001,1999.5,1002,2001.5]);await run(page);
  const json=await volumeJson(page);
  expect(json.settings.region).toEqual([1001,1999.5,1002,2001.5]);
  expect(json.result).toMatchObject({added:0,removed:1,net:-1,matchedCells:3,totalCells:4,uncomparedCells:1});
  expect(json.reviewArea.outsideCells).toBe(2);
  expect(json.reviewArea.selection).toContain('no fractional clipping');
  expect(json.before.loadedPoints).toBe(6);expect(json.after.loadedPoints).toBe(5);
  const d=page.waitForEvent('download');await page.locator('#volume-csv').click();
  const rows=(await text(await d)).trim().split('\n').slice(1).map(row=>row.split(',').map(Number));
  expect(rows).toHaveLength(3);expect(rows.every(row=>row[1]>=1001&&row[1]<=1002)).toBe(true);
  expect(rows.reduce((sum,row)=>sum+row[4],0)).toBe(-1);
  expect((await report(page)).sections.volume.notes.join(' ')).toContain('2 grid cells outside');
  await page.locator('#volume-area-max-x').fill('1001.1');
  await expect(page.locator('#volume-exports')).toBeHidden();expect((await report(page)).sections.volume).toBeUndefined();
  await page.locator('#volume-area-enabled').uncheck();await run(page);
  expect((await volumeJson(page)).result).toMatchObject({added:2,removed:1,totalCells:6});
  expect((await volumeJson(page)).reviewArea).toBeNull();
});
test('review area copies between volume and distance without computation and invalid or empty bounds do not imply surveyed coverage',async({page})=>{
  await setup(page);await area(page,[1000,2000,1001,2001]);
  await page.locator('#volume-send-distance-area').click();
  await expect(page.locator('#regions-overlap-enabled')).toBeChecked();
  expect(await page.locator('#regions-overlap-max-x').inputValue()).toBe('1001');
  expect(await page.evaluate(()=>(window as any).volumeCalls)).toBe(0);
  await page.getByText('Locate distance discrepancies',{exact:true}).click();
  await page.getByText('Expected overlap area',{exact:true}).click();
  await page.locator('#regions-overlap-max-x').fill('1002');
  await page.locator('#volume-use-distance-area').click();
  expect(await page.locator('#volume-area-max-x').inputValue()).toBe('1002');
  await run(page);expect((await volumeJson(page)).reviewArea.bounds).toEqual([1000,2000,1002,2001]);
  await page.locator('#volume-area-min-x').fill('');await page.locator('#volume-run').click();
  await expect(page.locator('#status')).toContainText('all four volume review bounds');
  await expect(page.locator('#volume-exports')).toBeHidden();
  expect(await page.evaluate(()=>(window as any).volumeCalls)).toBe(1);
  await area(page,[10,10,11,11]);await run(page);
  const empty=await volumeJson(page);
  expect(empty.result).toMatchObject({added:0,removed:0,matchedCells:0,totalCells:0,uncomparedCells:0});
  expect(empty.reviewArea.outsideCells).toBe(6);
  expect((await report(page)).sections.volume.metrics.coverage).toBeUndefined();
  await expect(page.locator('#volume-stats')).toContainText('unavailable');
  await expect(page.locator('#volume-scope')).toContainText('zero totals do not establish zero surface change');
  await expect(page.locator('#volume-mesh')).toBeDisabled();
  await expect(page.locator('#cloud-list > li')).toHaveCount(3); // Only the earlier successful compared cloud.
});
test('changing an in-flight review rectangle discards the staged volume instead of publishing stale area totals',async({page})=>{
  await setup(page);await area(page,[999,1999,1003,2002]);
  await page.evaluate(()=>{
    const send=Worker.prototype.postMessage;
    Worker.prototype.postMessage=function(message:any,...rest:any[]){
      if(message?.req?.kind==='volume'){
        const input=document.getElementById('volume-area-max-x') as HTMLInputElement;
        input.value='1001';input.dispatchEvent(new Event('input'));
      }
      return (send as any).call(this,message,...rest);
    };
  });
  await page.locator('#volume-run').click();await expect(page.locator('#status')).toContainText('inputs/settings changed');
  await expect(page.locator('#task')).toBeHidden();await expect(page.locator('#volume-exports')).toBeHidden();
  await expect(page.locator('#cloud-list > li')).toHaveCount(2);
  expect((await report(page)).sections.volume).toBeUndefined();
});
