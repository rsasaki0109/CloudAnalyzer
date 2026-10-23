import { expect, test, type Page, type Download } from '@playwright/test';

function ply(points:number[][]) {
  return Buffer.from(`ply\nformat ascii 1.0\nelement vertex ${points.length}\nproperty float x\nproperty float y\nproperty float z\nend_header\n`+points.map(p=>p.join(' ')).join('\n')+'\n');
}
const input=(name:string,buffer:Buffer)=>({name,mimeType:'application/octet-stream',buffer});
const before=Array.from({length:6},(_,i)=>[1000+i%3,2000+Math.floor(i/3),10]);
const sourceFiles=[input('before.ply',ply(before)),input('after.ply',ply(before.slice(0,5).map((p,i)=>[p[0],p[1],p[2]+[2,-1,0,0,0][i]])))];
async function bytes(d:Download){const parts:Buffer[]=[];for await(const part of await d.createReadStream())parts.push(part);return Buffer.concat(parts);}
async function download(page:Page,id:string){const d=page.waitForEvent('download');await page.locator(id).click();return bytes(await d);}
async function setup(page:Page) {
  await page.goto('/');await page.locator('#file-input').setInputFiles(sourceFiles);
  await expect(page.locator('#cloud-list > li')).toHaveCount(2);await expect(page.locator('#task')).toBeHidden();
  await page.locator('#volume-area-options > summary').click();await page.locator('#volume-options > summary').click();
  await page.locator('#volume-cell').fill('1');
}
async function saveArea(page:Page,name:string,bounds:number[]) {
  await page.locator('#volume-area-enabled').check();await page.locator('#volume-area-name').fill(name);
  for(const [i,key] of ['min-x','min-y','max-x','max-y'].entries())await page.locator(`#volume-area-${key}`).fill(String(bounds[i]));
  await page.locator('#volume-area-save').click();await expect(page.locator('#status')).toContainText('Saved area');
}
const fill=[999.9,1999.9,1000.1,2000.1],cut=[1000.9,1999.9,1002.1,2001.1];
test('saved areas produce independent exact f64 summaries without derived clouds and retain empty-area scope',async({page},info)=>{
  const errors:string[]=[];page.on('pageerror',error=>errors.push(error.message));await setup(page);
  await saveArea(page,'Fill area',fill);await saveArea(page,'Cut area',cut);
  await saveArea(page,'No survey',[10,10,11,11]);await saveArea(page,'Overlapping whole site',[999,1999,1003,2002]);
  await page.locator('#volume-areas-run').click();await expect(page.locator('#status')).toContainText('Computed 4 named area summaries');
  await expect(page.locator('#cloud-list > li')).toHaveCount(2);
  const json=JSON.parse((await download(page,'#volume-areas-json')).toString());
  expect(json.before.name).toBe('before.ply');expect(json.after.name).toBe('after.ply');expect(json.settings.region).toBeUndefined();
  expect(json.areas.map((r:any)=>[r.name,r.added,r.removed,r.net,r.matchedCells,r.uncomparedCells])).toEqual([
    ['Fill area',2,0,2,1,0],['Cut area',0,1,-1,3,1],['No survey',0,0,0,0,0],['Overlapping whole site',2,1,1,5,1]]);
  expect(json.areas[2].comparisonAvailable).toBe(false);expect(json).not.toHaveProperty('total');
  await expect(page.locator('#volume-areas-rows tr').nth(2)).toContainText('Unavailable');
  await page.locator('#volume-areas-result').screenshot({path:info.outputPath('named-area-summaries.png')});
  const csv=(await download(page,'#volume-areas-csv')).toString();expect(csv.trim().split('\n')).toHaveLength(5);
  expect(csv).toContain('"No survey",10,10,11,11,"m",false,,,,,,0,0,6');
  const qa=JSON.parse((await download(page,'#report-json')).toString());
  expect(qa.sections['volume-areas'].metrics['area_Fill%20area_fill'].value).toBe(2);
  expect(qa.sections['volume-areas'].metrics['area_No%20survey_fill']).toBeUndefined();
  const definitions=JSON.parse((await download(page,'#volume-areas-export')).toString());definitions.areas.reverse();
  await page.locator('#volume-areas-file').setInputFiles(input('reordered.json',Buffer.from(JSON.stringify(definitions))));
  await page.locator('#volume-areas-run').click();await expect(page.locator('#status')).toContainText('Computed 4 named area summaries');
  const reordered=JSON.parse((await download(page,'#report-json')).toString());
  expect(reordered.sections['volume-areas'].metrics['area_Fill%20area_fill'].value).toBe(2);
  const selectedBefore=await page.locator('#volume-before').inputValue(),selectedAfter=await page.locator('#volume-after').inputValue();
  await page.locator('#file-input').setInputFiles(input('unrelated.ply',ply([[0,0,0],[1,0,0],[0,1,0]])));
  await expect(page.locator('#cloud-list > li')).toHaveCount(3);
  await expect(page.locator('#volume-before')).toHaveValue(selectedBefore);await expect(page.locator('#volume-after')).toHaveValue(selectedAfter);
  await expect(page.locator('#volume-areas-result')).toBeVisible();
  await page.locator('#volume-cell').fill('2');await expect(page.locator('#volume-areas-result')).toBeHidden();
  expect(JSON.parse((await download(page,'#report-json')).toString()).sections['volume-areas']).toBeUndefined();
  await page.locator('#volume-areas-run').click();await expect(page.locator('#status')).toContainText('Computed 4 named area summaries');
  await page.locator('#cloud-list > li').first().locator('button[title="Remove"]').click();
  await expect(page.locator('#volume-areas-result')).toBeHidden();
  expect(JSON.parse((await download(page,'#report-json')).toString()).sections['volume-areas']).toBeUndefined();
  expect(errors).toEqual([]);
});
test('area definitions round-trip as JSON and projects, apply with their name, and reject malformed imports without replacement',async({page})=>{
  await setup(page);await saveArea(page,'区画 A',fill);await saveArea(page,'East',cut);
  const definitions=await download(page,'#volume-areas-export');
  const project=await download(page,'#project-save');expect(JSON.parse(project.toString()).reviewAreas).toEqual(JSON.parse(definitions.toString()).areas);
  await page.reload();await page.locator('#file-input').setInputFiles([input('project.cloudanalyzer.json',project),...sourceFiles]);
  await expect(page.locator('#status')).toContainText('Project restored');
  await page.locator('#volume-area-options').evaluate((el:HTMLDetailsElement)=>{el.open=true;});
  await expect(page.locator('#volume-areas option')).toHaveCount(2);
  await page.locator('#volume-areas').selectOption('区画 A');await page.locator('#volume-area-apply').click();
  await expect(page.locator('#volume-area-name')).toHaveValue('区画 A');await expect(page.locator('#volume-area-min-x')).toHaveValue(String(fill[0]));
  await page.locator('#volume-options').evaluate((el:HTMLDetailsElement)=>{el.open=true;});await page.locator('#volume-cell').fill('1');
  await page.locator('#volume-run').click();await expect(page.locator('#status')).toContainText('Volume: fill 2');
  const single=JSON.parse((await download(page,'#volume-json')).toString());expect(single.reviewArea.name).toBe('区画 A');
  await page.locator('#volume-areas-file').setInputFiles(input('bad-areas.json',Buffer.from(JSON.stringify({schema:'cloudanalyzer.named_review_areas.v1',areas:[{name:'bad',bounds:null}]}))));
  await expect(page.locator('#status')).toContainText('four XY bounds');await expect(page.locator('#volume-areas option')).toHaveCount(2);
  await page.locator('#volume-areas-file').setInputFiles(input('areas.json',definitions));
  await expect(page.locator('#volume-areas-result')).toBeHidden();await expect(page.locator('#volume-result')).toBeHidden();
  await page.locator('#volume-areas').selectOption('East');await page.locator('#volume-area-remove').click();
  await expect(page.locator('#volume-areas option')).toHaveCount(1);
});
test('cancellation and changed definitions discard all staged rows without adding scene clouds',async({page})=>{
  // Keep background project captures from taking the worker between the two runs.
  await page.addInitScript(()=>localStorage.setItem('cloudanalyzer-autosave-enabled','off'));
  await setup(page);await saveArea(page,'Fill',fill);await saveArea(page,'Cut',cut);
  await page.evaluate(()=>{
    const original=Worker.prototype.postMessage;(window as any).areaAction='cancel';(window as any).areaCalls=0;
    Worker.prototype.postMessage=function(message:any,...rest:any[]){
      if(message?.req?.kind==='volume'&&message.req.summaryOnly){
        (window as any).areaCalls++;
        if((window as any).areaAction==='cancel') {if((window as any).areaCalls===2)document.getElementById('task-cancel')!.click();}
        else {const el=document.getElementById('volume-area-name') as HTMLInputElement;el.value='edited';el.dispatchEvent(new Event('input'));}
      }
      return (original as any).call(this,message,...rest);
    };
  });
  await page.locator('#volume-areas-run').click();await expect(page.locator('#status')).toContainText('no partial summaries');
  expect(await page.evaluate(()=>(window as any).areaCalls)).toBe(2);
  await expect(page.locator('#task')).toBeHidden();await expect(page.locator('#volume-areas-result')).toBeHidden();
  await expect(page.locator('#cloud-list > li')).toHaveCount(2);
  await page.evaluate(()=>(window as any).areaAction='change');await page.locator('#volume-areas-run').click();
  await expect(page.locator('#status')).toContainText('inputs/settings changed');await expect(page.locator('#volume-areas-result')).toBeHidden();
  await expect(page.locator('#cloud-list > li')).toHaveCount(2);
});
