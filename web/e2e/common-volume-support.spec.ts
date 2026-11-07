import { expect, test, type Page } from '@playwright/test';
import { createHash } from 'node:crypto';
const file=(name:string,buffer:Buffer)=>({name,mimeType:'application/octet-stream',buffer});
const ply=(points:number[][])=>Buffer.from(`ply\nformat ascii 1.0\nelement vertex ${points.length}\nproperty float x\nproperty float y\nproperty float z\nend_header\n`+points.map(p=>p.join(' ')).join('\n')+'\n');
async function save(page:Page,id:string){const waiting=page.waitForEvent('download');await page.locator(id).click();const parts:Buffer[]=[];for await(const p of await(await waiting).createReadStream())parts.push(p);return Buffer.concat(parts);}
async function setup(page:Page,empty=false){
  await page.addInitScript(()=>localStorage.setItem('cloudanalyzer-autosave-enabled','off'));
  await page.goto('/');await page.locator('#file-input').setInputFiles([
    file('baseline.ply',ply([[0,0,10],[1,0,10],[2,0,10]])),
    file('visit-a.ply',ply(empty?[[0,0,12]]:[[0,0,12],[1,0,9]])),
    file('visit-b.ply',ply(empty?[[2,0,13]]:[[1,0,14],[2,0,13]]))]);
  await expect(page.locator('#cloud-list > li')).toHaveCount(3);await expect(page.locator('#task')).toBeHidden();
  for(const id of ['volume-area-options','volume-visits-options','volume-options'])await page.locator(`#${id}`).evaluate((el:HTMLDetailsElement)=>el.open=true);
  await page.locator('#volume-cell').fill('1');
  await page.locator('#volume-areas-file').setInputFiles(file('areas.json',Buffer.from(JSON.stringify({schema:'cloudanalyzer.named_review_areas.v1',areas:[{name:'All',bounds:[-1,-1,3,1]},{name:'Pair-only cell',bounds:[-.1,-.1,.1,.1]},{name:'Outside',bounds:[10,10,11,11]}]}))));
  const ids=await page.locator('#volume-visits option').evaluateAll((options:HTMLOptionElement[])=>options.map(o=>o.value));await page.locator('#volume-visits').selectOption(ids.slice(1));return ids;
}
async function run(page:Page){await page.locator('#volume-visits-run').click();await expect(page.locator('#status')).toContainText('Compared 2 visits across 3 areas');}
test('common support excludes cells present in only one visit, exports exact mask identity and retains independent mode',async({page},info)=>{
  const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));await setup(page);await run(page);
  const independent=JSON.parse((await save(page,'#volume-visits-json')).toString());expect(independent.support.mode).toBe('independent');
  expect(independent.rows.filter((r:any)=>r.name==='All').map((r:any)=>[r.added,r.removed,r.matchedCells])).toEqual([[2,1,2],[7,0,2]]);
  await page.locator('#volume-visits-support').selectOption('common_observed');await expect(page.locator('#volume-visits-result')).toBeHidden();await run(page);
  const common=JSON.parse((await save(page,'#volume-visits-json')).toString());
  expect(common.support.mode).toBe('common_observed');expect(common.support.commonGridCells).toBe(1);expect(common.support.commonGridArea).toBe(1);
  expect(common.rows.filter((r:any)=>r.name==='All').map((r:any)=>[r.added,r.removed,r.net,r.matchedCells,r.uncomparedCells])).toEqual([[0,1,-1,1,2],[4,0,4,1,2]]);
  expect(common.rows[1].comparisonAvailable).toBe(false);expect(common.rows[1].totalCells).toBe(1);expect(common.rows[1].outsideRegionCells).toBe(2);
  const mask=await save(page,'#volume-visits-mask');expect([...mask]).toEqual([0,1,0]);expect(common.support.mask.sha256).toBe(createHash('sha256').update(mask).digest('hex'));
  expect(common.sharedGrid).toMatchObject({min:[-.5,-.5],nx:3,ny:1,cell:1});expect(common.support.mask.byteLength).toBe(3);
  const csv=(await save(page,'#volume-visits-csv')).toString();expect(csv).toContain('support_mode,common_mask_sha256');expect(csv).toContain(`,common_observed,${common.support.mask.sha256}`);
  await expect(page.locator('#volume-visits-rows tr').nth(1)).toContainText('Unavailable');await expect(page.locator('#cloud-list > li')).toHaveCount(3);
  await page.locator('#volume-visits-result').screenshot({path:info.outputPath('common-support.png')});
  await page.locator('#volume-visits-support').selectOption('independent');await expect(page.locator('#volume-visits-result')).toBeHidden();await run(page);
  expect(JSON.parse((await save(page,'#volume-visits-json')).toString()).rows[0].matchedCells).toBe(2);await expect(page.locator('#volume-visits-mask')).toBeDisabled();expect(errors).toEqual([]);
});
test('empty intersection is unavailable with a zero mask, while interpolation, constant and mesh inputs fail preflight',async({page})=>{
  const ids=await setup(page,true);await page.locator('#volume-visits-support').selectOption('common_observed');await run(page);
  const json=JSON.parse((await save(page,'#volume-visits-json')).toString());expect(json.support.commonGridCells).toBe(0);expect(json.rows.every((r:any)=>!r.comparisonAvailable)).toBe(true);
  expect([...(await save(page,'#volume-visits-mask'))]).toEqual([0,0,0]);await expect(page.locator('#volume-visits-rows tr').first()).toContainText('Unavailable');
  await page.locator('#volume-fill').check();await page.locator('#volume-visits-run').click();await expect(page.locator('#status')).toContainText('interpolation disabled');
  await expect(page.locator('#volume-visits-result')).toBeHidden();await page.locator('#volume-fill').uncheck();await page.locator('#volume-before').selectOption('constant');
  await page.locator('#volume-visits-run').click();await expect(page.locator('#status')).toContainText('require point clouds');
  await page.locator('#volume-before').selectOption(ids[0]);await page.locator('#volume-cell').fill('.000001');await page.locator('#volume-visits-run').click();await expect(page.locator('#status')).toContainText('1,000,000');
  await page.locator('#file-input').setInputFiles(file('surface.obj',Buffer.from('v 0 0 10\nv 1 0 10\nv 0 1 10\nf 1 2 3\n')));
  await expect(page.locator('#cloud-list > li')).toHaveCount(4);await expect(page.locator('#task')).toBeHidden();
  await page.locator('#volume-visits').selectOption({label:'surface.obj (mesh)'});await page.locator('#volume-cell').fill('1');await page.locator('#volume-visits-run').click();await expect(page.locator('#status')).toContainText('require point clouds');
});
test('cancelling support preparation or later rows and changing settings discards all staged masks and results',async({page})=>{
  await setup(page);await page.locator('#volume-visits-support').selectOption('common_observed');
  await page.evaluate(()=>{const original=Worker.prototype.postMessage;(window as any).commonAction='cancel-prepare';(window as any).commonRows=0;
    Worker.prototype.postMessage=function(message:any,...rest:any[]){
      if(message?.req?.kind==='volume-support'){
        if((window as any).commonAction==='cancel-prepare')document.getElementById('task-cancel')!.click();
        if((window as any).commonAction==='change-prepare')document.getElementById('volume-cell')!.dispatchEvent(new Event('input'));
      }
      if(message?.req?.kind==='volume'&&message.req.supportMask){(window as any).commonRows++;if((window as any).commonAction==='cancel-row'&&(window as any).commonRows===2)document.getElementById('task-cancel')!.click();}
      return(original as any).call(this,message,...rest);
    };
  });
  await page.locator('#volume-visits-run').click();await expect(page.locator('#status')).toContainText('Cancelled; no partial summaries');await expect(page.locator('#volume-visits-result')).toBeHidden();
  expect(await page.evaluate(()=>(window as any).commonRows)).toBe(0);
  await page.evaluate(()=>(window as any).commonAction='change-prepare');await page.locator('#volume-visits-run').click();await expect(page.locator('#status')).toContainText('inputs/settings changed');
  await page.evaluate(()=>(window as any).commonAction='cancel-row');await page.locator('#volume-visits-run').click();await expect(page.locator('#status')).toContainText('Cancelled; no partial summaries');
  await expect(page.locator('#volume-visits-result')).toBeHidden();await expect(page.locator('#cloud-list > li')).toHaveCount(3);
});
