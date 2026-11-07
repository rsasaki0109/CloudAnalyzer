import { expect, test, type Page } from '@playwright/test';
import { createHash } from 'node:crypto';
const file=(name:string,buffer:Buffer)=>({name,mimeType:'application/octet-stream',buffer});
const ply=(zs:number[])=>Buffer.from(`ply\nformat ascii 1.0\nelement vertex ${zs.length}\nproperty float x\nproperty float y\nproperty float z\nend_header\n`+zs.map((z,x)=>`${x} 0 ${z}`).join('\n')+'\n');
async function save(page:Page,id:string){const waiting=page.waitForEvent('download');await page.locator(id).click();const parts:Buffer[]=[];for await(const p of await(await waiting).createReadStream())parts.push(p);return Buffer.concat(parts);}
async function setup(page:Page){
  await page.addInitScript(()=>localStorage.setItem('cloudanalyzer-autosave-enabled','off'));
  await page.goto('/');await page.locator('#file-input').setInputFiles([
    file('baseline.ply',ply([0,0,0])),file('visit-a.ply',ply([5,1,100])),file('visit-b.ply',ply([3,4])),file('visit-c.ply',ply([4,2]))]);
  await expect(page.locator('#cloud-list > li')).toHaveCount(4);await expect(page.locator('#task')).toBeHidden();
  for(const id of ['volume-area-options','volume-visits-options','volume-options'])await page.locator(`#${id}`).evaluate((el:HTMLDetailsElement)=>el.open=true);
  await page.locator('#volume-cell').fill('1');await page.locator('#volume-areas-file').setInputFiles(file('areas.json',Buffer.from(JSON.stringify({schema:'cloudanalyzer.named_review_areas.v1',areas:[
    {name:'All',bounds:[-1,-1,3,1]},{name:'Excluded',bounds:[1.9,-.1,2.1,.1]},{name:'Outside',bounds:[10,10,11,11]}]}))));
  const ids=await page.locator('#volume-visits option').evaluateAll((options:HTMLOptionElement[])=>options.map(o=>o.value));await page.locator('#volume-visits').selectOption(ids.slice(1));
  await page.locator('#volume-visits-support').selectOption('common_observed');await page.locator('#volume-visits-intervals').check();return ids;
}
async function run(page:Page){await page.locator('#volume-visits-run').click();await expect(page.locator('#task')).toBeHidden();await expect(page.locator('#volume-visits-result')).toBeVisible();}
test('direct interval cut/fill differs from baseline gross subtraction and follows explicit reversible visit order',async({page},info)=>{
  const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));const ids=await setup(page);await run(page);
  let json=JSON.parse((await save(page,'#volume-visits-json')).toString());expect(json.ordering.cloudIds).toEqual(ids.slice(1).map(Number));
  expect(json.rows.filter((r:any)=>r.name==='All').map((r:any)=>[r.added,r.removed,r.matchedCells])).toEqual([[6,0,2],[7,0,2],[6,0,2]]);
  const intervals=json.intervals.rows.filter((r:any)=>r.name==='All');expect(intervals.map((r:any)=>[r.fromVisit,r.toVisit,r.added,r.removed,r.net,r.matchedCells,r.uncomparedCells])).toEqual([
    ['visit-a.ply','visit-b.ply',3,2,1,2,1],['visit-b.ply','visit-c.ply',1,2,-1,2,1]]);
  expect(intervals[0].added).not.toBe(json.rows[3].added-json.rows[0].added);
  expect(json.intervals.rows[1].comparisonAvailable).toBe(false);await expect(page.locator('#volume-intervals-rows tr').nth(1)).toContainText('Unavailable');
  const mask=await save(page,'#volume-visits-mask');expect([...mask]).toEqual([1,1,0]);const hash=createHash('sha256').update(mask).digest('hex');expect(hash).toBe(json.support.mask.sha256);
  const csv=(await save(page,'#volume-intervals-csv')).toString();expect(csv).toContain('from_cloud_id,from_visit_name');expect(csv).toContain(`,common_observed,${hash}`);
  await page.locator('#volume-intervals-result').screenshot({path:info.outputPath('direct-intervals.png')});
  await page.getByRole('button',{name:'Move visit 1 later',exact:true}).click();await expect(page.locator('#volume-visits-result')).toBeHidden();
  await page.getByRole('button',{name:'Move visit 2 later',exact:true}).click();await page.getByRole('button',{name:'Move visit 1 later',exact:true}).click();
  await run(page);json=JSON.parse((await save(page,'#volume-visits-json')).toString());
  expect(json.visits.map((v:any)=>v.name)).toEqual(['visit-c.ply','visit-b.ply','visit-a.ply']);expect(json.support.mask.sha256).toBe(hash);
  expect(json.intervals.rows.filter((r:any)=>r.name==='All').map((r:any)=>[r.added,r.removed,r.net])).toEqual([[2,1,1],[2,3,-1]]);
  await page.locator('#file-input').setInputFiles(file('unrelated.ply',ply([99])));await expect(page.locator('#cloud-list > li')).toHaveCount(5);
  await expect(page.locator('#volume-visit-order > li').first()).toContainText('visit-c.ply');await expect(page.locator('#volume-intervals-result')).toBeVisible();
  await page.locator('#cloud-list > li').nth(2).locator('button[title="Remove"]').click();await expect(page.locator('#volume-visits-result')).toBeHidden();await expect(page.locator('#volume-intervals-result')).toBeHidden();expect(errors).toEqual([]);
});
test('intervals are opt-in, require common support and at least two visits, and preserve baseline-only behavior',async({page})=>{
  const ids=await setup(page);await page.locator('#volume-visits-intervals').uncheck();await run(page);
  expect(JSON.parse((await save(page,'#volume-visits-json')).toString()).intervals).toBeUndefined();await expect(page.locator('#volume-intervals-result')).toBeHidden();
  await page.locator('#volume-visits-intervals').check();await page.locator('#volume-visits-support').selectOption('independent');await expect(page.locator('#volume-visits-intervals')).not.toBeChecked();await expect(page.locator('#volume-visits-intervals')).toBeDisabled();
  await page.locator('#volume-visits-support').selectOption('common_observed');await page.locator('#volume-visits').selectOption(ids[1]);await expect(page.locator('#volume-visits-intervals')).toBeDisabled();
  await page.locator('#volume-visits-intervals').evaluate((el:HTMLInputElement)=>el.checked=true);await page.locator('#volume-visits-run').click();await expect(page.locator('#status')).toContainText('at least two visits');
});
test('cancellation or settings edits during direct intervals discard baseline rows, intervals and mask together',async({page})=>{
  await setup(page);await page.evaluate(()=>{const original=Worker.prototype.postMessage;(window as any).intervalAction='cancel';(window as any).intervalCalls=0;
    Worker.prototype.postMessage=function(message:any,...rest:any[]){
      if(message?.req?.kind==='volume'&&message.req.supportMask){(window as any).intervalCalls++;
        if((window as any).intervalCalls===10){
          if((window as any).intervalAction==='cancel')document.getElementById('task-cancel')!.click();
          else document.getElementById('volume-visits-intervals')!.dispatchEvent(new Event('change'));
        }
      }return(original as any).call(this,message,...rest);
    };
  });
  await page.locator('#volume-visits-run').click();await expect(page.locator('#status')).toContainText('Cancelled; no partial summaries');expect(await page.evaluate(()=>(window as any).intervalCalls)).toBe(10);
  await expect(page.locator('#volume-visits-result')).toBeHidden();await expect(page.locator('#volume-intervals-result')).toBeHidden();
  await page.evaluate(()=>{(window as any).intervalCalls=0;(window as any).intervalAction='change';});await page.locator('#volume-visits-run').click();await expect(page.locator('#status')).toContainText('inputs/settings changed');
  await expect(page.locator('#volume-visits-result')).toBeHidden();await expect(page.locator('#cloud-list > li')).toHaveCount(4);
});
