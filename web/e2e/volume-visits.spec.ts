import { expect, test, type Page, type Download } from '@playwright/test';
const ply=(points:number[][])=>Buffer.from(`ply\nformat ascii 1.0\nelement vertex ${points.length}\nproperty float x\nproperty float y\nproperty float z\nend_header\n`+points.map(p=>p.join(' ')).join('\n')+'\n');
const file=(name:string,buffer:Buffer)=>({name,mimeType:'application/octet-stream',buffer});
const baseline=Array.from({length:6},(_,i)=>[i%3,Math.floor(i/3),10]);
async function bytes(d:Download){const parts:Buffer[]=[];for await(const p of await d.createReadStream())parts.push(p);return Buffer.concat(parts);}
async function download(page:Page,id:string){const d=page.waitForEvent('download');await page.locator(id).click();return bytes(await d);}
async function setup(page:Page){
  // This suite exercises volume requests; project persistence has its own suite.
  await page.addInitScript(()=>localStorage.setItem('cloudanalyzer-autosave-enabled','off'));
  await page.goto('/');await page.locator('#file-input').setInputFiles([
    file('baseline.ply',ply(baseline)),
    file('2026-01.ply',ply(baseline.slice(0,5).map((p,i)=>[p[0],p[1],p[2]+[2,-1,0,0,0][i]]))),
    file('2026-02.ply',ply([...baseline.slice(0,4).map(p=>[p[0],p[1],p[2]+4]),[-.6,0,50]]))]);
  await expect(page.locator('#cloud-list > li')).toHaveCount(3);await expect(page.locator('#task')).toBeHidden();
  for(const id of ['volume-area-options','volume-visits-options','volume-options'])await page.locator(`#${id}`).evaluate((el:HTMLDetailsElement)=>el.open=true);
  await page.locator('#volume-cell').fill('1');
  await page.locator('#volume-areas-file').setInputFiles(file('areas.json',Buffer.from(JSON.stringify({schema:'cloudanalyzer.named_review_areas.v1',areas:[
    {name:'Whole',bounds:[-2,-1,3,2]},{name:'Shifted cell',bounds:[.39,-.1,.41,.1]},{name:'Absent',bounds:[10,10,11,11]}]}))));
  const values=await page.locator('#volume-visits option').evaluateAll((options:HTMLOptionElement[])=>options.map(o=>o.value));
  await page.locator('#volume-visits').selectOption(values.slice(1));return values;
}
test('repeated visits share the union grid and retain independent support, provenance and empty areas',async({page},info)=>{
  const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));await setup(page);
  await page.locator('#volume-visits-run').click();await expect(page.locator('#status')).toContainText('Compared 2 visits across 3 areas');
  await expect(page.locator('#cloud-list > li')).toHaveCount(3);
  const json=JSON.parse((await download(page,'#volume-visits-json')).toString());
  expect(json.baseline.name).toBe('baseline.ply');expect(json.visits.map((v:any)=>v.name)).toEqual(['2026-01.ply','2026-02.ply']);
  expect(json.sharedGrid.nx).toBe(4);expect(json.sharedGrid.ny).toBe(2);expect(json.sharedGrid.min[0]).toBeCloseTo(-1.1,6);
  expect(json.rows.map((r:any)=>[r.visitIndex,r.name,r.added,r.removed,r.matchedCells,r.uncomparedCells])).toEqual([
    [1,'Whole',2,1,5,3],[1,'Shifted cell',2,0,1,0],[1,'Absent',0,0,0,0],
    [2,'Whole',16,0,4,4],[2,'Shifted cell',4,0,1,0],[2,'Absent',0,0,0,0]]);
  expect(json.rows[2].comparisonAvailable).toBe(false);expect(json).not.toHaveProperty('increment');
  await expect(page.locator('#volume-visits-rows tr').nth(2)).toContainText('Unavailable');
  expect((await download(page,'#volume-visits-csv')).toString()).toContain('1,"2026-01.ply","Absent",10,10,11,11,"m",false,,,,,,0,0,8');
  await page.locator('#volume-visits-result').screenshot({path:info.outputPath('visit-summaries.png')});
  await page.locator('#file-input').setInputFiles(file('unrelated.ply',ply([[100,100,0]])));
  await expect(page.locator('#cloud-list > li')).toHaveCount(4);await expect(page.locator('#volume-visits-result')).toBeVisible();
  await expect(page.locator('#volume-visits option:checked')).toHaveCount(2);
  await page.locator('#volume-fill').check();await expect(page.locator('#volume-visits-result')).toBeHidden();
  await page.locator('#volume-fill').uncheck();await page.locator('#volume-visits-run').click();await expect(page.locator('#volume-visits-result')).toBeVisible();
  await page.locator('#cloud-list > li').nth(1).locator('button[title="Remove"]').click();await expect(page.locator('#volume-visits-result')).toBeHidden();
  expect(errors).toEqual([]);
});
test('rejects baseline as visit and grid over budget, supports constant baseline',async({page})=>{
  const values=await setup(page);
  await page.locator('#volume-visits').selectOption(values);await page.locator('#volume-visits-run').click();await expect(page.locator('#status')).toContainText('baseline cannot also be a visit');
  await expect(page.locator('#volume-visits-result')).toBeHidden();
  await page.locator('#volume-visits').selectOption(values.slice(1));await page.locator('#volume-cell').fill('.0001');
  await page.locator('#volume-visits-run').click();await expect(page.locator('#status')).toContainText('1,000,000');
  await page.locator('#volume-before').selectOption('constant');await page.locator('#volume-before-z').fill('10');await page.locator('#volume-cell').fill('1');
  await page.locator('#volume-visits-run').click();await expect(page.locator('#status')).toContainText('Compared 2 visits');
  const json=JSON.parse((await download(page,'#volume-visits-json')).toString());expect(json.baseline.constantHeight).toBe(10);
  expect(json.rows[3].added).toBe(56); // Four raised cells plus one point outside the surveyed baseline.
  await page.locator('#file-input').setInputFiles(Array.from({length:6},(_,i)=>file(`extra-${i}.ply`,ply([[0,0,10]]))));
  await expect(page.locator('#cloud-list > li')).toHaveCount(9);await expect(page.locator('#task')).toBeHidden();
  const all=await page.locator('#volume-visits option').evaluateAll((options:HTMLOptionElement[])=>options.map(o=>o.value));
  await page.locator('#volume-visits').selectOption(all);await expect(page.locator('#volume-visits-result')).toBeHidden();
  await page.locator('#volume-visits-run').click();await expect(page.locator('#status')).toContainText('Choose 1–8');
});
test('cancellation and settings mutation discard every staged visit row',async({page})=>{
  await setup(page);await page.evaluate(()=>{
    const original=Worker.prototype.postMessage;(window as any).visitAction='cancel';(window as any).visitCalls=0;
    Worker.prototype.postMessage=function(message:any,...rest:any[]){
      if(message?.req?.kind==='volume'&&message.req.sharedGrid){
        (window as any).visitCalls++;
        if((window as any).visitAction==='cancel'){if((window as any).visitCalls===4)document.getElementById('task-cancel')!.click();}
        else {const el=document.getElementById('volume-cell')!;el.dispatchEvent(new Event('input'));}
      }return (original as any).call(this,message,...rest);
    };
  });
  await page.locator('#volume-visits-run').click();await expect(page.locator('#status')).toContainText('Cancelled; no partial summaries published');
  expect(await page.evaluate(()=>(window as any).visitCalls)).toBe(4);await expect(page.locator('#volume-visits-result')).toBeHidden();await expect(page.locator('#task')).toBeHidden();
  await page.evaluate(()=>(window as any).visitAction='change');await page.locator('#volume-visits-run').click();
  await expect(page.locator('#status')).toContainText('inputs/settings changed');await expect(page.locator('#volume-visits-result')).toBeHidden();
  await expect(page.locator('#cloud-list > li')).toHaveCount(3);
});
