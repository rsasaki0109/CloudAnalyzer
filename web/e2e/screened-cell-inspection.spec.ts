import { test, expect, type Page } from '@playwright/test';
const ply=(name:string,zs:number[],origin=1000000)=>({name,mimeType:'application/octet-stream',buffer:Buffer.from(`ply\nformat ascii 1.0\nelement vertex ${zs.length}\nproperty double x\nproperty double y\nproperty double z\nend_header\n`+zs.map((z,x)=>`${origin+x*2} 2000000 ${z}`).join('\n')+'\n')});
async function output(page:Page,id:string){const pending=page.waitForEvent('download');await page.locator(id).click();const chunks:Buffer[]=[];for await(const c of await(await pending).createReadStream())chunks.push(c);return Buffer.concat(chunks).toString();}
async function setup(page:Page,zs:number[]){
 await page.addInitScript(()=>localStorage.setItem('cloudanalyzer-autosave-enabled','off'));await page.goto('/');await page.locator('#file-input').setInputFiles([ply('before.ply',zs.map(()=>0)),ply('after.ply',zs)]);await expect(page.locator('#cloud-list > li')).toHaveCount(2);await expect(page.locator('#task')).toBeHidden();
 for(const id of ['volume-options','height-review-options','volume-area-options','volume-visits-options'])await page.locator(`#${id}`).evaluate((el:HTMLDetailsElement)=>el.open=true);
 await page.locator('#volume-cell').fill('2');await page.locator('#height-review-enabled').check();await page.locator('#height-review-threshold').fill('1');
}
async function run(page:Page){await page.locator('#volume-run').click();await expect(page.locator('#volume-result')).toBeVisible();}
test('original f64 screened cells focus world-coordinate footprints and export without recomputing or changing totals',async({page},info)=>{
 await setup(page,[1+Number.EPSILON,-3,1]);await run(page);const original=await output(page,'#volume-json'),h=JSON.parse(original).result.heightReview;
 expect(h.cells.map((c:any)=>c.index)).toEqual([1,0]);expect(h.cells[1]).toMatchObject({x:1000000,y:2000000,before:0,after:1+Number.EPSILON,difference:1+Number.EPSILON});expect(h.unlistedCells).toBe(0);
 await page.evaluate(()=>{(window as any).cellWorkerCalls=0;const original=Worker.prototype.postMessage;Worker.prototype.postMessage=function(...args:any[]){(window as any).cellWorkerCalls++;return(original as any).apply(this,args);};});
 await page.locator('#height-review-inspect').click();await expect(page.locator('#height-cell-list > li')).toHaveCount(2);await expect(page.locator('#height-cell-context')).toContainText('2 listed of 2');await expect(page.locator('#height-cell-list')).toContainText('1.0000000000000002');
 const focus=page.getByRole('button',{name:'Focus screened cell 1',exact:true});await focus.click();await expect(page.locator('#status')).toContainText('Focused screened grid cell 1');expect(await output(page,'#volume-json')).toBe(original);expect(await page.evaluate(()=>(window as any).cellWorkerCalls)).toBe(0);
 const csv=await output(page,'#height-cell-csv');expect(csv).toContain('1000000,2000000,0,1.0000000000000002,1.0000000000000002,rise,0');
 await page.locator('#height-cell-inspection').screenshot({path:info.outputPath('cell-inspection.png')});
 await page.locator('#height-cell-close').click();await expect(page.locator('#height-cell-inspection')).toBeHidden();await expect(page.locator('#height-review-result')).toBeVisible();await page.locator('#height-review-inspect').click();await expect(page.locator('#height-cell-list > li')).toHaveCount(2);
 const old=await focus.elementHandle();await page.locator('#height-review-threshold').fill('2');await expect(page.locator('#height-cell-inspection')).toBeHidden();await expect(page.locator('#height-cell-list > li')).toHaveCount(0);await old!.evaluate((el:HTMLElement)=>el.click());await expect(page.locator('#height-cell-inspection')).toBeHidden();
});
test('the list is bounded with stable ties, complete totals include unlisted cells and equality produces no listed cells',async({page})=>{
 await setup(page,Array.from({length:70},(_,i)=>i%2?-2:2));await run(page);const h=JSON.parse(await output(page,'#volume-json')).result.heightReview;
 expect([h.riseCells,h.fallCells,h.cells.length,h.cellLimit,h.unlistedCells]).toEqual([35,35,50,50,20]);expect(h.cells.map((c:any)=>c.index)).toEqual(Array.from({length:50},(_,i)=>i));
 await page.locator('#height-review-inspect').click();await expect(page.locator('#height-cell-list > li')).toHaveCount(50);await expect(page.locator('#height-cell-context')).toContainText('20 unlisted');expect((await output(page,'#height-cell-csv')).trimEnd().split('\n')).toHaveLength(51);
 await page.locator('#height-review-threshold').fill('2');await run(page);await expect(page.locator('#height-review-inspect')).toBeDisabled();expect(JSON.parse(await output(page,'#volume-json')).result.heightReview.cells).toEqual([]);
});
test('named-area and direct interval rows inspect only their own bounds and common support and clear on input changes',async({page})=>{
 await setup(page,[5,1,100]);await page.locator('#file-input').setInputFiles(ply('visit-b.ply',[3,4]));await expect(page.locator('#cloud-list > li')).toHaveCount(3);await page.locator('#volume-cell').fill('2');
 await page.locator('#height-review-enabled').uncheck();await page.locator('#height-review-batch-enabled').check();await page.locator('#volume-areas-file').setInputFiles({name:'areas.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify({schema:'cloudanalyzer.named_review_areas.v1',areas:[{name:'All',bounds:[999999,1999999,1000005,2000001]},{name:'Excluded',bounds:[1000003.9,1999999,1000004.1,2000001]}]}))});
 const ids=await page.locator('#volume-visits option').evaluateAll((os:HTMLOptionElement[])=>os.map(o=>o.value));await page.locator('#volume-visits').selectOption(ids.slice(1));await page.locator('#volume-visits-support').selectOption('common_observed');await page.locator('#volume-visits-intervals').check();await page.locator('#volume-visits-run').click();await expect(page.locator('#volume-visits-result')).toBeVisible();
 const json=JSON.parse(await output(page,'#volume-visits-json'));expect(json.intervals.rows[0].heightReview.cells.map((c:any)=>[c.index,c.difference])).toEqual([[1,3],[0,-2]]);expect(json.intervals.rows[1].heightReview.cells).toEqual([]);
 await page.getByRole('button',{name:'Inspect screened cells: 1. after.ply → 2. visit-b.ply / All',exact:true}).click();await expect(page.locator('#height-cell-list > li')).toHaveCount(2);await expect(page.locator('#height-cell-context')).toContainText('after.ply → 2. visit-b.ply');
 await expect(page.getByRole('button',{name:'Inspect screened cells: 1. after.ply → 2. visit-b.ply / Excluded',exact:true})).toBeDisabled();
 await page.locator('#cloud-list > li').nth(2).locator('button[title="Remove"]').click();await expect(page.locator('#height-cell-inspection')).toBeHidden();await expect(page.locator('#volume-visits-result')).toBeHidden();
});
