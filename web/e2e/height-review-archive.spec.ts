import { test, expect, type Page } from '@playwright/test';
const file=(name:string,buffer:Buffer)=>({name,mimeType:'application/octet-stream',buffer});
const ply=(zs:number[])=>Buffer.from(`ply\nformat ascii 1.0\nelement vertex ${zs.length}\nproperty double x\nproperty double y\nproperty double z\nend_header\n`+zs.map((z,i)=>`${1000000+i*2} 2000000 ${z}`).join('\n')+'\n');
async function bytes(page:Page,id:string){const wait=page.waitForEvent('download');await page.locator(id).click();const chunks:Buffer[]=[];for await(const c of await(await wait).createReadStream())chunks.push(c);return Buffer.concat(chunks);}
async function setup(page:Page){
 await page.addInitScript(()=>localStorage.setItem('cloudanalyzer-autosave-enabled','off'));await page.goto('/');
 await page.locator('#file-input').setInputFiles([file('before.ply',ply([0,0,0])),file('after.ply',ply([1+Number.EPSILON,-3,1]))]);await expect(page.locator('#cloud-list > li')).toHaveCount(2);await expect(page.locator('#task')).toBeHidden();
 for(const id of ['volume-options','height-review-options','volume-area-options','volume-visits-options'])await page.locator(`#${id}`).evaluate((el:HTMLDetailsElement)=>el.open=true);
 await page.locator('#volume-cell').fill('2');await page.locator('#height-review-enabled').check();await page.locator('#height-review-threshold').fill('1');
}
async function prepare(page:Page){
 await setup(page);await page.locator('#volume-run').click();await expect(page.locator('#volume-result')).toBeVisible();await page.locator('#height-review-inspect').click();
 await page.getByRole('combobox',{name:'Review state for screened cell 1',exact:true}).selectOption('deferred');
 await page.getByRole('textbox',{name:'Review note for screened cell 1',exact:true}).fill('  =再確認\n<img src=x onerror=alert(1)>');
 return bytes(page,'#height-cell-reviews-json');
}
async function open(page:Page,data:Buffer,name='past.json'){
 await page.locator('#height-archive-options').evaluate((el:HTMLDetailsElement)=>el.open=true);
 await page.locator('#height-archive-file').setInputFiles(file(name,data));await expect(page.locator('#status')).toContainText('opened read-only');
}
test('actual review JSON opens without clouds and exports original bytes, exact CSV and standalone escaped offline HTML',async({page},info)=>{
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));const original=await prepare(page),csv=await bytes(page,'#height-cell-reviews-csv');
 await page.reload();await expect(page.locator('#cloud-list > li')).toHaveCount(0);await open(page,original);
 await expect(page.locator('#height-archive-list > details')).toHaveCount(1);await expect(page.locator('#height-archive-list li')).toHaveCount(0); // lazy rendering
 await page.locator('#height-archive-list > details > summary').click();await expect(page.locator('#height-archive-list li')).toHaveCount(2);
 await expect(page.locator('#height-archive-list')).toContainText('1.0000000000000002');await expect(page.locator('#height-archive-list')).toContainText('preview');
 await expect(page.locator('.height-archive-note').first()).toHaveText('  =再確認\n<img src=x onerror=alert(1)>');await expect(page.locator('#height-archive-list img')).toHaveCount(0);
 expect(await bytes(page,'#height-archive-json')).toEqual(original);expect(await bytes(page,'#height-archive-csv')).toEqual(csv);
 const html=(await bytes(page,'#height-archive-html')).toString();expect(html).toContain('&lt;img src=x onerror=alert(1)&gt;');expect(html).not.toContain('<img src=x');
 const context=await page.context().browser()!.newContext({offline:true});const offline=await context.newPage();await offline.setContent(html);
 await expect(offline.locator('h1')).toHaveText('Archived height-cell reviews');await expect(offline.locator('script,img')).toHaveCount(0);
 const snapshot=JSON.parse(await offline.locator('pre').textContent()||'{}');expect(snapshot.comparisons[0].cells[1].difference).toBe(1+Number.EPSILON);await context.close();
 expect(Object.keys(JSON.parse((await bytes(page,'#report-json')).toString()).sections).some(k=>k.startsWith('height-cell-reviews:'))).toBe(false);
 await page.locator('#height-archive-list > details').first().screenshot({path:info.outputPath('height-review-archive.png')});
 await page.locator('#height-archive-close').click();await expect(page.locator('#height-archive-result')).toBeHidden();await expect(page.locator('#cloud-list > li')).toHaveCount(0);expect(errors).toEqual([]);
});
test('invalid and oversized archives retain the previous file and never alter live decisions, volume or QA',async({page})=>{
 const original=await prepare(page),volume=await bytes(page,'#volume-json'),qa=JSON.parse((await bytes(page,'#report-json')).toString()).sections;await open(page,original);
 expect(await bytes(page,'#height-cell-reviews-json')).toEqual(original);expect(await bytes(page,'#volume-json')).toEqual(volume);expect(JSON.parse((await bytes(page,'#report-json')).toString()).sections).toEqual(qa);
 const bad=JSON.parse(original.toString());bad.comparisons[0].counts.deferred=99;await page.locator('#height-archive-file').setInputFiles(file('bad.json',Buffer.from(JSON.stringify(bad))));await expect(page.locator('#status')).toContainText('inconsistent');
 await expect(page.locator('#height-archive-name')).toHaveText('past.json');expect(await bytes(page,'#height-archive-json')).toEqual(original);
 await page.locator('#height-archive-file').evaluate((el:HTMLInputElement)=>{const transfer=new DataTransfer();transfer.items.add(new File([new Uint8Array(64*1024*1024+1)],'oversized.json'));el.files=transfer.files;el.dispatchEvent(new Event('change'));});await expect(page.locator('#status')).toContainText('64 MiB');
 await page.locator('#height-archive-file').setInputFiles(file('syntax.json',Buffer.from('{')));await expect(page.locator('#status')).toContainText('Could not open');expect(await bytes(page,'#height-archive-json')).toEqual(original);
 await page.locator('#height-review-threshold').fill('2');await expect(page.locator('#height-cell-inspection')).toBeHidden();await expect(page.locator('#height-archive-result')).toBeVisible();
 expect(Object.keys(JSON.parse((await bytes(page,'#report-json')).toString()).sections).some(k=>k.startsWith('height-cell-reviews:'))).toBe(false);
});
test('close and superseding reads prevent stale archives from reappearing',async({page})=>{
 const original=await prepare(page);await open(page,original);
 await page.evaluate(()=>{const original=File.prototype.text;File.prototype.text=function(){const file=this;if(file.name!=='delayed.json')return original.call(file);return new Promise(resolve=>{(window as any).releaseHeightArchive=async()=>resolve(await original.call(file));});};});
 await page.locator('#height-archive-file').setInputFiles(file('delayed.json',original));await page.waitForFunction(()=>!!(window as any).releaseHeightArchive);
 await page.locator('#height-archive-close').click();await page.evaluate(()=>(window as any).releaseHeightArchive());await expect(page.locator('#height-archive-result')).toBeHidden();
 await open(page,original,'current.json');await page.locator('#height-archive-file').setInputFiles(file('delayed.json',original));await open(page,original,'newest.json');await page.evaluate(()=>(window as any).releaseHeightArchive());await expect(page.locator('#height-archive-name')).toHaveText('newest.json');
});
test('actual common-observed interval archives retain support hash and adjacent pair without loading points into a fresh viewer',async({page})=>{
 await setup(page);await page.locator('#file-input').setInputFiles(file('visit-b.ply',ply([4,-1])));await expect(page.locator('#cloud-list > li')).toHaveCount(3);await page.locator('#volume-cell').fill('2');
 await page.locator('#height-review-batch-enabled').check();await page.locator('#volume-areas-file').setInputFiles(file('areas.json',Buffer.from(JSON.stringify({schema:'cloudanalyzer.named_review_areas.v1',areas:[{name:'All',bounds:[999999,1999999,1000005,2000001]}]}))));
 const ids=await page.locator('#volume-visits option').evaluateAll((os:HTMLOptionElement[])=>os.map(o=>o.value));await page.locator('#volume-visits').selectOption(ids.slice(1));await page.locator('#volume-visits-support').selectOption('common_observed');await page.locator('#volume-visits-intervals').check();await page.locator('#volume-visits-run').click();await expect(page.locator('#volume-visits-result')).toBeVisible();
 await page.getByRole('button',{name:'Inspect screened cells: 1. after.ply → 2. visit-b.ply / All',exact:true}).click();const original=await bytes(page,'#height-cell-reviews-json');
 const expected=JSON.parse(original.toString()).comparisons[0];expect(expected.before.name).toBe('after.ply');expect(expected.support.commonGridCells).toBe(2);
 await page.reload();await open(page,original);await page.locator('#height-archive-list > details > summary').click();await expect(page.locator('#height-archive-list')).toContainText(expected.support.sha256);await expect(page.locator('#height-archive-list')).toContainText('visit-b.ply');
 expect(await bytes(page,'#height-archive-json')).toEqual(original);await expect(page.locator('#cloud-list > li')).toHaveCount(0);
});
