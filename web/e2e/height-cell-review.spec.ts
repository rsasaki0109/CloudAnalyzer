import { test, expect, type Page } from '@playwright/test';
const ply=(name:string,zs:number[])=>({name,mimeType:'application/octet-stream',buffer:Buffer.from(`ply\nformat ascii 1.0\nelement vertex ${zs.length}\nproperty double x\nproperty double y\nproperty double z\nend_header\n`+zs.map((z,i)=>`${1000000+i*2} 2000000 ${z}`).join('\n')+'\n')});
async function output(page:Page,id:string){const pending=page.waitForEvent('download');await page.locator(id).click();const parts:Buffer[]=[];for await(const c of await(await pending).createReadStream())parts.push(c);return Buffer.concat(parts).toString();}
async function setup(page:Page,zs=[1+Number.EPSILON,-3,1]){
 await page.addInitScript(()=>localStorage.setItem('cloudanalyzer-autosave-enabled','off'));await page.goto('/');
 await page.locator('#file-input').setInputFiles([ply('before.ply',zs.map(()=>0)),ply('after.ply',zs)]);await expect(page.locator('#cloud-list > li')).toHaveCount(2);await expect(page.locator('#task')).toBeHidden();
 for(const id of ['volume-options','height-review-options','volume-area-options','volume-visits-options'])await page.locator(`#${id}`).evaluate((el:HTMLDetailsElement)=>el.open=true);
 await page.locator('#volume-cell').fill('2');await page.locator('#height-review-enabled').check();await page.locator('#height-review-threshold').fill('1');
}
async function inspect(page:Page){await page.locator('#volume-run').click();await expect(page.locator('#volume-result')).toBeVisible();await page.locator('#height-review-inspect').click();}
const state=(page:Page,n=1)=>page.getByRole('combobox',{name:`Review state for screened cell ${n}`,exact:true});
const note=(page:Page,n=1)=>page.getByRole('textbox',{name:`Review note for screened cell ${n}`,exact:true});
test('human cell notes survive close, focus and section edits with exact exports and escaped QA, leaving quantities unchanged',async({page},info)=>{
 const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));await setup(page);await inspect(page);const original=await output(page,'#volume-json');
 await state(page).selectOption('reviewed');await note(page).fill('目視確認\n<img src=x onerror=alert(1)>');await state(page,2).selectOption('deferred');await note(page,2).fill('  =再測量, "east"');
 await expect(page.locator('#height-cell-review-counts')).toContainText('0 unreviewed, 1 reviewed, 1 deferred');
 await page.getByRole('button',{name:'Focus screened cell 1',exact:true}).click();await page.locator('#height-cell-close').click();
 const closedQA=JSON.parse(await output(page,'#report-json'));expect(closedQA.sections['height-cell-reviews:1'].metrics.reviewed.value).toBe(1);
 await page.locator('#height-review-inspect').click();await expect(note(page)).toHaveValue('目視確認\n<img src=x onerror=alert(1)>');
 await page.getByRole('button',{name:'Compare point section for screened cell 1',exact:true}).click();await expect(page.locator('#regions-section-plot')).toBeVisible();
 await page.locator('#height-section-options').evaluate((el:HTMLDetailsElement)=>el.open=true);await page.locator('#height-section-width').fill('3');await expect(page.locator('#regions-section-plot')).toBeHidden();await expect(state(page)).toHaveValue('reviewed');
 const packet=JSON.parse(await output(page,'#height-cell-reviews-json'));expect(packet.schema).toBe('cloudanalyzer.height_cell_reviews.v1');expect(packet.comparisons).toHaveLength(1);
 const c=packet.comparisons[0];expect(c.before.name).toBe('before.ply');expect(c.after.name).toBe('after.ply');expect(c.settings).toMatchObject({cell:2,heightThreshold:1,fillEmpty:false});
 expect(c.cells[1]).toMatchObject({index:0,after:1+Number.EPSILON,status:'deferred',note:'  =再測量, "east"'});expect(c.cells[0].updatedAt).toMatch(/^\d{4}-\d{2}-\d{2}T/);expect(c.counts).toEqual({unreviewed:0,reviewed:1,deferred:1});
 const csv=await output(page,'#height-cell-reviews-csv');expect(csv).toContain('"\'  =再測量, ""east"""');expect(csv).toContain('1.0000000000000002');
 const html=await output(page,'#report-html');expect(html).toContain('&lt;img src=x onerror=alert(1)&gt;');expect(html).not.toContain('<img src=x');
 expect(await output(page,'#volume-json')).toBe(original);await page.locator('#height-cell-inspection').screenshot({path:info.outputPath('height-cell-review.png')});expect(errors).toEqual([]);
});
test('area and visit/interval records stay separate even with the same cell index and preserve pair, bounds and common support',async({page})=>{
 await setup(page,[5,1,100]);await page.locator('#file-input').setInputFiles(ply('visit-b.ply',[3,4]));await expect(page.locator('#cloud-list > li')).toHaveCount(3);await page.locator('#volume-cell').fill('2');
 await page.locator('#height-review-batch-enabled').check();await page.locator('#volume-areas-file').setInputFiles({name:'areas.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify({schema:'cloudanalyzer.named_review_areas.v1',areas:[{name:'All',bounds:[999999,1999999,1000005,2000001]},{name:'Same ground',bounds:[999999,1999999,1000005,2000001]}]}))});
 await page.locator('#volume-areas-run').click();await expect(page.locator('#volume-areas-result')).toBeVisible();
 await page.getByRole('button',{name:'Inspect screened cells: All',exact:true}).click();await state(page).selectOption('reviewed');await note(page).fill('Area A only');
 await page.getByRole('button',{name:'Inspect screened cells: Same ground',exact:true}).click();await expect(state(page)).toHaveValue('unreviewed');await note(page).fill('Area B');
 await page.getByRole('button',{name:'Inspect screened cells: All',exact:true}).click();await expect(note(page)).toHaveValue('Area A only');
 expect(JSON.parse(await output(page,'#height-cell-reviews-json')).comparisons).toHaveLength(2);
 const ids=await page.locator('#volume-visits option').evaluateAll((os:HTMLOptionElement[])=>os.map(o=>o.value));await page.locator('#volume-visits').selectOption(ids.slice(1));await page.locator('#volume-visits-support').selectOption('common_observed');await page.locator('#volume-visits-intervals').check();await page.locator('#volume-visits-run').click();await expect(page.locator('#volume-visits-result')).toBeVisible();
 await page.getByRole('button',{name:'Inspect screened cells: 1. after.ply → 2. visit-b.ply / All',exact:true}).click();await state(page).selectOption('deferred');await note(page).fill('Interval only');
 await page.getByRole('button',{name:'Inspect screened cells: 2. visit-b.ply / All',exact:true}).click();await expect(state(page)).toHaveValue('unreviewed');
 const comparisons=JSON.parse(await output(page,'#height-cell-reviews-json')).comparisons;expect(comparisons).toHaveLength(2); // previous analysis records were cleared
 expect(comparisons[0]).toMatchObject({before:{name:'after.ply'},after:{name:'visit-b.ply'},support:{mode:'common_observed'},settings:{region:[999999,1999999,1000005,2000001]}});
 expect(comparisons[0].support.sha256).toMatch(/^[0-9a-f]{64}$/);expect(comparisons[0].cells[0]).toMatchObject({index:1,difference:3,note:'Interval only',status:'deferred'});
 expect(comparisons[1].before.name).toBe('before.ply');expect(comparisons[1].cells[0].note).toBe('');
});
test('invalid edits preserve notes, reruns and settings reset decisions, and detached old controls cannot affect new results',async({page})=>{
 await setup(page);await inspect(page);await note(page).fill('Valid note');
 await note(page).evaluate((el:HTMLTextAreaElement)=>{el.value='x'.repeat(2001);el.dispatchEvent(new Event('input'));});await expect(page.locator('#status')).toContainText('2,000');await expect(note(page)).toHaveValue('Valid note');
 const old=await note(page).elementHandle();await inspect(page);await expect(note(page)).toHaveValue('');
 await old!.evaluate((el:HTMLTextAreaElement)=>{el.value='Old detached edit';el.dispatchEvent(new Event('input'));});
 expect(JSON.parse(await output(page,'#height-cell-reviews-json')).comparisons[0].cells[0].note).toBe('');
 await state(page).selectOption('reviewed');await page.locator('#height-review-threshold').fill('2');await expect(page.locator('#height-cell-inspection')).toBeHidden();
 expect(Object.keys(JSON.parse(await output(page,'#report-json')).sections).some(k=>k.startsWith('height-cell-reviews:'))).toBe(false);
 await inspect(page);await expect(state(page)).toHaveValue('unreviewed');await page.locator('#cloud-list > li').first().locator('button[title="Remove"]').click();await expect(page.locator('#height-cell-inspection')).toBeHidden();
 expect(Object.keys(JSON.parse(await output(page,'#report-json')).sections).some(k=>k.startsWith('height-cell-reviews:'))).toBe(false);
});
test('review counts and exports include only the bounded listed cells, with unlisted exceedances kept explicit',async({page})=>{
 await setup(page,Array.from({length:70},()=>2));await inspect(page);await expect(page.locator('#height-cell-list > li')).toHaveCount(50);
 const packet=JSON.parse(await output(page,'#height-cell-reviews-json')).comparisons[0];expect(packet.counts.unreviewed).toBe(50);expect(packet.screening).toMatchObject({riseCells:70,unlistedCells:20});expect(packet.cells).toHaveLength(50);
 expect((await output(page,'#height-cell-reviews-csv')).trimEnd().split('\n')).toHaveLength(51);
 expect(JSON.parse(await output(page,'#report-json')).sections['height-cell-reviews:1'].metrics.unlisted.value).toBe(20);
});
