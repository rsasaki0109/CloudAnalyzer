import { expect, test, type Page } from '@playwright/test';
const file=(name:string,buffer:Buffer)=>({name,mimeType:'application/octet-stream',buffer});
const ply=(z:number)=>Buffer.from(`ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 ${z}\n10 0 ${z}\n`);
async function bytes(page:Page,id:string){const waiting=page.waitForEvent('download');await page.locator(id).click();const parts:Buffer[]=[];for await(const c of await(await waiting).createReadStream())parts.push(c);return Buffer.concat(parts);}
async function prepare(page:Page){
  await page.addInitScript(()=>localStorage.setItem('cloudanalyzer-autosave-enabled','off'));
  await page.goto('/');await page.locator('#file-input').setInputFiles([file('truth.ply',ply(0)),file('map.ply',ply(1))]);
  await expect(page.locator('#cloud-list > li')).toHaveCount(2);await expect(page.locator('#task')).toBeHidden();
  await page.locator('#coverage-source').selectOption({label:'map.ply'});await page.locator('#coverage-reference').selectOption({label:'truth.ply'});
  await page.getByText('Locate distance discrepancies',{exact:true}).click();await page.locator('#regions-min').fill('1');await page.locator('#regions-cell').fill('2');
  await page.locator('#regions-both').click();await expect(page.locator('#status')).toContainText('ready in both directions');
  await page.locator('#regions-list > li').first().locator('select').selectOption('deferred');
  await page.locator('#regions-list > li').first().locator('textarea').fill('=Confirm\n<img src=x onerror=alert(1)>');
  return bytes(page,'#regions-review-json');
}
async function openPanel(page:Page){await page.locator('#review-archive-panel').evaluate((el:HTMLDetailsElement)=>el.open=true);}
async function open(page:Page,buffer:Buffer,name='past.json'){
  await openPanel(page);await page.locator('#review-archive-file').setInputFiles(file(name,buffer));await expect(page.locator('#status')).toContainText('opened read-only');
}
test('actual saved reviews reopen without clouds, export exact original bytes and keep historical records outside live QA',async({page},info)=>{
  const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));const original=await prepare(page);
  await page.reload();await expect(page.locator('#cloud-list > li')).toHaveCount(0);
  await open(page,original);await expect(page.locator('#review-archive-list > details')).toHaveCount(2);
  await expect(page.locator('#review-archive-list')).toContainText('deferred');await expect(page.locator('#review-archive-list')).toContainText('preview false');
  const direction=page.locator('#review-archive-list > details').first();
  await direction.getByText('Inputs and analysis conditions',{exact:true}).click();await expect(direction.locator('.hint')).toBeVisible();
  await direction.getByText('Inputs and analysis conditions',{exact:true}).click();
  await direction.locator('li summary').first().click();await expect(direction.locator('li').first()).toContainText('world XYZ bounds');
  await expect(page.locator('#review-archive-list .archive-comment').first()).toHaveText('=Confirm\n<img src=x onerror=alert(1)>');
  await expect(page.locator('#review-archive-list img')).toHaveCount(0);await expect(page.locator('#regions-result')).toBeHidden();
  expect(await bytes(page,'#review-archive-json')).toEqual(original);
  const csv=(await bytes(page,'#review-archive-csv')).toString();expect(csv).toContain(`"'=Confirm\n<img src=x onerror=alert(1)>"`);
  const html=(await bytes(page,'#review-archive-html')).toString();expect(html).toContain('&lt;img src=x onerror=alert(1)&gt;');expect(html).not.toContain('<img src=x');expect(html).toContain('Historical file only');
  const qa=JSON.parse((await bytes(page,'#report-json')).toString());expect(qa.sections['distance-candidate-reviews:forward']).toBeUndefined();
  await page.locator('#review-archive-panel').screenshot({path:info.outputPath('archived-reviews.png')});
  await page.locator('#review-archive-close').click();await expect(page.locator('#review-archive-result')).toBeHidden();await expect(page.locator('#cloud-list > li')).toHaveCount(0);expect(errors).toEqual([]);
});
test('archive import and close do not alter live decisions; invalid files leave the previous archive intact',async({page})=>{
  const original=await prepare(page);const before=await bytes(page,'#regions-review-json');await open(page,original);
  expect(await bytes(page,'#regions-review-json')).toEqual(before);
  const bad=JSON.parse(original.toString());bad.comparisons[0].review.counts.deferred=99;
  await page.locator('#review-archive-file').setInputFiles(file('bad.json',Buffer.from(JSON.stringify(bad))));await expect(page.locator('#status')).toContainText('inconsistent');
  await expect(page.locator('#review-archive-name')).toHaveText('past.json');expect(await bytes(page,'#review-archive-json')).toEqual(original);
  await page.locator('#review-archive-file').setInputFiles(file('too-large.json',Buffer.alloc(2*1024*1024+1,32)));await expect(page.locator('#status')).toContainText('2 MiB');
  await page.locator('#review-archive-file').setInputFiles(file('syntax.json',Buffer.from('{')));await expect(page.locator('#status')).toContainText('Could not open review archive');
  await page.locator('#regions-threshold').fill('.6');await expect(page.locator('#regions-result')).toBeHidden();await expect(page.locator('#review-archive-result')).toBeVisible();
  expect(await bytes(page,'#review-archive-json')).toEqual(original);await page.locator('#review-archive-close').click();await expect(page.locator('#review-archive-result')).toBeHidden();
});
test('closing or superseding an asynchronous file read prevents stale archives from reappearing',async({page})=>{
  const original=await prepare(page);await open(page,original);
  await page.evaluate(()=>{const originalText=File.prototype.text;File.prototype.text=function(){const file=this;if(file.name!=='delayed.json')return originalText.call(file);return new Promise(resolve=>{(window as any).releaseArchive=async()=>resolve(await originalText.call(file));});};});
  await page.locator('#review-archive-file').setInputFiles(file('delayed.json',original));await page.waitForFunction(()=>!!(window as any).releaseArchive);
  await page.locator('#review-archive-close').click();await page.evaluate(()=>(window as any).releaseArchive());await expect(page.locator('#review-archive-result')).toBeHidden();
  await open(page,original,'newest.json');await expect(page.locator('#review-archive-name')).toHaveText('newest.json');
  await page.locator('#review-archive-file').setInputFiles(file('delayed.json',original));
  await open(page,original,'superseding.json');await page.evaluate(()=>(window as any).releaseArchive());
  await expect(page.locator('#review-archive-name')).toHaveText('superseding.json');
});
