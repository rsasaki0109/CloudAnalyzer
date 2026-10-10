import { expect, test, type Page } from '@playwright/test';
import { readProjectSnapshot } from '../src/project-snapshot';
import { cloud, map } from '../tests/mapping-review-fixture.mjs';
import { writeFile } from 'node:fs/promises';

const input=(name:string,buffer:Buffer)=>({name,mimeType:'application/octet-stream',buffer});
async function copy(page:Page) {
  return page.evaluate(()=>new Promise<any>((resolve,reject)=>{
    const open=indexedDB.open('cloudanalyzer-recovery',1);
    open.onsuccess=()=>{const db=open.result,tx=db.transaction('projects','readonly'),request=tx.objectStore('projects').get('latest');request.onsuccess=()=>resolve({token:request.result.token,bytes:request.result.snapshot?.size});tx.oncomplete=()=>db.close();};
    open.onerror=()=>reject(open.error);
  }));
}
async function saved(page:Page) { await expect(page.locator('#project-save-status')).toContainText('Current point/mesh records are included'); }
async function prepare(page:Page) {
  await page.goto('/');await page.locator('#project-autosave-records').check();
  await page.locator('#file-input').setInputFiles(input('survey.ply',cloud));
  await expect(page.locator('#status')).toContainText('Loaded survey.ply');
  await page.locator('#filter-op').selectOption('random');await page.locator('#filter-percent').fill('50');await page.locator('#filter-run').click();
  await expect(page.locator('#cloud-list > li')).toHaveCount(2);
  await page.locator('#vm-file').setInputFiles(input('draft.json',Buffer.from(JSON.stringify(map))));
  await saved(page);
}
const warns=(page:Page)=>page.evaluate(()=>{const event=new Event('beforeunload',{cancelable:true});window.dispatchEvent(event);return event.defaultPrevented;});

test('processed results recover without source files and the saved browser copy downloads as an exact workspace',async({page,context})=>{
  await prepare(page);
  await page.locator('#file-input').setInputFiles(input('plane.obj',Buffer.from('v -1 -1 0\nv 20 -1 0\nv 20 20 0\nv -1 20 0\nf 1 2 3 4\n')));
  await expect(page.locator('#status')).toContainText('Loaded plane.obj');await saved(page);expect(await warns(page)).toBe(false);
  const before=await copy(page);expect(before.bytes).toBeGreaterThan(0);
  await page.close();const resumed=await context.newPage();await resumed.goto('/');
  await expect(resumed.locator('#project-recovery')).toBeVisible();
  await expect(resumed.locator('#project-autosave-records')).toBeChecked();
  const event=resumed.waitForEvent('download',d=>d.suggestedFilename()==='project.cloudanalyzer.zip');await resumed.locator('#project-download-recovery').click();
  const parts:Buffer[]=[];for await(const part of await (await event).createReadStream())parts.push(part);
  const files=await readProjectSnapshot(new File([Buffer.concat(parts)],'project.cloudanalyzer.zip'),new AbortController().signal);
  expect(files).toHaveLength(4);
  await resumed.locator('#project-resume').click();await expect(resumed.locator('#status')).toContainText('Project restored');
  await expect(resumed.locator('#cloud-list > li')).toHaveCount(3);
  await expect(resumed.locator('#cloud-list > li').nth(1).locator('.meta')).toContainText('2 points');
  await expect(resumed.locator('#cloud-list > li').last().locator('.meta')).toContainText('2 triangles');
  await expect(resumed.locator('#cloud-list > li').first().locator('input[type=checkbox]')).not.toBeChecked();
  await expect(resumed.locator('#cloud-list > li').last().locator('input[type=checkbox]')).toBeChecked();
  await expect(resumed.locator('#vm-status')).toContainText('1 lane');
  await resumed.locator('#c2c-compared').selectOption({label:'survey.ply'});
  await resumed.locator('#c2c-reference').selectOption({label:'plane.obj (mesh)'});await resumed.locator('#c2m-signed').check();
  await resumed.locator('#c2c-run').click();await expect(resumed.locator('#status')).toContainText('C2M distance computed for 4 points');
  await expect(resumed.locator('#c2c-stats tr',{hasText:'Mean'}).locator('td')).toHaveText('2');
  await saved(resumed);expect(await warns(resumed)).toBe(false);
});

test('a failed point-record commit keeps the previous ZIP and closing warning until retry',async({page})=>{
  await prepare(page);const before=await copy(page);
  await page.evaluate(()=>{const put=IDBObjectStore.prototype.put;(window as any).restorePut=()=>IDBObjectStore.prototype.put=put;IDBObjectStore.prototype.put=function(...args){if(this.name==='projects')throw new DOMException('point storage full','QuotaExceededError');return put.apply(this,args as any);};});
  await page.locator('#filter-percent').fill('25');await page.locator('#filter-run').click();
  await expect(page.locator('#cloud-list > li')).toHaveCount(3);
  await expect(page.locator('#project-save-status')).toContainText('point storage full');
  expect(await copy(page)).toEqual(before);expect(await warns(page)).toBe(true);
  await page.evaluate(()=>(window as any).restorePut());await page.locator('#project-autosave-retry').click();await saved(page);
  expect((await copy(page)).token).not.toBe(before.token);expect(await warns(page)).toBe(false);
});

test('changed browser point records are rejected while the current workspace stays intact',async({page})=>{
  await prepare(page);await page.reload();await expect(page.locator('#project-recovery')).toBeVisible();
  await page.locator('#file-input').setInputFiles(input('other.ply',cloud));await expect(page.locator('#status')).toContainText('Loaded other.ply');
  await page.evaluate(()=>new Promise<void>((resolve,reject)=>{
    const open=indexedDB.open('cloudanalyzer-recovery',1);open.onerror=()=>reject(open.error);
    open.onsuccess=()=>{const db=open.result,tx=db.transaction('projects','readwrite'),store=tx.objectStore('projects'),read=store.get('latest');
      read.onsuccess=()=>{const value=read.result;value.snapshot=new Blob(['corrupt ZIP']);store.put(value,'latest');};tx.oncomplete=()=>{db.close();resolve();};tx.onabort=()=>reject(tx.error);};
  }));
  await page.locator('#project-resume').click();await expect(page.locator('#status')).toContainText('Could not resume browser copy');
  await expect(page.locator('#cloud-list > li')).toHaveCount(1);await expect(page.locator('#cloud-list')).toContainText('other.ply');
  await expect(page.locator('#project-recovery')).toBeVisible();
});

test('NCLT full map and processed points resume from browser storage with exact native attributes',async({page,context})=>{
  const bundle=process.env.CA_REVIEW_NCLT_BUNDLE;
  test.skip(!bundle,'Set CA_REVIEW_NCLT_BUNDLE to the full generated-map review ZIP');test.setTimeout(120000);
  await page.goto('/');await page.locator('#project-autosave-records').check();
  await page.locator('#mapping-review-file').setInputFiles(bundle!);await expect(page.locator('#status')).toContainText('Opened generated maps');
  await page.locator('#filter-op').selectOption('random');await page.locator('#filter-percent').fill('1');await page.locator('#filter-run').click();
  await expect(page.locator('#cloud-list > li')).toHaveCount(2);await saved(page);
  await page.close();const resumed=await context.newPage();await resumed.goto('/');
  const event=resumed.waitForEvent('download');await resumed.locator('#project-download-recovery').click();
  const parts:Buffer[]=[];for await(const part of await (await event).createReadStream())parts.push(part);
  const zip=Buffer.concat(parts);await writeFile('/tmp/nclt-browser-recovery.zip',zip);
  const files=await readProjectSnapshot(new File([zip],'project.cloudanalyzer.zip'),new AbortController().signal);
  expect(Buffer.from(await files[1].arrayBuffer()).includes(Buffer.from('property float correction\n'))).toBe(true);
  await resumed.locator('#project-resume').click();
  await expect(resumed.locator('#status')).toContainText('Project restored');
  await expect(resumed.locator('#cloud-list > li').first().locator('.meta')).toContainText('646,309 points');
  await expect(resumed.locator('#cloud-list > li').last().locator('.meta')).toContainText('6,463 points');
  for(const index of [0,1]) {
    const li=resumed.locator('#cloud-list > li').nth(index),download=resumed.waitForEvent('download');
    await li.locator('button.icon').click();await li.locator('.save-formats').getByRole('button',{name:'PLY',exact:true}).click();
    const data:Buffer[]=[];for await(const part of await (await download).createReadStream())data.push(part);
    expect(Buffer.concat(data).equals(Buffer.from(await files[index+1].arrayBuffer()))).toBe(true);
  }
  await resumed.screenshot({path:'/tmp/nclt-browser-recovery-restored.png'});
});
