import { expect, test, type Page } from '@playwright/test';

const input=(name:string,buffer:Buffer)=>({name,mimeType:'application/octet-stream',buffer});
const cloud=(height=2)=>Buffer.from(`ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 ${height}\n10 0 ${height}\n0 10 ${height}\n10 10 ${height}\n`);
const map=(count=1)=>({format:'vectormap-ir',version:1,
  lanes:Array.from({length:count},(_,i)=>({id:i+7,kind:'driving',left:i*2+1,right:i*2+2})),
  boundaries:Array.from({length:count*2},(_,i)=>({id:i+1,kind:{type:'virtual'},geometry:[[0,i%2?-1.75:1.75,2],[10,i%2?-1.75:1.75,2]]})),
});
async function openMap(page:Page,count=1) {
  await page.locator('#vm-file').setInputFiles(input('map.json',Buffer.from(JSON.stringify(map(count)))));
  await expect(page.locator('#vm-status')).toContainText(`${count} lane`);
}
async function saved(page:Page) { await expect(page.locator('#project-save-status')).toContainText('Editing state saved in this browser'); }
async function recovery(page:Page) {
  return page.evaluate(()=>new Promise<any>((resolve,reject)=>{
    const opening=indexedDB.open('cloudanalyzer-recovery',1);
    opening.onerror=()=>reject(opening.error);
    opening.onsuccess=()=>{
      const db=opening.result,transaction=db.transaction('projects','readonly'),request=transaction.objectStore('projects').get('latest');
      request.onsuccess=()=>resolve(request.result);request.onerror=()=>reject(request.error);
      transaction.oncomplete=()=>db.close();
    };
  }));
}
const unloadWarns=(page:Page)=>page.evaluate(()=>{
  const event=new Event('beforeunload',{cancelable:true});window.dispatchEvent(event);return event.defaultPrevented;
});
async function exportProject(page:Page) {
  const downloaded=page.waitForEvent('download',d=>d.suggestedFilename()==='project.cloudanalyzer.json');
  await page.locator('#project-save').click();
  const pieces:Buffer[]=[];
  for await(const piece of await (await downloaded).createReadStream())pieces.push(piece);
  return JSON.parse(Buffer.concat(pieces).toString());
}

test('browser recovery preserves verified sources, graph constraints, saved reviews and an unsaved review draft',async({page,context})=>{
  const source=input('survey.ply',cloud()), poses=input('poses.kitti',Buffer.from('1 0 0 0 0 1 0 0 0 0 1 0\n1 0 0 1 0 1 0 0 0 0 1 0\n'));
  const scans=[input('000000.ply',cloud()),input('000001.ply',cloud())];
  await page.goto('/');
  await expect(page.locator('#project-save-status')).toContainText('Automatic saving ready');
  await page.locator('#file-input').setInputFiles(source);
  await expect(page.locator('#status')).toContainText('Loaded survey.ply');
  await openMap(page);
  await page.locator('#pg-files-input').setInputFiles([poses,...scans]);
  await expect(page.locator('#pg-stats')).toContainText('2 (2 with scans)');
  await page.locator('#pg-a').fill('1');await page.locator('#pg-fix').click();
  await expect(page.locator('#pg-fix')).toHaveText('Free A');
  await page.locator('#vm-review-next').click();
  await page.locator('#vm-review-state').selectOption('reviewed');
  await page.locator('#vm-review-notes').fill('保存した縁石');await page.locator('#vm-review-save').click();
  await page.locator('#vm-review-state').selectOption('deferred');
  await page.locator('#vm-review-notes').fill('まだ記録していない下書き\n再測量');
  await saved(page);
  const before=await recovery(page);
  expect(before.draft.notes).toContain('下書き');
  expect(before.project.reviews[0].notes).toBe('保存した縁石');
  expect(JSON.parse(before.project.poseGraph.snapshot).graph.nodes[1].fixed).toBe(true);
  expect(before.project.session.clouds[0].source.digest).toMatch(/^[0-9a-f]{64}$/);
  expect(JSON.stringify(before)).not.toContain('pointData');
  expect(await unloadWarns(page)).toBe(false);
  await page.close();const resumed=await context.newPage();await resumed.goto('/');
  await expect(resumed.locator('#project-recovery')).toBeVisible();
  await expect(resumed.locator('#cloud-list li')).toHaveCount(0);
  await resumed.locator('#project-resume').click();
  await expect(resumed.locator('#status')).toContainText('open matching source files');
  await resumed.locator('#file-input').setInputFiles(input('survey.ply',cloud(3)));
  await expect(resumed.locator('#status')).toContainText('open matching source files');
  await expect(resumed.locator('#cloud-list li')).toHaveCount(0);
  expect((await recovery(resumed)).token).toBe(before.token);
  await resumed.locator('#file-input').setInputFiles([source,poses,...scans]);
  await expect(resumed.locator('#status')).toContainText('Project restored');
  await expect(resumed.locator('#pg-fix')).toHaveText('Free A');
  await expect(resumed.locator('#vm-review-state')).toHaveValue('deferred');
  await expect(resumed.locator('#vm-review-notes')).toHaveValue(/下書き/);
  await resumed.locator('#vm-review-filter').selectOption('all');
  await expect(resumed.locator('#vm-review-rows')).toContainText('保存した縁石');
  await resumed.locator('#vm-review-save').click();await saved(resumed);
  const after=await recovery(resumed);
  expect(after.project.reviews[0].notes).toContain('下書き');expect(after.draft).toBeUndefined();
});

test('button-only changes to saved views, annotations and gates create fresh recovery copies',async({page})=>{
  await page.goto('/');
  // Local files let saving verify contents without relying on a remote ETag.
  const points=Buffer.from('ply\nformat ascii 1.0\nelement vertex 3600\nproperty float x\nproperty float y\nproperty float z\nend_header\n'+Array.from({length:3600},(_,i)=>`${i%60} ${Math.floor(i/60)} 2`).join('\n'));
  await page.locator('#file-input').setInputFiles([input('reference.ply',points),input('copy.ply',points)]);
  await expect(page.locator('#status')).toContainText('Loaded copy.ply');
  await page.locator('#c2c-run').click();
  await expect(page.locator('#status')).toContainText('C2C distance computed');
  await saved(page);
  await page.locator('#view-save').click();await saved(page);
  expect((await recovery(page)).project.session.views).toHaveLength(1);
  await page.locator('#gate-add').click();await saved(page);
  expect((await recovery(page)).project.session.gates).toHaveLength(1);
  await page.locator('#label').click();
  const canvas=page.locator('#viewport > canvas'),box=(await canvas.boundingBox())!;
  for(const [x,y] of [[.5,.5],[.45,.5],[.5,.45],[.55,.52]]) {
    await canvas.click({position:{x:box.width*x,y:box.height*y}});
    if(await page.locator('#note-list li').count())break;
  }
  await expect(page.locator('#note-list input')).toHaveValue(/^Z -?\d/);
  await saved(page);expect((await recovery(page)).project.session.labels).toHaveLength(1);
  await page.locator('#note-clear').click();
  await page.locator('#view-list button.remove').click();
  await page.locator('#gate-list button.remove').click();
  await saved(page);
  const after=(await recovery(page)).project.session;
  expect(after.labels).toHaveLength(0);expect(after.views).toHaveLength(0);expect(after.gates).toHaveLength(0);
});

test('previous browser copy stays protected while opening new work until explicitly discarded',async({page})=>{
  await page.goto('/');await openMap(page);await saved(page);
  const before=await recovery(page);await page.reload();
  await expect(page.locator('#project-recovery')).toBeVisible();
  await openMap(page,3);
  await expect(page.locator('#project-save-status')).toContainText('previous browser copy is protected');
  expect((await recovery(page)).token).toBe(before.token);
  await page.locator('#project-forget').click();await saved(page);
  const after=await recovery(page);expect(after.token).not.toBe(before.token);
  expect(JSON.parse(after.project.vectorMap).lanes).toHaveLength(3);
});

test('quota failures preserve the previous copy and unload warning; a portable export and retry still work',async({page})=>{
  const errors:string[]=[];page.on('pageerror',error=>errors.push(error.message));
  await page.goto('/');await openMap(page);await saved(page);const before=await recovery(page);
  await page.evaluate(()=>{
    const original=IDBObjectStore.prototype.put;(window as any).restorePut=()=>{IDBObjectStore.prototype.put=original;};
    IDBObjectStore.prototype.put=function(...args:Parameters<IDBObjectStore['put']>){if(this.name==='projects')throw new DOMException('storage full','QuotaExceededError');return original.apply(this,args);};
  });
  await page.locator('#vm-review-next').click();
  await page.locator('#vm-lane-speed').fill('30');await page.locator('#vm-lane-apply').click();
  await expect(page.locator('#project-save-status')).toContainText('storage full');
  expect((await recovery(page)).token).toBe(before.token);expect(await unloadWarns(page)).toBe(true);
  const exported=await exportProject(page);
  expect(JSON.parse(exported.vectorMap).lanes[0].speed_limit.kmh).toBe(30);
  expect(await unloadWarns(page)).toBe(false);
  await page.evaluate(()=>(window as any).restorePut());
  await page.locator('#project-autosave-retry').click();await saved(page);
  expect((await recovery(page)).token).not.toBe(before.token);expect(errors).toEqual([]);
});

test('a stale tab cannot overwrite or silently acknowledge another tab\'s browser copy',async({page,context})=>{
  await page.goto('/');await expect(page.locator('#project-save-status')).toContainText('Automatic saving ready');
  const other=await context.newPage();await other.goto('/');
  await expect(other.locator('#project-save-status')).toContainText('Automatic saving ready');
  await openMap(page);await saved(page);const before=await recovery(page);
  await openMap(other,3);
  await expect(other.locator('#project-save-status')).toContainText('Another tab changed');
  expect((await recovery(other)).token).toBe(before.token);expect(await unloadWarns(other)).toBe(true);
});

test('unavailable browser storage keeps manual saving usable and can be retried after access returns',async({page})=>{
  await page.addInitScript(()=>{
    const original=indexedDB.open.bind(indexedDB);(window as any).restoreOpen=()=>{indexedDB.open=original;};
    indexedDB.open=()=>{throw new DOMException('browser storage denied','SecurityError');};
  });
  await page.goto('/');await expect(page.locator('#project-save-status')).toContainText('browser storage denied');
  await openMap(page);
  expect(JSON.parse((await exportProject(page)).vectorMap).lanes).toHaveLength(1);
  await page.evaluate(()=>(window as any).restoreOpen());
  await page.locator('#project-autosave-retry').click();await saved(page);
  await page.locator('#project-autosave').uncheck();
  await page.reload();await expect(page.locator('#project-autosave')).not.toBeChecked();
});
