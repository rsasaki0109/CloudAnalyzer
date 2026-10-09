import { expect, test, type Page, type Download } from '@playwright/test';
import { readProjectSnapshot } from '../src/project-snapshot';
import { indexReviewZip, readReviewMember, writeReviewZip } from '../src/review-zip';
import { cloud, map } from '../tests/mapping-review-fixture.mjs';

async function bytes(download:Download) {
  const parts:Buffer[]=[];for await(const part of await download.createReadStream()) parts.push(part);
  return Buffer.concat(parts);
}
async function save(page:Page) {
  const event=page.waitForEvent('download',d=>d.suggestedFilename()==='project.cloudanalyzer.zip');
  await page.locator('#project-snapshot').click();
  return bytes(await event);
}
async function exported(page:Page,index:number) {
  const li=page.locator('#cloud-list > li').nth(index), event=page.waitForEvent('download');
  await li.locator('button.icon').click();
  await li.locator('.save-formats').getByRole('button',{name:'PLY',exact:true}).click();
  return bytes(await event);
}
const input=(name:string,buffer:Buffer)=>({name,mimeType:'application/octet-stream',buffer});
const read=(buffer:Buffer)=>readProjectSnapshot(new File([buffer],'project.cloudanalyzer.zip'),new AbortController().signal);

test('snapshot restores random-filter results, attributes, transformed coordinates, visibility and lane notes',async({page})=>{
  await page.goto('/');
  const header=Buffer.from('ply\nformat binary_little_endian 1.0\nelement vertex 8\nproperty double x\nproperty double y\nproperty double z\nproperty float intensity\nproperty float correction\nproperty uchar source\nend_header\n');
  const data=Buffer.alloc(8*33);
  for(let i=0;i<8;i++) {data.writeDoubleLE(500000.000123+i/100,33*i);data.writeDoubleLE(4000000.000456,33*i+8);data.writeDoubleLE(2.123456789,33*i+16);data.writeFloatLE(i/7,33*i+24);data.writeFloatLE(-0.1234567+i/100,33*i+28);data.writeUInt8(i+10,33*i+32);}
  await page.locator('#file-input').setInputFiles(input('survey.ply',Buffer.concat([header,data])));
  await expect(page.locator('#status')).toContainText('Loaded survey.ply');
  await page.locator('#filter-op').selectOption('random');await page.locator('#filter-percent').fill('50');await page.locator('#filter-run').click();
  await expect(page.locator('#cloud-list > li')).toHaveCount(2);
  await expect(page.locator('#cloud-list > li').last().locator('.meta')).toContainText('4 points');
  await page.locator('#align-panel summary').click();await page.locator('#align-matrix').fill('1 0 0 5\n0 1 0 0\n0 0 1 0\n0 0 0 1');await page.locator('#align-apply-matrix').click();
  await expect(page.locator('#status')).toContainText('Applied the matrix');
  await page.locator('#vm-file').setInputFiles(input('draft.json',Buffer.from(JSON.stringify(map))));
  await page.locator('#vm-review-next').click();await page.locator('#vm-review-notes').fill('Keep this lane draft');await page.locator('#vm-review-save').click();
  const zip=await save(page), files=await read(zip), project=JSON.parse(await files[0].text());
  expect(project.session.clouds.map((c:any)=>c.transforms)).toEqual([[],[]]);
  expect(project.session.clouds.map((c:any)=>c.visible)).toEqual([false,true]);
  const sourceBytes=Buffer.from(await files[1].arrayBuffer()), resultBytes=Buffer.from(await files[2].arrayBuffer());
  expect(resultBytes.toString('ascii',0,120)).toContain('element vertex 4');
  const offset=resultBytes.indexOf('end_header\n')+11;
  for(let i=0;i<4;i++) {
    expect(resultBytes.readDoubleLE(offset+i*33)).toBeGreaterThan(500005);
    expect(resultBytes.readDoubleLE(offset+i*33+8)).toBe(4000000.000456);
    expect(resultBytes.readDoubleLE(offset+i*33+16)).toBe(2.123456789);
    const originalIndex=Math.round((resultBytes.readDoubleLE(offset+i*33)-500005.000123)*100);
    expect(resultBytes.readFloatLE(offset+i*33+28)).toBe(Math.fround(-0.1234567+originalIndex/100));
    expect(resultBytes.readUInt8(offset+i*33+32)).toBe(originalIndex+10);
  }
  await page.reload();await page.locator('#file-input').setInputFiles(input('project.cloudanalyzer.zip',zip));
  await expect(page.locator('#status')).toContainText('Project restored');
  await expect(page.locator('#cloud-list > li')).toHaveCount(2);
  await expect(page.locator('#cloud-list > li').first().locator('input[type=checkbox]')).not.toBeChecked();
  await expect(page.locator('#cloud-list > li').last().locator('input[type=checkbox]')).toBeChecked();
  await expect(page.locator('#undo')).toBeDisabled();
  await page.locator('#vm-review-next').click();await expect(page.locator('#vm-review-notes')).toHaveValue('Keep this lane draft');
  expect(await exported(page,0)).toEqual(sourceBytes);expect(await exported(page,1)).toEqual(resultBytes);
  await page.locator('#filter-cloud').selectOption({label:project.session.clouds[1].name});
  await page.locator('#filter-op').selectOption('random');await page.locator('#filter-percent').fill('50');await page.locator('#filter-run').click();
  await expect(page.locator('#cloud-list > li')).toHaveCount(3);
  await expect(page.locator('#cloud-list > li').last().locator('.name')).toHaveAttribute('title',`${project.session.clouds[1].name}_random2`);
});

test('mesh snapshots retain triangle topology and signed distance behavior',async({page})=>{
  await page.goto('/');
  await page.locator('#file-input').setInputFiles([input('plane.obj',Buffer.from('v -1 -1 0\nv 20 -1 0\nv 20 20 0\nv -1 20 0\nf 1 2 3 4\n')),input('above.ply',cloud)]);
  await expect(page.locator('#status')).toContainText('Loaded above.ply');
  const zip=await save(page);
  await page.reload();await page.locator('#file-input').setInputFiles(input('project.cloudanalyzer.zip',zip));
  await expect(page.locator('#status')).toContainText('Project restored');
  await expect(page.locator('#cloud-list > li').first().locator('.meta')).toContainText('2 triangles');
  await page.locator('#c2m-signed').check();
  await page.locator('#c2c-run').click();await expect(page.locator('#status')).toContainText('C2M distance computed for 4 points');
  await expect(page.locator('#c2c-stats tr',{hasText:'Mean'}).locator('td')).toHaveText('2');
});

test('a metadata project does not clear the unload warning for unexported processed records',async({page})=>{
  await page.goto('/');await page.locator('#file-input').setInputFiles(input('source.ply',cloud));
  await expect(page.locator('#status')).toContainText('Loaded source.ply');
  await page.locator('#filter-op').selectOption('random');await page.locator('#filter-percent').fill('50');await page.locator('#filter-run').click();
  await expect(page.locator('#cloud-list > li')).toHaveCount(2);
  await expect(page.locator('#project-save-status')).toContainText('processed clouds/meshes are outside');
  const download=page.waitForEvent('download',d=>d.suggestedFilename()==='project.cloudanalyzer.json');
  await page.locator('#project-save').click();await download;
  const dialog=page.waitForEvent('dialog');const reloaded=page.reload();
  const prompt=await dialog;expect(prompt.type()).toBe('beforeunload');await prompt.accept();await reloaded;
});

test('changed snapshot evidence preserves existing map and clouds',async({page})=>{
  await page.goto('/');await page.locator('#file-input').setInputFiles(input('original.ply',cloud));
  await expect(page.locator('#status')).toContainText('Loaded original.ply');
  await page.locator('#vm-file').setInputFiles(input('draft.json',Buffer.from(JSON.stringify(map))));
  const zip=await save(page), file=new File([zip],'project.cloudanalyzer.zip'), signal=new AbortController().signal;
  const directory=await indexReviewZip(file,signal), entries:[string,Blob][]=[];
  for(const [name,member] of directory) entries.push([name,new Blob([await readReviewMember(file,member,signal)])]);
  const member=entries.find(([name])=>name.startsWith('clouds/'))!;
  const changed=new Uint8Array(await member[1].arrayBuffer());changed[changed.length-1]^=1;member[1]=new Blob([changed]);
  const bad=Buffer.from(await (await writeReviewZip(entries,signal)).arrayBuffer());
  await page.locator('#file-input').setInputFiles(input('project.cloudanalyzer.zip',bad));
  await expect(page.locator('#status')).toContainText('hash differs');
  await expect(page.locator('#cloud-list > li')).toHaveCount(1);await expect(page.locator('#cloud-list')).toContainText('original.ply');
  await expect(page.locator('#vm-status')).toContainText('1 lane');
});
