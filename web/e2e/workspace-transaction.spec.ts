import {expect,test,type Page,type Download} from '@playwright/test';
import {writeProjectSnapshot} from '../src/project-snapshot';
import {referenceFile} from '../src/source-reference';
import {cloud,map} from '../tests/mapping-review-fixture.mjs';
const input=(name:string,buffer:Buffer)=>({name,mimeType:'application/octet-stream',buffer});
async function bytes(d:Download){const parts:Buffer[]=[];for await(const part of await d.createReadStream())parts.push(part);return Buffer.concat(parts);}
async function project(page:Page){const d=page.waitForEvent('download',d=>d.suggestedFilename()==='project.cloudanalyzer.json');await page.locator('#project-save').click();return JSON.parse((await bytes(await d)).toString());}
async function exported(page:Page){const row=page.locator('#cloud-list > li').first();await row.locator('button[title="Save as…"]').click();const d=page.waitForEvent('download');await row.locator('.save-formats button',{hasText:'PLY'}).click();return bytes(await d);}
async function original(page:Page){
 await page.goto('/');await page.locator('#file-input').setInputFiles(input('original.ply',cloud));await expect(page.locator('#status')).toContainText('Loaded original.ply');
 await page.locator('#vm-file').setInputFiles(input('original-map.json',Buffer.from(JSON.stringify(map))));await expect(page.locator('#vm-status')).toContainText('1 lane');
 await page.locator('#filter-op').selectOption('random');await page.locator('#filter-percent').fill('50');await page.locator('#filter-run').click();await expect(page.locator('#cloud-list > li')).toHaveCount(2);await expect(page.locator('#undo')).toBeEnabled();
}
async function packed(template:any,files:File[],assets:File[]=[]){
 const p=structuredClone(template),base=p.session.clouds[0];p.session.clouds=[];
 for(const f of files)p.session.clouds.push({...base,name:`incoming-${f.name}`,source:await referenceFile(f),loadMaxPoints:0,transforms:[],distance:undefined});
 return Buffer.from(await(await writeProjectSnapshot(p,files,new AbortController().signal,assets)).arrayBuffer());
}
async function unchanged(page:Page,before:any,records:Buffer){
 await expect(page.locator('#cloud-list > li')).toHaveCount(2);await expect(page.locator('#cloud-list')).toContainText('original.ply');await expect(page.locator('#undo')).toBeEnabled();
 expect(await exported(page)).toEqual(records);const after=await project(page);expect(after.vectorMap).toEqual(before.vectorMap);expect(after.poseGraph).toEqual(before.poseGraph);for(const key of ['position','target'])for(let i=0;i<3;i++)expect(after.session.camera[key][i]).toBeCloseTo(before.session.camera[key][i],8);
}

test('a self-consistent ZIP with an invalid later PLY rolls back earlier native staging and preserves Undo',async({page})=>{
 await original(page);const before=await project(page),records=await exported(page);
 const zip=await packed(before,[new File([cloud],'valid.ply'),new File(['not a point cloud'],'invalid.ply')]);
 await page.locator('#file-input').setInputFiles(input('project.cloudanalyzer.zip',zip));await expect(page.locator('#status')).toContainText('Could not open workspace snapshot');
 await unchanged(page,before,records);await page.locator('#undo').click();await expect(page.locator('#cloud-list > li')).toHaveCount(1);
});

test('malformed saved graph inputs preserve the currently edited graph, map and points',async({page})=>{
 await original(page);
 const poses=Buffer.from('1 0 0 0 0 1 0 0 0 0 1 0\n1 0 0 1 0 1 0 0 0 0 1 0\n');
 await page.locator('#pg-files-input').setInputFiles([input('poses.kitti',poses),input('000000.ply',cloud),input('000001.ply',cloud)]);await expect(page.locator('#pg-stats')).toContainText('2 (2 with scans)');
 const before=await project(page),records=await exported(page),incoming=structuredClone(before),snapshot=JSON.parse(incoming.poseGraph.snapshot);snapshot.graph.nodes[0].pose.rotation=[];incoming.poseGraph.snapshot=JSON.stringify(snapshot);
 const zip=await packed(incoming,[new File([cloud],'valid.ply')],[new File([poses],'poses.kitti'),new File([cloud],'000000.ply'),new File([cloud],'000001.ply')]);
 await page.locator('#file-input').setInputFiles(input('project.cloudanalyzer.zip',zip));await expect(page.locator('#status')).toContainText('Could not open workspace snapshot');await unchanged(page,before,records);await expect(page.locator('#pg-stats')).toContainText('2 (2 with scans)');
});

function largeCloud(){
 const count=1500000,header=Buffer.from(`ply\nformat binary_little_endian 1.0\nelement vertex ${count}\nproperty double x\nproperty double y\nproperty double z\nend_header\n`),data=Buffer.alloc(count*24);
 for(let i=0;i<count;i++){data.writeDoubleLE(i%1000,i*24);data.writeDoubleLE(Math.floor(i/1000),i*24+8);data.writeDoubleLE(2,i*24+16);}
 return new File([header,data],'large.ply');
}

test('cancellation releases all staged clouds and leaves the original workspace and history intact',async({page})=>{
 test.setTimeout(180000);await original(page);const before=await project(page),records=await exported(page),zip=await packed(before,[new File([cloud],'first.ply'),largeCloud()]);
 for(let attempt=0;attempt<2;attempt++){
  // Cancel on the worker's second-cloud progress event. Polling and a later
  // Playwright click can race a fast parser and accidentally run after commit.
  await page.evaluate(()=>{
   const status=document.getElementById('status')!,observer=new MutationObserver(()=>{
    if(status.textContent?.includes('Staging incoming-large.ply')){
     observer.disconnect();document.getElementById('task-cancel')!.click();
    }
   });observer.observe(status,{childList:true,subtree:true,characterData:true});
  });
  await page.locator('#file-input').setInputFiles(input('project.cloudanalyzer.zip',zip));await expect(page.locator('#status')).toContainText('Could not open workspace snapshot: Cancelled');await unchanged(page,before,records);
 }
 await page.locator('#undo').click();await expect(page.locator('#cloud-list > li')).toHaveCount(1);
});

test('editing during native staging prevents replacement and retains the user change',async({page})=>{
 test.setTimeout(120000);await original(page);const before=await project(page),records=await exported(page),zip=await packed(before,[largeCloud()]);
 await page.locator('#file-input').setInputFiles(input('project.cloudanalyzer.zip',zip));await expect(page.locator('#status')).toContainText('incoming-large.ply');
 await page.locator('#point-size').evaluate((el:HTMLInputElement)=>{el.value='4';el.dispatchEvent(new Event('input',{bubbles:true}));});await expect(page.locator('#status')).toContainText('workspace changed during import');
 await expect(page.locator('#point-size')).toHaveValue('4');await unchanged(page,before,records);
});

test('a workspace cannot silently create a second cloud with an existing display name',async({page})=>{
 await original(page);const before=await project(page),records=await exported(page),p=structuredClone(before),file=new File([cloud],'new-source.ply');p.session.clouds=[{...p.session.clouds[0],source:await referenceFile(file),transforms:[],loadMaxPoints:0}];
 const zip=Buffer.from(await(await writeProjectSnapshot(p,[file],new AbortController().signal)).arrayBuffer());await page.locator('#file-input').setInputFiles(input('project.cloudanalyzer.zip',zip));await expect(page.locator('#status')).toContainText('same name');await unchanged(page,before,records);
});

test('matching extra inputs may precede the workspace ZIP without becoming extra clouds',async({page})=>{
 await original(page);const before=await project(page),file=new File([cloud],'valid.ply'),zip=await packed(before,[file]);
 await page.locator('#file-input').setInputFiles([input(file.name,cloud),input('project.cloudanalyzer.zip',zip)]);
 await expect(page.locator('#status')).toContainText('Project restored');await expect(page.locator('#cloud-list > li')).toHaveCount(3);await expect(page.locator('#undo')).toBeDisabled();
});

test('legacy workspaces wait for external graph inputs before restoring native state',async({page})=>{
 await original(page);const poses=Buffer.from('1 0 0 0 0 1 0 0 0 0 1 0\n1 0 0 1 0 1 0 0 0 0 1 0\n'),sources=[input('poses.kitti',poses),input('000000.ply',cloud),input('000001.ply',cloud)];
 await page.locator('#pg-files-input').setInputFiles(sources);await expect(page.locator('#pg-stats')).toContainText('2 (2 with scans)');const before=await project(page),zip=await packed(before,[new File([cloud],'valid.ply')]);
 await page.goto('/');await page.locator('#file-input').setInputFiles(input('project.cloudanalyzer.zip',zip));await expect(page.locator('#status')).toContainText('open matching source files');await expect(page.locator('#cloud-list > li')).toHaveCount(0);
 await page.locator('#file-input').setInputFiles(sources);await expect(page.locator('#status')).toContainText('Project restored');await expect(page.locator('#cloud-list > li')).toHaveCount(1);await expect(page.locator('#pg-stats')).toContainText('2 (2 with scans)');expect((await project(page)).poseGraph.snapshot).toEqual(before.poseGraph.snapshot);
});

test('actual NCLT workspace stages all native inputs and restores exact point exports and graph state',async({page},info)=>{
 const path=process.env.CLOUDANALYZER_TRANSACTION_ZIP;test.skip(!path,'Opt in with the previous complete NCLT workspace ZIP');test.setTimeout(600000);
 const {readFile,writeFile}=await import('node:fs/promises'),{readProjectSnapshot}=await import('../src/project-snapshot'),{createHash}=await import('node:crypto');
 const zip=await readFile(path!),files=await readProjectSnapshot(new File([zip],'project.cloudanalyzer.zip'),new AbortController().signal),before=JSON.parse(await files[0].text());
 const importPath=info.outputPath('project.cloudanalyzer.zip');await writeFile(importPath,zip);
 await page.goto('/');const started=Date.now();await page.locator('#file-input').setInputFiles(importPath);await expect(page.locator('#status')).toContainText(/Project restored|Could not open/,{timeout:540000});await expect(page.locator('#status')).toContainText('Project restored');
 const after=await project(page);expect(JSON.parse(after.poseGraph.snapshot)).toEqual(JSON.parse(before.poseGraph.snapshot));expect(JSON.parse(after.vectorMap)).toEqual(JSON.parse(before.vectorMap));
 await expect(page.locator('#cloud-list > li')).toHaveCount(2);const hashes=[];
 for(let i=0;i<2;i++){
  const row=page.locator('#cloud-list > li').nth(i);await row.locator('button[title="Save as…"]').click();const d=page.waitForEvent('download');await row.locator('.save-formats button',{hasText:'PLY'}).click();const data=await bytes(await d),expected=Buffer.from(await files.find(f=>f.name===before.session.clouds[i].source.name)!.arrayBuffer());
  expect(data).toEqual(expected);hashes.push({name:before.session.clouds[i].name,bytes:data.length,sha256:createHash('sha256').update(data).digest('hex')});
 }
 await expect(page.locator('#mapping-review-audit')).toBeDisabled();await expect(page.locator('#mapping-review-download')).toBeEnabled();
 await writeFile(info.outputPath('transaction-receipt.json'),JSON.stringify({zipBytes:zip.length,zipSha256:createHash('sha256').update(zip).digest('hex'),restoredMs:Date.now()-started,cloudExportsExact:true,poseGraphExact:true,hdMapExact:true,hashes},null,2));await page.screenshot({path:info.outputPath('restored-transaction.png')});
});
