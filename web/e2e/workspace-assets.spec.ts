import {expect,test,type Page,type Download} from '@playwright/test';
import {packageFixture,cloud} from '../tests/mapping-review-fixture.mjs';
import {readProjectSnapshot} from '../src/project-snapshot';
import {indexReviewZip,readReviewMember,writeReviewZip} from '../src/review-zip';
import {createHash} from 'node:crypto';
import {readFile,writeFile} from 'node:fs/promises';
const input=(name:string,buffer:Buffer)=>({name,mimeType:'application/octet-stream',buffer});
async function bytes(d:Download){const parts:Buffer[]=[];for await(const part of await d.createReadStream())parts.push(part);return Buffer.concat(parts);}
async function save(page:Page){const d=page.waitForEvent('download',d=>d.suggestedFilename()==='project.cloudanalyzer.zip');await page.locator('#project-snapshot').click();return bytes(await d);}
async function metadata(page:Page){const d=page.waitForEvent('download',d=>d.suggestedFilename()==='project.cloudanalyzer.json');await page.locator('#project-save').click();return JSON.parse((await bytes(await d)).toString());}
const signal=()=>new AbortController().signal;

test('one file resumes edited maps, original review archive and pose-graph constraints without reselecting sources',async({page})=>{
 await page.goto('/');
 const archive=packageFixture();
 // The original archive name may resemble a workspace ZIP; it must remain nested.
 await page.locator('#mapping-review-file').setInputFiles(input('original.cloudanalyzer.zip',archive));
 await expect(page.locator('#status')).toContainText('Opened generated maps');
 await page.locator('#pg-files-input').setInputFiles([input('poses.kitti',Buffer.from('1 0 0 0 0 1 0 0 0 0 1 0\n1 0 0 1 0 1 0 0 0 0 1 0\n')),input('000000.ply',cloud),input('000001.ply',cloud)]);
 await expect(page.locator('#pg-stats')).toContainText('2 (2 with scans)');
 await page.locator('#pg-a').fill('1');await page.locator('#pg-fix').click();
 await expect(page.locator('#pg-fix')).toHaveText('Free A');
 await page.locator('#vm-review-next').click();await page.locator('#vm-lane-speed').fill('30');await page.locator('#vm-lane-apply').click();
 await page.locator('#vm-review-notes').fill('Retain this edited geometry and original evidence separately');await page.locator('#vm-review-save').click();
 const before=await metadata(page),zip=await save(page),files=await readProjectSnapshot(new File([zip],'project.cloudanalyzer.zip'),signal());
 expect(files.map(f=>f.name)).toEqual(expect.arrayContaining(['poses.kitti','000000.ply','000001.ply','original.cloudanalyzer.zip']));
 expect(Buffer.from(await files.find(f=>f.name==='original.cloudanalyzer.zip')!.arrayBuffer())).toEqual(archive);
 await page.reload();await page.locator('#file-input').setInputFiles(input('project.cloudanalyzer.zip',zip));
 await expect(page.locator('#status')).toContainText('Project restored');
 await expect(page.locator('#cloud-list > li')).toHaveCount(1);
 await expect(page.locator('#pg-stats')).toContainText('2 (2 with scans)');await expect(page.locator('#pg-fix')).toHaveText('Free A');
 const after=await metadata(page);
 expect(JSON.parse(after.poseGraph.snapshot)).toEqual(JSON.parse(before.poseGraph.snapshot));expect(JSON.parse(after.vectorMap)).toEqual(JSON.parse(before.vectorMap));expect(after.reviews).toEqual(before.reviews);
 await expect(page.locator('#mapping-review-state')).toContainText('Current workspace edits need a new source check');
 await expect(page.locator('#mapping-review-audit')).toBeDisabled();
 const download=page.waitForEvent('download',d=>d.suggestedFilename()==='original.cloudanalyzer.zip');await page.locator('#mapping-review-download').click();expect(await bytes(await download)).toEqual(archive);
 // New review is of the edited map, not automatic reuse of the original saved audits.
 await expect(page.locator('#vm-quality-report')).not.toContainText('Saved low quantile');
});

test('changed original input bytes refuse the archive before changing existing workspace',async({page})=>{
 await page.goto('/');await page.locator('#mapping-review-file').setInputFiles(input('review.zip',packageFixture()));await expect(page.locator('#status')).toContainText('Opened generated maps');
 const zip=await save(page),file=new File([zip],'project.cloudanalyzer.zip'),directory=await indexReviewZip(file,signal(),2048),table:[string,Blob][]=[];
 for(const [name,member]of directory)table.push([name,new Blob([await readReviewMember(file,member,signal())])]);
 const asset=table.find(([name])=>name.startsWith('assets/'))!;const changed=new Uint8Array(await asset[1].arrayBuffer());changed[changed.length-1]^=1;asset[1]=new Blob([changed]);
 const bad=Buffer.from(await (await writeReviewZip(table,signal(),2048)).arrayBuffer());
 await page.locator('#file-input').setInputFiles(input('project.cloudanalyzer.zip',bad));await expect(page.locator('#status')).toContainText('hash differs');await expect(page.locator('#cloud-list > li')).toHaveCount(1);await expect(page.locator('#vm-status')).toContainText('1 lane');
});

test('actual NCLT graph inputs and original map evidence resume from one workspace file',async({page},info)=>{
 test.skip(process.env.CLOUDANALYZER_COMPLETE_WORKSPACE!=='1','Opt in for actual bundled NCLT');test.setTimeout(900000);
 await page.goto('/?demo=nclt');await expect(page.locator('#status')).toContainText(/nclt-2012-04-29_map is colored/,{timeout:540000});
 const stats=await page.locator('#pg-stats').textContent();
 const archivePath=process.env.CLOUDANALYZER_REVIEW_ZIP!;await page.locator('#mapping-review-file').setInputFiles(archivePath);await expect(page.locator('#status')).toContainText('Opened generated maps');
 const before=await metadata(page),zip=await save(page),packed=await readProjectSnapshot(new File([zip],'project.cloudanalyzer.zip'),signal());
 const source=packed.find(f=>f.name==='nclt-2012-04-29.mcap')!,original=await readFile('public/samples/nclt-2012-04-29.mcap');expect(Buffer.from(await source.arrayBuffer())).toEqual(original);
 const archive=await readFile(archivePath),embedded=packed.find(f=>f.name===archivePath.split('/').at(-1))!;expect(Buffer.from(await embedded.arrayBuffer())).toEqual(archive);
 const savedPath=info.outputPath('project.cloudanalyzer.zip');await writeFile(savedPath,zip);const start=Date.now();await page.goto('/');await page.locator('#file-input').setInputFiles(savedPath);await expect(page.locator('#status')).toContainText(/Project restored|Could not restore|Could not open/,{timeout:540000});await expect(page.locator('#status')).toContainText('Project restored');
 await expect(page.locator('#pg-stats')).toHaveText(stats!);const after=await metadata(page);expect(JSON.parse(after.poseGraph.snapshot)).toEqual(JSON.parse(before.poseGraph.snapshot));expect(JSON.parse(after.vectorMap)).toEqual(JSON.parse(before.vectorMap));
 await expect(page.locator('#mapping-review-audit')).toBeDisabled();
 const receipt={zipBytes:zip.length,zipSha256:createHash('sha256').update(zip).digest('hex'),inputBytes:original.length,inputSha256:createHash('sha256').update(original).digest('hex'),originalArchiveBytes:archive.length,originalArchiveSha256:createHash('sha256').update(archive).digest('hex'),nodes:JSON.parse(before.poseGraph.snapshot).graph.nodes.length,clouds:before.session.clouds.map((c:any)=>c.name),restoredMs:Date.now()-start,poseGraphExact:true,hdMapExact:true,archivedAuditsRemainOriginal:true};
 await writeFile(info.outputPath('complete-workspace-receipt.json'),JSON.stringify(receipt,null,2));await writeFile(process.env.CLOUDANALYZER_WORKSPACE_OUTPUT!,zip);await page.screenshot({path:info.outputPath('restored-complete-workspace.png')});
});
