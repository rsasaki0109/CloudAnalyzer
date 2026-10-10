import {expect,test,type Page,type Download} from '@playwright/test';
import {createHash} from 'node:crypto';
import {map} from '../tests/mapping-review-fixture.mjs';
const input=(name:string,buffer:Buffer)=>({name,mimeType:'application/octet-stream',buffer});
async function bytes(d:Download){const result:Buffer[]=[];for await(const part of await d.createReadStream())result.push(part);return Buffer.concat(result);}
async function exported(page:Page,index:number){const row=page.locator('#cloud-list > li').nth(index);await row.locator('button[title="Save as…"]').click();const d=page.waitForEvent('download');await row.locator('.save-formats button',{hasText:'PLY'}).click();return bytes(await d);}
async function project(page:Page){const d=page.waitForEvent('download');await page.locator('#project-save').click();return JSON.parse((await bytes(await d)).toString());}
async function processing(page:Page,index:number){const d=page.waitForEvent('download');await page.locator('#cloud-list > li').nth(index).locator('button[title="Download processing record"]').click();return JSON.parse((await bytes(await d)).toString());}
function fixture(count=65,offset=0){
 const header=Buffer.from('ply\nformat binary_little_endian 1.0\nelement vertex '+count+'\nproperty double x\nproperty double y\nproperty double z\nproperty float intensity\nproperty float correction\nend_header\n'),body=Buffer.alloc(count*32);
 for(let i=0;i<count;i++){body.writeDoubleLE(1000000+offset+(i%8)*0.2,i*32);body.writeDoubleLE(2000000+Math.floor(i/8)*0.2,i*32+8);body.writeDoubleLE(i===64?40:1,i*32+16);body.writeFloatLE(i+0.25,i*32+24);body.writeFloatLE(i/8,i*32+28);}return Buffer.concat([header,body]);
}
const plan={app:'CloudAnalyzer Filter Recipe',version:1,name:'Road cleanup',steps:[{op:'sor',neighbors:8,stdRatio:1},{op:'voxel',size:0.5}]};
async function openRecipe(page:Page,value:any=plan){await page.locator('#recipe-panel > summary').click();await page.locator('#recipe-file').setInputFiles(input('filter-recipe.json',Buffer.from(JSON.stringify(value))));await expect(page.locator('#recipe-steps > li')).toHaveCount(value.steps.length);}
async function run(page:Page){await page.locator('#recipe-run').click();await expect(page.locator('#status')).toContainText(/Recipe complete|Recipe failed/);}

test('recipe builder exports ordered steps and importing a recipe does not execute it',async({page})=>{
 await page.goto('/');await page.locator('#file-input').setInputFiles(input('scan.ply',fixture()));await expect(page.locator('#cloud-list > li')).toHaveCount(1);await page.locator('#recipe-panel > summary').click();await page.locator('#recipe-name').fill(plan.name);
 await page.locator('#filter-op').selectOption('sor');await page.locator('#recipe-add').click();await page.locator('#filter-op').selectOption('voxel');await page.locator('#filter-voxel').fill('0.5');await page.locator('#recipe-add').click();
 const d=page.waitForEvent('download');await page.locator('#recipe-export').click();const saved=await bytes(await d);expect(JSON.parse(saved.toString())).toEqual(plan);
 await page.locator('#recipe-steps button').first().click();await expect(page.locator('#recipe-steps > li')).toHaveCount(1);
 await page.locator('#recipe-file').setInputFiles(input('filter-recipe.json',saved));await expect(page.locator('#recipe-steps > li')).toHaveCount(2);await expect(page.locator('#cloud-list > li')).toHaveCount(1);await expect(page.locator('#undo')).toBeDisabled();
 await page.locator('#recipe-file').setInputFiles(input('bad.json',Buffer.from(JSON.stringify({...plan,steps:[{op:'random',percent:10}]}))));await expect(page.locator('#status')).toContainText('Could not open recipe');await expect(page.locator('#recipe-steps > li')).toHaveCount(2);
});

test('one batch matches individual filters, records hashes, and survives Undo, Redo and workspace reopening',async({page})=>{
 test.setTimeout(180000);const files=[input('scan-a.ply',fixture()),input('scan-b.ply',fixture(65,100))];
 await page.goto('/');await page.locator('#file-input').setInputFiles(files);await expect(page.locator('#cloud-list > li')).toHaveCount(2);
 const original=[await exported(page,0),await exported(page,1)],individual=[];
 for(let source=0;source<2;source++){
  await page.locator('#filter-cloud').selectOption({label:files[source].name});await page.locator('#filter-op').selectOption('sor');await page.locator('#filter-run').click();await expect(page.locator('#cloud-list > li')).toHaveCount(3+source*2);
  await page.locator('#filter-cloud').selectOption({label:files[source].name.replace('.ply','_sor')});await page.locator('#filter-op').selectOption('voxel');await page.locator('#filter-voxel').fill('0.5');await page.locator('#filter-run').click();await expect(page.locator('#cloud-list > li')).toHaveCount(4+source*2);individual.push(await exported(page,3+source*2));
 }
 await page.goto('/');await page.locator('#file-input').setInputFiles(files);await expect(page.locator('#cloud-list > li')).toHaveCount(2);await page.locator('#vm-file').setInputFiles(input('map.json',Buffer.from(JSON.stringify(map))));await expect(page.locator('#vm-status')).toContainText('1 lane');
 await openRecipe(page);await expect(page.locator('#cloud-list > li')).toHaveCount(2);await page.locator('#recipe-visible').click();await run(page);await expect(page.locator('#status')).toContainText('Recipe complete');await expect(page.locator('#cloud-list > li')).toHaveCount(4);
 const records=[];for(let i=0;i<2;i++){const output=await exported(page,2+i);expect(output).toEqual(individual[i]);const p=await processing(page,2+i);expect(p.recipe).toEqual(plan);expect(p.input.sha256).toEqual(createHash('sha256').update(original[i]).digest('hex'));expect(p.output.sha256).toEqual(createHash('sha256').update(output).digest('hex'));records.push(p);}
 await page.locator('#undo').click();await expect(page.locator('#cloud-list > li')).toHaveCount(2);await expect(page.locator('#cloud-list > li input[title="Show / hide"]:checked')).toHaveCount(2);
 await page.locator('#redo').click();await expect(page.locator('#cloud-list > li')).toHaveCount(4);expect(await exported(page,2)).toEqual(individual[0]);
 const state=await project(page),download=page.waitForEvent('download');await page.locator('#project-snapshot').click();const snapshot=await bytes(await download);
 await page.goto('/');await page.locator('#file-input').setInputFiles(input('project.cloudanalyzer.zip',snapshot));await expect(page.locator('#status')).toContainText('Project restored');await expect(page.locator('#cloud-list > li')).toHaveCount(4);expect((await project(page)).vectorMap).toEqual(state.vectorMap);
 for(let i=0;i<2;i++){expect(await processing(page,2+i)).toEqual(records[i]);expect(await exported(page,2+i)).toEqual(individual[i]);}
});

test('a native failure in a later recipe step keeps all originals and the preceding Undo step',async({page})=>{
 test.setTimeout(120000);await page.goto('/');await page.locator('#file-input').setInputFiles(input('scan.ply',fixture()));await expect(page.locator('#cloud-list > li')).toHaveCount(1);
 await page.locator('#filter-op').selectOption('random');await page.locator('#filter-percent').fill('50');await page.locator('#filter-run').click();await expect(page.locator('#cloud-list > li')).toHaveCount(2);
 const before=[await exported(page,0),await exported(page,1)];await openRecipe(page,{...plan,steps:[{op:'voxel',size:0.1},{op:'splat',minOpacity:0.1,maxSize:5}]});await page.locator('#recipe-clouds input').first().check();
 for(let attempt=0;attempt<2;attempt++){await run(page);await expect(page.locator('#status')).toContainText('not Gaussian splats');await expect(page.locator('#cloud-list > li')).toHaveCount(2);expect(await exported(page,0)).toEqual(before[0]);expect(await exported(page,1)).toEqual(before[1]);await expect(page.locator('#undo')).toBeEnabled();}
 await page.locator('#undo').click();await expect(page.locator('#cloud-list > li')).toHaveCount(1);
});

test('cancelling after the first recipe step keeps source records and the previous Undo available',async({page})=>{
 test.setTimeout(180000);await page.goto('/');await page.locator('#file-input').setInputFiles(input('scan.ply',fixture()));await expect(page.locator('#cloud-list > li')).toHaveCount(1);
 await page.locator('#filter-op').selectOption('random');await page.locator('#filter-run').click();await expect(page.locator('#cloud-list > li')).toHaveCount(2);
 const before=await exported(page,0);await openRecipe(page,{...plan,steps:[{op:'voxel',size:0.1},{op:'spatial',spacing:0.2}]});await page.locator('#recipe-clouds input').first().check();
 for(let attempt=0;attempt<2;attempt++){
  await page.evaluate(()=>{const status=document.getElementById('status')!,observer=new MutationObserver(()=>{if(status.textContent?.includes('step 2/2: spatial')){observer.disconnect();document.getElementById('task-cancel')!.click();}});observer.observe(status,{childList:true,subtree:true,characterData:true});});
  await run(page);await expect(page.locator('#status')).toContainText('Recipe failed: Cancelled');await expect(page.locator('#cloud-list > li')).toHaveCount(2);expect(await exported(page,0)).toEqual(before);await expect(page.locator('#undo')).toBeEnabled();
 }
 await page.locator('#undo').click();await expect(page.locator('#cloud-list > li')).toHaveCount(1);
});

test('a later source failure discards the already finished first result',async({page})=>{
 const properties='x y z nx ny nz f_dc_0 f_dc_1 f_dc_2 opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3'.split(' '),body=Buffer.alloc(4*properties.length*4);
 for(let i=0;i<4;i++)[i,0,0,0,0,0,1,0,-1,2,0,-1,-2,1,0,0,0].forEach((v,k)=>body.writeFloatLE(v,(i*properties.length+k)*4));
 const splats=Buffer.concat([Buffer.from('ply\nformat binary_little_endian 1.0\nelement vertex 4\n'+properties.map(p=>'property float '+p+'\n').join('')+'end_header\n'),body]);
 await page.goto('/');await page.locator('#file-input').setInputFiles([input('splats.ply',splats),input('plain.ply',fixture())]);await expect(page.locator('#cloud-list > li')).toHaveCount(2);
 const before=[await exported(page,0),await exported(page,1)];await openRecipe(page,{...plan,steps:[{op:'splat',minOpacity:0.1,maxSize:5}]});await page.locator('#recipe-visible').click();await run(page);await expect(page.locator('#status')).toContainText('not Gaussian splats');await expect(page.locator('#cloud-list > li')).toHaveCount(2);
 expect(await exported(page,0)).toEqual(before[0]);expect(await exported(page,1)).toEqual(before[1]);await expect(page.locator('#undo')).toBeDisabled();
});

test('changing current work during recipe staging discards results and retains the user edit',async({page})=>{
 await page.goto('/');await page.locator('#file-input').setInputFiles(input('scan.ply',fixture()));await expect(page.locator('#cloud-list > li')).toHaveCount(1);const before=await exported(page,0);
 await openRecipe(page,{...plan,steps:[{op:'voxel',size:0.1},{op:'spatial',spacing:0.2}]});await page.locator('#recipe-visible').click();
 await page.evaluate(()=>{const status=document.getElementById('status')!,observer=new MutationObserver(()=>{if(status.textContent?.includes('step 2/2: spatial')){observer.disconnect();const size=document.getElementById('point-size') as HTMLInputElement;size.value='4';size.dispatchEvent(new Event('input',{bubbles:true}));}});observer.observe(status,{childList:true,subtree:true,characterData:true});});
 await run(page);await expect(page.locator('#status')).toContainText('Current work changed');await expect(page.locator('#point-size')).toHaveValue('4');await expect(page.locator('#cloud-list > li')).toHaveCount(1);expect(await exported(page,0)).toEqual(before);await expect(page.locator('#undo')).toBeDisabled();
});

test('an insufficient Undo budget refuses the batch before hiding sources or losing previous Undo',async({page})=>{
 test.setTimeout(120000);await page.goto('/');await page.locator('#file-input').setInputFiles(input('small.ply',fixture()));await expect(page.locator('#cloud-list > li')).toHaveCount(1);
 await page.locator('#filter-op').selectOption('random');await page.locator('#filter-percent').fill('50');await page.locator('#filter-run').click();await expect(page.locator('#cloud-list > li')).toHaveCount(2);
 await page.locator('#file-input').setInputFiles(input('large.ply',fixture(10000)));await expect(page.locator('#cloud-list > li')).toHaveCount(3);const before=await exported(page,2);
 await page.locator('#memory-panel > summary').click();await page.locator('#memory-undo-budget').fill('1');await page.locator('#memory-apply').click();await page.locator('#memory-panel > summary').click();
 await openRecipe(page,{...plan,steps:[{op:'voxel',size:0.01}]});await page.locator('#recipe-clouds input').last().check();await run(page);await expect(page.locator('#status')).toContainText('Undo budget');await expect(page.locator('#cloud-list > li')).toHaveCount(3);expect(await exported(page,2)).toEqual(before);await expect(page.locator('#undo')).toBeEnabled();
 await page.locator('#undo').click();await expect(page.locator('#cloud-list > li')).toHaveCount(2);
});

test('very small voxels are refused before clipping distinct cells in both single and batch filters',async({page})=>{
 await page.goto('/');await page.locator('#file-input').setInputFiles(input('wide.ply',fixture(10000)));await expect(page.locator('#cloud-list > li')).toHaveCount(1);const before=await exported(page,0);
 await page.locator('#filter-op').selectOption('voxel');await page.locator('#filter-voxel').fill('0.000001');await page.locator('#filter-run').click();await expect(page.locator('#status')).toContainText('Voxel size is too small');await expect(page.locator('#cloud-list > li')).toHaveCount(1);
 await openRecipe(page,{...plan,steps:[{op:'voxel',size:0.000001}]});await page.locator('#recipe-visible').click();await run(page);await expect(page.locator('#status')).toContainText('Voxel size is too small');await expect(page.locator('#cloud-list > li')).toHaveCount(1);expect(await exported(page,0)).toEqual(before);
});

test('actual NCLT full map and display preview match individual filters and retain exact attributes and provenance',async({page},info)=>{
 const path=process.env.CLOUDANALYZER_RECIPE_PLY,archive=process.env.CLOUDANALYZER_RECIPE_REVIEW;
 test.skip(!path||!archive,'Opt in with the immutable full June map and its display-preview review ZIP');test.setTimeout(600000);
 const {writeFile,readFile}=await import('node:fs/promises');
 const wasmSha256=createHash('sha256').update(await readFile('src/wasm/ca_wasm_bg.wasm')).digest('hex');
 const actualPlan={...plan,steps:[{op:'sor',neighbors:8,stdRatio:1},{op:'voxel',size:0.2}]};
 async function load(){
  await page.goto('/');await page.locator('#file-input').setInputFiles(path!);await expect(page.locator('#status')).toContainText('Loaded local_map.ply',{timeout:180000});
  await page.locator('#mapping-review-file').setInputFiles(archive!);await expect(page.locator('#status')).toContainText('Opened generated maps',{timeout:180000});await expect(page.locator('#cloud-list > li')).toHaveCount(2);
  await page.locator('#memory-panel > summary').click();await page.locator('#memory-undo-budget').fill('256');await page.locator('#memory-apply').click();await page.locator('#memory-panel > summary').click();
 }
 await load();const state=await project(page),sourceRecords=[await exported(page,0),await exported(page,1)],individual:Buffer[]=[];
 for(let source=0;source<2;source++){
  await page.locator('#filter-cloud').selectOption({label:state.session.clouds[source].name});
  await page.locator('#filter-op').selectOption('sor');await page.locator('#filter-run').click();await expect(page.locator('#cloud-list > li')).toHaveCount(3+source*2,{timeout:180000});
  await page.locator('#filter-cloud').selectOption({label:state.session.clouds[source].name.replace(/\.[^.]+$/,'')+'_sor'});
  await page.locator('#filter-op').selectOption('voxel');await page.locator('#filter-voxel').fill('0.2');await page.locator('#filter-run').click();await expect(page.locator('#cloud-list > li')).toHaveCount(4+source*2,{timeout:180000});individual.push(await exported(page,3+source*2));
 }
 await load();await openRecipe(page,actualPlan);await page.locator('#recipe-visible').click();const started=Date.now();await page.locator('#recipe-run').click();await expect(page.locator('#status')).toContainText(/Recipe complete|Recipe failed/,{timeout:180000});await expect(page.locator('#status')).toContainText('Recipe complete');await expect(page.locator('#cloud-list > li')).toHaveCount(4);
 const records=[];const hash=(data:Buffer)=>createHash('sha256').update(data).digest('hex');
 function rows(data:Buffer){
  const end=data.indexOf('end_header\n')+11,header=data.subarray(0,end).toString(),properties=header.split('\n').filter(s=>s.startsWith('property '));
  expect(properties).toEqual(['property double x','property double y','property double z','property float intensity','property float correction']);
  const count=Number(header.match(/element vertex (\d+)/)![1]);expect(data.length-end).toEqual(count*32);
  return {end,count};
 }
 for(let i=0;i<2;i++){
  const output=await exported(page,2+i);expect(output.equals(individual[i])).toBe(true);const p=await processing(page,2+i);expect(p.recipe).toEqual(actualPlan);expect(p.wasmSha256).toEqual(wasmSha256);expect(p.input.sha256).toEqual(hash(sourceRecords[i]));expect(p.output.sha256).toEqual(hash(output));
  const source=rows(sourceRecords[i]),result=rows(output),originals=new Set<string>();
  for(let n=0;n<source.count;n++)originals.add(sourceRecords[i].subarray(source.end+n*32,source.end+(n+1)*32).toString('hex'));
  let unmatched=0;for(let n=0;n<result.count;n++)if(!originals.has(output.subarray(result.end+n*32,result.end+(n+1)*32).toString('hex')))unmatched++;
  expect(unmatched).toEqual(0);expect(source.count).toEqual([646309,161578][i]);expect(p.input.points).toEqual(source.count);expect(p.output.points).toEqual(result.count);records.push(p);
 }
 const after=await project(page);expect(after.vectorMap).toEqual(state.vectorMap);
 const d=page.waitForEvent('download');await page.locator('#project-snapshot').click();const zip=await bytes(await d),zipPath=info.outputPath('project.cloudanalyzer.zip');await writeFile(zipPath,zip);
 const {readProjectSnapshot}=await import('../src/project-snapshot');const members=await readProjectSnapshot(new File([zip],'project.cloudanalyzer.zip'),new AbortController().signal),saved=JSON.parse(await members[0].text());
 expect(saved.session.clouds[3].displayPreview).toBe(true);for(let i=0;i<2;i++)expect(saved.session.clouds[i+2].processing).toEqual(records[i]);
 await page.locator('#undo').click();await expect(page.locator('#cloud-list > li')).toHaveCount(2);await page.locator('#redo').click();await expect(page.locator('#cloud-list > li')).toHaveCount(4);
 await page.goto('/');await page.locator('#file-input').setInputFiles(zipPath);await expect(page.locator('#status')).toContainText(/Project restored|Could not open/,{timeout:180000});await expect(page.locator('#status')).toContainText('Project restored');expect((await project(page)).vectorMap).toEqual(state.vectorMap);
 for(let i=0;i<2;i++){expect(await processing(page,2+i)).toEqual(records[i]);expect((await exported(page,2+i)).equals(individual[i])).toBe(true);}
 const wasm=await readFile('src/wasm/ca_wasm_bg.wasm');
 await writeFile(info.outputPath('recipe-receipt.json'),JSON.stringify({recipe:actualPlan,wasmSha256:hash(wasm),originalInputSha256:hash(await readFile(path!)),originalReviewSha256:hash(await readFile(archive!)),sourcePoints:[646309,161578],records,individualFiltersExact:true,allOutputRecordsFromSource:true,hdMapUnchanged:true,previewFlagRetained:true,oneUndoRedo:true,workspaceRoundtripExact:true,zipBytes:zip.length,zipSha256:hash(zip),runAndVerificationMs:Date.now()-started},null,2));await page.screenshot({path:info.outputPath('recipe-workspace.png')});
});
