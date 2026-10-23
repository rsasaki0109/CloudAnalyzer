import { expect, test, type Page } from '@playwright/test';
import { writeFile } from 'node:fs/promises';
const file=(name:string,buffer:Buffer)=>({name,mimeType:'application/json',buffer});
const ply=(z:number)=>Buffer.from(`ply\nformat ascii 1.0\nelement vertex 2\nproperty float x\nproperty float y\nproperty float z\nend_header\n0 0 ${z}\n10 0 ${z}\n`);
async function bytes(page:Page,id:string){const pending=page.waitForEvent('download');await page.locator(id).click();const parts:Buffer[]=[];for await(const part of await(await pending).createReadStream())parts.push(part);return Buffer.concat(parts);}
async function setup(page:Page){
 await page.addInitScript(()=>localStorage.setItem('cloudanalyzer-autosave-enabled','off'));await page.goto('/');
 await page.locator('#file-input').setInputFiles([file('truth.ply',ply(0)),file('map.ply',ply(1))]);
 await expect(page.locator('#cloud-list > li')).toHaveCount(2);await expect(page.locator('#task')).toBeHidden();
 await page.locator('#coverage-source').selectOption({label:'map.ply'});await page.locator('#coverage-reference').selectOption({label:'truth.ply'});
 await page.getByText('Locate distance discrepancies',{exact:true}).click();await page.locator('#regions-min').fill('1');await page.locator('#regions-cell').fill('2');
 await page.locator('#regions-both').click();await expect(page.locator('#status')).toContainText('ready in both directions');
 const first=page.locator('#regions-list > li').first();await first.locator('select').selectOption('deferred');await first.locator('textarea').fill('Needs inspection');
 const a=await bytes(page,'#regions-review-json');
 await first.locator('select').selectOption('reviewed');await first.locator('textarea').fill('  =Checked, "east"\n<img src=x onerror=alert(1)>');
 return {a,b:await bytes(page,'#regions-review-json')};
}
async function openA(page:Page,value:Buffer,name='a.json'){
 await page.locator('#review-archive-panel').evaluate((el:HTMLDetailsElement)=>el.open=true);
 await page.locator('#review-archive-file').setInputFiles(file(name,value));await expect(page.locator('#status')).toContainText('opened read-only');
}
async function openB(page:Page,value:Buffer,name='b.json'){
 await page.locator('#review-compare-file').setInputFiles(file(name,value));await expect(page.locator('#status')).toContainText('comparison ready, read-only');
}
test('actual archived reviews compare read-only, export both records safely and reopen without clouds',async({page,context},info)=>{
 const {a,b}=await setup(page),before=await bytes(page,'#regions-review-json'),qa=JSON.parse((await bytes(page,'#report-json')).toString()).sections;
 await page.evaluate(()=>{(window as any).comparisonWorkerCalls=0;const original=Worker.prototype.postMessage;Worker.prototype.postMessage=function(...args:any[]){(window as any).comparisonWorkerCalls++;return(original as any).apply(this,args);};});
 await openA(page,a);await openB(page,b,'  =b.json');
 const dirs=page.locator('#review-compare-list > details');await expect(dirs).toHaveCount(2);
 await expect(dirs.first()).toContainText('1 status changes, 1 comment changes');await expect(dirs.first().locator('li').first()).toContainText('deferred → reviewed');await expect(dirs.nth(1)).toContainText('0 status changes, 0 comment changes');
 expect(await bytes(page,'#regions-review-json')).toEqual(before);expect(JSON.parse((await bytes(page,'#report-json')).toString()).sections).toEqual(qa);
 expect(await bytes(page,'#review-archive-json')).toEqual(a);expect(await bytes(page,'#review-compare-original')).toEqual(b);expect(await page.evaluate(()=>(window as any).comparisonWorkerCalls)).toBe(0);
 const result=JSON.parse((await bytes(page,'#review-compare-json')).toString());expect(result.directions.map((d:any)=>d.compatible)).toEqual([true,true]);expect(result.directions[0].rows[0].a.status).toBe('deferred');expect(result.directions[0].rows[0].b.status).toBe('reviewed');expect(result.directions[0].rows[1].changed).toEqual([]);
 const csv=await bytes(page,'#review-compare-csv');expect(csv.toString()).toContain(`"'  =b.json"`);expect(csv.toString()).toContain(`"'  =Checked, ""east""\n<img src=x onerror=alert(1)>"`);
 const html=await bytes(page,'#review-compare-html');await writeFile(info.outputPath('comparison.html'),html);await writeFile(info.outputPath('comparison.json'),JSON.stringify(result,null,2));await writeFile(info.outputPath('comparison.csv'),csv);
 const offline=await context.browser()!.newContext({offline:true}),report=await offline.newPage(),requests:string[]=[];report.on('request',r=>requests.push(r.url()));await report.setContent(html.toString());
 expect(JSON.parse((await report.locator('#comparison-json').textContent())!)).toEqual(result);await expect(report.locator('img,script,iframe')).toHaveCount(0);expect(requests).toEqual([]);await expect(report.locator('body')).toContainText('Needs inspection');await report.screenshot({path:info.outputPath('comparison.png'),fullPage:true});await offline.close();
 await page.reload();await expect(page.locator('#cloud-list > li')).toHaveCount(0);await openA(page,a);await openB(page,b);await expect(page.locator('#review-compare-list')).toContainText('deferred → reviewed');expect(JSON.parse((await bytes(page,'#report-json')).toString()).sections['distance-candidate-reviews:forward']).toBeUndefined();
 await page.locator('#review-compare-clear').click();await expect(page.locator('#review-compare-result')).toBeHidden();await expect(page.locator('#review-archive-result')).toBeVisible();
});
test('different diagnoses and missing directions have no transitions; invalid B retains the previous comparison',async({page})=>{
 const {a,b}=await setup(page);await openA(page,a);const other=JSON.parse(b.toString());other.comparisons=other.comparisons.slice(0,1);other.comparisons[0].review.analyzedAt='2026-10-25T00:00:00.000Z';const different=Buffer.from(JSON.stringify(other));
 await openB(page,different,'a.json');expect(JSON.parse((await bytes(page,'#review-compare-json')).toString()).directions.map((d:any)=>d.compatible)).toEqual([false,false]);await expect(page.locator('#review-compare-list')).toContainText('Analysis timestamps differ');await expect(page.locator('#review-compare-list')).toContainText('Direction is absent');await expect(page.locator('#review-compare-list li')).toHaveCount(0);expect((await bytes(page,'#review-compare-csv')).toString()).toContain('forward,false');
 const bad=JSON.parse(b.toString());bad.comparisons[0].review.counts.reviewed=999;
 for(const [name,value,message] of [['bad.json',Buffer.from(JSON.stringify(bad)),'inconsistent'],['big.json',Buffer.alloc(2*1024*1024+1),'2 MiB'],['syntax.json',Buffer.from('{'),'Could not compare']] as const){await page.locator('#review-compare-file').setInputFiles(file(name,value));await expect(page.locator('#status')).toContainText(message);expect(await bytes(page,'#review-compare-original')).toEqual(different);}
 await openB(page,b);await expect(page.locator('#review-compare-list li')).toHaveCount(4);await openA(page,a,'replacement.json');await expect(page.locator('#review-compare-result')).toBeHidden();
});
test('clear, superseding B, replacing A and closing A discard pending comparison reads',async({page})=>{
 const {a,b}=await setup(page);await openA(page,a);await page.evaluate(()=>{const original=File.prototype.text;File.prototype.text=function(){const file=this;if(!file.name.startsWith('delayed-'))return original.call(file);return new Promise(resolve=>{(window as any).releaseComparison=async()=>resolve(await original.call(file));});};});
 async function delay(){await page.evaluate(()=>delete(window as any).releaseComparison);await page.locator('#review-compare-file').setInputFiles(file('delayed-b.json',b));await page.waitForFunction(()=>!!(window as any).releaseComparison);}
 async function release(){await page.evaluate(()=>(window as any).releaseComparison());}
 await delay();await page.locator('#review-compare-clear').click();await release();await expect(page.locator('#review-compare-result')).toBeHidden();
 await delay();await openB(page,a,'newest.json');await release();await expect(page.locator('#review-compare-names')).toContainText('B: newest.json');
 await delay();await openA(page,a,'new-a.json');await release();await expect(page.locator('#review-compare-result')).toBeHidden();
 await delay();await page.locator('#review-archive-close').click();await release();await expect(page.locator('#review-archive-result')).toBeHidden();await expect(page.locator('#review-compare-result')).toBeHidden();
});
