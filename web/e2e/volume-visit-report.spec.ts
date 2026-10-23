import { test, expect, type Page } from '@playwright/test';
import { readFile } from 'node:fs/promises';
const ply=(name:string,zs:number[])=>({name,mimeType:'application/octet-stream',buffer:Buffer.from(`ply\nformat ascii 1.0\nelement vertex ${zs.length}\nproperty float x\nproperty float y\nproperty float z\nend_header\n`+zs.map((z,x)=>`${x} 0 ${z}`).join('\n')+'\n')});
async function save(page:Page,id:string,path:string){const pending=page.waitForEvent('download');await page.locator(id).click();await(await pending).saveAs(path);}
async function setup(page:Page){
 await page.addInitScript(()=>localStorage.setItem('cloudanalyzer-autosave-enabled','off'));
 await page.goto('/');await page.locator('#file-input').setInputFiles([ply('baseline.ply',[0,0,0]),ply('A.ply',[5,1,100]),ply('B.ply',[3,4])]);
 await expect(page.locator('#cloud-list > li')).toHaveCount(3);await expect(page.locator('#task')).toBeHidden();
 for(const id of ['volume-area-options','volume-visits-options','volume-options'])await page.locator(`#${id}`).evaluate((el:HTMLDetailsElement)=>el.open=true);
 await page.locator('#volume-cell').fill('1');
 await page.locator('#volume-areas-file').setInputFiles({name:'areas.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify({schema:'cloudanalyzer.named_review_areas.v1',areas:[{name:'All <img src=x onerror=alert(1)>',bounds:[-1,-1,3,1]},{name:'Empty',bounds:[1.9,-.1,2.1,.1]}]}))});
 const ids=await page.locator('#volume-visits option').evaluateAll((os:HTMLOptionElement[])=>os.map(o=>o.value));
 await page.locator('#volume-visits').selectOption(ids.slice(1));return ids;
}
test('downloaded common-support report opens offline with exact JSON, direct quantities, missing rows and inert labels',async({page,context},info)=>{
 await setup(page);await page.locator('#volume-visits-support').selectOption('common_observed');await page.locator('#volume-visits-intervals').check();
 await page.locator('#volume-visits-run').click();await expect(page.locator('#volume-visits-result')).toBeVisible();
 const jsonPath=info.outputPath('comparison.json'), htmlPath=info.outputPath('report.html');
 await save(page,'#volume-visits-json',jsonPath);await save(page,'#volume-visits-report',htmlPath);
 const snapshot=JSON.parse(await readFile(jsonPath,'utf8'));
 // Load the downloaded bytes into an offline document: managed browsers can
 // prohibit file:// navigation independently of the report's contents.
 const reportContext=await context.browser()!.newContext({offline:true});const report=await reportContext.newPage();
 const requests:string[]=[],errors:string[]=[];report.on('request',r=>requests.push(r.url()));report.on('pageerror',e=>errors.push(e.message));
 await report.setContent(await readFile(htmlPath,'utf8'));
 await expect(report.locator('h1')).toHaveText('2.5D survey volume report');
 expect(JSON.parse((await report.locator('#comparison-json').textContent())!)).toEqual(snapshot);
 await expect(report.locator('#interval-volumes tbody tr').first()).toContainText('1. A.ply → 2. B.ply');
 expect(await report.locator('#interval-volumes tbody tr').first().locator('td').allTextContents()).toEqual(['1. A.ply → 2. B.ply','All <img src=x onerror=alert(1)>','[-1, -1, 3, 1]','3','2','1','1','1','2','1','0']);
 await expect(report.locator('#baseline-volumes tbody tr').nth(1)).toContainText('Unavailable');
 await expect(report.locator('body')).toContainText(snapshot.support.mask.sha256);
 await expect(report.locator('img,script,iframe')).toHaveCount(0);expect(requests).toEqual([]);expect(errors).toEqual([]);
 await report.screenshot({path:info.outputPath('volume-report.png'),fullPage:true});
 await page.locator('#volume-cell').fill('2');await expect(page.locator('#volume-visits-report')).toBeHidden();
 await expect(report.locator('#interval-volumes tbody tr').first()).toContainText('1. A.ply');
 await report.emulateMedia({media:'print'});await expect(report.locator('details')).toBeHidden();
 await report.pdf({path:info.outputPath('volume-report.pdf'),format:'A4',landscape:true});await reportContext.close();
});
test('independent constant-baseline report retains interpolation and no-interval behavior and invalidates after input removal',async({page,context},info)=>{
 const ids=await setup(page);await page.locator('#volume-before').selectOption('constant');await page.locator('#volume-before-z').fill('0');
 await page.locator('#volume-fill').check();await page.locator('#volume-unit').selectOption('ft');
 await page.locator('#volume-visits-run').click();await expect(page.locator('#volume-visits-result')).toBeVisible();
 const path=info.outputPath('independent-report.html');await save(page,'#volume-visits-report',path);
 const reportContext=await context.browser()!.newContext({offline:true});const report=await reportContext.newPage();await report.setContent(await readFile(path,'utf8'));
 await expect(report.locator('body')).toContainText('Constant height 0');await expect(report.locator('body')).toContainText('enabled (estimated heights may be included)');
 await expect(report.locator('body')).toContainText('Each baseline/visit pair may have different support');await expect(report.locator('#interval-volumes')).toHaveCount(0);
 await expect(report.locator('#baseline-volumes thead')).toContainText('Fill (ft³)');
 const snapshot=JSON.parse((await report.locator('#comparison-json').textContent())!);expect(snapshot.baseline).toEqual({constantHeight:0});expect(snapshot.ordering.cloudIds).toEqual(ids.slice(1).map(Number));expect(snapshot.intervals).toBeUndefined();
 await page.locator('#cloud-list > li').nth(1).locator('button[title="Remove"]').click();await expect(page.locator('#volume-visits-report')).toBeHidden();await reportContext.close();
});
