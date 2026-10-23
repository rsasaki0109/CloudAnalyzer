import { test, expect, type Page } from '@playwright/test';
const ply=(name:string,zs:number[])=>({name,mimeType:'application/octet-stream',buffer:Buffer.from(`ply\nformat ascii 1.0\nelement vertex ${zs.length}\nproperty double x\nproperty double y\nproperty double z\nend_header\n`+zs.map((z,i)=>`${i*2} 0 ${z}`).join('\n')+'\n')});
async function output(page:Page,id:string){const pending=page.waitForEvent('download');await page.locator(id).click();const chunks:Buffer[]=[];for await(const c of await(await pending).createReadStream())chunks.push(c);return Buffer.concat(chunks).toString();}
async function setup(page:Page,before=[0,0,0,0,0,0],after=[2,-3,.5,-.5,0]){
 await page.addInitScript(()=>localStorage.setItem('cloudanalyzer-autosave-enabled','off'));await page.goto('/');
 await page.locator('#file-input').setInputFiles([ply('before.ply',before),ply('after.ply',after)]);await expect(page.locator('#cloud-list > li')).toHaveCount(2);await expect(page.locator('#task')).toBeHidden();
 for(const id of ['volume-options','height-review-options','volume-area-options'])await page.locator(`#${id}`).evaluate((el:HTMLDetailsElement)=>el.open=true);
 await page.locator('#volume-cell').fill('2');
}
async function run(page:Page){await page.locator('#volume-run').click();await expect(page.locator('#task')).toBeHidden();await expect(page.locator('#volume-result')).toBeVisible();}
test('strict height screening exports exact full-difference quantities with missing and bounded support, leaving totals and display unchanged',async({page},info)=>{
 await setup(page);await run(page);const unscreened=JSON.parse(await output(page,'#volume-json'));expect(unscreened.result.heightReview).toBeUndefined();await expect(page.locator('#height-review-result')).toBeHidden();
 await page.locator('#height-review-enabled').check();await page.locator('#height-review-threshold').fill('.5');await run(page);
 const screened=JSON.parse(await output(page,'#volume-json'));expect(screened.result.added).toBe(10);expect(screened.result.removed).toBe(14);expect(screened.result.matchedCells).toBe(5);expect(screened.result.uncomparedCells).toBe(1);expect(screened.comparedCellCloud.points).toBe(5);
 const h=screened.result.heightReview;expect(h).toMatchObject({threshold:.5,riseCells:1,fallCells:1,withinCells:3,riseArea:4,fallArea:4,riseVolume:8,fallVolume:12,comparisonAvailable:true});expect(h.scope).toContain('not calibrated survey uncertainty');expect(screened.settings.heightThreshold).toBe(.5);
 for(const key of ['added','removed','addedArea','removedArea','matchedCells','totalCells'])expect(screened.result[key]).toBe(unscreened.result[key]);
 expect(await output(page,'#height-review-csv')).toContain('0.5,"m",true,1,1,3,4,4,8,12,-4');const qa=JSON.parse(await output(page,'#report-json'));expect(qa.sections['height-change-screening'].metrics.rise_volume.value).toBe(8);
 await page.locator('#height-review-result').screenshot({path:info.outputPath('screening.png')});
 await page.locator('#volume-area-enabled').check();for(const [id,value] of [['min-x','-.1'],['min-y','-.1'],['max-x','2.1'],['max-y','.1']])await page.locator(`#volume-area-${id}`).fill(value);
 await run(page);const bounded=JSON.parse(await output(page,'#volume-json'));expect(bounded.result).toMatchObject({matchedCells:2,totalCells:2});expect(bounded.reviewArea.outsideCells).toBe(4);expect(bounded.result.heightReview).toMatchObject({riseCells:1,fallCells:1,withinCells:0});
 await page.locator('#height-review-threshold').fill('0');await expect(page.locator('#height-review-result')).toBeHidden();expect(JSON.parse(await output(page,'#report-json')).sections['height-change-screening']).toBeUndefined();
});
test('screening uses f64 near a boundary, and empty support is unavailable rather than zero change',async({page})=>{
 await setup(page,[0,0,0],[1+Number.EPSILON,-1-Number.EPSILON,1]);await page.locator('#volume-cell').fill('2');await page.locator('#height-review-enabled').check();await page.locator('#height-review-threshold').fill('1');await run(page);
 const json=JSON.parse(await output(page,'#volume-json'));expect(json.result.heightReview).toMatchObject({riseCells:1,fallCells:1,withinCells:1});const csv=await output(page,'#volume-csv');expect(csv).not.toContain('1.0000000000000002');
 await page.locator('#volume-area-enabled').check();for(const [id,value] of [['min-x','10'],['min-y','10'],['max-x','11'],['max-y','11']])await page.locator(`#volume-area-${id}`).fill(value);await run(page);
 await expect(page.locator('#height-review-result')).toContainText('Unavailable');expect(JSON.parse(await output(page,'#volume-json')).result.heightReview.comparisonAvailable).toBe(false);expect(await output(page,'#height-review-csv')).toContain('1,"m",false,0,0,0,,,,,');expect(JSON.parse(await output(page,'#report-json')).sections['height-change-screening'].metrics.rise_volume).toBeUndefined();
});
test('invalid thresholds fail before work, zero is valid, and in-flight screening edits discard staged results',async({page})=>{
 await setup(page);await page.locator('#height-review-enabled').check();await page.locator('#height-review-threshold').fill('-1');await page.locator('#volume-run').click();await expect(page.locator('#status')).toContainText('nonnegative');await expect(page.locator('#cloud-list > li')).toHaveCount(2);
 await page.locator('#volume-areas-file').setInputFiles({name:'areas.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify({schema:'cloudanalyzer.named_review_areas.v1',areas:[{name:'All',bounds:[-1,-1,11,1]}]}))});
 await page.locator('#volume-areas-run').click();await expect(page.locator('#volume-areas-result')).toBeVisible();
 const batch=JSON.parse(await output(page,'#volume-areas-json'));expect(batch.settings.heightThreshold).toBeUndefined();expect(batch.areas[0].heightReview).toBeUndefined();await expect(page.locator('#height-review-result')).toBeHidden();
 await page.locator('#height-review-threshold').fill('0');await run(page);expect(JSON.parse(await output(page,'#volume-json')).result.heightReview).toMatchObject({riseCells:2,fallCells:2,withinCells:1,riseVolume:10,fallVolume:14});
 await page.evaluate(()=>{const original=Worker.prototype.postMessage;Worker.prototype.postMessage=function(message:any,...rest:any[]){if(message?.req?.kind==='volume'){document.getElementById('height-review-threshold')!.dispatchEvent(new Event('input'));}return(original as any).call(this,message,...rest);};});
 await page.locator('#volume-run').click();await expect(page.locator('#status')).toContainText('settings changed');await expect(page.locator('#height-review-result')).toBeHidden();await expect(page.locator('#volume-result')).toBeHidden();
});
