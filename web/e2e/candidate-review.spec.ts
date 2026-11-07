import { expect, test, type Page } from '@playwright/test';
function ply(lifted:boolean) {
  const points=Array.from({length:27},(_,i)=>{const patch=Math.floor(i/9),j=i%9;return [patch*10+(j%3)*.1,Math.floor(j/3)*.1,lifted?[0,.5,1][patch]:0];});
  return Buffer.from(`ply\nformat ascii 1.0\nelement vertex 27\nproperty float x\nproperty float y\nproperty float z\nend_header\n`+points.map(p=>p.join(' ')).join('\n')+'\n');
}
const input=(name:string,buffer:Buffer)=>({name,mimeType:'application/octet-stream',buffer});
async function output(page:Page,id:string){const waiting=page.waitForEvent('download');await page.locator(id).click();const parts:Buffer[]=[];for await(const c of await(await waiting).createReadStream())parts.push(c);return Buffer.concat(parts).toString();}
async function setup(page:Page){
  await page.addInitScript(()=>localStorage.setItem('cloudanalyzer-autosave-enabled','off'));
  await page.goto('/');await page.locator('#file-input').setInputFiles([input('truth.ply',ply(false)),input('map.ply',ply(true))]);
  await expect(page.locator('#cloud-list > li')).toHaveCount(2);await expect(page.locator('#task')).toBeHidden();
  await page.locator('#coverage-source').selectOption({label:'map.ply'});await page.locator('#coverage-reference').selectOption({label:'truth.ply'});
  await page.getByText('Locate distance discrepancies',{exact:true}).click();
  await page.locator('#regions-cell').fill('2');await page.locator('#regions-min').fill('5');await page.locator('#regions-threshold').fill('.3');
  await page.evaluate(()=>{const original=Worker.prototype.postMessage;(window as any).reviewC2cCalls=0;
    Worker.prototype.postMessage=function(message:any,...rest:any[]){if(message?.req?.kind==='c2c')(window as any).reviewC2cCalls++;return(original as any).call(this,message,...rest);};});
  await page.locator('#regions-both').click();await expect(page.locator('#status')).toContainText('ready in both directions');
}
test('human reviews stay independent across directions, survive cell inspection and export with exact context',async({page},info)=>{
  const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));await setup(page);
  const first=page.locator('#regions-list > li').first();
  await first.locator('select').selectOption('reviewed');await first.locator('textarea').fill('目視確認済み\n<img src=x onerror=alert(1)>');
  await page.locator('#regions-list > li').nth(1).locator('select').selectOption('deferred');
  await page.locator('#regions-list > li').nth(1).locator('textarea').fill('=Needs another survey, "east"');
  await expect(page.locator('#regions-review-summary')).toContainText('1 reviewed, 1 deferred, 0 unreviewed');
  await first.getByRole('button',{name:'Inspect cells',exact:true}).click();await expect(page.locator('#status')).toContainText('cell inspection ready');
  await expect(first.locator('textarea')).toHaveValue('目視確認済み\n<img src=x onerror=alert(1)>');
  await page.locator('#regions-fine-cell').fill('.5');await expect(first.locator('select')).toHaveValue('reviewed');
  await page.locator('#regions-direction').selectOption('reverse');await expect(first.locator('select')).toHaveValue('unreviewed');
  await first.locator('select').selectOption('deferred');await first.locator('textarea').fill('Reverse requires coverage review');
  await page.locator('#regions-direction').selectOption('forward');await expect(first.locator('select')).toHaveValue('reviewed');
  const json=JSON.parse(await output(page,'#regions-review-json'));
  expect(json.schema).toBe('cloudanalyzer.distance_candidate_reviews.v1');expect(json.comparisons).toHaveLength(2);
  const [f,r]=json.comparisons;expect(f.source.name).toBe('map.ply');expect(r.source.name).toBe('truth.ply');
  expect(f.analysis.threshold).toBe(.3);expect(f.analysis.cell).toBe(2);expect(f.analysis.minPoints).toBe(5);expect(f.analysis.overlap).toBeNull();
  expect(f.review.counts).toEqual({unreviewed:0,reviewed:1,deferred:1});expect(r.review.counts).toEqual({unreviewed:1,reviewed:0,deferred:1});
  expect(f.review.candidates[0].bounds[2]).toBe(1);expect(f.review.candidates[0].note).toBe('目視確認済み\n<img src=x onerror=alert(1)>');
  expect(Number.isFinite(Date.parse(f.review.candidates[0].updatedAt))).toBe(true);expect(f.source.loadedPoints).toBe(27);
  const original=JSON.parse(await output(page,'#regions-json'));expect(original.review).toEqual(f.review);
  const both=JSON.parse(await output(page,'#regions-both-json'));expect(both.reverse.review).toEqual(r.review);
  const csv=await output(page,'#regions-review-csv');expect(csv).toContain('forward,');expect(csv).toContain('reverse,');expect(csv).toContain('"\'=Needs another survey, ""east"""');
  const qa=JSON.parse(await output(page,'#report-json'));expect(qa.sections['distance-candidate-reviews:forward'].metrics.reviewed.value).toBe(1);
  expect(qa.sections['distance-candidate-reviews:reverse'].metrics.deferred.value).toBe(1);
  const html=await output(page,'#report-html');expect(html).toContain('&lt;img src=x onerror=alert(1)&gt;');expect(html).not.toContain('<img src=x');
  expect(await page.evaluate(()=>(window as any).reviewC2cCalls)).toBe(2);
  await page.locator('#regions-result').screenshot({path:info.outputPath('candidate-reviews.png')});expect(errors).toEqual([]);
});
test('reruns reset decisions even with cached distances; changed settings and input removal clear reports',async({page})=>{
  await setup(page);const first=page.locator('#regions-list > li').first();await first.locator('select').selectOption('reviewed');await first.locator('textarea').fill('Old snapshot decision');
  await page.locator('#regions-both').click();await expect(page.locator('#status')).toContainText('ready in both directions');
  await expect(first.locator('select')).toHaveValue('unreviewed');await expect(first.locator('textarea')).toHaveValue('');
  let receipt=JSON.parse(await output(page,'#regions-review-json'));expect(receipt.comparisons[0].review.candidates[0].updatedAt).toBeNull();expect(receipt.comparisons[0].reusedDistances).toBe(true);
  expect(await page.evaluate(()=>(window as any).reviewC2cCalls)).toBe(2);
  await first.locator('select').selectOption('deferred');await page.locator('#regions-threshold').fill('.6');await expect(page.locator('#regions-result')).toBeHidden();
  let qa=JSON.parse(await output(page,'#report-json'));expect(qa.sections['distance-candidate-reviews:forward']).toBeUndefined();expect(qa.sections['distance-candidate-reviews:reverse']).toBeUndefined();
  await page.locator('#regions-run').click();await expect(page.locator('#status')).toContainText('Distance regions ready');
  await expect(page.locator('#regions-list > li')).toHaveCount(1);await expect(first.locator('select')).toHaveValue('unreviewed');
  await first.locator('textarea').fill('New snapshot');await page.locator('#cloud-list > li').first().locator('button[title="Remove"]').click();
  await expect(page.locator('#regions-result')).toBeHidden();qa=JSON.parse(await output(page,'#report-json'));expect(qa.sections['distance-candidate-reviews:forward']).toBeUndefined();
});
test('empty candidate list does not imply all scene changes reviewed and invalid note edits cannot replace evidence',async({page})=>{
  await setup(page);const first=page.locator('#regions-list > li').first();await first.locator('textarea').fill('Valid note');
  await first.locator('textarea').evaluate((el:HTMLTextAreaElement)=>{el.value='x'.repeat(2001);el.dispatchEvent(new Event('input'));});
  await expect(page.locator('#status')).toContainText('2,000');expect(JSON.parse(await output(page,'#regions-review-json')).comparisons[0].review.candidates[0].note).toBe('Valid note');
  await page.locator('#regions-threshold').fill('10');await page.locator('#regions-run').click();await expect(page.locator('#status')).toContainText('Distance regions ready');
  const json=JSON.parse(await output(page,'#regions-review-json'));expect(json.comparisons[0].review.candidates).toEqual([]);
  expect(json.comparisons[0].review.counts).toEqual({unreviewed:0,reviewed:0,deferred:0});expect(json.scope).toContain('listed distance candidates only');
  expect((await output(page,'#regions-review-csv')).trim().split('\n')).toHaveLength(1);
});
test('unlisted eligible regions stay outside human review counts and exports',async({page})=>{
  await page.addInitScript(()=>localStorage.setItem('cloudanalyzer-autosave-enabled','off'));
  const isolated=(z:number)=>Buffer.from(`ply\nformat ascii 1.0\nelement vertex 53\nproperty float x\nproperty float y\nproperty float z\nend_header\n`+Array.from({length:53},(_,i)=>`${i*4} 0 ${z}`).join('\n')+'\n');
  await page.goto('/');await page.locator('#file-input').setInputFiles([input('truth.ply',isolated(0)),input('map.ply',isolated(1))]);
  await expect(page.locator('#cloud-list > li')).toHaveCount(2);await expect(page.locator('#task')).toBeHidden();
  await page.locator('#coverage-source').selectOption({label:'map.ply'});await page.locator('#coverage-reference').selectOption({label:'truth.ply'});
  await page.getByText('Locate distance discrepancies',{exact:true}).click();await page.locator('#regions-cell').fill('1');await page.locator('#regions-min').fill('1');
  await page.locator('#regions-run').click();await expect(page.locator('#status')).toContainText('Distance regions ready');
  await expect(page.locator('#regions-list > li')).toHaveCount(50);await expect(page.locator('#regions-review-summary')).toContainText('3 eligible regions unlisted');
  const packet=JSON.parse(await output(page,'#regions-review-json')).comparisons[0];expect(packet.review.counts.unreviewed).toBe(50);expect(packet.review.candidates).toHaveLength(50);expect(packet.analysis.unlistedEligiblePoints).toBe(3);
  const qa=JSON.parse(await output(page,'#report-json'));expect(qa.sections['distance-candidate-reviews:forward'].metrics.unlisted.value).toBe(3);
});
