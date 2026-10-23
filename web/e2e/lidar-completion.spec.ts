import { expect, test, type Page, type Download } from '@playwright/test';
import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { completionFixture } from '../tests/lidar-completion-fixture.mjs';
const input = (name: string, buffer: Buffer) => ({name,mimeType:'application/octet-stream',buffer});
const files = async (f: ReturnType<typeof completionFixture>) => Promise.all(f.files().map(async (file: File)=>input(file.name,Buffer.from(await file.arrayBuffer()))));
async function bytes(d: Download) {const parts: Buffer[]=[];for await(const part of await d.createReadStream())parts.push(part);return Buffer.concat(parts);}
async function download(page: Page, button: string) {
  const d=page.waitForEvent('download');await page.locator(button).click();return bytes(await d);
}
async function open(page: Page, f=completionFixture()) {
  await page.locator('#lidar-ai-files').setInputFiles(await files(f));
  await expect(page.locator('#status')).toContainText('Opened AI scan comparison');return f;
}
test('completion separates baseline/added candidates, switches all six views and restores added-only after Undo',async({page})=>{
  await page.goto('/');await page.locator('#lidar-ai-panel > summary').click();const f=await open(page);
  await expect(page.locator('#cloud-list > li')).toHaveCount(4);
  await expect(page.locator('#lidar-ai-summary')).toContainText('2 baseline candidates + 1 added');
  await expect(page.locator('#lidar-ai-state')).toContainText('Full score/seed/veto verification requires');
  for(const [mode,count] of [['highlight',3],['original',1],['retained',1],['candidates',2],['baseline-candidates',1],['added',1]] as const) {
    await page.locator('#lidar-ai-view').selectOption(mode);
    await expect(page.locator('#cloud-list input[title="Show / hide"]:checked')).toHaveCount(count);
  }
  await expect(page.locator('#cloud-list > li').filter({has:page.locator('input:checked')})).toContainText('scan_AI_added');
  await page.locator('#lidar-ai-focus').click();
  expect(await download(page,'#lidar-ai-download-added')).toEqual(f.addedRecords);
  expect(await download(page,'#lidar-ai-download-candidates')).toEqual(f.candidates);
  expect(await download(page,'#lidar-ai-download-retained')).toEqual(f.retained);
  await page.locator('#undo').click();await expect(page.locator('#cloud-list > li')).toHaveCount(0);
  await page.locator('#redo').click();await expect(page.locator('#cloud-list > li')).toHaveCount(4);
  await expect(page.locator('#cloud-list input[title="Show / hide"]:checked')).toHaveCount(1);
  await expect(page.locator('#cloud-list > li').filter({has:page.locator('input:checked')})).toContainText('scan_AI_added');
});
test('bad completion and cancellation preserve prior comparison, downloads and Undo',async({page})=>{
  await page.goto('/');await page.locator('#lidar-ai-panel > summary').click();const original=await open(page);
  const bad=completionFixture();bad.report.completed_mask_sha256='0'.repeat(64);
  await page.locator('#lidar-ai-files').setInputFiles(await files(bad));
  await expect(page.locator('#status')).toContainText('Could not open AI result');
  await expect(page.locator('#cloud-list > li')).toHaveCount(4);
  expect(await download(page,'#lidar-ai-download-added')).toEqual(original.addedRecords);
  await page.evaluate(()=>{
    const digest=crypto.subtle.digest.bind(crypto.subtle);
    crypto.subtle.digest=async(...args:Parameters<SubtleCrypto['digest']>)=>{document.getElementById('task-cancel')!.click();return digest(...args);};
  });
  await page.locator('#lidar-ai-files').setInputFiles(await files(completionFixture()));
  await expect(page.locator('#status')).toContainText('Cancelled');
  await expect(page.locator('#cloud-list > li')).toHaveCount(4);
  await page.locator('#undo').click();await expect(page.locator('#cloud-list > li')).toHaveCount(0);
});
test('zero additions support an empty added-only view/download with no focus target',async({page})=>{
  await page.goto('/');await page.locator('#lidar-ai-panel > summary').click();
  await open(page,completionFixture({added:[0,0,0,0]}));
  await expect(page.locator('#cloud-list > li')).toHaveCount(3);
  await page.locator('#lidar-ai-view').selectOption('added');
  await expect(page.locator('#cloud-list input[title="Show / hide"]:checked')).toHaveCount(0);
  await expect(page.locator('#lidar-ai-focus')).toBeDisabled();
  expect((await download(page,'#lidar-ai-download-added')).length).toBe(0);
  const baseline=completionFixture().baseline;
  const nativeFiles=await Promise.all(baseline.files().map(async(file:File)=>input(file.name,Buffer.from(await file.arrayBuffer()))));
  await page.locator('#lidar-ai-files').setInputFiles(nativeFiles);
  await expect(page.locator('#lidar-ai-summary')).toContainText('4 source points · 2 candidates · 2 retained');
  await expect(page.locator('#lidar-ai-download-added')).toBeHidden();
  await expect(page.locator('#lidar-ai-view option[value="added"]')).toHaveJSProperty('disabled',true);
  await expect(page.locator('#lidar-ai-view')).toHaveValue('highlight');
});
test('actual frozen completion downloads preserve exact added/completed records and source through Undo',async({page},info)=>{
  const scan=process.env.LIDAR_COMPLETION_SCAN,base=process.env.LIDAR_COMPLETION_BASE,result=process.env.LIDAR_COMPLETION_RESULT;
  test.skip(!scan||!base||!result,'Opt in with local frozen files; no dataset/model downloads');test.setTimeout(120000);
  const raw=await readFile(scan!),nativeReport=await readFile(join(base!,'report.json'));
  const metaBytes=await readFile(join(result!,'completion-report.json')),meta=JSON.parse(metaBytes.toString());
  const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
  await page.setViewportSize({width:1600,height:1000});await page.goto('/');await page.locator('#lidar-ai-panel > summary').click();
  await page.locator('#lidar-ai-files').setInputFiles([input('scan.bin',raw),input('report.json',nativeReport),
    input('candidate-mask.u8',await readFile(join(base!,'candidate-mask.u8'))),input('completion-report.json',metaBytes),
    input('completed-mask.u8',await readFile(join(result!,'completed-mask.u8'))),input('added-mask.u8',await readFile(join(result!,'added-mask.u8')))]);
  await expect(page.locator('#status')).toContainText('Opened AI scan comparison');
  const addedMask=await readFile(join(result!,'added-mask.u8'));
  const expectedAdded=Buffer.concat([...addedMask].flatMap((v,i)=>v?[raw.subarray(i*16,i*16+16)]:[]));
  expect(await download(page,'#lidar-ai-download-added')).toEqual(expectedAdded);
  expect(await download(page,'#lidar-ai-download-candidates')).toEqual(await readFile(join(result!,'candidates.bin')));
  expect(await download(page,'#lidar-ai-download-retained')).toEqual(await readFile(join(result!,'retained.bin')));
  await page.getByRole('button',{name:'Top',exact:true}).click();
  await page.locator('#lidar-ai-focus').click();
  await page.locator('#lidar-ai-panel').evaluate(el=>{
    const sidebar=document.getElementById('sidebar')!;
    sidebar.scrollTop+=el.getBoundingClientRect().top-sidebar.getBoundingClientRect().top-8;
  });
  await page.screenshot({path:info.outputPath('completion-highlight.png')});
  await page.locator('#lidar-ai-view').selectOption('added');await page.locator('#lidar-ai-focus').click();
  await page.locator('#lidar-ai-panel').evaluate(el=>{
    const sidebar=document.getElementById('sidebar')!;
    sidebar.scrollTop+=el.getBoundingClientRect().top-sidebar.getBoundingClientRect().top-8;
  });
  await page.screenshot({path:info.outputPath('completion-added.png')});
  await page.locator('#undo').click();await expect(page.locator('#cloud-list > li')).toHaveCount(0);
  await page.locator('#redo').click();await expect(page.locator('#cloud-list > li')).toHaveCount(4);
  expect(await download(page,'#lidar-ai-download-added')).toEqual(expectedAdded);
  expect(await readFile(scan!)).toEqual(raw);expect(errors).toEqual([]);
  await writeFile(info.outputPath('completion-review-receipt.json'),JSON.stringify({sourceSha256:meta.input_sha256,
    baseline:meta.counts.original_removed,added:meta.counts.added,completed:meta.counts.total_removed,
    exactDownloads:true,sourceUnchanged:true,undoRedoAddedView:true,browserErrors:errors,
    fullProbabilityRuleRerun:false,semanticAccuracyMeasuredByBrowser:false},null,2));
});
