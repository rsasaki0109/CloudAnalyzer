import { expect, test, type Page, type Download } from '@playwright/test';
import { readFile, writeFile } from 'node:fs/promises';
import { join } from 'node:path';
import { aiFixture } from '../tests/lidar-ai-fixture.mjs';
const input = (name: string, buffer: Buffer) => ({name,mimeType:'application/octet-stream',buffer});
const files = (f: ReturnType<typeof aiFixture>) => [input('scan.bin',f.raw),input('candidate-mask.u8',f.mask),input('report.json',Buffer.from(JSON.stringify(f.report)))];
async function bytes(d: Download) {const parts: Buffer[]=[];for await(const part of await d.createReadStream())parts.push(part);return Buffer.concat(parts);}
async function open(page: Page, f=aiFixture()) {
  await page.locator('#lidar-ai-files').setInputFiles(files(f));
  await expect(page.locator('#status')).toContainText('Opened AI scan comparison');
  return f;
}
async function download(page: Page, button: string) {
  const d=page.waitForEvent('download');await page.locator(button).click();return bytes(await d);
}
test('verified comparison highlights candidates, switches views and undoes/redoes as one operation', async ({page}) => {
  await page.goto('/');await page.locator('#lidar-ai-panel > summary').click();
  const f=await open(page);
  await expect(page.locator('#cloud-list > li')).toHaveCount(3);
  await expect(page.locator('#cloud-list input[title="Show / hide"]:checked')).toHaveCount(2);
  await expect(page.locator('#lidar-ai-summary')).toContainText('9 source points · 1 candidates · 8 retained');
  for(const [mode,label] of [['original','scan_AI_original'],['retained','scan_AI_retained'],['candidates','scan_AI_candidates']]) {
    await page.locator('#lidar-ai-view').selectOption(mode);
    await expect(page.locator('#cloud-list input[title="Show / hide"]:checked')).toHaveCount(1);
    await expect(page.locator('#cloud-list > li').filter({has:page.locator('input:checked')})).toContainText(label);
  }
  await page.locator('#lidar-ai-view').selectOption('highlight');
  await page.locator('#lidar-ai-focus').click();
  expect(await download(page,'#lidar-ai-download-retained')).toEqual(f.retained);
  expect(await download(page,'#lidar-ai-download-candidates')).toEqual(f.candidates);
  await page.locator('#lidar-ai-view').selectOption('original');
  await page.locator('#undo').click();await expect(page.locator('#cloud-list > li')).toHaveCount(0);
  await expect(page.locator('#lidar-ai-view')).toBeDisabled();
  await page.locator('#redo').click();await expect(page.locator('#cloud-list > li')).toHaveCount(3);
  await expect(page.locator('#lidar-ai-view')).toBeEnabled();
  await expect(page.locator('#cloud-list input[title="Show / hide"]:checked')).toHaveCount(1);
  await expect(page.locator('#cloud-list > li').filter({has:page.locator('input:checked')})).toContainText('scan_AI_original');
  await page.locator('#lidar-ai-view').selectOption('retained');
  expect(await download(page,'#lidar-ai-download-retained')).toEqual(f.retained);
});
test('bad source/mask leaves prior clouds, verified downloads and Undo intact', async ({page}) => {
  await page.goto('/');await page.locator('#lidar-ai-panel > summary').click();const original=await open(page);
  for(const mutate of [(f:any)=>f.raw[0]^=1,(f:any)=>{f.mask[0]=0;f.mask[2]=1;}]) {
    const f=aiFixture();mutate(f);await page.locator('#lidar-ai-files').setInputFiles(files(f));
    await expect(page.locator('#status')).toContainText('Could not open AI result');
    await expect(page.locator('#cloud-list > li')).toHaveCount(3);
    expect(await download(page,'#lidar-ai-download-retained')).toEqual(original.retained);
  }
  await page.locator('#undo').click();await expect(page.locator('#cloud-list > li')).toHaveCount(0);
});
test('zero candidates opens without parsing an empty point file, and removal disables stale comparison controls', async ({page}) => {
  await page.goto('/');await page.locator('#lidar-ai-panel > summary').click();
  await open(page,aiFixture({mask:Array(9).fill(0)}));
  await expect(page.locator('#cloud-list > li')).toHaveCount(2);
  expect((await download(page,'#lidar-ai-download-candidates')).length).toBe(0);
  await page.locator('#cloud-list > li').last().locator('button[title="Remove"]').click();
  await expect(page.locator('#lidar-ai-view')).toBeDisabled();
});
test('cancellation during verification adds no clouds and preserves an earlier Undo step', async ({page}) => {
  await page.goto('/');await page.locator('#lidar-ai-panel > summary').click();await open(page);
  await page.evaluate(() => {
    const original=crypto.subtle.digest.bind(crypto.subtle);
    crypto.subtle.digest = async (...args: Parameters<SubtleCrypto['digest']>) => {
      document.getElementById('task-cancel')!.click();
      return original(...args);
    };
  });
  await page.locator('#lidar-ai-files').setInputFiles(files(aiFixture()));
  await expect(page.locator('#status')).toContainText('Cancelled');
  await expect(page.locator('#cloud-list > li')).toHaveCount(3);
  await page.locator('#undo').click();await expect(page.locator('#cloud-list > li')).toHaveCount(0);
});

test('an insufficient Undo budget rejects staged clouds without changing the existing source', async ({page}) => {
  await page.goto('/');
  const f=aiFixture();await page.locator('#file-input').setInputFiles(input('existing.bin',f.raw));
  await expect(page.locator('#cloud-list > li')).toHaveCount(1);
  await page.locator('#memory-panel > summary').click();await page.locator('#memory-steps').fill('0');
  await page.locator('#memory-apply').click();await page.locator('#lidar-ai-panel > summary').click();
  await page.locator('#lidar-ai-files').setInputFiles(files(f));
  await expect(page.locator('#status')).toContainText('Undo budget');
  await expect(page.locator('#cloud-list > li')).toHaveCount(1);
  await expect(page.locator('#cloud-list')).toContainText('existing.bin');
  await expect(page.locator('#cloud-list input[title="Show / hide"]')).toBeChecked();
  await page.locator('#memory-steps').fill('20');await page.locator('#memory-apply').click();
  await open(page);await expect(page.locator('#cloud-list > li')).toHaveCount(4);
  await page.locator('#undo').click();await expect(page.locator('#cloud-list > li')).toHaveCount(1);
});

test('actual native GPU result keeps exact downloads and source hash through browser review and Undo', async ({page},info) => {
  const scan=process.env.LIDAR_AI_SCAN, result=process.env.LIDAR_AI_RESULT;
  test.skip(!scan||!result,'Opt in with the local native scan/result; no model or dataset downloads in CI');
  test.setTimeout(120000);
  const raw=await readFile(scan!), mask=await readFile(join(result!,'candidate-mask.u8'));
  const reportBytes=await readFile(join(result!,'report.json')), report=JSON.parse(reportBytes.toString());
  const errors: string[]=[];page.on('pageerror',e=>errors.push(e.message));
  await page.setViewportSize({width:1600,height:1000});
  await page.goto('/');await page.locator('#lidar-ai-panel > summary').click();
  await page.locator('#lidar-ai-files').setInputFiles([input('000000.bin',raw),input('candidate-mask.u8',mask),input('report.json',reportBytes)]);
  await expect(page.locator('#status')).toContainText('Opened AI scan comparison');
  await expect(page.locator('#cloud-list > li')).toHaveCount(3);
  const kept=await download(page,'#lidar-ai-download-retained'), candidates=await download(page,'#lidar-ai-download-candidates');
  expect(kept).toEqual(await readFile(join(result!,'retained.bin')));
  expect(candidates).toEqual(await readFile(join(result!,'candidates.bin')));
  await page.getByRole('button',{name:'Top',exact:true}).click();
  await page.locator('#lidar-ai-focus').click();
  await page.locator('#lidar-ai-panel > summary').scrollIntoViewIfNeeded();
  await page.screenshot({path:info.outputPath('ai-scan-highlight.png')});
  await page.locator('#lidar-ai-view').selectOption('retained');
  await page.screenshot({path:info.outputPath('ai-scan-retained.png')});
  await page.locator('#undo').click();await expect(page.locator('#cloud-list > li')).toHaveCount(0);
  await page.locator('#redo').click();await expect(page.locator('#cloud-list > li')).toHaveCount(3);
  expect(await download(page,'#lidar-ai-download-retained')).toEqual(kept);
  expect(await readFile(scan!)).toEqual(raw);expect(errors).toEqual([]);
  await writeFile(info.outputPath('review-receipt.json'),JSON.stringify({inputSha256:report.input_sha256,
    points:report.projection.points,candidates:report.selection.removed,retained:report.selection.retained,
    exactNativeDownloads:true,sourceUnchanged:true,undoRedo:true,browserErrors:errors,
    accuracyMeasured:false},null,2));
});
