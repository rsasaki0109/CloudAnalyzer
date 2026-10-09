import { expect, test, type Download } from '@playwright/test';
import { readProjectSnapshot } from '../src/project-snapshot';
import { writeFile } from 'node:fs/promises';

async function bytes(download:Download) {
  const parts:Buffer[]=[];for await(const part of await download.createReadStream()) parts.push(part);
  return Buffer.concat(parts);
}
test('NCLT full point map and processed result reopen together from one workspace snapshot',async({page})=>{
  const bundle=process.env.CA_REVIEW_NCLT_BUNDLE;
  test.skip(!bundle,'Set CA_REVIEW_NCLT_BUNDLE to the exact full generated-map review ZIP');
  test.setTimeout(120000);
  await page.goto('/');await page.locator('#mapping-review-file').setInputFiles(bundle!);
  await expect(page.locator('#status')).toContainText('Opened generated maps');
  await page.locator('#filter-op').selectOption('random');await page.locator('#filter-percent').fill('1');await page.locator('#filter-run').click();
  await expect(page.locator('#cloud-list > li')).toHaveCount(2);
  await expect(page.locator('#cloud-list > li').last().locator('.meta')).toContainText('6,463 points');
  const event=page.waitForEvent('download',d=>d.suggestedFilename()==='project.cloudanalyzer.zip');await page.locator('#project-snapshot').click();
  const zip=await bytes(await event);
  await writeFile('/tmp/nclt-workspace-snapshot.zip',zip);
  const files=await readProjectSnapshot(new File([zip],'project.cloudanalyzer.zip'),new AbortController().signal);
  expect(Buffer.from(await files[1].arrayBuffer()).includes(Buffer.from('property float correction\n'))).toBe(true);
  const project=JSON.parse(await files[0].text()), lanes=JSON.parse(project.vectorMap).lanes.length;
  await page.reload();await page.locator('#file-input').setInputFiles({name:'project.cloudanalyzer.zip',mimeType:'application/zip',buffer:zip});
  await expect(page.locator('#status')).toContainText('Project restored');
  await expect(page.locator('#cloud-list > li')).toHaveCount(2);
  await expect(page.locator('#cloud-list > li').first().locator('.meta')).toContainText('646,309 points');
  await expect(page.locator('#cloud-list > li').last().locator('.meta')).toContainText('6,463 points');
  await expect(page.locator('#cloud-list > li').first().locator('input[type=checkbox]')).not.toBeChecked();
  await expect(page.locator('#cloud-list > li').last().locator('input[type=checkbox]')).toBeChecked();
  await expect(page.locator('#vm-status')).toContainText(`${lanes} lanes`);
  for(const index of [0,1]) {
    const li=page.locator('#cloud-list > li').nth(index), exported=page.waitForEvent('download');
    await li.locator('button.icon').click();await li.locator('.save-formats').getByRole('button',{name:'PLY',exact:true}).click();
    expect((await bytes(await exported)).equals(Buffer.from(await files[index+1].arrayBuffer()))).toBe(true);
  }
  await page.screenshot({path:'/tmp/nclt-workspace-snapshot-restored.png'});
});
