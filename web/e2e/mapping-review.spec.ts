import { expect, test } from '@playwright/test';
import { packageFixture, packagePreview, previewFixture, cloud, map } from '../tests/mapping-review-fixture.mjs';
const input=(buffer:Buffer)=>({name:'generated-review.zip',mimeType:'application/zip',buffer});

test('opens the verified pair, focuses saved source failures and switches all four protocols', async ({page})=>{
  await page.goto('/');
  await page.locator('#mapping-review-file').setInputFiles(input(packageFixture()));
  await expect(page.locator('#status')).toContainText('Opened generated maps');
  await expect(page.locator('#cloud-list')).toContainText('map.ply');
  await expect(page.locator('#vm-status')).toContainText('1 lane');
  await expect(page.locator('#mapping-review-summary')).toContainText('10.0 / 20.0 m; requested extent unmet');
  await expect(page.locator('#vm-quality-report')).toContainText('Saved low quantile / editable IR');
  await expect(page.locator('#vm-quality-lanes')).toContainText('Lane 7');
  await page.locator('#vm-quality-next').click();
  await expect(page.locator('#vm-quality-problem-detail')).toContainText('Lane 7, left boundary');
  await page.locator('#vm-quality-focus').click();
  await page.locator('#mapping-review-audit').selectOption('1');
  await expect(page.locator('#vm-quality-report')).toContainText('low quantile / reopened OSM');
  await page.locator('#mapping-review-audit').selectOption('2');
  await expect(page.locator('#vm-quality-report')).toContainText('ground consensus / editable IR');
  await expect(page.locator('#vm-quality-lanes button')).toHaveCount(0);
  await page.locator('#mapping-review-audit').selectOption('3');
  await expect(page.locator('#vm-quality-report')).toContainText('ground consensus / reopened OSM');
  await page.locator('#vm-review-next').click();
  await page.locator('#vm-lane-speed').fill('30');
  await page.locator('#vm-lane-apply').click();
  await expect(page.locator('#mapping-review-audit')).toBeDisabled();
  await expect(page.locator('#mapping-review-state')).toContainText('HD map changed');
  await expect(page.locator('#vm-quality-report')).toContainText('has not been checked');
});

test('invalid evidence preserves existing map, cloud and saved lane review', async ({page})=>{
  await page.goto('/');
  await page.locator('#file-input').setInputFiles({name:'original.ply',mimeType:'application/octet-stream',buffer:cloud});
  await expect(page.locator('#status')).toContainText('Loaded original.ply');
  await page.locator('#vm-file').setInputFiles({name:'original.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(map))});
  await page.locator('#vm-review-next').click();
  await page.locator('#vm-review-notes').fill('Preserve this review');
  await page.locator('#vm-review-save').click();
  const changed=packageFixture(d=>d.entries.find(([name]:[string,Buffer])=>name===d.manifest.roles.graph)[1]=Buffer.from('GRAPH'));
  await page.locator('#mapping-review-file').setInputFiles(input(changed));
  await expect(page.locator('#status')).toContainText('hash differs');
  await expect(page.locator('#cloud-list > li')).toHaveCount(1);
  await expect(page.locator('#cloud-list')).toContainText('original.ply');
  await expect(page.locator('#vm-review-notes')).toHaveValue('Preserve this review');
  await expect(page.locator('#vm-status')).toContainText('1 lane');
});

test('removing the imported point source invalidates saved audits', async ({page})=>{
  await page.goto('/');
  await page.locator('#mapping-review-file').setInputFiles(input(packageFixture()));
  await expect(page.locator('#status')).toContainText('Opened generated maps');
  await page.locator('#cloud-list button[title="Remove"]').click();
  await expect(page.locator('#mapping-review-audit')).toBeDisabled();
  await expect(page.locator('#mapping-review-state')).toContainText('removed');
});

test('preview shows full-source saved audits, blocks new source checks and persists the distinction in a project', async ({page})=>{
  await page.goto('/');
  await page.locator('#mapping-review-file').setInputFiles(input(packagePreview()));
  await expect(page.locator('#status')).toContainText('Opened generated maps');
  await expect(page.locator('#mapping-review-summary')).toContainText('2 loaded point records (display preview from 4)');
  await expect(page.locator('#mapping-review-state')).toContainText('original 4 points');
  await expect(page.locator('#vm-quality-check')).toBeDisabled();
  await expect(page.locator('#vm-quality-lanes button')).toHaveCount(1);
  await page.locator('#mapping-review-audit').selectOption('2');
  await expect(page.locator('#vm-quality-report')).toContainText('ground consensus / editable IR');
  await expect(page.locator('#vm-quality-lanes button')).toHaveCount(0);
  const download=page.waitForEvent('download',d=>d.suggestedFilename()==='project.cloudanalyzer.json');
  await page.locator('#project-save').click();
  const chunks:Buffer[]=[];for await(const part of await (await download).createReadStream()) chunks.push(part);
  const project=Buffer.concat(chunks);expect(JSON.parse(project.toString()).session.clouds[0].displayPreview).toBe(true);
  await page.reload();
  await page.locator('#file-input').setInputFiles({name:'project.cloudanalyzer.json',mimeType:'application/json',buffer:project});
  await expect(page.locator('#status')).toContainText('open matching source files');
  const preview=previewFixture().entries.find(([name]:[string,Buffer])=>name==='files/display-preview.ply')[1];
  await page.locator('#file-input').setInputFiles({name:'display-preview.ply',mimeType:'application/octet-stream',buffer:preview});
  await expect(page.locator('#status')).toContainText('Project restored');
  await expect(page.locator('#vm-quality-check')).toBeDisabled();
  await page.locator('#file-input').setInputFiles({name:'full-source.ply',mimeType:'application/octet-stream',buffer:cloud});
  await expect(page.locator('#status')).toContainText('Loaded full-source.ply');
  await page.locator('#vm-quality summary').click();
  const option=await page.locator('#vm-quality-cloud option').evaluateAll(nodes=>nodes.map(n=>({text:n.textContent,value:(n as HTMLOptionElement).value})).find(n=>n.text?.includes('full-source')));
  await page.locator('#vm-quality-cloud').selectOption(option!.value);
  await expect(page.locator('#vm-quality-check')).toBeEnabled();
});
