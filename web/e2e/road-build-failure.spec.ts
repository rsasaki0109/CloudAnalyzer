import { expect, test, type Page } from '@playwright/test';
import { readFileSync } from 'node:fs';

// This NCLT-derived reference exercises a returning curve on a synthetic flat
// source. The original fixture's attribution remains alongside the Rust data.
const fixture = JSON.parse(readFileSync(new URL('../../rust/crates/ca-core/tests/fixtures/vector-map/nclt-returning-section.json', import.meta.url), 'utf8'));
const reference: number[][] = fixture.reference.map(([x, y]: number[]) => [500000 + x, 4000000 + y, 13]);

async function savedMap(page: Page) {
  const ready = page.waitForEvent('download', d => d.suggestedFilename() === 'project.cloudanalyzer.json');
  await page.locator('#project-save').click();
  const chunks: Buffer[] = [];
  for await (const chunk of await (await ready).createReadStream()) chunks.push(chunk);
  return JSON.parse(JSON.parse(Buffer.concat(chunks).toString()).vectorMap);
}

async function sources(page: Page) {
  await page.goto('/');
  const xs = reference.map(p => p[0]), ys = reference.map(p => p[1]);
  const points: number[][] = [];
  for (let x = Math.floor(Math.min(...xs)) - 12; x <= Math.max(...xs) + 12; x += 0.5)
    for (let y = Math.floor(Math.min(...ys)) - 12; y <= Math.max(...ys) + 12; y += 0.5) points.push([x, y, 12]);
  const header = `ply\nformat binary_little_endian 1.0\nelement vertex ${points.length}\nproperty double x\nproperty double y\nproperty double z\nend_header\n`;
  const data = Buffer.alloc(points.length * 24);
  points.forEach((p, i) => p.forEach((v, a) => data.writeDoubleLE(v, i * 24 + a * 8)));
  await page.locator('#file-input').setInputFiles([
    {name: 'survey.ply', mimeType: 'application/octet-stream', buffer: Buffer.concat([Buffer.from(header), data])},
    {name: 'return.csv', mimeType: 'text/csv', buffer: Buffer.from('timestamp,x,y,z\n' + reference.map((p, i) => [i, ...p].join(',')).join('\n'))},
  ]);
  await expect(page.locator('#status')).toContainText(`trajectory of ${reference.length} poses`);
  await page.getByText('Build from a trajectory', {exact: true}).click();
  await page.locator('#vm-discover-after-build').uncheck();
  await page.getByText('Road options', {exact: true}).click();
  await page.locator('#vm-fit-boundaries').uncheck();
}

test('rejected road keeps the existing map and provides a bounded preview, settings and retry', async ({page}, info) => {
  const errors: string[] = [];
  page.on('pageerror', e => errors.push(e.message));
  await sources(page);
  const map = {format: 'vectormap-ir', version: 1, metadata: {},
    lanes: [{id: 3, kind: 'driving', left: 1, right: 2, speed_limit: {kmh: 40}}],
    boundaries: [
      {id: 1, kind: {type: 'virtual'}, geometry: [[500100, 4000001.75, 12], [500110, 4000001.75, 12]]},
      {id: 2, kind: {type: 'virtual'}, geometry: [[500100, 3999998.25, 12], [500110, 3999998.25, 12]]},
    ]};
  await page.locator('#vm-file').setInputFiles({name: 'existing.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(map))});
  await expect(page.locator('#status')).toContainText('Opened existing.json');
  const before = await savedMap(page);
  await page.locator('#vm-build').click();
  await expect(page.locator('#status')).toContainText('ambiguous travel directions');
  await expect(page.locator('#status')).not.toContainText('"diagnostic"');
  await expect(page.locator('#vm-build-failure')).toBeVisible();
  await expect(page.locator('#vm-status')).toContainText('1 lane');
  expect(await savedMap(page)).toEqual(before);
  await expect(page.locator('#vm-forward')).toHaveValue('1');
  await expect(page.locator('#vm-backward')).toHaveValue('1');
  await page.locator('#vm-failure-focus').click();
  await page.screenshot({path: info.outputPath('survey-frame-preview.png')});
  await page.locator('#vm-failure-options').click();
  await expect(page.locator('#vm-forward')).toBeFocused();
  await page.locator('#vm-backward').fill('0');
  await expect(page.locator('#vm-failure-stale')).toBeVisible();
  await expect(page.locator('#vm-failure-settings')).toContainText('1 forward / 1 backward');
  await page.locator('#vm-failure-retry').click();
  await expect(page.locator('#status')).toContainText('Draft roads added');
  await expect(page.locator('#vm-build-failure')).toBeHidden();
  // A failed draft adds no undo entry. One undo removes only the successful retry.
  await page.locator('#vm-undo').click();
  await expect(page.locator('#vm-status')).toContainText('1 lane');
  expect(await savedMap(page)).toEqual(before);
  await page.locator('#vm-review-next').click();
  await expect(page.locator('#vm-lane-speed')).toHaveValue('40');
  expect(errors).toEqual([]);
});

test('changed settings during a build do not resurrect a stale failure preview', async ({page}) => {
  await sources(page);
  await page.evaluate(() => {
    (document.getElementById('vm-build') as HTMLButtonElement).click();
    const width = document.getElementById('vm-width') as HTMLInputElement;
    width.value = '3'; width.dispatchEvent(new Event('input', {bubbles: true}));
  });
  await expect(page.locator('#status')).toContainText('ambiguous travel directions');
  await expect(page.locator('#vm-build-failure')).toBeHidden();
  await page.locator('#vm-build').click();
  await expect(page.locator('#status')).toContainText('Boundary search margin must be between 0 and 1.35 m');
  await expect(page.locator('#vm-search-margin')).toBeFocused();
  await expect(page.locator('#vm-search-margin')).toHaveValue('1.5');
  await page.locator('#vm-search-margin').fill('1.3');
  await page.locator('#vm-backward').fill('0');
  await page.locator('#vm-build').click();
  await expect(page.locator('#status')).toContainText('Draft roads added');
  await expect(page.locator('#vm-build-failure')).toBeHidden();
  await page.locator('#vm-clear').click();
  await page.locator('#vm-width').fill('3.5');
  await page.locator('#vm-search-margin').fill('1.5');
  await page.locator('#vm-backward').fill('1');
  await page.locator('#vm-build').click();
  await expect(page.locator('#vm-build-failure')).toBeVisible();
  await expect(page.locator('#vm-failure-settings')).toContainText('3.5 m per lane');
  await page.locator('#vm-failure-dismiss').click();
  await expect(page.locator('#vm-build-failure')).toBeHidden();
  await page.locator('#vm-build').click();
  await expect(page.locator('#vm-build-failure')).toBeVisible();
  await page.locator('#vm-road').click();
  await expect(page.locator('#vm-build-failure')).toBeHidden();
  await page.locator('#vm-build').click();
  await expect(page.locator('#vm-build-failure')).toBeVisible();
  await page.locator('#file-input').setInputFiles({name: 'straight.csv', mimeType: 'text/csv',
    buffer: Buffer.from('timestamp,x,y,z\n0,500000,4000000,13\n1,500010,4000000,13\n')});
  await expect(page.locator('#status')).toContainText('trajectory of 2 poses');
  await expect(page.locator('#vm-build-failure')).toBeHidden();
  // Plain validation failures retain the existing error message and have no preview.
  await page.locator('#vm-width').fill('0');
  await page.locator('#vm-build').click();
  await expect(page.locator('#status')).toContainText('Could not build draft roads');
  await expect(page.locator('#vm-build')).toBeEnabled();
  await expect(page.locator('#vm-build-failure')).toBeHidden();
});
