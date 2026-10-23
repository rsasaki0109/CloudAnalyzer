import { test, expect, type Page } from '@playwright/test';

const ply = (name: string, zs: number[]) => ({ name, mimeType: 'application/octet-stream', buffer: Buffer.from(
  `ply\nformat ascii 1.0\nelement vertex ${zs.length}\nproperty double x\nproperty double y\nproperty double z\nend_header\n` +
  zs.map((z, i) => `${1000000 + 2 * i} 2000000 ${z}`).join('\n') + '\n') });
async function output(page: Page, id: string) {
  const pending = page.waitForEvent('download'); await page.locator(id).click(); const chunks: Buffer[] = [];
  for await (const chunk of await (await pending).createReadStream()) chunks.push(chunk);
  return Buffer.concat(chunks).toString();
}
async function setup(page: Page) {
  await page.addInitScript(() => localStorage.setItem('cloudanalyzer-autosave-enabled', 'off')); await page.goto('/');
  await page.locator('#file-input').setInputFiles([ply('before.ply', [0, 0, 0]), ply('after.ply', [5, 1, 100])]);
  await expect(page.locator('#cloud-list > li')).toHaveCount(2); await expect(page.locator('#task')).toBeHidden();
  for (const id of ['volume-options', 'height-review-options', 'volume-area-options', 'volume-visits-options'])
    await page.locator(`#${id}`).evaluate((el: HTMLDetailsElement) => el.open = true);
  await page.locator('#volume-cell').fill('2'); await page.locator('#height-review-enabled').check();
  await page.locator('#height-review-threshold').fill('1');
}
async function inspect(page: Page) {
  await page.locator('#volume-run').click(); await expect(page.locator('#volume-result')).toBeVisible();
  await page.locator('#height-review-inspect').click();
}
async function section(page: Page) {
  await page.getByRole('button', { name: 'Compare point section for screened cell 1', exact: true }).click();
  await expect(page.locator('#regions-section-plot')).toBeVisible();
  return JSON.parse(await output(page, '#regions-section-json'));
}
test('height cell section uses its hidden original pair and exports band points, provenance and separate QA without altering volume', async ({ page }, info) => {
  const errors: string[] = []; page.on('pageerror', e => errors.push(e.message)); await setup(page); await inspect(page);
  const original = await output(page, '#volume-json');
  await page.locator('#file-input').setInputFiles(ply('unrelated.ply', [900, 900, 900]));
  await expect(page.locator('#cloud-list > li')).toHaveCount(4); // includes the existing volume cell cloud
  await page.locator('#cloud-list > li').first().locator('input[type=checkbox]').uncheck();
  const result = await section(page);
  expect(result.schema).toBe('cloudanalyzer.height_cell_section.v1');
  expect(result.before).toMatchObject({ name: 'before.ply', selectedPoints: 3, zMax: 0, keepEvery: 1, preview: false });
  expect(result.after).toMatchObject({ name: 'after.ply', selectedPoints: 3, zMax: 100 });
  expect(result.parent.cell).toMatchObject({ index: 2, x: 1000004, y: 2000000, difference: 100 });
  expect(result.scope).toContain('f32'); expect(result.scope).toContain('not clipped');
  const rows = (await output(page, '#regions-section-csv')).trim().split('\n').slice(1).map(r => r.split(','));
  expect(rows).toHaveLength(6); expect(rows.filter(r => r[0] === 'after').map(r => Number(r[5]))).toEqual([5, 1, 100]);
  expect(rows.filter(r => r[0] === 'before').every(r => Number(r[5]) === 0)).toBe(true);
  const qa = JSON.parse(await output(page, '#report-json')).sections;
  expect(qa['height-cell-section'].metrics.beforePoints.value).toBe(3); expect(qa['distance-section']).toBeUndefined();
  expect(await output(page, '#volume-json')).toBe(original);
  await page.locator('#regions-section-width').evaluate((el: HTMLInputElement) => { el.value = '4'; el.dispatchEvent(new Event('input')); }); // independent distance controls
  await expect(page.locator('#regions-section-plot')).toBeVisible();
  await page.locator('#height-section-options').evaluate((el: HTMLDetailsElement) => el.open = true);
  await page.locator('#height-section-axis').selectOption('y'); await expect(page.locator('#regions-section-plot')).toBeHidden();
  await expect(page.locator('#height-cell-inspection')).toBeVisible(); await expect(page.locator('#volume-result')).toBeVisible();
  const narrow = await section(page); expect(narrow.settings.axis).toBe('y'); expect(narrow.after.selectedPoints).toBe(1);
  await page.locator('#regions-section-plot').screenshot({ path: info.outputPath('height-cell-section.png') });
  expect(await page.locator('#regions-section-canvas').evaluate(el => el.getBoundingClientRect().height)).toBeGreaterThan(40);
  await page.locator('#height-cell-close').click(); await expect(page.locator('#regions-section-plot')).toBeHidden();
  expect(JSON.parse(await output(page, '#report-json')).sections['height-cell-section']).toBeUndefined(); expect(errors).toEqual([]);
});
test('direct interval selects adjacent visits and retains raw band points outside parent bounds and the common mask', async ({ page }) => {
  await setup(page); await page.locator('#file-input').setInputFiles(ply('visit-b.ply', [3, 4]));
  await expect(page.locator('#cloud-list > li')).toHaveCount(3);
  await page.locator('#volume-cell').fill('2');
  await page.locator('#height-review-enabled').uncheck(); await page.locator('#height-review-batch-enabled').check();
  const bounds = [1000001.9, 1999999.9, 1000002.1, 2000000.1];
  await page.locator('#volume-areas-file').setInputFiles({ name: 'areas.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify({ schema: 'cloudanalyzer.named_review_areas.v1', areas: [{ name: 'Small', bounds }] })) });
  const ids = await page.locator('#volume-visits option').evaluateAll((os: HTMLOptionElement[]) => os.map(o => o.value));
  await page.locator('#volume-visits').selectOption(ids.slice(1)); await page.locator('#volume-visits-support').selectOption('common_observed');
  await page.locator('#volume-visits-intervals').check(); await page.locator('#volume-visits-run').click();
  await expect(page.locator('#volume-visits-result')).toBeVisible();
  await page.getByRole('button', { name: 'Inspect screened cells: 1. after.ply → 2. visit-b.ply / Small', exact: true }).click();
  const result = await section(page);
  expect(result.before).toMatchObject({ name: 'after.ply', selectedPoints: 3, zMax: 100 });
  expect(result.after).toMatchObject({ name: 'visit-b.ply', selectedPoints: 2, zMax: 4 });
  expect(result.parent).toMatchObject({ supportMode: 'common_observed', cell: { index: 1, difference: 3 } });
  expect(result.parent.maskSha256).toMatch(/^[0-9a-f]{64}$/); expect(result.parentReviewBounds).toEqual(bounds);
  await page.locator('#cloud-list > li').nth(2).locator('button[title="Remove"]').click();
  await expect(page.locator('#regions-section-plot')).toBeHidden(); await expect(page.locator('#height-cell-inspection')).toBeHidden();
});
for (const action of ['cancel', 'settings', 'close'] as const) test(`height section ${action} discards in-flight worker output and QA`, async ({ page }) => {
  await setup(page); await inspect(page);
  await page.evaluate(action => {
    const original = Worker.prototype.postMessage;
    Worker.prototype.postMessage = function (message: any, ...rest: any[]) {
      if (message?.source?.positions && message?.reference?.positions) {
        if (action === 'cancel') document.getElementById('task-cancel')!.click();
        else if (action === 'close') document.getElementById('height-cell-close')!.click();
        else { const el = document.getElementById('height-section-width') as HTMLInputElement; el.value = '4'; el.dispatchEvent(new Event('input')); }
      }
      return (original as any).call(this, message, ...rest);
    };
  }, action);
  await page.getByRole('button', { name: 'Compare point section for screened cell 1', exact: true }).click();
  await expect(page.locator('#task')).toBeHidden(); await expect(page.locator('#regions-section-plot')).toBeHidden();
  await expect(page.locator('#volume-result')).toBeVisible();
  expect(JSON.parse(await output(page, '#report-json')).sections['height-cell-section']).toBeUndefined();
});
test('constant reference heights retain cell inspection but cannot offer original point-pair sections', async ({ page }) => {
  await setup(page); await page.locator('#volume-before').selectOption('constant'); await inspect(page);
  await expect(page.getByRole('button', { name: 'Focus screened cell 1', exact: true })).toBeEnabled();
  await expect(page.getByRole('button', { name: 'Compare point section for screened cell 1', exact: true })).toBeDisabled();
});
