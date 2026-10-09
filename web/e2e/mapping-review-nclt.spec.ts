import { expect, test } from '@playwright/test';

const bundle = process.env.CA_REVIEW_NCLT_BUNDLE;
test('reviews the exact adopted NCLT pair and its retained legacy failures', async ({page})=>{
  test.skip(!bundle, 'Set CA_REVIEW_NCLT_BUNDLE to the exported NCLT review ZIP (large data stays outside git)');
  await page.goto('/');
  await page.locator('#mapping-review-file').setInputFiles(bundle!);
  await expect(page.locator('#status')).toContainText('Opened generated maps');
  await expect(page.locator('#mapping-review-summary')).toContainText('646,309 loaded point records');
  await expect(page.locator('#mapping-review-summary')).toContainText('124.0 / 252.6 m; requested extent unmet');
  await expect(page.locator('#vm-quality-lanes button')).toHaveCount(9);
  await page.locator('#vm-quality-next').click();
  await expect(page.locator('#vm-quality-problem-detail')).toContainText('height disagreement');
  await page.locator('#vm-quality-focus').click();
  await page.locator('#mapping-review-audit').selectOption('2');
  await expect(page.locator('#vm-quality-report')).toContainText('ground consensus / editable IR');
  await expect(page.locator('#vm-quality-lanes button')).toHaveCount(0);
  await page.locator('#mapping-review-audit').selectOption('0');
  await expect(page.locator('#vm-quality-lanes button')).toHaveCount(9);
  await page.screenshot({path:'/tmp/nclt-browser-generated-maps.png'});
});
