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

test('reviews the NCLT display subset while retaining full-source audit failures', async ({page})=>{
  const preview=process.env.CA_REVIEW_NCLT_PREVIEW_BUNDLE;
  test.skip(!preview,'Set CA_REVIEW_NCLT_PREVIEW_BUNDLE to the exported display-only NCLT ZIP');
  await page.goto('/');
  await page.locator('#mapping-review-file').setInputFiles(preview!);
  await expect(page.locator('#status')).toContainText('Opened generated maps');
  await expect(page.locator('#mapping-review-summary')).toContainText('161,578 loaded point records (display preview from 646,309)');
  await expect(page.locator('#mapping-review-summary')).toContainText('124.0 / 252.6 m; requested extent unmet');
  await expect(page.locator('#mapping-review-state')).toContainText('original 646,309 points');
  await expect(page.locator('#vm-quality-check')).toBeDisabled();
  await expect(page.locator('#vm-quality-lanes button')).toHaveCount(9);
  await page.locator('#vm-quality-next').click();
  await expect(page.locator('#vm-quality-problem-detail')).toContainText('height disagreement');
  await page.locator('#vm-quality-focus').click();
  for(const value of ['1','2','3','0']) {
    await page.locator('#mapping-review-audit').selectOption(value);
    await expect(page.locator('#vm-quality-lanes button')).toHaveCount(value==='0'||value==='1'?9:0);
    await expect(page.locator('#vm-quality-check')).toBeDisabled();
  }
  await page.screenshot({path:'/tmp/nclt-browser-display-preview.png'});
});
