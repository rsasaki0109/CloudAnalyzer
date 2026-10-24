import { expect, test, type Page } from '@playwright/test';

async function prepareHeldSnapshot(page: Page) {
  await page.addInitScript(() => localStorage.setItem('cloudanalyzer-autosave-enabled', 'off'));
  await page.goto('/?demo=volume');
  await expect(page.locator('#status')).toContainText(/Volume: fill 40\d\.\d+ m³, cut 4\d\.\d+ m³/);
  await expect(page.locator('#task')).toBeHidden();
  await page.locator('#volume-options').evaluate((el: HTMLDetailsElement) => el.open = true);
  await page.locator('#volume-cell').fill('0.5');
  await expect(page.locator('#volume-result')).toBeHidden();
  await page.evaluate(() => {
    const original = Worker.prototype.postMessage;
    let held = false;
    (window as any).demoVolumeRequests = 0;
    Worker.prototype.postMessage = function (message: any, ...rest: any[]) {
      if (message?.req?.kind === 'volume') (window as any).demoVolumeRequests++;
      if (!held && message?.req?.kind === 'vm' && message.req.op === 'json') {
        held = true;
        // The API has registered its pending request before calling postMessage.
        // Hold the actual read until the test explicitly releases it.
        (window as any).releaseDemoSnapshot = () => (original as any).call(this, message, ...rest);
        return;
      }
      return (original as any).call(this, message, ...rest);
    };
  });
  await page.locator('#project-autosave').check();
  await page.waitForFunction(() => typeof (window as any).releaseDemoSnapshot === 'function');
  // The welcome buttons hide after loading. Invoke the registered demo handler
  // directly to exercise its supported already-loaded-input path.
  await page.locator('[data-demo="volume"]').evaluate((el: HTMLButtonElement) => el.click());
  await expect(page.locator('#status')).toContainText('Waiting for background worker requests');
  await expect(page.locator('#task')).toBeVisible();
  expect(await page.evaluate(() => (window as any).demoVolumeRequests)).toBe(0);
}

test('a demo waits for an in-flight autosave read, then starts exactly once after release', async ({ page }) => {
  await prepareHeldSnapshot(page);
  await page.evaluate(() => (window as any).releaseDemoSnapshot());
  await expect(page.locator('#status')).toContainText(/Volume: fill 40\d\.\d+ m³, cut 4\d\.\d+ m³/);
  await expect(page.locator('#task')).toBeHidden();
  expect(await page.evaluate(() => (window as any).demoVolumeRequests)).toBe(1);
});

test('cancelling a waiting demo starts no volume request and leaves a later retry usable', async ({ page }) => {
  await prepareHeldSnapshot(page);
  await page.locator('#task-cancel').click();
  await expect(page.locator('#status')).toHaveText('Demo cancelled');
  await expect(page.locator('#task')).toBeHidden();
  await expect(page.locator('#volume-result')).toBeHidden();
  expect(await page.evaluate(() => (window as any).demoVolumeRequests)).toBe(0);
  await page.evaluate(() => (window as any).releaseDemoSnapshot());
  await page.locator('[data-demo="volume"]').evaluate((el: HTMLButtonElement) => el.click());
  await expect(page.locator('#status')).toContainText(/Volume: fill 40\d\.\d+ m³, cut 4\d\.\d+ m³/);
  expect(await page.evaluate(() => (window as any).demoVolumeRequests)).toBe(1);
});
