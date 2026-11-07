import { expect, test } from "@playwright/test";

// Every other test depends on this one (see playwright.config.ts). If the app
// cannot start, the run stops here with the page errors, instead of each test
// timing out on its own.
test("the app loads without page errors and the worker answers", async ({ page }) => {
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(error.stack ?? error.message));
  await page.goto("/");
  try {
    // While loading, the vector map asks the worker for its view and checks
    // it, so this text needs the page, the worker and the WebAssembly core.
    await expect(page.locator("#vm-status")).toHaveText("No map yet.");
  } finally {
    expect(errors, "page errors while the app loaded").toEqual([]);
  }
});
