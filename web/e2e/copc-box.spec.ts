import { expect, test, type Page, type Download } from "@playwright/test";
import { readFileSync } from "node:fs";

const copc = readFileSync(new URL("./fixtures/small.copc.laz", import.meta.url));
const bounds = [8, 8, -1, 20, 24, 10];
async function box(page: Page, cap = 200000) {
  for (const [i, axis] of ["xmin", "ymin", "zmin", "xmax", "ymax", "zmax"].entries())
    await page.locator(`#copc-box-${axis}`).fill(String(bounds[i]));
  await page.locator("#copc-box-limit").fill(String(cap));
}
async function bytes(download: Download) {
  const out: Buffer[] = [];
  for await (const chunk of await download.createReadStream()) out.push(chunk as Buffer);
  return Buffer.concat(out);
}
async function save(page: Page, index: number) {
  const row = page.locator(".cloud-list li").nth(index);
  await row.getByTitle("Save as…", { exact: true }).click();
  const pending = page.waitForEvent("download");
  await row.locator(".save-formats").getByRole("button", { name: "LAS", exact: true }).click();
  return bytes(await pending);
}
function selectedRecords(las: Buffer) {
  const start = las.readUInt32LE(96), stride = las.readUInt16LE(105);
  const count = Number(las.readBigUInt64LE(247));
  const out: string[] = [];
  for (let i = 0; i < count; i++) {
    const offset = start + i * stride;
    const xyz = [0, 1, 2].map((a) => las.readInt32LE(offset + 4 * a) * las.readDoubleLE(131 + 8 * a) + las.readDoubleLE(155 + 8 * a));
    if (xyz.every((v, a) => v >= bounds[a] && v <= bounds[a + 3]))
      out.push(JSON.stringify([...xyz.map((x) => x.toFixed(6)), las.readUInt16LE(offset + 12), las[offset + 16]]));
  }
  return out.sort();
}
test.beforeEach(async ({ page }) => { await page.goto("/"); });

test("full-density box matches full-source XYZ, intensity and classes; Undo restores visibility", async ({ page }) => {
  await page.locator("#file-input").setInputFiles({ name: "small.copc.laz", mimeType: "application/octet-stream", buffer: copc });
  await expect(page.locator("#status")).toContainText("42,000 of 42,000");
  const expected = selectedRecords(await save(page, 0));
  expect(expected).toHaveLength(1965);
  await box(page);
  await page.locator("#copc-box-run").click();
  await expect(page.locator("#copc-box-result")).toContainText("1,965 points");
  expect(selectedRecords(await save(page, 1))).toEqual(expected);
  await expect(page.locator(".cloud-list li").first().locator('input[type="checkbox"]')).not.toBeChecked();
  await page.locator("#undo").click();
  await expect(page.locator(".cloud-list li")).toHaveCount(1);
  await expect(page.locator(".cloud-list li input[type=checkbox]")).toBeChecked();
});

test("reads deeper nodes than the display budget and rejects overflowing or empty selections atomically", async ({ page }) => {
  await page.evaluate(() => { const s = document.querySelector<HTMLSelectElement>("#max-points")!; s.add(new Option("5k", "5000")); s.value = "5000"; });
  await page.locator("#file-input").setInputFiles({ name: "small.copc.laz", mimeType: "application/octet-stream", buffer: copc });
  await expect(page.locator("#status")).toContainText("2,000 of 42,000");
  await box(page, 100);
  await page.locator("#copc-box-run").click();
  await expect(page.locator("#status")).toContainText("exceeds 100 selected points");
  await expect(page.locator(".cloud-list li")).toHaveCount(1);
  await box(page);
  await page.locator("#copc-box-xmin").fill("100");
  await page.locator("#copc-box-xmax").fill("110");
  await page.locator("#copc-box-run").click();
  await expect(page.locator("#status")).toContainText("no points inside");
  await expect(page.locator(".cloud-list li")).toHaveCount(1);
  await box(page);
  await page.locator("#copc-box-run").click();
  await expect(page.locator("#copc-box-result")).toContainText("1,965 points");
});

for (const mode of ["etag", "size", "cancel", "missing-etag", "success", "stale"] as const) {
  test(`HTTP full-density selection handles ${mode} without publishing output`, async ({ page }) => {
    let selecting = false;
    let reached: (() => void) | undefined;
    const held = new Promise<void>((resolve) => { reached = resolve; });
    let release: (() => void) | undefined;
    const wait = new Promise<void>((resolve) => { release = resolve; });
    await page.route((url) => url.pathname === "/remote/box.copc.laz", async (route) => {
      const headers = route.request().headers();
      const match = /bytes=(\d+)-(\d+)/.exec(headers.range ?? "")!;
      const start = Number(match[1]), end = Math.min(Number(match[2]), copc.length - 1);
      if (selecting && (mode === "cancel" || mode === "stale")) { reached!(); await wait; }
      if (selecting && mode !== "missing-etag") expect(headers["if-match"]).toBe('"source-v1"');
      await route.fulfill({ status: 206, headers: {
        "content-range": `bytes ${start}-${end}/${copc.length + (selecting && mode === "size" ? 1 : 0)}`,
        ...(mode === "missing-etag" ? {} : { etag: selecting && mode === "etag" ? '"source-v2"' : '"source-v1"' }),
      }, body: copc.subarray(start, end + 1) }).catch(() => {});
    });
    await page.goto("/?url=/remote/box.copc.laz");
    await expect(page.locator("#status")).toContainText("42,000 of 42,000");
    selecting = true;
    await box(page);
    await page.locator("#copc-box-run").click();
    if (mode === "cancel") { await held; await page.locator("#task-cancel").click(); }
    if (mode === "stale") { await held; await page.locator("#copc-box-limit").fill("200001"); release!(); }
    if (mode === "success") {
      await expect(page.locator("#copc-box-result")).toContainText("1,965 points");
      await expect(page.locator(".cloud-list li")).toHaveCount(2);
      return;
    }
    await expect(page.locator("#status")).toContainText(mode === "etag" ? "ETag changed" : mode === "size" ? "size changed" : mode === "cancel" ? "selection cancelled" : mode === "stale" ? "result discarded" : "exposed strong ETag");
    release!();
    await expect(page.locator(".cloud-list li")).toHaveCount(1);
    await expect(page.locator("#copc-box-run")).toBeEnabled();
  });
}
