import { expect, type Page, test } from "@playwright/test";

/** A binary little-endian PLY with float x/y/z. */
function ply(points: [number, number, number][]): Buffer {
  const header =
    "ply\nformat binary_little_endian 1.0\n" +
    `element vertex ${points.length}\n` +
    "property float x\nproperty float y\nproperty float z\nend_header\n";
  const body = Buffer.alloc(points.length * 12);
  points.forEach((p, i) => p.forEach((v, a) => body.writeFloatLE(v, i * 12 + a * 4)));
  return Buffer.concat([Buffer.from(header), body]);
}

/** An uncompressed LAS 1.2 file (point format 1) with intensity and classes. */
function las(points: { xyz: [number, number, number]; intensity: number; cls: number }[]): Buffer {
  const header = Buffer.alloc(227);
  header.write("LASF", 0);
  header[24] = 1;
  header[25] = 2;
  header.writeUInt16LE(227, 94);
  header.writeUInt32LE(227, 96);
  header[104] = 1;
  header.writeUInt16LE(28, 105);
  header.writeUInt32LE(points.length, 107);
  for (let a = 0; a < 3; a++) header.writeDoubleLE(0.001, 131 + 8 * a);
  const body = Buffer.alloc(points.length * 28);
  points.forEach((p, i) => {
    const o = i * 28;
    p.xyz.forEach((v, a) => body.writeInt32LE(Math.round(v / 0.001), o + 4 * a));
    body.writeUInt16LE(p.intensity, o + 12);
    body[o + 15] = p.cls;
  });
  return Buffer.concat([header, body]);
}

/** A wavy grid of `n` x `n` points, optionally lifted. */
function grid(n: number, lift = 0): [number, number, number][] {
  const out: [number, number, number][] = [];
  for (let j = 0; j < n; j++) {
    for (let i = 0; i < n; i++) out.push([i * 0.1, j * 0.1, Math.sin(i / 7) * 0.2 + lift]);
  }
  return out;
}

async function open(page: Page, files: { name: string; buffer: Buffer }[]): Promise<void> {
  await page
    .locator("#file-input")
    .setInputFiles(files.map((f) => ({ name: f.name, mimeType: "application/octet-stream", buffer: f.buffer })));
}

const status = (page: Page) => page.locator("#status");

test.beforeEach(async ({ page }) => {
  page.on("pageerror", (err) => {
    throw err;
  });
  await page.goto("/");
});

test("sample: loads two LiDAR scans and computes C2C", async ({ page }) => {
  await page.getByRole("button", { name: "Try a sample" }).click();
  await expect(status(page)).toContainText("C2C distance computed for 34,370 points");
  await expect(page.locator("#c2c-stats")).toContainText("0.022061");
  await expect(page.locator("#colorbar")).toBeVisible();
  await expect(page.locator(".cloud-list li")).toHaveCount(2);
});

test("PLY: C2C between a grid and a lifted copy, then export", async ({ page }) => {
  await open(page, [
    { name: "reference.ply", buffer: ply(grid(60)) },
    { name: "lifted.ply", buffer: ply(grid(60, 0.5)) },
  ]);
  await expect(status(page)).toContainText("Loaded lifted.ply: 3,600 points");
  await page.locator("#c2c-run").click();
  await expect(status(page)).toContainText("C2C distance computed for 3,600 points");
  // Each lifted point is 0.5 above its twin; on the wavy grid a neighbour can
  // be slightly closer, so 0.5 is the maximum.
  await expect(page.locator("#c2c-stats")).toContainText(/Max\s*0\.5(?!\d)/);

  const download = page.waitForEvent("download");
  await page.locator("#export-ply").click();
  const file = await download;
  expect(file.suggestedFilename()).toBe("lifted_C2C.ply");
  const stream = await file.createReadStream();
  const chunks: Buffer[] = [];
  for await (const chunk of stream) chunks.push(chunk as Buffer);
  const header = Buffer.concat(chunks).subarray(0, 300).toString("latin1");
  expect(header).toContain("element vertex 3600");
  expect(header).toContain("property float scalar_C2C_distance");
});

test("LAS: intensity and classification with a class filter", async ({ page }) => {
  const points = grid(40).map((xyz, i) => ({ xyz, intensity: 100 + (i % 50), cls: i % 10 === 0 ? 6 : 2 }));
  await open(page, [{ name: "classes.las", buffer: las(points) }]);
  await expect(status(page)).toContainText("Loaded classes.las: 1,600 points");
  // No RGB, so the cloud starts colored by intensity.
  await expect(page.locator(".cloud-list select")).toHaveValue("intensity");
  const classes = page.locator("#class-list");
  await expect(classes).toContainText("2 · Ground");
  await expect(classes).toContainText("1,440");
  await expect(classes).toContainText("6 · Building");
  await expect(classes).toContainText("160");
  await classes.locator("input").first().uncheck();
  await expect(classes.locator("input").first()).not.toBeChecked();
});

test("clipping box crops a cloud", async ({ page }) => {
  await open(page, [{ name: "grid.ply", buffer: ply(grid(50)) }]);
  await expect(status(page)).toContainText("Loaded grid.ply: 2,500 points");
  await page.locator("#clip-enabled").check();
  const x = page.locator('.clip-axis[data-axis="0"] input');
  await x.nth(0).fill("0");
  await x.nth(1).fill("500");
  await page.locator("#clip-crop").click();
  await expect(status(page)).toContainText("Cropped: grid_crop");
  // Half of the 50 columns (x = 0.0 … 2.4), within one column of rounding.
  const meta = await page.locator(".cloud-list li").nth(1).locator(".meta").textContent();
  const kept = Number(meta?.match(/([\d,]+) points/)?.[1].replace(/,/g, ""));
  expect(kept).toBeGreaterThanOrEqual(1200);
  expect(kept).toBeLessThanOrEqual(1300);
});

test("mesh: signed C2M against an OBJ plane", async ({ page }) => {
  const obj = Buffer.from("v -1 -1 0\nv 10 -1 0\nv 10 10 0\nv -1 10 0\nf 1 2 3 4\n");
  const above = grid(20, 0.25).map(([x, y]) => [x, y, 0.25] as [number, number, number]);
  await open(page, [
    { name: "plane.obj", buffer: obj },
    { name: "above.ply", buffer: ply(above) },
  ]);
  await expect(status(page)).toContainText("Loaded above.ply");
  await expect(page.locator("#c2c-reference")).toHaveValue(/.+/);
  await page.locator("#c2c-run").click();
  await expect(status(page)).toContainText("C2M distance computed for 400 points");
  await expect(page.locator("#c2c-stats")).toContainText(/Mean\s*0\.25(?!\d)/);
});

test("filters: SOR drops outliers, voxel subsampling keeps one point per voxel", async ({ page }) => {
  const noisy = [...grid(60), [50, 50, 20], [-30, 10, 5], [10, -40, 8]] as [number, number, number][];
  await open(page, [{ name: "noisy.ply", buffer: ply(noisy) }]);
  await expect(status(page)).toContainText("Loaded noisy.ply: 3,603 points");
  await page.locator("#filter-op").selectOption("sor");
  await page.locator("#filter-run").click();
  await expect(status(page)).toContainText("noisy_sor: kept 3,600 of 3,603 points (3 removed)");

  // Voxel subsample the cleaned cloud: 60 x 60 points at 0.1 spacing, 0.5 voxels.
  await page.locator("#filter-cloud").selectOption({ label: "noisy_sor" });
  await page.locator("#filter-op").selectOption("voxel");
  await page.locator("#filter-voxel").fill("0.5");
  await page.locator("#filter-run").click();
  await expect(status(page)).toContainText("kept 144 of 3,600 points");
  await expect(page.locator(".cloud-list li")).toHaveCount(3);
});

test("1.2M points: octree and SOR run on the worker pool with exact results", async ({ page }) => {
  // A flat 1100 x 1100 grid plus five far outliers.
  const points: [number, number, number][] = [];
  for (let j = 0; j < 1100; j++) for (let i = 0; i < 1100; i++) points.push([i * 0.1, j * 0.1, 0]);
  points.push([300, 300, 100], [-200, 50, 40], [50, -250, -60], [400, -100, 0], [-150, -150, 150]);
  await open(page, [{ name: "big.ply", buffer: ply(points) }]);
  await expect(status(page)).toContainText(/Loaded big\.ply: 1,210,005 points .*index .* on \d+ workers/, {
    timeout: 60_000,
  });
  await page.locator("#filter-op").selectOption("sor");
  await page.locator("#filter-run").click();
  await expect(status(page)).toContainText("big_sor: kept 1,210,000 of 1,210,005 points (5 removed)", {
    timeout: 60_000,
  });
});

test("large files keep every n-th point above the max-points setting", async ({ page }) => {
  // A tiny limit stands in for a multi-gigabyte file.
  await page.evaluate(() => {
    const select = document.getElementById("max-points") as HTMLSelectElement;
    select.add(new Option("1k", "1000"));
    select.value = "1000";
  });
  await open(page, [{ name: "big.ply", buffer: ply(grid(60)) }]);
  await expect(status(page)).toContainText("Loaded big.ply: 900 of 3,600 points (1 in 4)");
  const points = grid(40).map((xyz) => ({ xyz, intensity: 7, cls: 2 }));
  await open(page, [{ name: "big.las", buffer: las(points) }]);
  await expect(status(page)).toContainText("Loaded big.las: 800 of 1,600 points (1 in 2)");
  await expect(page.locator(".cloud-list")).toContainText("800 points (1 in 2)");
});

test("volume: a 2 x 2 x 1 m mound over flat ground is 4 m³ of fill", async ({ page }) => {
  const site = (mound: boolean) => {
    const out: [number, number, number][] = [];
    for (let j = 0; j < 60; j++) {
      for (let i = 0; i < 60; i++) {
        const [x, y] = [i * 0.1, j * 0.1];
        const inside = x >= 0.95 && x < 2.95 && y >= 0.95 && y < 2.95;
        out.push([x, y, mound && inside ? 1 : 0]);
      }
    }
    return out;
  };
  await open(page, [
    { name: "ground.ply", buffer: ply(site(false)) },
    { name: "surveyed.ply", buffer: ply(site(true)) },
  ]);
  await expect(status(page)).toContainText("Loaded surveyed.ply");
  await page.locator("#volume-panel summary").click();
  await page.locator("#volume-cell").fill("0.5");
  await page.locator("#volume-run").click();
  await expect(status(page)).toContainText("Volume: fill 4 m³, cut 0 m³, net 4 m³");
  await expect(page.locator("#volume-stats")).toContainText(/Fill area\s*4 m²/);
  // The per-cell differences are added as a cloud colored by height difference.
  await expect(page.locator(".cloud-list li")).toHaveCount(3);
  await expect(page.locator("#colorbar-title")).toContainText("Height difference");
});

test("ground extraction (CSF) separates a box from flat ground", async ({ page }) => {
  const scene: [number, number, number][] = [];
  for (let j = 0; j < 80; j++) {
    for (let i = 0; i < 80; i++) {
      const [x, y] = [i * 0.25, j * 0.25];
      const box = x >= 8 && x < 12 && y >= 8 && y < 12;
      scene.push([x, y, box ? 3 : 0]);
    }
  }
  await open(page, [{ name: "site.ply", buffer: ply(scene) }]);
  await expect(status(page)).toContainText("Loaded site.ply: 6,400 points");
  await page.locator("#filter-op").selectOption("ground");
  await page.locator("#filter-run").click();
  await expect(status(page)).toContainText("site_csf: 6,144 of 6,400 points are ground");
  // The classified copy is shown by class, with ground and "other" in the class panel.
  await expect(page.locator("#class-list")).toContainText("2 · Ground");
  await expect(page.locator("#class-list")).toContainText("256");
});

test("M3C2: a flat grid lifted by 0.3 m changes by 0.3 m along the normal", async ({ page }) => {
  const flat = (lift: number) => grid(60).map(([x, y]) => [x, y, lift] as [number, number, number]);
  await open(page, [
    { name: "before.ply", buffer: ply(flat(0)) },
    { name: "after.ply", buffer: ply(flat(0.3)) },
  ]);
  await expect(status(page)).toContainText("Loaded after.ply: 3,600 points");
  await page.locator("#distance-method").selectOption("m3c2");
  await page.locator("#m3c2-normal").fill("0.5");
  await page.locator("#m3c2-projection").fill("0.25");
  await page.locator("#m3c2-depth").fill("1");
  await page.locator("#m3c2-core").fill("0.5");
  await page.locator("#c2c-run").click();
  await expect(status(page)).toContainText(/M3C2 at [\d,]+ core points/);
  await expect(status(page)).toContainText("(100.0 %)");
  await expect(page.locator("#c2c-stats")).toContainText(/Mean\s*0\.3(0*)(?!\d)/);
  await expect(page.locator(".cloud-list li")).toHaveCount(3);

  const download = page.waitForEvent("download");
  await page.locator("#export-ply").click();
  const file = await download;
  expect(file.suggestedFilename()).toBe("after_m3c2_M3C2.ply");
  const chunks: Buffer[] = [];
  for await (const chunk of await file.createReadStream()) chunks.push(chunk as Buffer);
  const header = Buffer.concat(chunks).subarray(0, 400).toString("latin1");
  expect(header).toContain("property float m3c2_distance");
  expect(header).toContain("property uchar significant");
});

test.describe("phone", () => {
  test.use({ viewport: { width: 390, height: 844 }, hasTouch: true, isMobile: true });

  test("panels are a bottom sheet; tapping a point picks it, double tap centres it", async ({ page }) => {
    const sheet = page.locator("#sidebar");
    await expect(sheet).toBeInViewport();
    await page.getByRole("button", { name: "Try a sample" }).click();
    await expect(status(page)).toContainText("C2C distance computed");
    // The sheet closes when the first cloud arrives, leaving the full view.
    await expect(sheet).not.toBeInViewport();
    const canvas = page.locator("#viewport canvas");
    const box = (await canvas.boundingBox())!;
    expect(box.height).toBeGreaterThan(600);

    await canvas.tap({ position: { x: box.width / 2, y: box.height / 2 } });
    await expect(status(page)).toContainText("Picked");
    await canvas.tap({ position: { x: box.width / 2, y: box.height / 2 } });

    await page.locator("#panels").tap();
    await expect(sheet).toBeInViewport();
    await expect(page.locator("#pick-panel")).toBeVisible();
    await page.locator("#panels").tap();
    await expect(sheet).not.toBeInViewport();
  });
});
