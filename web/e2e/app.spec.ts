import { type Download, expect, type Page, test } from "@playwright/test";
import { mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

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

/** An uncompressed LAS 1.2 file (point format 1, 1 mm scale) with intensity and classes. */
function las(
  points: { xyz: [number, number, number]; intensity: number; cls: number }[],
  offset: [number, number, number] = [0, 0, 0],
): Buffer {
  const header = Buffer.alloc(227);
  header.write("LASF", 0);
  header[24] = 1;
  header[25] = 2;
  header.writeUInt16LE(227, 94);
  header.writeUInt32LE(227, 96);
  header[104] = 1;
  header.writeUInt16LE(28, 105);
  header.writeUInt32LE(points.length, 107);
  for (let a = 0; a < 3; a++) {
    header.writeDoubleLE(0.001, 131 + 8 * a);
    header.writeDoubleLE(offset[a], 155 + 8 * a);
  }
  const body = Buffer.alloc(points.length * 28);
  points.forEach((p, i) => {
    const o = i * 28;
    p.xyz.forEach((v, a) => body.writeInt32LE(Math.round((v - offset[a]) / 0.001), o + 4 * a));
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

async function bytesOf(download: Download): Promise<Buffer> {
  const chunks: Buffer[] = [];
  for await (const chunk of await download.createReadStream()) chunks.push(chunk as Buffer);
  return Buffer.concat(chunks);
}

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

test("LAS export: UTM points, intensity, classes and C2C saved to LAS / LAZ and read back", async ({ page }) => {
  // Georeferenced coordinates on the fixture's 1 mm grid.
  const utm = (lift: number) =>
    grid(60, lift).map((p, i) => ({
      xyz: [368_000 + p[0], 3_955_000 + p[1], 40 + p[2]] as [number, number, number],
      intensity: 100 + (i % 50),
      cls: i % 10 === 0 ? 6 : 2,
    }));
  const lifted = utm(0.5);
  const utmOffset: [number, number, number] = [368_000, 3_955_000, 0];
  await open(page, [
    { name: "reference.las", buffer: las(utm(0), utmOffset) },
    { name: "lifted.las", buffer: las(lifted, utmOffset) },
  ]);
  await expect(status(page)).toContainText("Loaded lifted.las: 3,600 points");
  await page.locator("#c2c-run").click();
  await expect(status(page)).toContainText("C2C distance computed for 3,600 points");

  // Save from the cloud's ⤓ menu.
  const download = page.waitForEvent("download");
  await page.locator(".cloud-list li").nth(1).locator("button.icon").click();
  await page.locator(".cloud-list .save-formats").getByRole("button", { name: "LAS" }).click();
  const file = await download;
  expect(file.suggestedFilename()).toBe("lifted_C2C.las");
  const saved = await bytesOf(file);
  expect(saved.subarray(0, 4).toString("latin1")).toBe("LASF");
  expect([saved[24], saved[25], saved[104]]).toEqual([1, 4, 6]);
  const recordLen = saved.readUInt16LE(105);
  expect(recordLen).toBe(34); // format 6 + one float extra byte
  expect(saved.toString("latin1", 375 + 54 + 4, 375 + 54 + 16)).toBe("C2C_distance");
  expect(saved.readDoubleLE(147)).toBe(0.001);

  // Every point comes back (in octree order) on the 1 mm grid, with its attributes.
  const expected = new Map(lifted.map((p) => [p.xyz.map((v) => Math.round(v * 1000)).join(","), p]));
  const count = Number(saved.readBigUInt64LE(247));
  expect(count).toBe(3600);
  const data = saved.readUInt32LE(96);
  const bad: number[] = [];
  for (let i = 0; i < count; i++) {
    const o = data + i * recordLen;
    const xyz = [0, 1, 2].map((a) => saved.readInt32LE(o + 4 * a) * saved.readDoubleLE(131 + 8 * a) + saved.readDoubleLE(155 + 8 * a));
    const p = expected.get(xyz.map((v) => Math.round(v * 1000)).join(","));
    const distance = saved.readFloatLE(o + 30);
    const ok =
      p !== undefined &&
      xyz.every((v, a) => Math.abs(v - Math.round(p.xyz[a] * 1000) / 1000) < 1e-6) &&
      saved.readUInt16LE(o + 12) === p.intensity &&
      saved[o + 16] === p.cls &&
      distance >= 0 &&
      distance <= 0.5001;
    if (!ok) bad.push(i);
  }
  expect(bad).toEqual([]);

  await open(page, [{ name: "again.las", buffer: saved }]);
  await expect(status(page)).toContainText("Loaded again.las: 3,600 points");
  // The class list counts all three clouds: 3 × 3,240 ground points.
  await expect(page.locator("#class-list")).toContainText("2 · Ground9,720");

  // The distance panel's LAZ export compresses the same records.
  await page.locator(".cloud-list li").nth(1).locator("select").selectOption("c2c");
  const lazDownload = page.waitForEvent("download");
  await page.locator("#export-laz").click();
  const laz = await bytesOf(await lazDownload);
  expect(laz[104]).toBe(0x86);
  expect(laz.length).toBeLessThan(saved.length / 2);
  await open(page, [{ name: "again.laz", buffer: laz }]);
  await expect(status(page)).toContainText("Loaded again.laz: 3,600 points");
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

test("mesh: a grid meshed by 2.5D Delaunay, saved as OBJ, and used for C2M", async ({ page }) => {
  await open(page, [
    { name: "wave.ply", buffer: ply(grid(30)) },
    { name: "lifted.ply", buffer: ply(grid(30, 0.5)) },
  ]);
  await expect(status(page)).toContainText("Loaded lifted.ply");
  await page.locator("#mesh-cloud").selectOption({ label: "wave.ply" });
  await page.locator("#mesh-run").click();
  // A 30 x 30 grid has 29 x 29 cells of two triangles each.
  await expect(status(page)).toContainText(/wave_mesh: 1,682 triangles from 900 points, 0 longer than 0\.4\d* removed/);
  const mesh = page.locator(".cloud-list li").last();
  await expect(mesh.locator(".meta")).toHaveText("1,682 triangles · mesh");

  const download = page.waitForEvent("download");
  await mesh.locator("button.icon").click();
  await mesh.locator(".save-formats").getByRole("button", { name: "OBJ" }).click();
  const file = await download;
  expect(file.suggestedFilename()).toBe("wave_mesh.obj");
  const lines = (await bytesOf(file)).toString("utf8").split("\n");
  expect(lines.filter((l) => l.startsWith("v ")).length).toBe(900);
  expect(lines.filter((l) => l.startsWith("f ")).length).toBe(1682);

  // Every lifted point is 0.5 above a vertex; the wavy surface slopes a
  // little, so the distance along its normal is a little less. Normals face up.
  await page.locator("#c2c-compared").selectOption({ label: "lifted.ply" });
  await page.locator("#c2c-reference").selectOption({ label: "wave_mesh (mesh)" });
  await page.locator("#c2c-run").click();
  await expect(status(page)).toContainText("C2M distance computed for 900 points");
  const stats = (await page.locator("#c2c-stats").textContent()) ?? "";
  const stat = (name: string) => Number(stats.match(new RegExp(`${name}\\s*(-?[\\d.]+)`))?.[1]);
  expect(stat("Min")).toBeGreaterThan(0.45);
  expect(stat("Max")).toBeLessThanOrEqual(0.5 + 1e-6);
  expect(stat("Mean")).toBeCloseTo(0.49, 1);
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

/**
 * A 3D Gaussian Splatting PLY (as INRIA's trainer writes it, SH degree 0):
 * one Gaussian per point with the given opacity logit and log scale.
 */
function splatPly(splats: { xyz: [number, number, number]; opacity: number; scale: number }[]): Buffer {
  const names = "x y z nx ny nz f_dc_0 f_dc_1 f_dc_2 opacity scale_0 scale_1 scale_2 rot_0 rot_1 rot_2 rot_3".split(" ");
  const header =
    "ply\nformat binary_little_endian 1.0\n" +
    `element vertex ${splats.length}\n` +
    names.map((n) => `property float ${n}\n`).join("") +
    "end_header\n";
  const body = Buffer.alloc(splats.length * names.length * 4);
  splats.forEach((s, i) => {
    const v = [...s.xyz, 0, 0, 0, 1, 0, -1, s.opacity, s.scale, s.scale - 1, s.scale - 2, 1, 0, 0, 0];
    v.forEach((x, k) => body.writeFloatLE(x, (i * names.length + k) * 4));
  });
  return Buffer.concat([Buffer.from(header), body]);
}

test("Gaussian splats: read as points, cleaned up, compared with C2C", async ({ page }) => {
  // 100 Gaussians on the grid: 20 nearly transparent, 5 huge floaters.
  const splats = grid(10).map((xyz, i) => ({
    xyz,
    opacity: i % 5 === 0 ? -4 : 3,
    scale: i % 20 === 1 ? 3 : -4,
  }));
  await open(page, [
    { name: "reference.ply", buffer: ply(grid(10)) },
    { name: "scene.ply", buffer: splatPly(splats) },
  ]);
  await expect(status(page)).toContainText("Loaded scene.ply: 100 points (Gaussian splat centers)");
  const mode = page.locator(".cloud-list li", { hasText: "scene.ply" }).getByTitle("Color by");
  await expect(mode).toHaveValue("rgb");
  await mode.selectOption("opacity");

  await page.locator("#filter-cloud").selectOption({ label: "scene.ply" });
  await page.locator("#filter-op").selectOption("splat");
  await page.locator("#filter-opacity").fill("0.1");
  await page.locator("#filter-size").fill("1");
  await page.locator("#filter-run").click();
  await expect(status(page)).toContainText("scene_clean: kept 75 of 100 points (25 removed)");
  // The cleanup is not offered for an ordinary cloud.
  await page.locator("#filter-cloud").selectOption({ label: "reference.ply" });
  await expect(page.locator('#filter-op option[value="splat"]')).toHaveJSProperty("hidden", true);
  await expect(page.locator("#filter-op")).toHaveValue("voxel");

  // The splat centres are an ordinary cloud: they lie on the reference grid.
  await page.locator("#c2c-compared").selectOption({ label: "scene_clean" });
  await page.locator("#c2c-reference").selectOption({ label: "reference.ply" });
  await page.locator("#c2c-run").click();
  await expect(status(page)).toContainText("C2C distance computed for 75 points");
  await expect(page.locator("#c2c-stats")).toContainText(/Max\s*0(?!\.0*[1-9])/);

  // The headerless .splat format: 32 bytes per Gaussian.
  const records = Buffer.alloc(64);
  records.writeFloatLE(5, 32);
  records.writeUInt32LE(0xff0000ff, 56); // rgba = ff 00 00 ff (red, opaque)
  await open(page, [{ name: "tiny.splat", buffer: records }]);
  await expect(status(page)).toContainText("Loaded tiny.splat: 2 points (Gaussian splat centers)");
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

test("full detail: a thinned LAS / LAZ shows every point of the file where zoomed in", async ({ page }) => {
  // 200,000 points in rows, as a scanner writes them; the limit keeps 1 in 4.
  const points: { xyz: [number, number, number]; intensity: number; cls: number }[] = [];
  for (let j = 0; j < 400; j++) {
    for (let i = 0; i < 500; i++) points.push({ xyz: [i * 0.1, j * 0.1, Math.sin(i / 7) * 0.2], intensity: i % 50, cls: 2 });
  }
  const setMaxPoints = (value: string) =>
    page.evaluate((value) => {
      const select = document.getElementById("max-points") as HTMLSelectElement;
      if (![...select.options].some((o) => o.value === value)) select.add(new Option(value, value));
      select.value = value;
    }, value);
  const drawn = page.locator("#drawn");
  const drawnPoints = async () => {
    const [, n, unit] = /Drawing ([\d.]+)([kM]?) of/.exec((await drawn.textContent()) ?? "")!;
    return Number(n) * ({ k: 1e3, M: 1e6 }[unit] ?? 1);
  };
  const zoomIn = async () => {
    await page.locator('[data-view="top"]').click();
    await page.locator("#fit").click();
    const box = (await page.locator("#viewport > canvas").boundingBox())!;
    await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
    for (let k = 0; k < 6; k++) await page.mouse.wheel(0, -300);
  };

  // A LAZ made by the app from the whole file (octree order, 50,000-point chunks).
  await setMaxPoints("0");
  await open(page, [{ name: "all.las", buffer: las(points) }]);
  await expect(status(page)).toContainText("Loaded all.las: 200,000 points");
  await expect(drawn).not.toContainText("full detail");
  const download = page.waitForEvent("download");
  await page.locator(".cloud-list li").first().locator("button.icon").click();
  await page.locator(".cloud-list .save-formats").getByRole("button", { name: "LAZ" }).click();
  const laz = await bytesOf(await download);

  for (const file of [
    { name: "big.las", buffer: las(points) },
    { name: "big.laz", buffer: laz },
  ]) {
    await page.goto("/");
    await setMaxPoints("50000");
    await open(page, [file]);
    await expect(status(page)).toContainText(`Loaded ${file.name}: 50,000 of 200,000 points (1 in 4)`);
    await zoomIn();
    await expect(drawn).toContainText(/of 200k points · full detail: [1-9]\d* chunks?$/);
    // More than every loaded point: the view comes from the file.
    expect(await drawnPoints()).toBeGreaterThan(50_000);

    // Colored by class, the chunks follow; switched off, only the loaded points are drawn.
    await page.locator(".cloud-list li select").selectOption("classification");
    await expect(drawn).toContainText("full detail");
    await page.locator("#full-detail").uncheck();
    await expect(drawn).toContainText(/Drawing [\d.]+k? of 50k points$/);
    expect(await drawnPoints()).toBeLessThanOrEqual(50_000);
    await page.locator("#full-detail").check();
    await expect(drawn).toContainText("full detail");
  }
});

test("volume: a 2 x 2 x 1 m mound over flat ground is 4 m³ of fill", async ({ page }) => {
  const site = (mound: boolean) => {
    const out: [number, number, number][] = [];
    for (let j = 0; j < 60; j++) {
      for (let i = 0; i < 60; i++) {
        const [x, y] = [i * 0.1, j * 0.1];
        // Cells are centred on multiples of 0.5 (the first point is a cell
        // centre), so their edges are at 0.25 + 0.5 k.
        const inside = x >= 1.25 && x < 3.25 && y >= 1.25 && y < 3.25;
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

test("rasterize: a sloped plane saves as a GeoTIFF and a PNG", async ({ page }) => {
  // z = 0.5 x over 4 x 4 m. With 0.5 m cells centred on multiples of 0.5,
  // the top-left cell holds x = 0, 0.1, 0.2: mean height 0.05.
  const points: [number, number, number][] = [];
  for (let j = 0; j < 40; j++) for (let i = 0; i < 40; i++) points.push([i * 0.1, j * 0.1, i * 0.05]);
  await open(page, [{ name: "slope.ply", buffer: ply(points) }]);
  await expect(status(page)).toContainText("Loaded slope.ply");
  await page.locator("#raster-panel summary").click();
  await page.locator("#raster-cell").fill("0.5");
  await page.locator("#raster-run").click();
  await expect(status(page)).toContainText(/Raster: 9 × 9 cells, 81 with points \(100\.0 %\) in \d+ ms/);
  await expect(page.locator(".cloud-list li")).toHaveCount(2);
  await expect(page.locator("#colorbar-title")).toContainText("Height · slope_raster0.5");

  const tiff = page.waitForEvent("download");
  await page.locator("#raster-tiff").click();
  const file = await tiff;
  expect(file.suggestedFilename()).toBe("slope_raster0.5.tif");
  const chunks: Buffer[] = [];
  for await (const chunk of await file.createReadStream()) chunks.push(chunk as Buffer);
  const bytes = Buffer.concat(chunks);
  expect(bytes.subarray(0, 4).toString("latin1")).toBe("II*\0");
  // Header, 15 tags and their values take 296 bytes; then 81 Float32 cells.
  expect(bytes.length).toBe(296 + 81 * 4);
  expect(bytes.readFloatLE(296)).toBeCloseTo(0.05, 5);

  const png = page.waitForEvent("download");
  await page.locator("#raster-png").click();
  expect((await png).suggestedFilename()).toBe("slope_raster0.5.png");
  await expect(status(page)).toContainText("Saved slope_raster0.5.png (9 × 9 px)");
});

test("rasterize: ground points only give a terrain model with the roof filled in", async ({ page }) => {
  const points: { xyz: [number, number, number]; intensity: number; cls: number }[] = [];
  for (let j = 0; j < 10; j++) {
    for (let i = 0; i < 10; i++) {
      const roof = i >= 3 && i < 7 && j >= 3 && j < 7;
      points.push({ xyz: [i, j, roof ? 5 : 0], intensity: 1, cls: roof ? 6 : 2 });
    }
  }
  await open(page, [{ name: "site.las", buffer: las(points) }]);
  await expect(status(page)).toContainText("Loaded site.las");
  await expect(page.locator("#raster-class-row")).toBeVisible();
  await page.locator("#raster-class").selectOption("2");
  await page.locator("#raster-panel summary").click();
  await page.locator("#raster-cell").fill("1");
  await page.locator("#raster-fill").check();
  await page.locator("#raster-run").click();
  await expect(status(page)).toContainText("Raster: 10 × 10 cells, 84 with points (84.0 %), 16 filled");
  await expect(page.locator("#c2c-stats")).toContainText(/Max\s*0(?!\.)/);
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

test("trajectories: TUM files are drawn and give the Python module's ATE", async ({ page }) => {
  // The Rust parity fixture (see rust/crates/ca-core/tests/trajectory.rs).
  const fixture = (name: string) =>
    readFileSync(new URL(`../../rust/crates/ca-core/tests/trajectory/${name}`, import.meta.url));
  await open(page, [
    { name: "reference.tum", buffer: fixture("reference.tum") },
    { name: "estimate.tum", buffer: fixture("estimate.tum") },
    // A .txt point cloud still opens as a cloud.
    { name: "points.txt", buffer: Buffer.from("0 0 0\n1 0 0\n0 1 0\n1 1 1\n") },
  ]);
  await expect(status(page)).toContainText("Loaded points.txt: 4 points");
  const rows = page.locator("#trajectory-list li");
  await expect(rows).toHaveCount(2);
  await expect(rows.nth(1)).toContainText("39 poses · TUM");
  await expect(page.locator("#cloud-list li")).toHaveCount(1);

  // Defaults: the newest trajectory against the first, SE(3) alignment.
  await page.locator("#trajectory-run").click();
  const stats = page.locator("#trajectory-stats");
  // Python's evaluate_trajectory(align_rigid=True) on these files: ATE RMSE 0.0388736953.
  await expect(stats).toContainText(/ATE RMSE\s*0\.038874(?!\d)/);
  await expect(stats).toContainText(/Matched poses\s*20 of 20/);
  await expect(rows).toHaveCount(3);
  await expect(rows.nth(2)).toContainText("estimate_ate");
  await expect(rows.nth(1).locator("input[type=checkbox]")).not.toBeChecked();
  // The alignment can move a cloud (e.g. a map built from the estimate), undoably.
  await page.locator("#trajectory-apply").click();
  await expect(status(page)).toContainText("Moved points.txt with the SE(3) alignment");
  await page.keyboard.press("Control+z");
  await expect(status(page)).toContainText("Undid the trajectory alignment");

  const download = page.waitForEvent("download");
  await page.locator("#trajectory-csv").click();
  const file = await download;
  expect(file.suggestedFilename()).toBe("estimate_ate.csv");
  const chunks: Buffer[] = [];
  for await (const chunk of await file.createReadStream()) chunks.push(chunk as Buffer);
  const lines = Buffer.concat(chunks).toString("utf8").trim().split("\n");
  expect(lines[0]).toBe("timestamp,x,y,z,reference_x,reference_y,reference_z,ate,ate_rotation_deg");
  expect(lines).toHaveLength(21);

  // Sim(3) fits the fixture's scale; the result replaces the previous one.
  await page.locator("#trajectory-align").selectOption("sim3");
  await page.locator("#trajectory-run").click();
  await expect(stats).toContainText(/Sim\(3\), scale 1\.03/);
  await expect(rows).toHaveCount(3);
  await expect(page.locator("#trajectory-apply")).toBeDisabled();
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
    const canvas = page.locator("#viewport > canvas");
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

test("share link: reopens the sample with its C2C and view settings", async ({ page, context }) => {
  await page.getByRole("button", { name: "Try a sample" }).click();
  await expect(status(page)).toContainText("C2C distance computed");
  await page.locator("#point-size").fill("5");
  await page.locator("#share").click();
  const link = await page.locator("#share-link").inputValue();
  expect(link).toContain("#session=");

  const other = await context.newPage();
  await other.goto(link);
  await expect(other.locator("#status")).toContainText("Session restored (2 clouds)");
  await expect(other.locator(".cloud-list li")).toHaveCount(2);
  await expect(other.locator("#c2c-stats")).toContainText("0.022061");
  await expect(other.locator("#point-size")).toHaveValue("5");
});

test("session file: restores visibility once its files are opened", async ({ page, context }) => {
  const files = [
    { name: "reference.ply", buffer: ply(grid(30)) },
    { name: "lifted.ply", buffer: ply(grid(30, 0.5)) },
  ];
  await open(page, files);
  await expect(status(page)).toContainText("Loaded lifted.ply");
  await page.locator(".cloud-list li").first().locator('input[type="checkbox"]').uncheck();
  const download = page.waitForEvent("download");
  await page.locator("#session-save").click();
  const saved = await download;
  expect(saved.suggestedFilename()).toBe("session.cloudanalyzer.json");
  const chunks: Buffer[] = [];
  for await (const chunk of await saved.createReadStream()) chunks.push(chunk as Buffer);
  const session = Buffer.concat(chunks);

  // The session alone asks for its files; opening them finishes the restore.
  const other = await context.newPage();
  await other.goto("/");
  await open(other, [{ name: "session.cloudanalyzer.json", buffer: session }]);
  await expect(other.locator("#status")).toContainText("Session: open reference.ply, lifted.ply");
  await open(other, files);
  await expect(other.locator("#status")).toContainText("Session restored (2 clouds)");
  const boxes = other.locator('.cloud-list li input[type="checkbox"]');
  await expect(boxes.first()).not.toBeChecked();
  await expect(boxes.last()).toBeChecked();
});

test("?url= opens a cloud from a URL", async ({ page }) => {
  await page.goto("/?url=samples/lidar_reference.pcd");
  await expect(status(page)).toContainText("Loaded lidar_reference.pcd");
  await expect(page.locator(".cloud-list li")).toHaveCount(1);
});

test("profile: a line across two flat grids plots both, saves CSV and survives a share link", async ({
  page,
  context,
}) => {
  const flat = (z: number) => grid(60).map(([x, y]) => [x, y, z] as [number, number, number]);
  await open(page, [
    { name: "low.ply", buffer: ply(flat(0)) },
    { name: "high.ply", buffer: ply(flat(1)) },
  ]);
  await expect(status(page)).toContainText("Loaded high.ply");
  await page.locator('[data-view="top"]').click();
  await page.locator("#fit").click();
  await page.locator("#profile-width").fill("0.2");
  await page.locator("#profile-draw").click();
  const canvas = page.locator("#viewport > canvas");
  const box = (await canvas.boundingBox())!;
  await canvas.click({ position: { x: box.width * 0.35, y: box.height / 2 } });
  await canvas.click({ position: { x: box.width * 0.65, y: box.height / 2 } });
  await page.keyboard.press("Enter");
  await expect(status(page)).toContainText(/Profile: [\d,]+ points from 2 clouds within 0\.2/);
  await expect(page.locator("#profile-plot")).toBeVisible();
  await expect(page.locator("#profile-legend")).toContainText("low.ply");

  const download = page.waitForEvent("download");
  await page.locator("#profile-csv").click();
  const file = await download;
  expect(file.suggestedFilename()).toBe("profile.csv");
  const chunks: Buffer[] = [];
  for await (const chunk of await file.createReadStream()) chunks.push(chunk as Buffer);
  const csv = Buffer.concat(chunks).toString("utf8").trim().split("\n");
  expect(csv[0]).toBe("cloud,distance,x,y,z");
  // Every point of "high" is at z = 1, every point of "low" at z = 0.
  for (const row of csv.slice(1)) {
    const [name, , , , z] = row.split(",");
    expect(Number(z)).toBeCloseTo(name === "high.ply" ? 1 : 0, 5);
  }

  // A share link carries the line (these local files are then asked for).
  await page.locator("#share").click();
  const link = await page.locator("#share-link").inputValue();
  const other = await context.newPage();
  await other.goto(link);
  await expect(other.locator("#status")).toContainText("Session: open low.ply, high.ply");
  await open(other, [
    { name: "low.ply", buffer: ply(flat(0)) },
    { name: "high.ply", buffer: ply(flat(1)) },
  ]);
  await expect(other.locator("#status")).toContainText("Session restored");
  await expect(other.locator("#profile-plot")).toBeVisible();
  await expect(other.locator("#profile-width")).toHaveValue("0.2");
});

test("merge and split: by class, and back into the merged files", async ({ page }) => {
  const lasPoints = Array.from({ length: 100 }, (_, i) => ({
    xyz: [i % 10, Math.floor(i / 10), 0] as [number, number, number],
    intensity: 100,
    cls: i < 60 ? 2 : 6,
  }));
  const plyPoints = Array.from({ length: 30 }, (_, i) => [i % 6, Math.floor(i / 6), 3] as [number, number, number]);
  await open(page, [
    { name: "a.las", buffer: las(lasPoints) },
    { name: "b.ply", buffer: ply(plyPoints) },
  ]);
  await expect(status(page)).toContainText("Loaded b.ply");
  await page.locator("#merge-run").click();
  await expect(status(page)).toContainText("merged_2: 130 points from a.las, b.ply");

  await page.locator("#split-cloud").selectOption({ label: "merged_2" });
  await page.locator("#split-by").selectOption("classification");
  await page.locator("#split-run").click();
  // b.ply had no classes, so its points are class 0.
  await expect(status(page)).toContainText("Split merged_2 into 3 clouds:");
  await expect(status(page)).toContainText("(30)");
  await expect(status(page)).toContainText("(60)");
  await expect(status(page)).toContainText("(40)");

  await page.locator("#split-cloud").selectOption({ label: "merged_2" });
  await page.locator("#split-by").selectOption("source");
  await page.locator("#split-run").click();
  await expect(status(page)).toContainText("Split merged_2 into 2 clouds: a.las, b.ply");
  const names = page.locator(".cloud-list li .name");
  await expect(names.filter({ hasText: /^a\.las100 points/ })).toHaveCount(2);
  await expect(names.filter({ hasText: /^b\.ply30 points/ })).toHaveCount(2);
});

test("shapes: RANSAC finds a floor and a wall, and the result splits by segment", async ({ page }) => {
  // A 6 x 6 m floor, a wall at x = 2 from 1 m up, and three stray points.
  const scene: [number, number, number][] = [];
  for (let j = 0; j < 60; j++) for (let i = 0; i < 60; i++) scene.push([i * 0.1, j * 0.1, 0]);
  for (let j = 0; j < 30; j++) for (let i = 0; i < 60; i++) scene.push([2, i * 0.1, 1 + j * 0.1]);
  scene.push([5, 5, 3], [0.5, 4, 2], [4, 1, 1.5]);
  await open(page, [{ name: "room.ply", buffer: ply(scene) }]);
  await expect(status(page)).toContainText("Loaded room.ply: 5,403 points");
  await page.locator("#shapes-distance").fill("0.02");
  await page.locator("#shapes-min").fill("500");
  await page.locator("#shapes-run").click();
  await expect(status(page)).toContainText("2 planes in room.ply: 5,400 of 5,403 points, 3 left");
  const table = page.locator("#shapes-stats");
  await expect(table.locator("tr")).toHaveCount(3);
  await expect(table.locator("tr").nth(0)).toContainText("plane 1 (3,600)");
  await expect(table.locator("tr").nth(0)).toContainText("normal (0.000, 0.000, 1.000), d 0");
  await expect(table.locator("tr").nth(1)).toContainText("plane 2 (1,800)");
  await expect(table.locator("tr").nth(1)).toContainText("normal (1.000, 0.000, 0.000), d -2");
  await expect(table.locator("tr").nth(2)).toContainText("rest (3)");

  // The segment is the cloud's source, so Merge / split takes it apart.
  await page.locator("#split-cloud").selectOption({ label: "room_planes" });
  await page.locator("#split-by").selectOption("source");
  await page.locator("#split-run").click();
  await expect(status(page)).toContainText("Split room_planes into 3 clouds: room_plane1, room_plane2, room_rest");
});

test("clusters: two separated grids and stray points, a cloud per cluster", async ({ page }) => {
  const blobs = [...grid(20), ...grid(20).map(([x, y, z]) => [x + 10, y, z] as [number, number, number])];
  blobs.push([5, 0, 0], [5, 1.5, 1], [-4, 1, 0]);
  await open(page, [{ name: "blobs.ply", buffer: ply(blobs) }]);
  await expect(status(page)).toContainText("Loaded blobs.ply: 803 points");
  await page.locator("#shapes-method").selectOption("cluster");
  // Defaults: three times the point spacing, at least 10 points.
  await expect(page.locator("#shapes-min")).toHaveValue("10");
  await page.locator("#shapes-output").selectOption("split");
  await page.locator("#shapes-run").click();
  await expect(status(page)).toContainText("2 clusters in blobs.ply: 800 of 803 points, 3 noise");
  await expect(page.locator("#shapes-stats")).toContainText("cluster 1 (400)");
  await expect(page.locator("#shapes-stats")).toContainText("noise (3)");
  const names = page.locator(".cloud-list li .name");
  await expect(names).toHaveCount(4);
  await expect(names.filter({ hasText: /^blobs_cluster1400 points/ })).toHaveCount(1);
  await expect(names.filter({ hasText: /^blobs_cluster2400 points/ })).toHaveCount(1);
  await expect(names.filter({ hasText: /^blobs_noise3 points/ })).toHaveCount(1);
});

test("normals: estimated, shaded, saved to PLY and read back", async ({ page }) => {
  await open(page, [{ name: "wave.ply", buffer: ply(grid(60)) }]);
  await expect(status(page)).toContainText("Loaded wave.ply");
  await page.locator("#normals-run").click();
  await expect(status(page)).toContainText("Normals of 3,600 points");
  const mode = page.locator(".cloud-list li").first().locator("select");
  await expect(mode).toHaveValue("shade");

  const download = page.waitForEvent("download");
  await page.locator(".cloud-list li").first().locator("button.icon").click();
  await page.locator(".cloud-list .save-formats").getByRole("button", { name: "PLY" }).click();
  const file = await download;
  const chunks: Buffer[] = [];
  for await (const chunk of await file.createReadStream()) chunks.push(chunk as Buffer);
  const saved = Buffer.concat(chunks);
  const header = saved.subarray(0, 400).toString("latin1");
  for (const n of ["nx", "ny", "nz"]) expect(header).toContain(`property float ${n}`);

  await open(page, [{ name: "again.ply", buffer: saved }]);
  await expect(status(page)).toContainText("Loaded again.ply");
  const again = page.locator(".cloud-list li").last().locator("select");
  await expect(again.locator('option[value="normal"]')).toHaveCount(1);
  await again.selectOption("normal");
});

test("progress and cancel: a stalled download can be stopped; memory is shown", async ({ page }) => {
  // A server that never finishes answering.
  await page.route("**/stalled/slow.ply", () => new Promise(() => {}));
  await page.locator("#url-input").fill("/stalled/slow.ply");
  await page.locator("#url-open").click();
  await expect(page.locator("#task")).toBeVisible();
  await expect(status(page)).toContainText("Downloading slow.ply");
  await page.locator("#task-cancel").click();
  await expect(status(page)).toContainText("Stopped downloading slow.ply");
  await expect(page.locator("#task")).toBeHidden();

  // A normal load hides the bar again and reports the worker's memory.
  await open(page, [{ name: "small.ply", buffer: ply(grid(20)) }]);
  await expect(status(page)).toContainText("Loaded small.ply");
  await expect(page.locator("#task")).toBeHidden();
  await expect(page.locator("#memory")).toContainText(/WASM \d+ MB/);
});

test("demos: ?demo= loads synthetic samples and runs the analysis", async ({ page }) => {
  await page.goto("/?demo=volume");
  // A 4 m paraboloid heap (π r² h / 2 ≈ 402 m³) and a 45 m³ pit.
  await expect(status(page)).toContainText(/Volume: fill 40\d\.\d+ m³, cut 4\d\.\d+ m³/);
  await page.goto("/");
  await page.locator('[data-demo="ground"]').click();
  await expect(status(page)).toContainText(/town_csf: [\d,]+ of 69,200 points are ground/);
});

test.describe("COPC", () => {
  const copc = readFileSync(new URL("./fixtures/small.copc.laz", import.meta.url));

  test("a local COPC file loads its octree levels", async ({ page }) => {
    await open(page, [{ name: "small.copc.laz", buffer: copc }]);
    await expect(status(page)).toContainText("Loaded small.copc.laz: 42,000 of 42,000 points (COPC levels 0–2)");
    await expect(page.locator("#class-panel")).toBeVisible();
  });

  test("a remote COPC file is read with range requests", async ({ page }) => {
    const ranges: string[] = [];
    await page.route((url) => url.pathname === "/remote/small.copc.laz", (route) => {
      const range = route.request().headers().range;
      ranges.push(range ?? "none");
      const match = /bytes=(\d+)-(\d+)/.exec(range ?? "");
      if (!match) return route.fulfill({ status: 200, body: copc });
      const [start, end] = [Number(match[1]), Math.min(Number(match[2]), copc.length - 1)];
      return route.fulfill({
        status: 206,
        headers: { "content-range": `bytes ${start}-${end}/${copc.length}`, "accept-ranges": "bytes" },
        body: copc.subarray(start, end + 1),
      });
    });
    await page.goto("/?url=/remote/small.copc.laz");
    await expect(status(page)).toContainText("Loaded small.copc.laz: 42,000 of 42,000 points (COPC levels 0–2)");
    // Never the whole file at once.
    expect(ranges.every((r) => r.startsWith("bytes="))).toBe(true);
    expect(ranges.length).toBeGreaterThan(2);
  });
});

test("E57: scans are merged, split back per scan, and saved as E57", async ({ page }) => {
  const e57 = readFileSync(new URL("./fixtures/scans.e57", import.meta.url));
  await open(page, [{ name: "scans.e57", buffer: e57 }]);
  await expect(status(page)).toContainText("Loaded scans.e57: 3,000 points");
  await expect(page.locator(".cloud-list li .meta")).toContainText("· RGB");

  await page.locator("#split-cloud").selectOption({ label: "scans.e57" });
  await page.locator("#split-by").selectOption("source");
  await page.locator("#split-run").click();
  await expect(status(page)).toContainText("Split scans.e57 into 2 clouds: station1, station2");

  const download = page.waitForEvent("download");
  const station2 = page.locator(".cloud-list li", { hasText: "station2" });
  await station2.locator("button.icon").click();
  await station2.locator(".save-formats").getByRole("button", { name: "E57" }).click();
  const file = await download;
  expect(file.suggestedFilename()).toBe("station2.e57");
  const chunks: Buffer[] = [];
  for await (const chunk of await file.createReadStream()) chunks.push(chunk as Buffer);
  const saved = Buffer.concat(chunks);
  expect(saved.subarray(0, 8).toString("latin1")).toBe("ASTM-E57");
  await open(page, [{ name: "saved.e57", buffer: saved }]);
  await expect(status(page)).toContainText("Loaded saved.e57: 1,500 points");
});

test("labels: added by clicking, edited, saved in a PNG and restored from a share link", async ({ page, context }) => {
  await page.getByRole("button", { name: "Try a sample" }).click();
  await expect(status(page)).toContainText("C2C distance computed");
  await page.locator("#label").click();
  const canvas = page.locator("#viewport > canvas");
  const box = (await canvas.boundingBox())!;
  // Try a few spots until one hits a point.
  for (const [fx, fy] of [[0.5, 0.5], [0.45, 0.5], [0.5, 0.45], [0.55, 0.52]]) {
    await canvas.click({ position: { x: box.width * fx, y: box.height * fy } });
    if ((await page.locator("#note-list li").count()) > 0) break;
  }
  const input = page.locator("#note-list input").first();
  await expect(input).toHaveValue(/^Z -?\d/);
  await input.fill("Tree trunk");
  await expect(page.locator("#labels span.note")).toHaveText("Tree trunk");

  const download = page.waitForEvent("download");
  await page.locator("#save-image").click();
  const file = await download;
  expect(file.suggestedFilename()).toBe("cloudanalyzer-view.png");
  const chunks: Buffer[] = [];
  for await (const chunk of await file.createReadStream()) chunks.push(chunk as Buffer);
  const png = Buffer.concat(chunks);
  expect(png.subarray(1, 4).toString("latin1")).toBe("PNG");
  expect(png.length).toBeGreaterThan(10_000);

  await page.locator("#share").click();
  const link = await page.locator("#share-link").inputValue();
  const other = await context.newPage();
  await other.goto(link);
  await expect(other.locator("#status")).toContainText("Session restored");
  await expect(other.locator("#note-list input")).toHaveValue("Tree trunk");
  await expect(other.locator("#labels span.note")).toHaveText("Tree trunk");
});

test("display: adaptive point size, background colour and saved views survive a share link", async ({ page, context }) => {
  await page.getByRole("button", { name: "Try a sample" }).click();
  await expect(status(page)).toContainText("C2C distance computed");
  await page.locator("#point-size-mode").selectOption("adaptive");
  await expect(page.locator("#edl")).toBeDisabled();
  await page.locator("#background").evaluate((el: HTMLInputElement) => {
    el.value = "#ffffff";
    el.dispatchEvent(new Event("input"));
  });

  await page.locator('[data-view="top"]').click();
  await page.locator("#view-save").click();
  await page.locator("#view-list input").first().fill("From above");
  await page.locator('[data-view="side"]').click();
  await page.locator("#view-save").click();
  await expect(page.locator("#view-list li")).toHaveCount(2);

  await page.locator("#share").click();
  const link = await page.locator("#share-link").inputValue();
  const other = await context.newPage();
  await other.goto(link);
  await expect(other.locator("#status")).toContainText("Session restored");
  await expect(other.locator("#point-size-mode")).toHaveValue("adaptive");
  await expect(other.locator("#edl")).toBeDisabled();
  await expect(other.locator("#background")).toHaveValue("#ffffff");
  await expect(other.locator("#view-list input").first()).toHaveValue("From above");
  await other.locator("#view-list li").first().getByRole("button", { name: "Go" }).click();
});

test("tools: measuring, labeling and drawing a profile take turns; Esc leaves them", async ({ page }) => {
  await open(page, [{ name: "grid.ply", buffer: ply(grid(60)) }]);
  await expect(status(page)).toContainText("Loaded grid.ply");
  await page.locator("[data-view=top]").click();
  const measure = page.locator("#measure");
  const label = page.locator("#label");
  const draw = page.locator("#profile-draw");

  await page.keyboard.press("m");
  await expect(measure).toHaveAttribute("aria-pressed", "true");
  await expect(page.locator("#measure-hint")).toHaveText("Click the first point.");
  const canvas = page.locator("#viewport > canvas");
  const box = (await canvas.boundingBox())!;
  for (const fx of [0.4, 0.6]) {
    await canvas.click({ position: { x: box.width * fx, y: box.height * 0.5 } });
    if (fx === 0.4) await expect(page.locator("#measure-hint")).toHaveText(/second point/);
  }
  await expect(page.locator("#measure-list li")).toHaveCount(1);
  await expect(status(page)).toContainText("Distance:");

  // Another tool replaces it.
  await label.click();
  await expect(label).toHaveAttribute("aria-pressed", "true");
  await expect(measure).toHaveAttribute("aria-pressed", "false");
  await draw.click();
  await expect(draw).toHaveText("Finish line");
  await expect(label).toHaveAttribute("aria-pressed", "false");

  // Esc cancels the line and leaves the tool; a second Esc does nothing more.
  await page.keyboard.press("Escape");
  await expect(draw).toHaveText("Draw line");
  await expect(page.locator("#viewport")).not.toHaveClass(/measuring/);
  await expect(page.locator("#measure-list li")).toHaveCount(1);
});

test("segment: a lasso keeps or splits points; undo and redo restore the list", async ({ page }) => {
  await open(page, [{ name: "grid.ply", buffer: ply(grid(60)) }]);
  await expect(status(page)).toContainText("Loaded grid.ply");
  await page.locator("[data-view=top]").click();
  const canvas = page.locator("#viewport > canvas");
  const box = (await canvas.boundingBox())!;
  const lasso = async () => {
    await page.keyboard.press("s");
    await expect(page.locator("#segment-bar")).toBeVisible();
    for (const [fx, fy] of [[0.45, 0.45], [0.55, 0.45], [0.55, 0.55], [0.45, 0.55]]) {
      await canvas.click({ position: { x: box.width * fx, y: box.height * fy } });
    }
  };
  const rows = page.locator("#cloud-list li");
  const count = async (i: number) =>
    Number((await rows.nth(i).locator(".meta").textContent())!.match(/^([\d,]+) points/)![1].replace(/,/g, ""));

  await lasso();
  await page.keyboard.press("Enter");
  await expect(status(page)).toContainText(/Segmented: grid_segmented \([\d,]+\)/);
  await expect(page.locator("#segment-bar")).toBeHidden();
  await expect(rows).toHaveCount(2);
  const inside = await count(1);
  expect(inside).toBeGreaterThan(0);
  expect(inside).toBeLessThan(3600);
  await expect(rows.nth(0).locator("input[type=checkbox]")).not.toBeChecked();

  await page.keyboard.press("Control+z");
  await expect(status(page)).toContainText("Undid the lasso cut");
  await expect(rows).toHaveCount(1);
  await expect(rows.nth(0).locator("input[type=checkbox]")).toBeChecked();
  await page.keyboard.press("Control+Shift+z");
  await expect(status(page)).toContainText("Redid the lasso cut");
  await expect(rows).toHaveCount(2);
  expect(await count(1)).toBe(inside);
  await page.locator("#undo").click();
  await expect(rows).toHaveCount(1);

  // Split: both parts, together every point.
  await lasso();
  await page.locator('[data-keep="both"]').click();
  await expect(rows).toHaveCount(3);
  expect((await count(1)) + (await count(2))).toBe(3600);
  expect(await count(1)).toBe(inside);
  // A new step drops the redo history.
  await page.locator("#undo").click();
  await expect(page.locator("#redo")).toBeEnabled();
  await lasso();
  await page.locator('[data-keep="outside"]').click();
  await expect(status(page)).toContainText("grid_remaining");
  expect(await count(1)).toBe(3600 - inside);
  await expect(page.locator("#redo")).toBeDisabled();
});

test("manual alignment: a typed matrix, point pairs and the gizmo, all undoable", async ({ page }) => {
  // Dense enough that a click always lands within picking range of a point.
  const shifted = grid(100).map(([x, y, z]) => [x + 1, y + 0.5, z + 0.2] as [number, number, number]);
  await open(page, [
    { name: "reference.ply", buffer: ply(grid(100)) },
    { name: "moved.ply", buffer: ply(shifted) },
  ]);
  await expect(status(page)).toContainText("Loaded moved.ply");
  await expect(page.locator("#align-moving")).toHaveValue(/\d+/);
  await expect(page.locator("#align-moving option:checked")).toHaveText("moved.ply");
  await expect(page.locator("#align-reference option:checked")).toHaveText("reference.ply");

  // A typed rigid matrix moves the copy back onto the reference exactly.
  await page.locator("#align-panel summary").click();
  await page.locator("#align-matrix").fill("1 0 0 -1\n0 1 0 -0.5\n0 0 1 -0.2\n0 0 0 1");
  await page.locator("#align-apply-matrix").click();
  await expect(status(page)).toContainText("Applied the matrix to moved.ply");
  await page.locator("#c2c-run").click();
  await expect(status(page)).toContainText("C2C distance computed");
  expect(Number(await page.locator("#c2c-stats tr", { hasText: "Max" }).locator("td").textContent())).toBeLessThan(1e-5);
  // A scale is refused: undo could not invert it.
  await page.locator("#align-matrix").fill("2 0 0 0\n0 2 0 0\n0 0 2 0\n0 0 0 1");
  await page.locator("#align-apply-matrix").click();
  await expect(status(page)).toContainText("Only rigid transforms");
  await page.keyboard.press("Control+z");
  await expect(status(page)).toContainText("Undid the matrix");

  // Point pairs: picks must land on the right cloud; three pairs enable Align.
  await page.locator("[data-view=top]").click();
  await page.locator("#cloud-list li").nth(0).locator("input[type=checkbox]").uncheck();
  const canvas = page.locator("#viewport > canvas");
  const box = (await canvas.boundingBox())!;
  const spots = [[0.45, 0.45], [0.55, 0.47], [0.5, 0.56]];
  await page.locator("#align-pick").click();
  const rows = page.locator("#cloud-list li");
  for (const [fx, fy] of spots) {
    await rows.nth(0).locator("input[type=checkbox]").uncheck();
    await rows.nth(1).locator("input[type=checkbox]").check();
    await canvas.click({ position: { x: box.width * fx, y: box.height * fy } });
      await expect(page.locator("#align-hint")).toContainText("Now the same spot on reference.ply");
    await rows.nth(1).locator("input[type=checkbox]").uncheck();
    await rows.nth(0).locator("input[type=checkbox]").check();
    await canvas.click({ position: { x: box.width * fx, y: box.height * fy } });
  }
  await expect(page.locator("#align-pairs li")).toHaveCount(3);
  await page.keyboard.press("Escape");
  await expect(page.locator("#align-run")).toBeEnabled();
  await page.locator("#align-run").click();
  await expect(status(page)).toContainText(/Aligned moved\.ply with 3 pairs: RMS/);
  await expect(page.locator("#align-pairs li").first()).toContainText("residual");
  await expect(page.locator("#undo")).toHaveAttribute("title", /Undo the point-pair alignment/);

  // The gizmo: apply without dragging leaves the cloud where it is, cancel ends it.
  await page.locator("#gizmo-translate").click();
  await expect(page.locator("#gizmo-apply")).toBeEnabled();
  await page.locator("#gizmo-rotate").click();
  await expect(page.locator("#gizmo-rotate")).toHaveAttribute("aria-pressed", "true");
  await page.locator("#gizmo-cancel").click();
  await expect(page.locator("#gizmo-apply")).toBeDisabled();
  await page.locator("#gizmo-translate").click();
  await page.locator("#gizmo-apply").click();
  await expect(status(page)).toContainText("Moved moved.ply");
  await expect(page.locator("#undo")).toHaveAttribute("title", /Undo the manual move/);
});

test("QA report: gates on the latest results pass or fail the HTML and JSON report", async ({ page, context }) => {
  await open(page, [
    { name: "reference.ply", buffer: ply(grid(40)) },
    { name: "lifted.ply", buffer: ply(grid(40, 0.05)) },
  ]);
  await expect(status(page)).toContainText("Loaded lifted.ply");
  await expect(page.locator("#gate-add")).toBeDisabled();
  await page.locator("#c2c-run").click();
  await expect(status(page)).toContainText("C2C distance computed");

  // Two gates on the C2C result: the mean passes, the max fails.
  await page.locator("#gate-add").click();
  await page.locator("#gate-add").click();
  const gates = page.locator("#gate-list li");
  await expect(gates).toHaveCount(2);
  const setGate = async (i: number, metric: string, op: string, threshold: string) => {
    const row = gates.nth(i);
    await row.locator("select").first().selectOption({ label: metric });
    await row.locator("select").nth(1).selectOption(op);
    await row.locator("input").fill(threshold);
    await row.locator("input").blur();
  };
  await setGate(0, "C2C lifted.ply → reference.ply · Mean", "<=", "0.1");
  await setGate(1, "C2C lifted.ply → reference.ply · Max", "<=", "0.01");
  await expect(gates.nth(0).locator(".badge")).toHaveText(/^pass · 0\.05/);
  await expect(gates.nth(1).locator(".badge")).toHaveText(/^fail · 0\.05/);

  const html = page.waitForEvent("download");
  await page.locator("#report-html").click();
  const htmlFile = await html;
  expect(htmlFile.suggestedFilename()).toBe("cloudanalyzer-report.html");
  const htmlText = await new Response(await htmlFile.createReadStream() as unknown as ReadableStream).text();
  expect(htmlText).toContain("FAIL · 1 passed, 1 failed");
  expect(htmlText).toContain("C2C lifted.ply → reference.ply");
  expect(htmlText).toContain("data:image/png;base64,");
  await expect(status(page)).toContainText("FAIL (1 of 2 gates passed)");

  const json = page.waitForEvent("download");
  await page.locator("#report-json").click();
  const report = JSON.parse(await new Response(await (await json).createReadStream() as unknown as ReadableStream).text());
  expect(report.schema_version).toBe("cloudanalyzer.web_report.v0.1");
  expect(report.gate_summary.schema_version).toBe("cloudanalyzer.gate_summary.v0.1");
  expect(report.gate_summary).toMatchObject({ passed: false, exit_code: 1, pass_count: 1, fail_count: 1 });
  expect(report.checks.map((c: { status: string }) => c.status)).toEqual(["pass", "fail"]);
  expect(report.sections["distance:lifted.ply"].metrics.mean.value).toBeCloseTo(0.05, 4);
  expect(report.clouds).toHaveLength(2);

  // Gates travel in share links.
  await page.locator("#share").click();
  const other = await context.newPage();
  await other.goto(await page.locator("#share-link").inputValue());
  await expect(other.locator("#gate-list li")).toHaveCount(2);
  await expect(other.locator("#gate-list li").first().locator("input")).toHaveValue("0.1");
});

test("large coordinates: a UTM cloud opened after a local one picks, measures and cuts exactly", async ({ page }) => {
  // The local cloud comes first, so the global shift is zero and the UTM
  // cloud must be drawn relative to its own shift (float32 would round y
  // to 0.25 m here).
  await open(page, [{ name: "grid.ply", buffer: ply(grid(20)) }]);
  await expect(status(page)).toContainText("Loaded grid.ply");
  const utm: { xyz: [number, number, number]; intensity: number; cls: number }[] = [];
  for (let j = 0; j <= 10; j++) {
    for (let i = 0; i <= 10; i++) {
      utm.push({ xyz: [368000 + i * 0.01, 3955000 + j * 0.01, 50.123], intensity: 0, cls: 2 });
    }
  }
  await open(page, [{ name: "utm.las", buffer: las(utm, [368000, 3955000, 0]) }]);
  await expect(status(page)).toContainText("Loaded utm.las: 121 points");
  await expect(page.locator("#shift")).toHaveText("");
  const rows = page.locator("#cloud-list li");
  await rows.nth(0).locator("input[type=checkbox]").uncheck();
  await page.locator("[data-view=top]").click();
  await page.locator("#fit").click();

  // Fit frames the 10 cm square's bounding sphere: at the target, half the
  // view height spans radius / cos(25°) (50° field of view).
  const canvas = page.locator("#viewport > canvas");
  const box = (await canvas.boundingBox())!;
  const halfSpan = Math.hypot(0.05, 0.05) / Math.cos((25 * Math.PI) / 180);
  const at = (dx: number, dy: number) => ({
    x: box.width / 2 + (dx / halfSpan) * (box.height / 2),
    y: box.height / 2 - (dy / halfSpan) * (box.height / 2),
  });

  await canvas.click({ position: at(0, 0) });
  const info = page.locator("#pick-info");
  await expect(info).toContainText("368000.050");
  await expect(info).toContainText("3955000.050");
  await expect(info).toContainText("50.123");

  await page.keyboard.press("m");
  await canvas.click({ position: at(-0.02, -0.01) });
  await expect(page.locator("#measure-hint")).toHaveText(/second point/);
  await canvas.click({ position: at(0.01, 0.03) });
  const measured = page.locator("#measure-list li");
  await expect(measured).toHaveCount(1);
  await expect(measured.locator(".distance")).toHaveText("0.05");
  await expect(measured.locator(".delta")).toHaveText("ΔX 0.03  ΔY 0.04  ΔZ 0");
  await page.keyboard.press("Escape");

  // A lasso around the 3 x 3 points 2-4 cm from the corner.
  await page.keyboard.press("s");
  for (const [dx, dy] of [[-0.035, -0.035], [-0.005, -0.035], [-0.005, -0.005], [-0.035, -0.005]]) {
    await canvas.click({ position: at(dx, dy) });
  }
  await page.keyboard.press("Enter");
  await expect(status(page)).toContainText("Segmented: utm_segmented (9)");
});

test("scalar fields: histogram, color by field, range filter and a calculator field saved to PLY", async ({ page }) => {
  // 400 points; intensity 0..399 so a range keeps an exact count.
  const points = grid(20).map((xyz, i) => ({ xyz, intensity: i, cls: 1 }));
  await open(page, [
    { name: "scan.las", buffer: las(points) },
    { name: "lifted.ply", buffer: ply(grid(20, 0.25)) },
  ]);
  await expect(status(page)).toContainText("Loaded lifted.ply");
  await page.locator("#c2c-compared").selectOption({ label: "scan.las" });
  await page.locator("#c2c-reference").selectOption({ label: "lifted.ply" });
  await page.locator("#c2c-run").click();
  await expect(status(page)).toContainText("C2C distance computed");

  await page.locator("#field-cloud").selectOption({ label: "scan.las" });
  await expect(page.locator("#field-name option")).toHaveText(["C2C distance", "intensity", "Z"]);
  await page.locator("#field-name").selectOption("intensity");
  await expect(page.locator("#field-stats tr", { hasText: "Max" }).locator("td")).toHaveText("399");
  await expect(page.locator("#field-histogram canvas")).toBeVisible();

  await page.locator("#field-name").selectOption("Z");
  await page.locator("#field-color").click();
  await expect(page.locator("#colorbar-title")).toHaveText("Z · scan.las");

  // Keep intensity 100..199: exactly 100 points.
  await page.locator("#field-name").selectOption("intensity");
  await page.locator("#field-lo").fill("100");
  await page.locator("#field-hi").fill("199");
  await page.locator("#field-keep").click();
  await expect(status(page)).toContainText("scan_in: 100 of 400 points with intensity within 100 … 199");
  await page.keyboard.press("Control+z");
  await expect(status(page)).toContainText("Undid the intensity filter");

  // Calculator: the distance in millimetres, colored and exported.
  await page.locator("#field-cloud").selectOption({ label: "scan.las" });
  await page.locator("#field-panel summary").click();
  await page.locator("#field-calc-name").fill("mm");
  await page.locator("#field-expr").fill("abs([C2C distance]) * 1000");
  await page.locator("#field-calc").click();
  await expect(status(page)).toContainText("Computed mm = abs([C2C distance]) * 1000 for 400 points");
  await expect(page.locator("#colorbar-title")).toHaveText("mm · scan.las");
  await expect(page.locator("#field-stats tr", { hasText: "Mean" }).locator("td")).toHaveText(/^2\d\d/);
  await page.locator("#field-expr").fill("abs(");
  await page.locator("#field-calc").click();
  await expect(status(page)).toContainText("Calculator: unexpected end");

  const row = page.locator(".cloud-list li", { hasText: "scan.las" });
  await expect(row.locator("select").first().locator("option", { hasText: "mm" })).toHaveCount(1);
  const download = page.waitForEvent("download");
  await row.locator("button.icon").click();
  await row.locator(".save-formats").getByRole("button", { name: "PLY" }).click();
  const chunks: Buffer[] = [];
  for await (const chunk of await (await download).createReadStream()) chunks.push(chunk as Buffer);
  const header = Buffer.concat(chunks).subarray(0, 600).toString("latin1");
  expect(header).toMatch(/property float mm/);
});

test("subsampling: minimum distance and octree level", async ({ page }) => {
  await open(page, [{ name: "dense.ply", buffer: ply(grid(64)) }]);
  await expect(status(page)).toContainText("Loaded dense.ply: 4,096 points");
  await page.locator("#filter-op").selectOption("octree");
  await expect(page.locator("#filter-level-hint")).toContainText("Cells of");
  await page.locator("#filter-level").fill("3");
  await page.locator("#filter-run").click();
  // 64 x 64 points over an 8 x 8 grid of cells: one point per cell.
  await expect(status(page)).toContainText("dense_octree3: kept 64 of 4,096 points");
  await page.keyboard.press("Control+z");
  await page.locator("#filter-op").selectOption("spatial");
  await page.locator("#filter-spacing").fill("0.25");
  await page.locator("#filter-run").click();
  await expect(status(page)).toContainText(/dense_space0\.25: kept [\d,]+ of 4,096 points/);
  const kept = Number(/kept ([\d,]+)/.exec((await status(page).textContent())!)![1].replace(/,/g, ""));
  // Points 0.1 apart thinned to 0.25: between a 0.3 grid (22 x 22) and a 0.2 grid.
  expect(kept).toBeGreaterThanOrEqual(22 * 22);
  expect(kept).toBeLessThan(32 * 32);
});

type Pose2 = { x: number; y: number; yaw: number };

/**
 * A courtyard with pillars scanned from 24 poses around a 20 m square: the
 * true poses, dead-reckoned odometry with a yaw bias and 1 % scale error,
 * and each pose's scan as a KITTI Velodyne .bin.
 */
function courtyard(): { truth: Pose2[]; drifted: Pose2[]; scan: (p: Pose2) => Buffer } {
  const world: number[][] = [];
  for (let a = -20; a <= 20; a += 0.5) {
    for (let z = 0; z <= 4; z += 0.5) world.push([a, -20, z], [a, 20, z], [-20, a, z], [20, a, z]);
  }
  for (const [cx, cy] of [[8, 8], [-8, 8], [8, -8], [-8, -8], [0, 12], [5, -3]]) {
    for (let k = 0; k < 16; k++) {
      const t = (k / 16) * 2 * Math.PI;
      for (let z = 0; z <= 4; z += 0.5) world.push([cx + 0.5 * Math.cos(t), cy + 0.5 * Math.sin(t), z]);
    }
  }
  for (let x = -20; x <= 20; x += 1) for (let y = -20; y <= 20; y += 1) world.push([x, y, 0.1 * Math.sin(x / 3)]);

  const truth: Pose2[] = [];
  for (let k = 0; k < 24; k++) {
    const s = (k * 80) / 24;
    const side = Math.floor(s / 20);
    const d = s - side * 20;
    const [x, y] = [[-10 + d, -10], [10, -10 + d], [10 - d, 10], [-10, 10 - d]][side];
    truth.push({ x, y, yaw: (side * Math.PI) / 2 });
  }
  // Odometry with a steady yaw bias and 1 % scale error, dead-reckoned.
  const drifted: Pose2[] = [truth[0]];
  for (let k = 1; k < 24; k++) {
    const [a, b, prev] = [truth[k - 1], truth[k], drifted[k - 1]];
    const [dx, dy] = [b.x - a.x, b.y - a.y];
    const [lx, ly] = [Math.cos(a.yaw) * dx + Math.sin(a.yaw) * dy, -Math.sin(a.yaw) * dx + Math.cos(a.yaw) * dy];
    const yaw = prev.yaw + (b.yaw - a.yaw) + 0.004;
    drifted.push({
      x: prev.x + 1.01 * (Math.cos(prev.yaw) * lx - Math.sin(prev.yaw) * ly),
      y: prev.y + 1.01 * (Math.sin(prev.yaw) * lx + Math.cos(prev.yaw) * ly),
      yaw,
    });
  }
  const scan = (p: Pose2) => {
    const [c, s] = [Math.cos(p.yaw), Math.sin(p.yaw)];
    const body = Buffer.alloc(world.length * 16);
    world.forEach(([x, y, z], i) => {
      const [dx, dy] = [x - p.x, y - p.y];
      [c * dx + s * dy, -s * dx + c * dy, z, 0.5].forEach((v, a) => body.writeFloatLE(v, i * 16 + a * 4));
    });
    return body;
  };
  return { truth, drifted, scan };
}

/** A KITTI pose line (3x4, row-major) for a planar pose. */
function kittiLine(p: Pose2): string {
  const [c, s] = [Math.cos(p.yaw), Math.sin(p.yaw)];
  return `${c} ${-s} 0 ${p.x} ${s} ${c} 0 ${p.y} 0 0 1 0`;
}

/** Scan files named by frame number. */
function scanFiles(poses: Pose2[], scan: (p: Pose2) => Buffer) {
  return poses.map((p, k) => ({
    name: `${String(k).padStart(6, "0")}.bin`,
    mimeType: "application/octet-stream",
    buffer: scan(p),
  }));
}

test("pose graph: a drifted loop closed with ICP, by hand and found automatically, undone and turned into a map", async ({ page }) => {
  const { truth, drifted, scan } = courtyard();
  await page
    .locator("#pg-files-input")
    .setInputFiles([
      { name: "poses.txt", mimeType: "text/plain", buffer: Buffer.from(`${drifted.map(kittiLine).join("\n")}\n`) },
      ...scanFiles(truth, scan),
    ]);
  await expect(status(page)).toContainText("Opened poses.txt: 24 poses");
  const stats = page.locator("#pg-stats");
  await expect(stats).toContainText("24 (24 with scans)");
  await expect(stats).toContainText("23 odometry, 0 loops");

  /** Largest distance of the saved KITTI poses from the truth. */
  const saveError = async () => {
    const download = page.waitForEvent("download");
    await page.locator("#pg-save-kitti").click();
    const rows = (await bytesOf(await download)).toString().trim().split("\n");
    expect(rows).toHaveLength(24);
    return Math.max(
      ...rows.map((row, k) => {
        const m = row.split(" ").map(Number);
        return Math.hypot(m[3] - truth[k].x, m[7] - truth[k].y);
      }),
    );
  };
  expect(await saveError()).toBeGreaterThan(1);

  await page.locator("#pg-a").fill("0");
  await page.locator("#pg-b").fill("23");
  await page.locator("#pg-loop").click();
  await expect(status(page)).toContainText(/Loop 0 – 23 added .*χ²/);
  await expect(stats).toContainText("23 odometry, 1 loops");
  expect(await saveError()).toBeLessThan(0.3);

  await page.locator("#pg-undo").click();
  await expect(status(page)).toContainText("Removed the last loop");
  await expect(stats).toContainText("23 odometry, 0 loops");
  expect(await saveError()).toBeGreaterThan(1);

  // Found automatically: the last poses come back near the first ones.
  await page.locator("#pose-graph-panel summary", { hasText: "Find loops automatically" }).click();
  await page.locator("#pg-find").click();
  await expect(status(page)).toContainText(/Added [1-9]\d* of \d+ candidate loops?; χ²/);
  expect(await saveError()).toBeLessThan(0.3);
  await expect(page.locator("#pg-loop-list li").first()).toBeVisible();
  await page.locator("#pg-undo").click();
  await expect(status(page)).toContainText(/Removed the (last loop|\d+ loops found)/);
  await expect(stats).toContainText("23 odometry, 0 loops");

  await page.locator("#pg-map").click();
  await expect(status(page)).toContainText(/Added poses_map: [\d,]+ points/);
  await expect(page.locator("#cloud-list li")).toHaveCount(1);
});

test("pose graph: a wrong loop in a g2o file shows as the worst edge and is removed", async ({ page }) => {
  const { truth, scan } = courtyard();
  // The true poses and odometry, plus a loop claiming poses 3 and 15 (opposite sides) coincide.
  const quat = (yaw: number) => `0 0 ${Math.sin(yaw / 2)} ${Math.cos(yaw / 2)}`;
  const info = "100 0 0 0 0 0 100 0 0 0 0 100 0 0 0 1000 0 0 1000 0 1000";
  const relative = (a: Pose2, b: Pose2) => {
    const [dx, dy] = [b.x - a.x, b.y - a.y];
    const [c, s] = [Math.cos(a.yaw), Math.sin(a.yaw)];
    return `${c * dx + s * dy} ${-s * dx + c * dy} 0 ${quat(b.yaw - a.yaw)}`;
  };
  const lines = truth.map((p, k) => `VERTEX_SE3:QUAT ${k} ${p.x} ${p.y} 0 ${quat(p.yaw)}`);
  for (let k = 1; k < truth.length; k++) lines.push(`EDGE_SE3:QUAT ${k - 1} ${k} ${relative(truth[k - 1], truth[k])} ${info}`);
  lines.push(`EDGE_SE3:QUAT 0 23 ${relative(truth[0], truth[23])} ${info}`);
  lines.push(`EDGE_SE3:QUAT 3 15 0 0 0 0 0 0 1 ${info}`);
  await page
    .locator("#pg-files-input")
    .setInputFiles([
      { name: "graph.g2o", mimeType: "text/plain", buffer: Buffer.from(`${lines.join("\n")}\n`) },
      ...scanFiles(truth, scan),
    ]);
  await expect(status(page)).toContainText("Opened graph.g2o: 24 poses");
  const stats = page.locator("#pg-stats");
  await expect(stats).toContainText("23 odometry, 2 loops");

  // Optimised, the wrong loop still disagrees most and heads the list.
  await page.locator("#pg-optimize").click();
  await expect(status(page)).toContainText("Optimised: χ²");
  const loops = page.locator("#pg-loop-list li");
  await expect(loops).toHaveCount(2);
  await expect(loops.nth(0)).toContainText("3 – 15");
  await expect(loops.nth(1)).toContainText("0 – 23");
  const errorOf = async (row: number) => Number((await loops.nth(row).locator(".meta").textContent())!.replace("error ", ""));
  expect(await errorOf(0)).toBeGreaterThan(10);

  // Selecting a loop fills A and B.
  await loops.nth(0).locator(".link").click();
  await expect(page.locator("#pg-a")).toHaveValue("3");
  await expect(page.locator("#pg-b")).toHaveValue("15");

  await loops.nth(0).locator(".remove").click();
  await expect(status(page)).toContainText("Removed loop 3 – 15 and optimised");
  await expect(loops).toHaveCount(1);
  await expect(stats).toContainText("23 odometry, 1 loops");
  expect(await errorOf(0)).toBeLessThan(0.01);

  await page.locator("#pg-undo").click();
  await expect(status(page)).toContainText("Put back 1 removed edge");
  await expect(loops).toHaveCount(2);
  await expect(loops.nth(0)).toContainText("3 – 15");

  // Removing by threshold catches the same loop.
  await page.locator("#pg-prune-limit").fill("5");
  await page.locator("#pg-prune").click();
  await expect(status(page)).toContainText("Removed 1 loop with an error above 5 and optimised");
  await expect(loops).toHaveCount(1);

  // Height colors and error colors can be switched without errors.
  await page.locator("#pg-colors").selectOption("height");
  await page.locator("#pg-edge-errors").uncheck();
  await page.locator("#pg-show-scans").uncheck();
  await page.locator("#pg-show-scans").check();
  await expect(page.locator("#pg-legend")).toBeHidden();

  const download = page.waitForEvent("download");
  await page.locator("#pg-save-g2o").click();
  const saved = (await bytesOf(await download)).toString();
  expect(saved.match(/^EDGE_SE3:QUAT/gm)).toHaveLength(24);
});

test("pose graph: a second session in another frame is joined at a shared place", async ({ page }) => {
  const { truth, scan } = courtyard();
  // Session A: the first half, in the true frame. Session B: the second
  // half, recorded in its own frame (turned 120° and shifted).
  const [a, b] = [truth.slice(0, 12), truth.slice(12)];
  const turn = (2 * Math.PI) / 3;
  const inB = (p: Pose2): Pose2 => ({
    x: Math.cos(turn) * p.x - Math.sin(turn) * p.y + 50,
    y: Math.sin(turn) * p.x + Math.cos(turn) * p.y - 7,
    yaw: p.yaw + turn,
  });
  await page
    .locator("#pg-files-input")
    .setInputFiles([
      { name: "a.txt", mimeType: "text/plain", buffer: Buffer.from(`${a.map(kittiLine).join("\n")}\n`) },
      ...scanFiles(a, scan),
    ]);
  await expect(status(page)).toContainText("Opened a.txt: 12 poses");

  // The other session as a folder; its scans are numbered from 0 too.
  const dir = mkdtempSync(join(tmpdir(), "pg-merge-"));
  writeFileSync(join(dir, "b.txt"), `${b.map(inB).map(kittiLine).join("\n")}\n`);
  for (const f of scanFiles(b, scan)) writeFileSync(join(dir, f.name), f.buffer);
  await page.locator("#pose-graph-panel summary", { hasText: "Join another graph" }).click();
  await page.locator("#pg-merge-here").fill("11");
  await page.locator("#pg-merge-input").setInputFiles(dir);
  await expect(status(page)).toContainText(/Joined b\.txt \(12 poses\) at node 11: overlap \d+ %/);
  await expect(page.locator("#pg-stats")).toContainText("24 (24 with scans)");
  await expect(page.locator("#pg-stats")).toContainText("22 odometry, 1 loops");

  const download = page.waitForEvent("download");
  await page.locator("#pg-save-kitti").click();
  const rows = (await bytesOf(await download)).toString().trim().split("\n");
  expect(rows).toHaveLength(24);
  rows.forEach((row, k) => {
    const m = row.split(" ").map(Number);
    expect(Math.hypot(m[3] - truth[k].x, m[7] - truth[k].y)).toBeLessThan(0.05);
  });
  const g2o = page.waitForEvent("download");
  await page.locator("#pg-save-g2o").click();
  // B's ids were shifted past A's.
  expect((await bytesOf(await g2o)).toString()).toContain("VERTEX_SE3:QUAT 23 ");
});
