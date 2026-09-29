// Screenshots and the animated GIF for the README. Not part of the test
// suite: run `npm run media` (after `npm run build`), which writes the
// images to docs/images/web/ (see scripts/readme-gif.mjs for the GIF).

import { expect, type Page, test } from "@playwright/test";
import { mkdirSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";

const OUT = new URL("../../docs/images/web/", import.meta.url);
const FRAMES = new URL("../media-frames/", import.meta.url);
mkdirSync(OUT, { recursive: true });

test.describe.configure({ mode: "serial" });
test.use({ viewport: { width: 1280, height: 760 } });

const status = (page: Page) => page.locator("#status");

/**
 * A real drive for the pose graph GIFs: a PandaSet scene (CC BY 4.0) made
 * with `python scripts/fetch_pandaset.py 019 <dir>`. Those GIFs are skipped without it.
 */
const PANDASET = process.env.PANDASET_DIR;

/**
 * A real drive with loops: an NCLT session (Open Database License) made with
 * `python scripts/prepare_nclt.py 2012-04-29 <dir>`. Those pictures are skipped without it.
 */
const NCLT = process.env.NCLT_DIR;

/** Open the NCLT session; `query` is added to the app's URL (e.g. to slow the glide). */
async function openNclt(page: Page, query: string): Promise<void> {
  await page.goto(`/${query}`);
  await page.locator("#pg-folder-input").setInputFiles(`${NCLT}/velodyne`);
  await expect(status(page)).toContainText("Opened kiss_poses.txt", { timeout: 900_000 });
  await page.locator("#round-points").check();
}

/**
 * Two NCLT seasons of the same campus, for the seasons GIF: `NCLT_SEASONS=<summer dir>,<winter dir>`
 * (2012-06-15, and keyframes 200-2199 of 2012-12-01, from scripts/prepare_nclt.py), joined where
 * summer keyframe 1219 and winter keyframe 850 stand within 0.1 m of each other.
 */
const SEASONS = process.env.NCLT_SEASONS?.split(",");

async function openPandaSet(page: Page): Promise<void> {
  await page.goto("/");
  await page.locator("#pg-folder-input").setInputFiles(`${PANDASET}/velodyne`);
  await expect(status(page)).toContainText("Opened poses.txt", { timeout: 300_000 });
  await page.locator("#round-points").check();
}

/** Look along the street from above and behind node `node`. */
async function streetView(page: Page, node: number, steps: number): Promise<void> {
  await page.locator("[data-view=iso]").click();
  await page.locator("#pg-a").fill(String(node));
  await page.locator("#pg-goto").click();
  await zoom(page, steps);
  await page.locator("#pg-a").fill("");
}
const canvas = (page: Page) => page.locator("#viewport > canvas");

async function shot(page: Page, name: string): Promise<void> {
  // Let LOD refinement and damping settle.
  await page.waitForTimeout(1200);
  await page.screenshot({ path: fileURLToPath(new URL(`${name}.jpg`, OUT)), type: "jpeg", quality: 88 });
}

/** Scroll the sidebar so a panel is in view. */
async function showPanel(page: Page, id: string): Promise<void> {
  await page.locator(`#${id}`).evaluate((el) => el.scrollIntoView({ block: "start" }));
}

async function open(page: Page, files: { name: string; buffer: Buffer }[]): Promise<void> {
  await page.locator("#file-input").setInputFiles(
    files.map((f) => ({ name: f.name, mimeType: "application/octet-stream", buffer: f.buffer })),
  );
}

function ply(points: number[][], colors?: number[][]): Buffer {
  const rgb = colors ? "property uchar red\nproperty uchar green\nproperty uchar blue\n" : "";
  const header =
    `ply\nformat binary_little_endian 1.0\nelement vertex ${points.length}\n` +
    `property float x\nproperty float y\nproperty float z\n${rgb}end_header\n`;
  const stride = colors ? 15 : 12;
  const body = Buffer.alloc(points.length * stride);
  points.forEach((p, i) => {
    p.forEach((v, a) => body.writeFloatLE(v, i * stride + a * 4));
    colors?.[i].forEach((c, a) => body.writeUInt8(c, i * stride + 12 + a));
  });
  return Buffer.concat([Buffer.from(header), body]);
}

/** A small room: floor, two walls and two columns, with a little noise. */
function room(): Buffer {
  let s = 7;
  const rnd = () => ((s = (s * 1103515245 + 12345) % 2147483648) / 2147483648);
  const pts: number[][] = [];
  const n = () => (rnd() - 0.5) * 0.01;
  for (let i = 0; i < 120000; i++) pts.push([rnd() * 8, rnd() * 6, n()]);
  for (let i = 0; i < 45000; i++) pts.push([n(), rnd() * 6, rnd() * 3]);
  for (let i = 0; i < 60000; i++) pts.push([rnd() * 8, 6 + n(), rnd() * 3]);
  for (const [cx, cy] of [[3, 2.5], [5.5, 3.5]]) {
    for (let i = 0; i < 15000; i++) {
      const a = rnd() * 2 * Math.PI;
      pts.push([cx + 0.25 * Math.cos(a) + n(), cy + 0.25 * Math.sin(a) + n(), rnd() * 3]);
    }
  }
  return ply(pts);
}

async function orbitFrames(page: Page, prefix: string, count: number, dx: number): Promise<void> {
  const box = (await canvas(page).boundingBox())!;
  const [x, y] = [box.x + box.width / 2, box.y + box.height / 2];
  mkdirSync(FRAMES, { recursive: true });
  await page.mouse.move(x, y);
  await page.mouse.down();
  for (let i = 0; i < count; i++) {
    await page.mouse.move(x + dx * (i + 1), y, { steps: 2 });
    await page.waitForTimeout(60);
    await page.screenshot({ path: fileURLToPath(new URL(`${prefix}-${String(i).padStart(3, "0")}.png`, FRAMES)) });
  }
  await page.mouse.up();
}

/** Zoom toward the view centre with the mouse wheel. */
async function zoom(page: Page, steps: number): Promise<void> {
  const box = (await canvas(page).boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  for (let i = 0; i < steps; i++) {
    await page.mouse.wheel(0, -200);
    await page.waitForTimeout(80);
  }
}

async function demo(page: Page, name: string, done: RegExp | string): Promise<void> {
  await page.goto(`/?demo=${name}`);
  await expect(status(page)).toContainText(done, { timeout: 120_000 });
}

test("distance, volume, ground and M3C2 demos, with orbit frames for the GIF", async ({ page }) => {
  test.setTimeout(600_000);
  await demo(page, "c2c", "C2C distance computed");
  await page.locator("#point-size").fill("3");
  await page.locator("[data-view=iso]").click();
  await page.locator("#fit").click();
  await zoom(page, 6);
  await shot(page, "c2c");
  await orbitFrames(page, "a-c2c", 24, 12);

  await demo(page, "m3c2", "M3C2 at");
  await shot(page, "m3c2");
  await orbitFrames(page, "b-m3c2", 24, 12);

  await demo(page, "volume", "Volume:");
  await showPanel(page, "volume-panel");
  await shot(page, "volume");
  await orbitFrames(page, "c-volume", 24, 12);

  await demo(page, "ground", "are ground");
  await shot(page, "ground");
  await orbitFrames(page, "d-ground", 24, 12);

});

test("pose graph: a campus drive's loops closed, and how far each point moved", async ({ page }) => {
  test.skip(!NCLT, "set NCLT_DIR (scripts/prepare_nclt.py)");
  test.setTimeout(900_000);
  await openNclt(page, "");
  await page.locator("#pose-graph-panel summary", { hasText: "Find loops automatically" }).click();
  await page.locator("#pg-find").click();
  await expect(status(page)).toContainText(/Added \d+ of/, { timeout: 300_000 });
  await page.locator("#pg-map-voxel").fill("0.3");
  await page.locator("#pg-compare").click();
  await expect(status(page)).toContainText("colored by how far", { timeout: 300_000 });
  await page.locator("#pg-show-scans").uncheck();
  await page.locator("[data-view=iso]").click();
  await page.locator("#fit").click();
  await zoom(page, 4);
  await shot(page, "posegraph");
  await orbitFrames(page, "e-posegraph", 24, 12);
});

/** Frames of the view every `ms` milliseconds while `during` runs. */
async function framesWhile(page: Page, prefix: string, ms: number, during: Promise<unknown>): Promise<void> {
  mkdirSync(FRAMES, { recursive: true });
  let done = false;
  void during.finally(() => (done = true));
  for (let i = 0; !done && i < 400; i++) {
    await page.screenshot({ path: fileURLToPath(new URL(`${prefix}-${String(i).padStart(3, "0")}.png`, FRAMES)) });
    await page.waitForTimeout(ms);
  }
  await during;
}

/** Where node A's yellow marker is on screen (see the pose graph panel), or the view centre. */
async function markerA(page: Page): Promise<{ x: number; y: number }> {
  const box = (await canvas(page).boundingBox())!;
  const shot = (await canvas(page).screenshot()).toString("base64");
  const found = await page.evaluate(async (png) => {
    const image = new Image();
    image.src = `data:image/png;base64,${png}`;
    await image.decode();
    const c = document.createElement("canvas");
    [c.width, c.height] = [image.width, image.height];
    const g = c.getContext("2d")!;
    g.drawImage(image, 0, 0);
    const { data } = g.getImageData(0, 0, c.width, c.height);
    let [sx, sy, n] = [0, 0, 0];
    for (let i = 0; i < data.length; i += 4) {
      if (data[i] > 240 && data[i + 1] > 215 && data[i + 1] < 250 && data[i + 2] < 90) {
        sx += (i / 4) % c.width;
        sy += Math.floor(i / 4 / c.width);
        n++;
      }
    }
    return n ? { x: sx / n / c.width, y: sy / n / c.height } : null;
  }, shot);
  return found ? { x: box.x + found.x * box.width, y: box.y + found.y * box.height } : { x: box.x + box.width / 2, y: box.y + box.height / 2 };
}

test("pose graph: LiDAR odometry of a real drive replayed", async ({ page }) => {
  test.skip(!PANDASET, "set PANDASET_DIR (scripts/fetch_pandaset.py)");
  test.setTimeout(900_000);
  await openPandaSet(page);
  await page.locator("#point-size").fill("2");
  await page.locator("#pg-colors").selectOption("height");
  await page.locator("#pg-axes").check();
  await streetView(page, 40, 3);
  // Slow, so each frame catches a keyframe or two (a screenshot takes most of a second); the GIF plays it fast.
  await page.locator("#pg-play-rate").fill("2");
  await page.locator("#pg-play").click();
  await framesWhile(page, "f-odometry", 0, expect(page.locator("#pg-play")).toHaveText("Play", { timeout: 300_000 }));
});

test("pose graph: a loop on a real campus closed by hand", async ({ page }) => {
  test.skip(!NCLT, "set NCLT_DIR (scripts/prepare_nclt.py)");
  test.setTimeout(900_000);
  // Corrections glide in over 10 s here, so the frames catch them moving.
  await openNclt(page, "?glide=10000");
  await page.locator("#point-size").fill("2");
  await page.locator("#pg-colors").selectOption("height");

  // Where the drive comes back after a kilometre, the two passes' scans side by side (Align by hand):
  // the odometry left them metres apart. ICP lines them up; the loop then pulls the drive together.
  const frames = async (prefix: string, count: number) => {
    await page.waitForTimeout(600);
    for (let i = 0; i < count; i++) {
      await page.screenshot({ path: fileURLToPath(new URL(`${prefix}-${String(i).padStart(3, "0")}.png`, FRAMES)) });
    }
  };
  await page.locator("[data-view=top]").click();
  await page.locator("#pg-a").fill("469");
  await page.locator("#pg-b").fill("1410");
  await page.locator("#pg-align").click();
  await expect(page.locator("#pg-align-panel")).toBeVisible();
  await zoom(page, 20);
  await frames("g-loop-a", 8);
  await page.locator("#pg-align-icp").click();
  await expect(page.locator("#pg-align-info")).toContainText("ICP RMS", { timeout: 60_000 });
  await frames("g-loop-b", 8);
  await page.locator("#pg-align-accept").click();
  await framesWhile(page, "g-loop-c", 0, expect(status(page)).toContainText(/Loop 469 – 1410 added by hand; χ²/, { timeout: 120_000 }));
  await frames("g-loop-d", 10);
});

test("pose graph: traffic removed from a real street's map", async ({ page }) => {
  test.skip(!PANDASET, "set PANDASET_DIR (scripts/fetch_pandaset.py)");
  test.setTimeout(600_000);
  await openPandaSet(page);
  await page.locator("#point-size").fill("2");
  // The map with every scan: the traffic that drove past leaves ghost trails along the lanes.
  await page.locator("#pg-map-voxel").fill("0.1");
  await page.locator("#pg-map").click();
  await expect(status(page)).toContainText("Added poses_map", { timeout: 120_000 });
  await page.locator("#pg-show-scans").uncheck();
  await streetView(page, 40, 6);
  await orbitFrames(page, "h-dynamic-a", 14, 10);
  // Left out: the ghosts turn red (their own cloud) and the map is clean.
  await page.locator("#pg-dynamic").click();
  await expect(status(page)).toContainText(/dynamic points of/, { timeout: 180_000 });
  await page.locator("#cloud-list li").first().locator("input[type=checkbox]").uncheck();
  await orbitFrames(page, "h-dynamic-b", 14, 10);
  await page.locator("#cloud-list li").last().locator("input[type=checkbox]").uncheck();
  await orbitFrames(page, "h-dynamic-c", 12, 10);
});

test("lasso segmentation and a cross-section profile on the town", async ({ page }) => {
  test.setTimeout(300_000);
  await page.goto("/?url=samples/town.ply");
  await expect(status(page)).toContainText("Loaded town.ply", { timeout: 60_000 });
  await page.locator("#point-size").fill("4");
  await page.locator("[data-view=top]").click();
  await page.locator("#fit").click();
  const box = (await canvas(page).boundingBox())!;
  await page.keyboard.press("s");
  for (const [fx, fy] of [[0.35, 0.3], [0.62, 0.26], [0.7, 0.55], [0.55, 0.72], [0.33, 0.62]]) {
    await canvas(page).click({ position: { x: box.width * fx, y: box.height * fy } });
  }
  await shot(page, "lasso");
  await page.keyboard.press("Escape");

  await showPanel(page, "profile-panel");
  await page.locator("#profile-width").fill("2");
  await page.locator("#profile-draw").click();
  await canvas(page).click({ position: { x: box.width * 0.2, y: box.height * 0.5 } });
  await canvas(page).click({ position: { x: box.width * 0.8, y: box.height * 0.45 } });
  await page.keyboard.press("Enter");
  await expect(status(page)).toContainText("Profile:");
  await page.locator("[data-view=iso]").click();
  await shot(page, "profile");
});

test("shapes, mesh, raster and scalar fields", async ({ page }) => {
  test.setTimeout(300_000);
  await page.goto("/");
  await open(page, [{ name: "room.ply", buffer: room() }]);
  await expect(status(page)).toContainText("Loaded room.ply");
  await page.locator("#point-size").fill("4");
  await page.locator("[data-view=iso]").click();
  await showPanel(page, "shapes-panel");
  await page.locator("#shapes-min").fill("3000");
  await page.locator("#shapes-max").fill("5");
  await page.locator("#shapes-run").click();
  await expect(status(page)).toContainText("in room.ply", { timeout: 60_000 });
  await shot(page, "shapes");

  await page.goto("/?url=samples/stockpile_after.ply");
  await expect(status(page)).toContainText("Loaded stockpile_after.ply", { timeout: 60_000 });
  await showPanel(page, "mesh-panel");
  await page.locator("#mesh-run").click();
  await expect(status(page)).toContainText("triangles", { timeout: 60_000 });
  await page.locator("[data-view=iso]").click();
  await shot(page, "mesh");

  await page.goto("/?url=samples/town.ply");
  await expect(status(page)).toContainText("Loaded town.ply", { timeout: 60_000 });
  await showPanel(page, "raster-panel");
  await page.locator("#raster-run").click();
  await expect(status(page)).toContainText("Raster:", { timeout: 60_000 });
  await page.locator("#point-size").fill("5");
  await page.locator("[data-view=iso]").click();
  await shot(page, "raster");

  await page.goto("/?url=samples/town.ply");
  await expect(status(page)).toContainText("Loaded town.ply", { timeout: 60_000 });
  await page.locator("#point-size").fill("4");
  await showPanel(page, "field-panel");
  await page.locator("#field-name").selectOption("Z");
  await page.locator("#field-color").click();
  await page.locator("[data-view=iso]").click();
  await shot(page, "fields");
});

test("trajectory evaluation and the QA report", async ({ page, browser }) => {
  test.setTimeout(300_000);
  await page.goto("/");
  const dir = new URL("../../rust/crates/ca-core/tests/trajectory/", import.meta.url);
  await open(page, [
    { name: "reference.tum", buffer: readFileSync(new URL("reference.tum", dir)) },
    { name: "estimate.tum", buffer: readFileSync(new URL("estimate.tum", dir)) },
  ]);
  await showPanel(page, "trajectory-panel");
  await page.locator("#trajectory-estimate").selectOption({ label: "estimate.tum" });
  await page.locator("#trajectory-reference").selectOption({ label: "reference.tum" });
  await page.locator("#trajectory-run").click();
  await expect(page.locator("#trajectory-result")).toBeVisible({ timeout: 60_000 });
  await page.locator("[data-view=iso]").click();
  await page.locator("#fit").click();
  await zoom(page, 2);
  await shot(page, "trajectory");

  await demo(page, "volume", "Volume:");
  await page.locator("#c2c-compared").selectOption({ label: "stockpile_after.ply" });
  await page.locator("#c2c-reference").selectOption({ label: "stockpile_before.ply" });
  await page.locator("#c2c-run").click();
  await expect(status(page)).toContainText("C2C distance computed");
  await page.locator("#gate-add").click();
  await page.locator("#gate-add").click();
  const gates = page.locator("#gate-list li");
  await gates.nth(0).locator("select").first().selectOption({ label: "C2C stockpile_after.ply → stockpile_before.ply · Median" });
  await gates.nth(0).locator("input").fill("0.1");
  await gates.nth(1).locator("select").first().selectOption({ label: "Volume stockpile_after.ply vs stockpile_before.ply · Net" });
  await gates.nth(1).locator("select").nth(1).selectOption(">=");
  await gates.nth(1).locator("input").fill("300");
  await gates.nth(1).locator("input").blur();
  const download = page.waitForEvent("download");
  await page.locator("#report-html").click();
  const html = await new Response((await (await download).createReadStream()) as unknown as ReadableStream).text();
  const report = await browser.newPage({ viewport: { width: 1000, height: 1100 } });
  await report.setContent(html);
  await report.screenshot({ path: fileURLToPath(new URL("report.jpg", OUT)), type: "jpeg", quality: 88 });
});

test("manual alignment with the gizmo", async ({ page }) => {
  test.setTimeout(300_000);
  await demo(page, "c2c", "C2C distance computed");
  await page.locator("#point-size").fill("3");
  await page.locator("[data-view=iso]").click();
  await page.locator("#fit").click();
  await zoom(page, 6);
  await showPanel(page, "align-panel");
  await page.locator("#align-panel summary").click();
  await page.locator("#gizmo-rotate").click();
  await shot(page, "align");
});

test("pose graph: the same campus street in summer and in winter", async ({ page }) => {
  test.skip(!SEASONS, "set NCLT_SEASONS (scripts/prepare_nclt.py, two sessions)");
  test.setTimeout(1_800_000);
  const [summer, winter] = SEASONS!;
  await page.goto("/");
  await page.locator("#pg-folder-input").setInputFiles(`${summer}/velodyne`);
  await expect(status(page)).toContainText("Opened kiss_poses.txt", { timeout: 900_000 });
  const summerNodes = readFileSync(`${summer}/velodyne/kiss_poses.txt`, "utf8").trim().split("\n").length;
  await page.locator("#pose-graph-panel summary", { hasText: "Find loops automatically" }).click();
  await page.locator("#pg-find").click();
  await expect(status(page)).toContainText(/Added \d+ of/, { timeout: 600_000 });
  // Winter joins where both drives pass the same spot, then loops tie the seasons together.
  await page.locator("#pose-graph-panel summary", { hasText: "Join another graph" }).click();
  await page.locator("#pg-merge-here").fill("1219");
  await page.locator("#pg-merge-there").fill("850");
  await page.locator("#pg-merge-input").setInputFiles(`${winter}/velodyne`);
  await expect(status(page)).toContainText("Joined", { timeout: 900_000 });
  await page.locator("#pg-find").click();
  await expect(status(page)).toContainText(/Added \d+ of/, { timeout: 600_000 });
  // The IMU's up direction for both drives, winter's keyframes after summer's.
  const gravity = mkdtempSync(join(tmpdir(), "seasons-"));
  const lines = (dir: string, offset: number) =>
    readFileSync(`${dir}/gravity/gravity.txt`, "utf8")
      .trim()
      .split("\n")
      .map((line) => line.replace(/^\d+/, (k) => String(Number(k) + offset)));
  writeFileSync(join(gravity, "gravity.txt"), [...lines(summer, 0), ...lines(winter, summerNodes)].join("\n") + "\n");
  await page.locator("#pose-graph-panel summary", { hasText: "IMU gravity" }).click();
  await page.locator("#pg-gravity-input").setInputFiles(gravity);
  await expect(status(page)).toContainText("Gravity tied", { timeout: 300_000 });
  // Each season's map where they meet, and the largest change between them: a tree-lined street.
  await page.locator("#pg-map-voxel").fill("0.3");
  await page.locator("#pg-parts").click();
  await expect(status(page)).toContainText("M3C2 at", { timeout: 1_800_000 });
  await page.locator("#pg-show-scans").uncheck();
  await page.locator("#round-points").check();
  const maps = page.locator("#cloud-list li");
  // The seasons' own look, not the M3C2 colour bar.
  await maps.nth(2).locator("select").first().selectOption("solid");
  await page.locator("#pg-change-list li .link").first().click();
  await page.waitForTimeout(2000);
  await page.locator("[data-view=iso]").click();
  await zoom(page, 18);
  const show = async (k: number) => {
    for (let i = 0; i < 3; i++) {
      const box = maps.nth(i).locator("input[type=checkbox]").first();
      if ((await box.isChecked()) !== (i === k)) await box.click();
    }
    await page.waitForTimeout(1500);
  };
  // June, then December, then June and December again: leaves, then bare branches.
  for (const [round, k] of [0, 1, 0, 1].entries()) {
    await show(k);
    await orbitFrames(page, `i-seasons-${round}`, 6, 4);
  }
});
