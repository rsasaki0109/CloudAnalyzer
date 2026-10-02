// Actual production UI capture. Surveyed approaches are imported; connections
// are generated from cloud support, selected, edited, undone and exported.
import { expect, test } from "@playwright/test";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const MAP = process.env.VECTOR_MAP_PLANNING_DIR;
const JUNCTION = process.env.VECTOR_MAP_JUNCTION_DIR;
const FRAMES = fileURLToPath(new URL("../media-frames/vector-map-intersection/", import.meta.url));
const CHOSEN = [[113,116],[113,122],[113,124],[123,112],[123,116],[123,124],
  [125,112],[125,116],[125,122],[9178,112],[9178,122],[9178,124]];
test.use({ viewport: { width: 1280, height: 760 } });

test("vector map intersection README GIF on the real planning survey", async ({ page }) => {
  test.skip(!MAP || !JUNCTION, "Set VECTOR_MAP_PLANNING_DIR and VECTOR_MAP_JUNCTION_DIR to external survey and ablation inputs");
  test.setTimeout(240_000);
  mkdirSync(FRAMES, { recursive: true });
  const input = JSON.parse(readFileSync(`${JUNCTION}/input.json`, "utf8"));
  const expected = JSON.parse(readFileSync(`${JUNCTION}/report.json`, "utf8"));
  const errors: string[] = [];
  page.on("pageerror", (error) => errors.push(String(error)));
  let exports = 0;
  const exportMap = async () => {
    if (exports > 0 && exports % 4 === 0) await page.waitForTimeout(1100);
    exports++;
    const osm = page.waitForEvent("download", (d) => d.suggestedFilename() === "lanelet2_map.osm");
    const yaml = page.waitForEvent("download", (d) => d.suggestedFilename() === "map_projector_info.yaml");
    await page.locator("#vm-export").click();
    const [map, projector] = await Promise.all([osm, yaml]);
    const path = await map.path(), projectorPath = await projector.path();
    if (!path || !projectorPath) throw new Error("Export download unavailable");
    expect(readFileSync(projectorPath, "utf8")).toContain("MGRS");
    return readFileSync(path);
  };
  await page.goto("/");
  await page.locator("#file-input").setInputFiles(`${MAP}/pointcloud_map.pcd`);
  await expect(page.locator("#status")).toContainText("Loaded pointcloud_map.pcd", { timeout: 120_000 });
  await page.locator('#cloud-list select[title="Color by"]').selectOption("solid");
  await page.locator("#vm-file").setInputFiles(`${JUNCTION}/input.json`);
  await expect(page.locator("#status")).toContainText("Opened input.json: 72 lanes");
  await page.locator("#vector-map-panel").getByText("Map display", { exact: true }).click();
  await page.locator("#vm-context").fill("28");
  await page.locator("#vm-plan").click();
  const imported = await exportMap();
  await page.locator("#vector-map-panel").getByText("Draft junction connections", { exact: true }).click();
  await page.locator("#vm-junction-preview").click();
  await expect(page.locator("#vm-junction-report")).toContainText("83 ground-supported candidates", { timeout: 120_000 });
  const labels = await page.locator("#vm-junction-candidates label").allTextContents();
  const pairs = labels.map((s) => s.match(/(\d+) → (\d+)/)!.slice(1).map(Number));
  expect(new Set(pairs.map(JSON.stringify))).toEqual(new Set(expected.generated_pairs.map(JSON.stringify)));
  expect((await exportMap()).equals(imported)).toBe(true);
  await page.locator("#vm-junction-none").click();
  const focus = pairs.findIndex(([from, to]) => from === 113 && to === 122);
  expect(focus).toBeGreaterThanOrEqual(0);
  await page.locator("#vm-junction-candidates button").nth(focus).click();
  await page.waitForTimeout(1000);

  // Captions/cursor explain real actions; all cloud/map pixels are from the app.
  await page.evaluate(() => {
    const caption = document.createElement("div");
    caption.id = "capture-caption";
    caption.style.cssText = "position:fixed;left:300px;top:92px;padding:10px 16px;background:#111a29f0;color:white;font:600 20px sans-serif;border:1px solid #3a5975;border-radius:8px;z-index:1000;pointer-events:none";
    document.body.append(caption);
    const cursor = document.createElement("div");
    cursor.id = "capture-cursor";
    cursor.style.cssText = "position:fixed;width:20px;height:20px;border:2px solid #00e5ff;border-radius:50%;transform:translate(-50%,-50%);pointer-events:none;z-index:1001;display:none";
    document.body.append(cursor);
    document.addEventListener("pointermove", (e) => { cursor.style.left = `${e.clientX}px`; cursor.style.top = `${e.clientY}px`; });
  });
  const frames: { name: string; duration: number }[] = [];
  const shot = async (text: string, duration: number) => {
    await page.locator("#capture-caption").evaluate((el, text) => { el.textContent = text; }, text);
    await page.waitForTimeout(100);
    const name = `${String(frames.length).padStart(3, "0")}.png`;
    await page.screenshot({ path: `${FRAMES}/${name}` });
    frames.push({ name, duration });
  };
  await shot("1. Surveyed approaches, signals & crossings", 1.7);
  for (const [from, to] of CHOSEN) {
    const i = pairs.findIndex(([a, b]) => a === from && b === to);
    expect(i).toBeGreaterThanOrEqual(0);
    await page.locator("#vm-junction-candidates input").nth(i).check();
  }
  await shot("2. Preview point-supported connection drafts", 2);
  await page.locator("#vm-junction-apply").click();
  await expect(page.locator("#status")).toContainText("12 draft connections added");
  const connected = await exportMap();
  expect(connected.toString().match(/k="cloudanalyzer_review_required" v="yes"/g)?.length).toBe(12);
  await shot("3. Add 12 connections · review required", 1.8);
  await page.locator("#vm-select").click();
  const box = (await page.locator("#viewport > canvas").boundingBox())!;
  // Select a visible turning lane through the actual nearest-lane UI. The
  // overlapping center also has straight connections, which are not this shot.
  let turnSelected = false;
  for (const [x, y] of [[0.48,0.64],[0.52,0.59],[0.55,0.58],[0.55,0.44],[0.45,0.56],[0.46,0.40],[0.40,0.48]]) {
    await page.mouse.click(box.x + box.width * x, box.y + box.height * y);
    if (/turns (left|right)/.test(await page.locator("#vm-lane-info").textContent() ?? "")) { turnSelected = true; break; }
  }
  expect(turnSelected).toBe(true);
  await expect(page.locator(".vm-map-label.selected")).toBeVisible();
  await shot("Trace incoming → selected → outgoing lanes", 1.8);
  await page.keyboard.press("Escape");
  await page.locator('[data-view="iso"]').click();
  await page.waitForTimeout(1000);
  await shot("4. Inspect crossings & saved signal geometry", 2);
  await page.locator('[data-view="top"]').click();
  for (const key of ["surfaces", "directions", "regulations", "labels"]) await page.locator(`#vm-show-${key}`).uncheck();
  await page.locator("#vm-vertices").click();
  await page.waitForTimeout(800);
  await page.screenshot({ path: `${FRAMES}/handles.png` });

  // Find real visible handles. Accept only a surveyed boundary used by multiple
  // imported lanes, through UI hit feedback. Do not poke the worker/map state.
  const usage = new Map<number, number>();
  for (const lane of input.lanes) for (const ref of [lane.left, lane.right]) {
    const id = typeof ref === "number" ? ref : ref.boundary;
    usage.set(id, (usage.get(id) ?? 0) + 1);
  }
  const screenshot = (await page.screenshot()).toString("base64");
  const candidates = await page.evaluate(async ({ screenshot, box }) => {
    const image = new Image(); image.src = `data:image/png;base64,${screenshot}`; await image.decode();
    const canvas = document.createElement("canvas"); canvas.width = image.width; canvas.height = image.height;
    const ctx = canvas.getContext("2d")!; ctx.drawImage(image, 0, 0);
    const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    const pixels: { x: number; y: number; distance: number }[] = [];
    for (let y = Math.floor(box.y + 90); y < box.y + box.height - 70; y++) for (let x = Math.floor(box.x + 70); x < box.x + box.width - 70; x++) {
      const i = (y * canvas.width + x) * 4;
      if (data[i] > 235 && data[i+1] > 205 && data[i+2] < 95) pixels.push({ x,y,distance:(x-box.x-box.width/2)**2+(y-box.y-box.height/2)**2 });
    }
    const hits: { x: number; y: number }[] = [];
    // Prefer approach vertices outside the dense new turning-lane handles.
    for (const p of pixels.sort((a,b)=>Math.abs(Math.sqrt(a.distance)-210)-Math.abs(Math.sqrt(b.distance)-210))) if (hits.every(q=>Math.hypot(p.x-q.x,p.y-q.y)>8)) hits.push(p);
    return hits.slice(0,200);
  }, { screenshot, box });
  let vertex: { x: number; y: number } | undefined;
  let boundary: number | undefined;
  for (const candidate of candidates) {
    await page.mouse.move(candidate.x, candidate.y); await page.mouse.down();
    const match = /Boundary (\d+), vertex (\d+):/.exec(await page.locator("#vm-hint").textContent() ?? "");
    if (match && (usage.get(Number(match[1])) ?? 0) > 1) { vertex = candidate; boundary = Number(match[1]); break; }
    await page.mouse.up();
  }
  if (!vertex) throw new Error("No shared boundary handle selected");
  await page.locator("#capture-cursor").evaluate((el) => { (el as HTMLElement).style.display = "block"; });
  await shot("5. Edit a shared boundary · both lanes update", 0.8);
  for (let i = 1; i <= 8; i++) {
    await page.mouse.move(vertex.x + 8*i/8, vertex.y + 4*i/8);
    await shot("5. Edit a shared boundary · both lanes update", 0.12);
  }
  await page.mouse.up();
  await expect(page.locator("#status")).toContainText("vertex moved");
  expect((await exportMap()).equals(connected)).toBe(false);
  await shot("The saved map changes with the edit", 0.8);
  await page.locator("#capture-cursor").evaluate((el) => { (el as HTMLElement).style.display = "none"; });
  await page.keyboard.press("Escape");
  await page.locator("#vm-undo").click();
  await expect(page.locator("#status")).toContainText("Undone");
  expect((await exportMap()).equals(connected)).toBe(true);
  for (const key of ["surfaces", "directions", "regulations", "labels"]) await page.locator(`#vm-show-${key}`).check();
  await shot("Undo restores the original draft exactly", 1.1);
  await page.locator('[data-view="iso"]').click();
  await page.waitForTimeout(900);
  const final = await exportMap();
  expect(final.equals(connected)).toBe(true);
  await shot("6. Save Lanelet2 + MGRS projector metadata", 2);
  await shot("Generated connections · surveyed context · manual review", 2);
  await page.locator("#vm-undo").click();
  await expect(page.locator("#status")).toContainText("Undone");
  expect((await exportMap()).equals(imported)).toBe(true);
  await page.locator("#vm-file").setInputFiles({ name: "intersection.osm", mimeType: "application/xml", buffer: final });
  await expect(page.locator("#status")).toContainText("Opened intersection.osm: 84 lanes");
  expect((await exportMap()).toString().match(/k="cloudanalyzer_review_required" v="yes"/g)?.length).toBe(12);
  expect(errors).toEqual([]);
  writeFileSync(`${FRAMES}/verification.json`, JSON.stringify({ points: expected.cloud_points, importedLanes:72, importedSignals:34, importedCrosswalks:4, addedConnections:CHOSEN, sharedBoundary:boundary, changedEdit:true, exactEditUndo:true, exactBatchUndo:true, mgrsRoundtrip:true, errors }, null, 2));
  writeFileSync(`${FRAMES}/frames.json`, JSON.stringify(frames, null, 2));
  writeFileSync(`${FRAMES}/concat.txt`, frames.map(f=>`file '${f.name}'\nduration ${f.duration}\n`).join("") + `file '${frames.at(-1)!.name}'\n`);
});
