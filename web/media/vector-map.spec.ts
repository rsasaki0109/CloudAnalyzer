// Reproduce the README GIF from a real PandaSet drive, without distributing inputs.
import { expect, test } from "@playwright/test";
import { mkdirSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const INPUT = process.env.VECTOR_MAP_DEMO_DIR;
const FRAMES = fileURLToPath(new URL("../media-frames/vector-map/", import.meta.url));
test.use({ viewport: { width: 1280, height: 760 } });

test("vector map README GIF on real PandaSet", async ({ page }) => {
  test.skip(!INPUT, "Set VECTOR_MAP_DEMO_DIR after preparing PandaSet 019");
  test.setTimeout(180_000);
  mkdirSync(FRAMES, { recursive: true });
  const status = page.locator("#status");
  await page.goto("/");
  await page.locator("#file-input").setInputFiles([`${INPUT}/map.pcd`, `${INPUT}/trajectory.csv`]);
  await expect(status).toContainText("trajectory of 80 poses", { timeout: 120_000 });
  await page.locator('[data-view="top"]').click();
  await page.locator("#vector-map-panel").getByText("Road options", { exact: true }).click();
  await page.locator("#vm-traffic").selectOption("right");
  await page.locator("#vector-map-panel").getByText("Build from a trajectory", { exact: true }).click();

  // Capture-only captions and a cursor ring; all geometry and actions are the app's.
  await page.evaluate(() => {
    const caption = document.createElement("div");
    caption.id = "capture-caption";
    caption.style.cssText = "position:fixed;left:300px;top:92px;padding:10px 16px;background:#111a29eb;color:white;font:600 22px sans-serif;border-radius:8px;z-index:1000;pointer-events:none";
    document.body.append(caption);
    const cursor = document.createElement("div");
    cursor.id = "capture-cursor";
    cursor.style.cssText = "position:fixed;width:20px;height:20px;border:2px solid #00e5ff;border-radius:50%;transform:translate(-50%,-50%);pointer-events:none;z-index:1001;display:none";
    document.body.append(cursor);
    document.addEventListener("pointermove", (e) => {
      cursor.style.left = `${e.clientX}px`;
      cursor.style.top = `${e.clientY}px`;
    });
  });
  let index = 0;
  const frames: { name: string; duration: number }[] = [];
  const shot = async (text: string, duration = 0.3) => {
    await page.locator("#capture-caption").evaluate((el, text) => { el.textContent = text; }, text);
    await page.waitForTimeout(100);
    const name = `${String(index++).padStart(3, "0")}.png`;
    await page.screenshot({ path: `${FRAMES}/${name}` });
    frames.push({ name, duration });
  };
  await shot("1. Real LiDAR + recorded trajectory", 1.5);
  await page.locator("#vm-build").click();
  await expect(status).toContainText("Draft roads added", { timeout: 120_000 });
  await page.locator("#vm-fit").click();
  await page.waitForTimeout(1200);
  await shot("2. Generate a lane draft", 2);
  await page.locator("#vm-vertices").click();
  const box = (await page.locator("#viewport > canvas").boundingBox())!;
  await page.mouse.move(box.x + box.width / 2, box.y + box.height / 2);
  await page.mouse.wheel(0, -240);
  await page.waitForTimeout(1200);
  await shot("3. Review and drag shared vertices", 1);

  // Locate visible handles in the actual screenshot, then use UI feedback to
  // select an interior point of shared boundary 2. No worker internals required.
  const screenshot = (await page.screenshot()).toString("base64");
  const candidates = await page.evaluate(async ({ screenshot, box }) => {
    const image = new Image();
    image.src = `data:image/png;base64,${screenshot}`;
    await image.decode();
    const canvas = document.createElement("canvas");
    canvas.width = image.width;
    canvas.height = image.height;
    const ctx = canvas.getContext("2d")!;
    ctx.drawImage(image, 0, 0);
    const data = ctx.getImageData(0, 0, canvas.width, canvas.height).data;
    const pixels: { x: number; y: number; distance: number }[] = [];
    for (let y = Math.floor(box.y + 80); y < box.y + box.height - 60; y++) {
      for (let x = Math.floor(box.x + 80); x < box.x + box.width - 80; x++) {
        const i = (y * canvas.width + x) * 4;
        if (data[i] > 235 && data[i + 1] > 205 && data[i + 2] < 95) {
          pixels.push({ x, y, distance: (x - box.x - box.width / 2) ** 2 + (y - box.y - box.height / 2) ** 2 });
        }
      }
    }
    const hits: { x: number; y: number }[] = [];
    for (const p of pixels.sort((a, b) => a.distance - b.distance)) {
      if (hits.every((q) => Math.hypot(p.x - q.x, p.y - q.y) > 8)) hits.push(p);
    }
    return hits.slice(0, 100);
  }, { screenshot, box });
  await page.locator("#capture-cursor").evaluate((el) => { (el as HTMLElement).style.display = "block"; });
  let vertex: { x: number; y: number } | undefined;
  for (const candidate of candidates) {
    await page.mouse.move(candidate.x, candidate.y);
    await page.mouse.down();
    const match = /Boundary (\d+), vertex (\d+):/.exec(await page.locator("#vm-hint").textContent() ?? "");
    if (match && Number(match[1]) === 2 && Number(match[2]) >= 3 && Number(match[2]) <= 18) {
      vertex = candidate;
      break;
    }
    await page.mouse.up();
    await page.locator("#vm-vertices").click();
    await page.locator("#vm-vertices").click();
  }
  if (!vertex) throw new Error("No shared interior vertex was selected");
  await shot("3. Review and drag shared vertices", 0.5);
  for (let i = 1; i <= 12; i++) {
    await page.mouse.move(vertex.x + 18 * i / 12, vertex.y + 8 * i / 12);
    await shot("3. Review and drag shared vertices", 0.12);
  }
  await page.mouse.up();
  await expect(status).toContainText("vertex moved");
  await shot("Shared lanes update together", 1.5);
  await page.locator("#capture-cursor").evaluate((el) => { (el as HTMLElement).style.display = "none"; });
  await page.keyboard.press("Escape");
  const osm = page.waitForEvent("download", (d) => d.suggestedFilename() === "lanelet2_map.osm");
  const yaml = page.waitForEvent("download", (d) => d.suggestedFilename() === "map_projector_info.yaml");
  await page.locator("#vm-export").click();
  await Promise.all([osm, yaml]);
  await shot("4. Save Lanelet2 + Autoware metadata", 2);
  await page.locator("#vm-fit").click();
  await page.waitForTimeout(1000);
  await shot("A draft to review, edit and validate", 2);
  writeFileSync(`${FRAMES}/frames.json`, JSON.stringify(frames, null, 2));
  writeFileSync(`${FRAMES}/concat.txt`, frames.map((f) => `file '${f.name}'\nduration ${f.duration}\n`).join("") + `file '${frames.at(-1)!.name}'\n`);
});
