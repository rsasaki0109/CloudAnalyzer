// Optional real-source regeneration, not an imported expected vector map.
import { expect, test } from "@playwright/test";
import { mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
const SOURCE = process.env.VECTOR_MAP_SURFACE_SOURCE;
const PROOF = process.env.VECTOR_MAP_SURFACE_PROOF;
const EVALUATION = process.env.VECTOR_MAP_SURFACE_EVALUATION;
const OUTPUT = fileURLToPath(new URL("../media-frames/vector-map-surface/", import.meta.url));
test.use({ viewport: { width: 1280, height: 760 }, actionTimeout: 30000 });

test("real branch source-footprint generation agrees with native and undoes", async ({ page }) => {
  test.skip(!SOURCE || !PROOF || !EVALUATION, "Set source, frozen trajectory proof and native comparison directories");
  test.setTimeout(240000); mkdirSync(OUTPUT, { recursive: true });
  const comparison = JSON.parse(readFileSync(`${EVALUATION}/comparison.json`, "utf8"));
  expect(comparison.reference_inputs).toEqual([]);
  const branch = comparison.cases[3];
  const proof = JSON.parse(readFileSync(`${PROOF}/roads-report.json`, "utf8"));
  const errors: string[] = []; page.on("pageerror", e => errors.push(String(e)));
  await page.goto("/"); await page.locator("#file-input").setInputFiles(SOURCE!);
  await expect(page.locator("#status")).toContainText("Loaded", { timeout: 120000 });
  await expect(page.locator("#vm-status")).toContainText("No map yet");
  await page.locator("#file-input").setInputFiles(`${PROOF}/${proof.builds[3].csv}`);
  await expect(page.locator("#vm-trajectory option")).toHaveCount(1);
  await page.locator("#vector-map-panel").getByText("Road options", { exact: true }).click();
  await page.locator("#vm-forward").fill("1"); await page.locator("#vm-backward").fill("1");
  await page.locator("#vm-width").fill("3.5"); await page.locator("#vm-segment").fill("0");
  await page.locator("#vector-map-panel").getByText("Build from a trajectory", { exact: true }).click();
  await page.locator("#vm-discover-after-build").uncheck();
  await expect(page.locator("#vm-source-surface")).not.toBeChecked();
  await page.locator("#vector-map-panel").getByText("Map display", { exact: true }).click();
  await page.locator("#vm-show-labels").uncheck();
  await page.locator('#cloud-list select[title="Color by"]').first().selectOption("intensity");
  await page.locator("#edl").uncheck();
  await page.locator("#clip-enabled").check();
  await page.locator('[aria-label="Z maximum"]').fill("140");
  const exportMap = async () => {
    await page.waitForTimeout(1100);
    const wait = page.waitForEvent("download", d => d.suggestedFilename() === "lanelet2_map.osm");
    await page.locator("#vm-export").click(); return readFileSync((await (await wait).path())!, "utf8");
  };
  const checks: any[] = [];
  for (const mode of ["before", "after"] as const) {
    if (mode === "after") await page.locator("#vm-source-surface").check();
    await page.locator("#vm-build").click();
    await expect(page.locator("#status")).toContainText("Draft roads added", { timeout: 120000 });
    await expect(page.locator("#vm-status")).toContainText("2 lanes");
    if (mode === "after") await expect(page.locator("#vm-build-report")).toContainText("deferred");
    const xml = await exportMap();
    writeFileSync(`${OUTPUT}/${mode}.osm`, xml);
    const curves = await page.evaluate(xml => {
      const root = new DOMParser().parseFromString(xml, "text/xml");
      const nodes = new Map([...root.querySelectorAll("node")].map(n => {
        const tags = new Map([...n.querySelectorAll("tag")].map(t => [t.getAttribute("k"), t.getAttribute("v")]));
        return [n.getAttribute("id"), ["local_x", "local_y", "ele"].map(k => Number(tags.get(k)))] as const;
      }));
      return [...root.querySelectorAll("way")].map(w => [...w.querySelectorAll("nd")].map(n => nodes.get(n.getAttribute("ref"))!));
    }, xml);
    const native = JSON.parse(readFileSync(`${EVALUATION}/${branch[mode].map}`, "utf8"));
    const selected = new Set(branch[mode].added_lane_ids);
    const ids = new Set(native.lanes.filter((l: any) => selected.has(l.id)).flatMap((l: any) => [l.left, l.right].map(v => typeof v === "number" ? v : v.boundary)));
    const expected: number[][][] = native.boundaries.filter((b: any) => ids.has(b.id)).map((b: any) => b.geometry);
    const sort = (values: number[][][]) => values.sort((a, b) => a[0][0] - b[0][0] || a[0][1] - b[0][1]);
    sort(curves); sort(expected); expect(curves.length).toBe(expected.length);
    let maximumError = 0;
    for (let i = 0; i < curves.length; i++) {
      expect(curves[i].length).toBe(expected[i].length);
      for (let j = 0; j < curves[i].length; j++) for (let k = 0; k < 3; k++) maximumError = Math.max(maximumError, Math.abs(curves[i][j][k] - expected[i][j][k]));
    }
    expect(maximumError).toBeLessThan(1e-7);
    if (mode === "before") await page.locator("#vm-quality summary").click();
    await page.locator("#vm-quality-check").click();
    const review = branch[mode].source_quality.source_review_lane_ids.length;
    await expect(page.locator("#vm-quality-report")).toContainText(`2 lanes checked; ${review} need source review; 0 omitted`, { timeout: 120000 });
    expect(await exportMap()).toBe(xml);
    await page.locator("#vm-plan").click(); await page.locator("#vm-fit").click();
    await page.locator("#vm-quality-report").scrollIntoViewIfNeeded();
    await page.screenshot({ path: `${OUTPUT}/${mode}.png` });
    checks.push({ mode, maximumNativeWebCoordinateError: maximumError, sourceReviewLanes: review, auditLeavesExportUnchanged: true });
    await page.locator("#vm-undo").click();
    await expect(page.locator("#vm-status")).toContainText("No map yet");
    await expect(page.locator("#vm-undo")).toBeDisabled();
  }
  expect(errors).toEqual([]);
  writeFileSync(`${OUTPUT}/verification.json`, JSON.stringify({ runtimeSourceCommit: comparison.source_commit, sourceSha256: comparison.source_sha256, generationReferenceInputs: [], laneCountsExplicit: true, checks, undoRestoresEmptyMap: true, pageErrors: errors }, null, 2));
});
