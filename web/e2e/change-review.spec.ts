import { expect, test, type Page } from "@playwright/test";

const input = (name: string, buffer: Buffer) => ({ name, mimeType: "application/octet-stream", buffer });
function scan(lift: number, poseX: number, patches = 2) {
  const points: string[] = [];
  for (let patch = 0; patch < patches; patch++) {
    for (let x = 0; x <= 20; x++) for (let y = 0; y <= 20; y++) {
      points.push(`${patch * 8 + x * .2 - poseX} ${y * .2} ${lift}`);
    }
  }
  return Buffer.from(`ply\nformat ascii 1.0\nelement vertex ${points.length}\nproperty float x\nproperty float y\nproperty float z\nend_header\n${points.join("\n")}\n`);
}
async function saved(page: Page, id: string) {
  const waiting = page.waitForEvent("download");
  await page.locator(id).click();
  const parts: Buffer[] = [];
  for await (const part of await (await waiting).createReadStream()) parts.push(part as Buffer);
  return JSON.parse(Buffer.concat(parts).toString());
}
async function setup(page: Page, lift = .8, patches = 2) {
  await page.addInitScript(() => localStorage.setItem("cloudanalyzer-autosave-enabled", "off"));
  await page.goto("/");
  const poses = Array.from({ length: 4 }, (_, i) => `1 0 0 ${i * .1} 0 1 0 0 0 0 1 0\n`).join("");
  await page.locator("#pg-files-input").setInputFiles([
    input("survey.kitti", Buffer.from(poses)),
    ...Array.from({ length: 4 }, (_, i) => input(`${String(i).padStart(6, "0")}.ply`, scan(i < 2 ? 0 : lift, i * .1, patches))),
  ]);
  await expect(page.locator("#pg-stats")).toContainText("4 (4 with scans)");
  await page.locator("#pg-split").fill("2");
  await page.locator("#pg-parts").click();
  await expect(page.locator("#pg-change-export")).toBeEnabled();
  await expect(page.locator("#pg-changes")).toBeVisible();
}

test("two-pass review preserves camera across views and exports judgments with measured evidence", async ({ page }, info) => {
  const errors: string[] = []; page.on("pageerror", e => errors.push(e.message));
  await setup(page);
  const initial = await saved(page, "#pg-change-export");
  expect(initial.candidates.length).toBeGreaterThanOrEqual(2);
  await expect(page.locator("#pg-change-quality")).toContainText("0 cross-part loop constraints");
  await expect(page.locator("#pg-change-quality")).toContainText("weighted residual unavailable");
  await page.locator("#pg-change-list button").first().click();
  const before = await saved(page, "#project-save");
  for (const view of ["before", "after", "difference"]) {
    await page.locator("#pg-change-view").selectOption(view);
    const camera = (await saved(page, "#project-save")).session.camera;
    for (const key of ["position", "target"]) for (let axis = 0; axis < 3; axis++) {
      expect(camera[key][axis]).toBeCloseTo(before.session.camera[key][axis], 10);
    }
    const rows = page.locator("#cloud-list > li");
    for (let i = 0; i < 3; i++) await expect(rows.nth(i).locator('input[type="checkbox"]').first()).toBeChecked({ checked: i === ["before", "after", "difference"].indexOf(view) });
  }
  await page.locator("#pg-change-decision").selectOption("misalignment");
  await page.locator("#pg-change-note").fill('確認\n<img src=x onerror=alert(1)>');
  await page.locator("#pg-change-list button").nth(1).click();
  await page.locator("#pg-change-decision").selectOption("change");
  await page.locator("#pg-change-list button").first().click();
  await expect(page.locator("#pg-change-note")).toHaveValue('確認\n<img src=x onerror=alert(1)>');
  const review = await saved(page, "#pg-change-export");
  expect(review.schema).toBe("cloudanalyzer.survey_change_review.v1");
  expect(review.counts).toEqual({ unreviewed: initial.candidates.length - 2, change: 1, misalignment: 1, insufficient: 0, deferred: 0 });
  expect(review.context.firstNodes).toEqual([0, 1]); expect(review.context.secondNodes).toEqual([2, 3]);
  expect(review.context.mapVoxel).toBe(.3); expect(review.context.m3c2.normalRadius).toBe(1);
  expect(review.diagnostics.measured).toBeGreaterThan(0); expect(review.diagnostics.weightedRms).toBeNull();
  expect(Math.abs(review.candidates[0].meanChange)).toBeCloseTo(.8, 3);
  expect(review.candidates[0].note).toContain('<img src=x'); expect(review.staleReason).toBeNull();
  await page.setViewportSize({ width: 1440, height: 1400 });
  await page.locator("#pg-changes").scrollIntoViewIfNeeded();
  await page.locator("#pg-changes").screenshot({ path: info.outputPath("survey-change-review.png") });
  await page.locator("#pg-edge-errors").uncheck();
  await expect(page.locator("#pg-change-stale")).toBeHidden();
  // Fixation is a real graph change; old decisions stay available only as historical evidence.
  await page.locator("#pg-a").fill("1");
  await page.locator("#pg-fix").click();
  await expect(page.locator("#pg-change-stale")).toContainText("Graph poses or constraints changed");
  await expect(page.locator("#pg-change-view")).toBeDisabled();
  await expect(page.locator("#pg-change-decision")).toBeDisabled();
  const historical = await saved(page, "#pg-change-export");
  expect(historical.counts).toEqual(review.counts); expect(historical.staleReason).toBeTruthy();
  await page.locator("#pg-close").click();
  await expect(page.locator("#pg-body")).toBeHidden();
  await expect(page.locator("#pg-change-export")).toBeVisible();
  expect((await saved(page, "#pg-change-export")).counts).toEqual(review.counts);
  expect(errors).toEqual([]);
});

test("all candidates can be reviewed across pages; reruns reset and cloud removal marks history stale", async ({ page }) => {
  await setup(page, .8, 53);
  const initial = await saved(page, "#pg-change-export");
  const total = initial.candidates.length;
  expect(total).toBeGreaterThan(50);
  await expect(page.locator("#pg-change-list li")).toHaveCount(50);
  await page.locator("#pg-change-next").click();
  await expect(page.locator("#pg-change-list li")).toHaveCount(Math.min(50, total - 50));
  await page.locator("#pg-change-list button").last().click();
  await page.locator("#pg-change-decision").selectOption("deferred");
  await page.locator("#pg-change-note").fill("Need another survey");
  await page.locator("#pg-change-prev").click(); await page.locator("#pg-change-next").click();
  await expect(page.locator("#pg-change-note")).toHaveValue("Need another survey");
  let review = await saved(page, "#pg-change-export");
  expect(review.candidates).toHaveLength(total);
  const reviewed = Math.min(100, total) - 1;
  expect(reviewed).toBeGreaterThanOrEqual(50); expect(review.candidates[reviewed].decision).toBe("deferred");
  await page.locator("#pg-parts").click(); await expect(page.locator("#pg-change-export")).toBeEnabled();
  review = await saved(page, "#pg-change-export"); expect(review.counts.unreviewed).toBe(total); expect(review.counts.deferred).toBe(0);
  await page.locator("#cloud-list > li").last().locator('button[title="Remove"]').click();
  await expect(page.locator("#pg-change-stale")).toContainText("changed or removed");
  await expect(page.locator("#pg-change-view")).toBeDisabled();
});

test("no candidate never implies unchanged scene; notes reject invalid edits without replacing saved judgment", async ({ page }) => {
  await setup(page);
  await page.locator("#pg-change-note").fill("Valid note");
  await page.locator("#pg-change-note").evaluate((el: HTMLTextAreaElement) => {
    el.value = "x".repeat(2001); el.dispatchEvent(new Event("input"));
  });
  await expect(page.locator("#status")).toContainText("2,000");
  expect((await saved(page, "#pg-change-export")).candidates[0].note).toBe("Valid note");
  await setup(page, 0);
  await expect(page.locator("#pg-change-list li")).toHaveCount(0);
  await expect(page.locator("#pg-changes-hint")).toContainText("Unmeasured areas and smaller changes still need inspection");
  await expect(page.locator("#pg-change-editor")).toBeHidden();
  const review = await saved(page, "#pg-change-export"); expect(review.candidates).toEqual([]); expect(review.counts.change).toBe(0);
});

async function openArchive(page: Page, record: any, name = "saved-review.json") {
  const section = page.locator("#pg-review-archive");
  if (!(await section.evaluate(el => (el as HTMLDetailsElement).open))) await section.locator("summary").first().click();
  await page.locator("#pg-review-file").setInputFiles(input(name, Buffer.from(JSON.stringify(record))));
}

test("saved decisions resume after rerunning the exact comparison with new cloud IDs", async ({ page }) => {
  const errors: string[] = []; page.on("pageerror", e => errors.push(e.message));
  await setup(page);
  await page.locator("#pg-change-decision").selectOption("misalignment");
  await page.locator("#pg-change-note").fill('確認\n<img src=x onerror=alert(1)>');
  const original = await saved(page, "#pg-change-export");
  expect(original.identity.algorithm).toBe("sha256-ply-graph-v1");
  await page.locator("#pg-parts").click();
  await expect(page.locator("#pg-change-export")).toBeEnabled();
  const rerun = await saved(page, "#pg-change-export");
  expect(rerun.sources.before.id).not.toBe(original.sources.before.id);
  expect(rerun.identity).toEqual(original.identity);
  expect(rerun.counts.misalignment).toBe(0);
  await openArchive(page, original);
  await expect(page.locator("#pg-review-match")).toContainText("Comparison contents match");
  await expect(page.locator("#pg-change-note")).toHaveValue("");
  await expect(page.locator("#pg-review-resume")).toBeEnabled();
  const before = await saved(page, "#project-save");
  await page.locator("#pg-review-resume").click();
  await expect(page.locator("#status")).toContainText("decisions resumed");
  await expect(page.locator("#pg-change-note")).toHaveValue(original.candidates[0].note);
  const resumed = await saved(page, "#pg-change-export");
  expect(resumed.candidates).toEqual(original.candidates);
  expect(resumed.sources).toEqual(rerun.sources); expect(resumed.analyzedAt).toBe(rerun.analyzedAt);
  const after = await saved(page, "#project-save");
  expect(after.poseGraph).toEqual(before.poseGraph);
  expect(after.session.clouds).toEqual(before.session.clouds);
  expect(await page.locator("#pg-review-list img").count()).toBe(0);
  // A fresh page has no retained review state; the file and new comparison must suffice.
  await setup(page);
  await openArchive(page, original);
  await expect(page.locator("#pg-review-resume")).toBeEnabled();
  await page.locator("#pg-review-resume").click();
  expect((await saved(page, "#pg-change-export")).candidates).toEqual(original.candidates);
  expect(errors).toEqual([]);
});

test("changed contents, edited geometry and historical reviews cannot overwrite live decisions", async ({ page }) => {
  await setup(page);
  await page.locator("#pg-change-decision").selectOption("change");
  const original = await saved(page, "#pg-change-export");
  await setup(page, .9);
  await page.locator("#pg-change-note").fill("Keep current judgment");
  const current = await saved(page, "#pg-change-export");
  await openArchive(page, original);
  await expect(page.locator("#pg-review-resume")).toBeDisabled();
  await expect(page.locator("#pg-review-match")).toContainText("does not match");
  expect((await saved(page, "#pg-change-export")).candidates).toEqual(current.candidates);
  const edited = structuredClone(current); edited.candidates[0].meanChange += .01;
  await openArchive(page, edited, "edited-geometry.json");
  await expect(page.locator("#pg-review-name")).toHaveText("edited-geometry.json");
  await expect(page.locator("#pg-review-resume")).toBeDisabled();
  const stale = structuredClone(current); stale.staleReason = "Graph closed";
  await openArchive(page, stale, "historical.json");
  await expect(page.locator("#pg-review-match")).toContainText("Historical reviews cannot resume");
  // Even a synthetic click bypassing the disabled control must not apply old decisions.
  await page.locator("#pg-review-resume").dispatchEvent("click");
  expect((await saved(page, "#pg-change-export")).candidates).toEqual(current.candidates);
});

test("legacy files open without live clouds; invalid or late reads preserve the current history", async ({ page }) => {
  await setup(page);
  await page.locator("#pg-change-note").fill("Legacy note");
  const legacy = await saved(page, "#pg-change-export"); delete legacy.identity;
  await page.goto("/");
  await openArchive(page, legacy, "legacy.json");
  await expect(page.locator("#pg-review-result")).toBeVisible();
  await expect(page.locator("#pg-review-resume")).toBeDisabled();
  await expect(page.locator("#pg-review-summary")).toContainText(legacy.context.graph);
  await expect(page.locator("#pg-review-match")).toContainText("no content identity");
  await page.locator("#pg-review-list details").first().locator("summary").click();
  await expect(page.locator("#pg-review-list")).toContainText("Legacy note");
  const invalid = structuredClone(legacy); invalid.counts.change++;
  await openArchive(page, invalid, "bad.json");
  await expect(page.locator("#status")).toContainText("Could not open survey review");
  await expect(page.locator("#pg-review-name")).toHaveText("legacy.json");
  await page.locator("#pg-review-file").setInputFiles(input("oversized.json", Buffer.alloc(16 * 1024 * 1024 + 1, 32)));
  await expect(page.locator("#status")).toContainText("16 MiB limit");
  await expect(page.locator("#pg-review-name")).toHaveText("legacy.json");
  await page.evaluate(() => {
    const original = File.prototype.text;
    File.prototype.text = function () {
      const file = this;
      if (file.name !== "slow-review.json") return original.call(file);
      return new Promise(resolve => { (window as any).resolveSurveyRead = () => original.call(file).then(resolve); });
    };
  });
  await openArchive(page, legacy, "slow-review.json");
  await page.locator("#pg-review-close").click();
  await page.evaluate(() => (window as any).resolveSurveyRead());
  await expect(page.locator("#pg-review-result")).toBeHidden();
  await expect(page.locator("#cloud-list > li")).toHaveCount(0);
});

test("comparison still works when secure-context hashing is unavailable", async ({ page }) => {
  await page.addInitScript(() => Object.defineProperty(window.crypto, "subtle", { value: undefined }));
  await setup(page);
  await expect(page.locator("#pg-change-quality")).toContainText("HTTPS or localhost");
  await page.locator("#pg-change-note").fill("Historical review still available");
  const record = await saved(page, "#pg-change-export");
  expect(record.identity).toBeNull();
  expect(record.candidates[0].note).toBe("Historical review still available");
  await openArchive(page, record);
  await expect(page.locator("#pg-review-match")).toContainText("no content identity");
  await expect(page.locator("#pg-review-resume")).toBeDisabled();
});

test("filtering keeps the selected note editor stable and exports every candidate", async ({ page }) => {
  await setup(page);
  const initial = await saved(page, "#pg-change-export"), before = await saved(page, "#project-save");
  await page.evaluate(() => {
    const original = Worker.prototype.postMessage; (window as any).filterComputeCalls = 0;
    Worker.prototype.postMessage = function (message: any, ...rest: any[]) {
      if (["m3c2", "change-objects", "pg-map"].includes(message?.req?.kind)) (window as any).filterComputeCalls++;
      return (original as any).call(this, message, ...rest);
    };
  });
  await page.locator("#pg-change-filter").selectOption("unreviewed");
  await page.locator("#pg-change-decision").selectOption("change");
  await expect(page.locator('#pg-change-list button[data-rank="1"]')).toHaveCount(0);
  await expect(page.locator("#pg-change-selected")).toHaveText("Candidate #1");
  await expect(page.locator("#pg-change-selected-context")).toContainText("outside this page or filter");
  await page.locator("#pg-change-note").fill("判定後にメモを追記\n根拠となる点群を確認");
  await page.locator("#pg-change-filter").selectOption("misalignment");
  await expect(page.locator("#pg-change-list li")).toHaveCount(0);
  await expect(page.locator("#pg-change-empty")).toBeVisible();
  await expect(page.locator("#pg-change-note")).toHaveValue("判定後にメモを追記\n根拠となる点群を確認");
  const record = await saved(page, "#pg-change-export");
  expect(record.candidates).toHaveLength(initial.candidates.length); expect(record.counts.change).toBe(1);
  expect(record.candidates[0].note).toContain("メモを追記"); expect(record.identity).toEqual(initial.identity);
  await page.locator("#pg-change-show-selected").click();
  await expect(page.locator("#pg-change-filter")).toHaveValue("all");
  await expect(page.locator('#pg-change-list button[data-rank="1"]')).toHaveAttribute("aria-pressed", "true");
  expect(await page.evaluate(() => (window as any).filterComputeCalls)).toBe(0);
  const after = await saved(page, "#project-save");
  for (const key of ["position", "target"]) for (let axis = 0; axis < 3; axis++) {
    expect(after.session.camera[key][axis]).toBeCloseTo(before.session.camera[key][axis], 10);
  }
});

test("restored decisions refresh filters and next-unreviewed skips across original pages and wraps", async ({ page }) => {
  await setup(page, .8, 53);
  const original = await saved(page, "#pg-change-export"), total = original.candidates.length;
  expect(total).toBeGreaterThan(50);
  for (const candidate of original.candidates) if (![1, total].includes(candidate.rank)) {
    candidate.decision = "change"; candidate.updatedAt = original.analyzedAt;
  }
  original.counts = { unreviewed: 2, change: total - 2, misalignment: 0, insufficient: 0, deferred: 0 };
  await openArchive(page, original);
  await page.locator("#pg-change-filter").selectOption("unreviewed");
  await page.locator("#pg-review-resume").click();
  await expect(page.locator("#pg-change-list li")).toHaveCount(2);
  await expect(page.locator("#pg-change-filter-summary")).toContainText(`2 of ${total}`);
  await page.locator("#pg-change-next-unreviewed").click();
  await expect(page.locator("#pg-change-selected")).toHaveText(`Candidate #${total}`);
  await expect(page.locator(`#pg-change-list button[data-rank="${total}"]`)).toHaveAttribute("aria-pressed", "true");
  await page.locator("#pg-change-next-unreviewed").click();
  await expect(page.locator("#pg-change-selected")).toHaveText("Candidate #1");
  await page.locator("#pg-change-decision").selectOption("deferred");
  await page.locator("#pg-change-next-unreviewed").click();
  await expect(page.locator("#pg-change-next-unreviewed")).toBeDisabled();
  await page.locator("#pg-change-decision").selectOption("change");
  await expect(page.locator("#pg-change-empty")).toBeVisible();
  await expect(page.locator("#pg-change-next-unreviewed")).toBeDisabled();
  await page.locator("#pg-change-filter").selectOption("deferred");
  await expect(page.locator("#pg-change-list li")).toHaveCount(1);
  await expect(page.locator('#pg-change-list button[data-rank="1"]')).toBeVisible();
  await page.locator("#pg-change-filter").selectOption("decided");
  await expect(page.locator("#pg-change-list li")).toHaveCount(50);
  await page.locator("#pg-change-show-selected").click();
  await expect(page.locator(`#pg-change-list button[data-rank="${total}"]`)).toBeVisible();
  const record = await saved(page, "#pg-change-export");
  expect(record.candidates).toHaveLength(total); expect(record.counts.unreviewed).toBe(0);
  expect(record.counts.deferred).toBe(1); expect(record.counts.change).toBe(total - 1);
});

function relatedMap(bounds: number[], offset = 0) {
  const z = (bounds[2] + bounds[5]) / 2;
  return { format: "vectormap-ir", version: 1,
    lanes: [101, 102, 103].map((id, i) => ({ id, kind: "driving", left: 1001 + i * 2, right: 1002 + i * 2 })),
    boundaries: Array.from({ length: 6 }, (_, i) => {
      const lane = Math.floor(i / 2), x = offset + (lane === 2 ? 10000 : 0);
      const y = i % 2 ? bounds[1] - 2 : bounds[4] + 2, height = z + (lane === 1 ? 8 : 0);
      return { id: 1001 + i, kind: { type: "virtual" }, geometry: [[bounds[0] - 2 + x, y, height], [bounds[3] + 2 + x, y, height]] };
    }) };
}
async function loadRelatedMap(page: Page, bounds: number[], offset = 0) {
  await page.locator("#vm-file").setInputFiles(input("roads.json", Buffer.from(JSON.stringify(relatedMap(bounds, offset)))));
  await expect(page.locator("#vm-status")).toContainText("3 lane");
}

test("candidate boxes link to lane surfaces with height margins and selection leaves both datasets unchanged", async ({ page }) => {
  const errors: string[] = []; page.on("pageerror", e => errors.push(e.message));
  await setup(page);
  const original = await saved(page, "#pg-change-export");
  await loadRelatedMap(page, original.candidates[0].bounds);
  await page.locator("#pg-change-lanes summary").click();
  await expect(page.locator("#pg-change-lane-hint")).toContainText("Confirm a shared Cartesian metre frame");
  await expect(page.locator("#pg-change-lane-list li")).toHaveCount(0);
  const before = await saved(page, "#project-save");
  await page.locator("#pg-change-lane-frame").check();
  await expect(page.locator("#pg-change-lane-hint")).toContainText("1 overlapping lane surface");
  await expect(page.locator("#pg-change-lane-hint")).toContainText("3 of 3 lanes checked");
  await expect(page.locator("#pg-change-lane-list li")).toHaveCount(1);
  await page.getByRole("button", { name: "Inspect lane 101", exact: true }).click();
  await expect(page.locator("#vm-lane-title")).toHaveText("Lane 101");
  const after = await saved(page, "#project-save");
  expect(after.vectorMap).toEqual(before.vectorMap); expect(after.poseGraph).toEqual(before.poseGraph);
  expect((await saved(page, "#pg-change-export")).candidates).toEqual(original.candidates);
  await page.locator("#pg-change-lane-z").fill("8");
  await expect(page.locator("#pg-change-lane-list li")).toHaveCount(2);
  await expect(page.getByRole("button", { name: "Inspect lane 102", exact: true })).toBeVisible();
  await page.locator("#pg-change-lane-xy").fill("");
  await expect(page.locator("#pg-change-lane-list li")).toHaveCount(0);
  await expect(page.locator("#pg-change-lane-hint")).toContainText("Enter both margins");
  await page.locator("#pg-change-lane-xy").fill("0");
  await expect(page.locator("#pg-change-lane-list li")).toHaveCount(2);
  expect(errors).toEqual([]);
});

test("map edits and same-ID replacements clear lane associations without changing human judgments", async ({ page }) => {
  await setup(page);
  await page.locator("#pg-change-note").fill("Keep this judgment");
  const original = await saved(page, "#pg-change-export");
  await loadRelatedMap(page, original.candidates[0].bounds);
  await page.locator("#pg-change-lanes summary").click();
  await page.locator("#pg-change-lane-frame").check();
  const oldButton = await page.getByRole("button", { name: "Inspect lane 101", exact: true }).elementHandle();
  await page.getByRole("button", { name: "Inspect lane 101", exact: true }).click();
  await page.locator("#vm-lane-speed").fill("30"); await page.locator("#vm-lane-apply").click();
  await expect(page.locator("#pg-change-lane-frame")).not.toBeChecked();
  await expect(page.locator("#pg-change-lane-list li")).toHaveCount(0);
  await expect(page.locator("#pg-change-stale")).toBeHidden();
  await loadRelatedMap(page, original.candidates[0].bounds, 10000);
  await oldButton!.evaluate(el => (el as HTMLButtonElement).click());
  await expect(page.locator("#vm-lane")).toBeHidden();
  await page.locator("#pg-change-lane-frame").check();
  await expect(page.locator("#pg-change-lane-hint")).toContainText("0 overlapping lane surfaces");
  await oldButton!.evaluate(el => (el as HTMLButtonElement).click());
  await expect(page.locator("#vm-lane")).toBeHidden();
  await expect(page.locator("#pg-change-lane-hint")).toContainText("does not establish");
  expect((await saved(page, "#pg-change-export")).candidates).toEqual(original.candidates);
  await page.locator("#pg-a").fill("1"); await page.locator("#pg-fix").click();
  await expect(page.locator("#pg-change-lane-hint")).toContainText("Historical comparison");
  await expect(page.locator("#pg-change-lane-frame")).toBeDisabled();
});

async function recordFirstLaneTask(page: Page) {
  await page.getByRole("button", { name: "Record lane 101", exact: true }).click();
  await expect(page.locator("#pg-lane-task-export")).toBeEnabled();
}
async function setupLaneTasks(page: Page) {
  await setup(page);
  const review = await saved(page, "#pg-change-export");
  await loadRelatedMap(page, review.candidates[0].bounds);
  await page.locator("#pg-change-lanes summary").click();
  await page.locator("#pg-change-lane-frame").check();
  return review;
}

test("explicit lane follow-up records save decisions and snapshots without editing either map or survey", async ({ page }) => {
  const errors: string[] = []; page.on("pageerror", e => errors.push(e.message));
  const review = await setupLaneTasks(page);
  const before = await saved(page, "#project-save");
  await recordFirstLaneTask(page);
  await expect(page.locator("#pg-lane-task-summary")).toContainText("1 follow-up records");
  await recordFirstLaneTask(page);
  await expect(page.locator("#pg-lane-task-list li")).toHaveCount(1);
  await page.locator("#pg-lane-task-state").selectOption("update-needed");
  await page.locator("#pg-lane-task-note").fill('境界を再確認\n<img src=x onerror=alert(1)>');
  await page.locator("#pg-lane-task-save").click();
  const file = await saved(page, "#pg-lane-task-export");
  expect(file.schema).toBe("cloudanalyzer.survey_lane_tasks.v1"); expect(file.tasks).toHaveLength(1);
  expect(file.tasks[0].status).toBe("update-needed"); expect(file.tasks[0].note).toContain("境界を再確認");
  expect(file.tasks[0].candidate).toEqual(review.candidates[0]);
  expect(file.tasks[0].comparison.identity).toEqual(review.identity);
  expect(file.tasks[0].mapSha256).toMatch(/^[a-f0-9]{64}$/);
  expect(file.tasks[0].lane.id).toBe(101); expect(file.tasks[0].overlap.zMargin).toBe(1);
  expect(file.tasks[0].coreSupport).toEqual({protocol:'xy-lane-z-v1',status:'complete',total:review.candidates[0].points,
    supported:review.candidates[0].points,heightMarginM:1,reason:null});
  await expect(page.locator('#pg-lane-task-source')).toContainText(`Saved core support: ${review.candidates[0].points} / ${review.candidates[0].points}`);
  await page.locator("#pg-lane-task-inspect").click();
  await expect(page.locator("#vm-lane-title")).toHaveText("Lane 101");
  await expect(page.locator("#vm-review-state")).toHaveValue("unreviewed");
  const after = await saved(page, "#project-save");
  expect(after.vectorMap).toEqual(before.vectorMap); expect(after.poseGraph).toEqual(before.poseGraph);
  expect((await saved(page, "#pg-change-export")).candidates).toEqual(review.candidates);
  expect(errors).toEqual([]);
});

test("changed source judgments and same-ID map replacements preserve only historical lane tasks", async ({ page }) => {
  const review = await setupLaneTasks(page);
  await recordFirstLaneTask(page);
  await page.locator("#pg-change-note").fill("New source judgment");
  await expect(page.locator("#pg-lane-task-state")).toBeDisabled();
  let file = await saved(page, "#pg-lane-task-export");
  expect(file.tasks[0].staleReason).toBeTruthy(); expect(file.tasks[0].candidate.note).toBe("");
  await recordFirstLaneTask(page);
  await expect(page.locator("#pg-lane-task-list li")).toHaveCount(2);
  await page.locator("#pg-lane-task-inspect").click();
  await page.locator("#vm-lane-speed").fill("30"); await page.locator("#vm-lane-apply").click();
  await expect(page.locator("#pg-lane-task-summary")).toContainText("2 historical");
  await expect(page.locator("#pg-lane-task-inspect")).toBeDisabled();
  await loadRelatedMap(page, review.candidates[0].bounds);
  await expect(page.locator("#pg-lane-task-state")).toBeDisabled();
  await expect(page.locator("#vm-lane")).toBeHidden();
  file = await saved(page, "#pg-lane-task-export");
  expect(file.tasks.every((task: any) => !!task.staleReason)).toBe(true);
  expect(file.tasks[1].candidate.note).toBe("New source judgment");
});

test("lane task archives reopen as history and invalid files never replace the current queue", async ({ page }) => {
  await setupLaneTasks(page); await recordFirstLaneTask(page);
  await page.locator("#pg-lane-task-state").selectOption("deferred");
  await page.locator("#pg-lane-task-note").fill('後日確認\n<img src=x onerror=alert(1)>');
  await page.locator("#pg-lane-task-save").click();
  const file = await saved(page, "#pg-lane-task-export");
  await page.locator("#pg-lane-task-file").setInputFiles(input("tasks.json", Buffer.from(JSON.stringify(file))));
  await expect(page.locator("#pg-lane-task-archive-name")).toContainText("tasks.json");
  await expect(page.locator("#pg-lane-task-list li")).toHaveCount(1);
  await expect(page.locator("#pg-lane-task-state")).toHaveValue("deferred");
  await page.goto("/"); await page.locator("#pg-lane-tasks summary").first().click();
  await page.locator("#pg-lane-task-file").setInputFiles(input("tasks.json", Buffer.from(JSON.stringify(file))));
  await expect(page.locator("#pg-lane-task-archive-name")).toContainText("1 historical follow-up records");
  await page.locator("#pg-lane-task-archive-list summary").first().click();
  await expect(page.locator("#pg-lane-task-archive-list")).toContainText("後日確認");
  await expect(page.locator("#pg-lane-task-archive-list img")).toHaveCount(0);
  await expect(page.locator('#pg-lane-task-archive-list')).toContainText('Saved core support:');
  const legacy=structuredClone(file);delete legacy.tasks[0].coreSupport;
  await page.locator('#pg-lane-task-file').setInputFiles(input('legacy.json',Buffer.from(JSON.stringify(legacy))));
  await page.locator('#pg-lane-task-archive-list summary').first().click();
  await expect(page.locator('#pg-lane-task-archive-list')).toContainText('Core support not recorded');
  await page.locator('#pg-lane-task-file').setInputFiles(input('tasks.json',Buffer.from(JSON.stringify(file))));
  await expect(page.locator("#pg-lane-task-list li")).toHaveCount(0);
  await expect(page.locator("#pg-lane-task-editor")).toBeHidden();
  const invalid = structuredClone(file); invalid.counts.deferred++;
  await page.locator("#pg-lane-task-file").setInputFiles(input("bad.json", Buffer.from(JSON.stringify(invalid))));
  await expect(page.locator("#status")).toContainText("Could not open lane tasks");
  await expect(page.locator("#pg-lane-task-archive-name")).toContainText("tasks.json");
});

test("a map change during lane-task hashing cannot commit a task against reused IDs", async ({ page }) => {
  await setupLaneTasks(page);
  await page.getByRole("button", { name: "Inspect lane 101", exact: true }).click();
  await page.evaluate(() => {
    const original = crypto.subtle.digest.bind(crypto.subtle);
    Object.defineProperty(crypto.subtle, "digest", { configurable: true, value: (...args: any[]) => new Promise(resolve => {
      (window as any).finishLaneTaskHash = () => (original as any)(...args).then(resolve);
    }) });
  });
  await page.getByRole("button", { name: "Record lane 101", exact: true }).click();
  await expect.poll(() => page.evaluate(() => typeof (window as any).finishLaneTaskHash)).toBe("function");
  await page.locator("#vm-lane-speed").fill("40"); await page.locator("#vm-lane-apply").click();
  await expect(page.locator("#pg-change-lane-frame")).not.toBeChecked();
  await page.evaluate(() => (window as any).finishLaneTaskHash());
  await expect(page.locator("#status")).toContainText("Could not record lane task");
  await expect(page.locator("#pg-lane-task-list li")).toHaveCount(0);
  await expect(page.locator("#pg-lane-task-export")).toBeDisabled();
});

test("both join inputs stay disabled throughout a pose-graph operation", async ({ page }) => {
  await setup(page);
  await page.locator('#pose-graph-panel summary',{hasText:'Join another graph'}).click();
  await expect(page.locator('#pg-merge')).toBeEnabled();
  await expect(page.locator('#pg-merge-files')).toBeEnabled();
  await page.evaluate(()=>{
    const original=Worker.prototype.postMessage;
    Worker.prototype.postMessage=function(message:any,...rest:any[]){
      if(message?.req?.kind==='pg-optimize') {
        (window as any).releaseJoinControlCheck=()=> (original as any).call(this,message,...rest);
        return;
      }
      return (original as any).call(this,message,...rest);
    };
  });
  await page.locator('#pg-optimize').click();
  await expect(page.locator('#pg-merge')).toBeDisabled();
  await expect(page.locator('#pg-merge-files')).toBeDisabled();
  await page.evaluate(()=> (window as any).releaseJoinControlCheck());
  await expect(page.locator('#pg-optimize')).toBeEnabled();
  await expect(page.locator('#pg-merge')).toBeEnabled();
  await expect(page.locator('#pg-merge-files')).toBeEnabled();
});

async function changedCoreXs(page: Page, review: any): Promise<number[]> {
  const row=page.locator('#cloud-list > li').filter({hasText:review.sources.difference.name});
  await row.locator('button[title="Save with distances…"]').click();
  const ready=page.waitForEvent('download'); await row.locator('.save-formats button',{hasText:'PLY'}).click();
  const parts:Buffer[]=[];for await(const part of await(await ready).createReadStream())parts.push(part as Buffer);
  const raw=Buffer.concat(parts), end=raw.indexOf('end_header\n')+'end_header\n'.length, header=raw.subarray(0,end).toString();
  const sizes:Record<string,number>={double:8,float:4,uchar:1};
  let stride=0;const fields:Record<string,number>={};
  for(const line of header.split('\n')) if(line.startsWith('property ')) {
    const [,type,name]=line.split(' ');expect(sizes[type]).toBeTruthy();fields[name]=stride;stride+=sizes[type];
  }
  const count=Number(/element vertex (\d+)/.exec(header)![1]),xs:number[]=[];
  for(let i=0;i<count;i++) if(raw.readFloatLE(end+i*stride+fields.change_object)===review.candidates[0].rank)xs.push(raw.readDoubleLE(end+i*stride+fields.x));
  expect(xs).toHaveLength(review.candidates[0].points);return [...new Set(xs)].sort((a,b)=>a-b);
}

test("original changed-core support appears in the lane list and overlay without changing saved data", async ({page},info)=>{
  const review=await setupLaneTasks(page), total=review.candidates[0].points;
  const metric=page.locator('[data-core-lane="101"]');
  await expect(metric).toHaveAttribute('data-core-status','complete');
  await expect(metric).toHaveAttribute('data-core-supported',String(total));
  await expect(metric).toHaveAttribute('data-core-total',String(total));
  const before=await saved(page,'#project-save');
  await page.getByRole('button',{name:'Show changed cores for lane 101',exact:true}).click();
  await expect(page.locator('#pg-core-overlay-hint')).toContainText(`${total} of ${total}`);
  await expect(page.locator('#pg-core-overlay-clear')).toBeVisible();
  await page.screenshot({path:info.outputPath('changed-core-overlay.png')});
  const after=await saved(page,'#project-save');expect(after.vectorMap).toEqual(before.vectorMap);expect(after.poseGraph).toEqual(before.poseGraph);
  expect((await saved(page,'#pg-change-export')).candidates).toEqual(review.candidates);
  await page.locator('#pg-change-list button').nth(1).click();
  await expect(page.locator('#pg-core-overlay-clear')).toBeHidden();
  await expect(page.locator('#pg-core-overlay-hint')).toBeEmpty();
});

test("box-only links remain visible with zero individual-core support", async ({page})=>{
  const review=await setupLaneTasks(page), xs=await changedCoreXs(page,review);
  let index=0;for(let i=1;i<xs.length-1;i++)if(xs[i+1]-xs[i]>xs[index+1]-xs[index])index=i;
  const gap=xs[index+1]-xs[index], x=(xs[index+1]+xs[index])/2,b=review.candidates[0].bounds,z=(b[2]+b[5])/2;
  const map={format:'vectormap-ir',version:1,lanes:[{id:201,kind:'driving',left:1,right:2}],boundaries:[
    {id:1,kind:{type:'virtual'},geometry:[[x-gap/4,b[1]-1,z],[x-gap/4,b[4]+1,z]]},
    {id:2,kind:{type:'virtual'},geometry:[[x+gap/4,b[1]-1,z],[x+gap/4,b[4]+1,z]]}]};
  await page.locator('#vm-file').setInputFiles(input('empty-gap.json',Buffer.from(JSON.stringify(map))));
  await expect(page.locator('#vm-status')).toContainText('1 lane');
  await page.locator('#pg-change-lane-frame').check();
  const metric=page.locator('[data-core-lane="201"]');
  await expect(metric).toHaveAttribute('data-core-status','complete');
  await expect(metric).toHaveAttribute('data-core-supported','0');
  await expect(metric).toContainText('box overlap only');
  await expect(page.getByRole('button',{name:'Record lane 201',exact:true})).toBeEnabled();
  await page.getByRole('button',{name:'Show changed cores for lane 201',exact:true}).click();
  await expect(page.locator('#pg-core-overlay-hint')).toContainText(`0 of ${review.candidates[0].points}`);
  await page.getByRole('button',{name:'Record lane 201',exact:true}).click();
  await expect(page.locator('#pg-lane-task-export')).toBeEnabled();
  const file=await saved(page,'#pg-lane-task-export');
  expect(file.tasks[0].coreSupport).toMatchObject({status:'complete',supported:0,total:review.candidates[0].points});
  await page.locator('#pg-lane-task-file').setInputFiles(input('zero.json',Buffer.from(JSON.stringify(file))));
  await page.locator('#pg-lane-task-archive-list summary').first().click();
  await expect(page.locator('#pg-lane-task-archive-list')).toContainText('box overlap only');
});

test("unavailable or superseded core evidence never appears as zero or leaks across map replacement", async ({page})=>{
  await setup(page);const review=await saved(page,'#pg-change-export');await loadRelatedMap(page,review.candidates[0].bounds);
  await page.locator('#pg-change-lanes summary').click();
  await page.evaluate(()=>{
    const original=Worker.prototype.postMessage;
    Worker.prototype.postMessage=function(message:any,...rest:any[]){
      if(message?.req?.kind==='change-core-support'&&!(window as any).coreFailureInjected){message.req.expectedPoints++;(window as any).coreFailureInjected=true;}
      return(original as any).call(this,message,...rest);
    };
  });
  await page.locator('#pg-change-lane-frame').check();
  const metric=page.locator('[data-core-lane="101"]');
  await expect(metric).toHaveAttribute('data-core-status','unavailable');
  await expect(metric).not.toHaveAttribute('data-core-supported','0');
  await expect(page.getByRole('button',{name:'Show changed cores for lane 101',exact:true})).toBeDisabled();
  await page.evaluate(()=>{
    const original=Worker.prototype.postMessage;
    Worker.prototype.postMessage=function(message:any,...rest:any[]){
      if(message?.req?.kind==='change-core-support') {(window as any).releaseOldCoreQuery=()=> (original as any).call(this,message,...rest);return;}
      return(original as any).call(this,message,...rest);
    };
  });
  await page.locator('#pg-change-lane-z').fill('2');
  await expect.poll(()=>page.evaluate(()=>typeof(window as any).releaseOldCoreQuery)).toBe('function');
  await loadRelatedMap(page,review.candidates[0].bounds,10000);
  await page.evaluate(()=>(window as any).releaseOldCoreQuery());
  await expect(page.locator('#pg-change-lane-list li')).toHaveCount(0);
  await expect(page.locator('#pg-core-overlay-clear')).toBeHidden();
  await expect(page.locator('#pg-change-lane-frame')).not.toBeChecked();
});


test("recording saves unavailable evidence explicitly and rejects a map replaced during core collection",async({page})=>{
  await setupLaneTasks(page);
  await expect(page.locator('[data-core-lane="101"]')).toHaveAttribute('data-core-status','complete');
  await page.evaluate(()=>{
    const original=Worker.prototype.postMessage;
    Worker.prototype.postMessage=function(message:any,...rest:any[]){
      if(message?.req?.kind==='change-core-support')message.req.expectedPoints++;
      return(original as any).call(this,message,...rest);
    };
  });
  await recordFirstLaneTask(page);
  const file=await saved(page,'#pg-lane-task-export');
  expect(file.tasks[0].coreSupport).toMatchObject({status:'unavailable',supported:null});
  expect(file.tasks[0].coreSupport.reason).toContain('count');
  await page.locator('#pg-lane-task-file').setInputFiles(input('unavailable.json',Buffer.from(JSON.stringify(file))));
  await page.locator('#pg-lane-task-archive-list summary').first().click();
  await expect(page.locator('#pg-lane-task-archive-list')).toContainText('Saved core support unavailable:');

  await page.goto('/');const review=await setupLaneTasks(page);
  await expect(page.locator('[data-core-lane="101"]')).toHaveAttribute('data-core-status','complete');
  await page.evaluate(()=>{
    const original=Worker.prototype.postMessage;
    Worker.prototype.postMessage=function(message:any,...rest:any[]){
      if(message?.req?.kind==='change-core-support'){
        (window as any).releaseTaskCore=()=> (original as any).call(this,message,...rest);return;
      }
      return(original as any).call(this,message,...rest);
    };
  });
  await page.getByRole('button',{name:'Record lane 101',exact:true}).click();
  await expect.poll(()=>page.evaluate(()=>typeof(window as any).releaseTaskCore)).toBe('function');
  await loadRelatedMap(page,review.candidates[0].bounds,10000);
  await page.evaluate(()=>(window as any).releaseTaskCore());
  await expect(page.locator('#status')).toContainText('Could not record lane task');
  await expect(page.locator('#pg-lane-task-list li')).toHaveCount(0);
});

test('project lane tasks resume only after matching current evidence and retain saved decisions', async ({ page }) => {
  await setupLaneTasks(page);
  await recordFirstLaneTask(page);
  await page.locator('#pg-lane-task-state').selectOption('deferred');
  await page.locator('#pg-lane-task-note').fill('Resume this saved decision');
  await page.locator('#pg-lane-task-save').click();
  const original = await saved(page, '#pg-lane-task-export');
  const project = await saved(page, '#project-save');
  await page.reload();
  const poses = Array.from({ length: 4 }, (_, i) => `1 0 0 ${i * .1} 0 1 0 0 0 0 1 0\n`).join('');
  await page.locator('#file-input').setInputFiles([
    input('project.cloudanalyzer.json', Buffer.from(JSON.stringify(project))),
    input('survey.kitti', Buffer.from(poses)),
    ...Array.from({ length: 4 }, (_, i) => input(`${String(i).padStart(6, '0')}.ply`, scan(i < 2 ? 0 : .8, i * .1))),
  ]);
  await expect(page.locator('#status')).toContainText('Project restored');
  await expect(page.locator('#pg-lane-task-state')).toBeDisabled();
  await page.locator('#pg-split').fill('2');
  await page.locator('#pg-parts').click();
  await expect(page.locator('#pg-change-export')).toBeEnabled();
  if (!(await page.locator('#pg-review-archive').evaluate(el => (el as HTMLDetailsElement).open))) await page.locator('#pg-review-archive > summary').click();
  await expect(page.locator('#pg-review-resume')).toBeEnabled({timeout:1000});
  await page.locator('#pg-review-resume').click();
  await page.locator('#pg-change-list button').first().click();
  await page.locator('#pg-change-lanes summary').click();
  await page.locator('#pg-change-lane-frame').check();
  await page.locator('#pg-change-lane-z').fill('2');
  await page.getByRole('button', { name: 'Resume saved task for lane 101', exact: true }).click();
  await expect(page.locator('#status')).toContainText('evidence differs');
  await expect(page.locator('#pg-lane-task-state')).toBeDisabled();
  await page.locator('#pg-change-lane-z').fill('1');
  await page.getByRole('button', { name: 'Resume saved task for lane 101', exact: true }).click();
  await expect(page.locator('#status')).toContainText('Resumed task');
  await expect(page.locator('#pg-lane-task-state')).toBeEnabled();
  expect((await saved(page, '#pg-lane-task-export')).tasks).toEqual(original.tasks);
  await page.locator('#pg-lane-task-note').fill('Continued review');
  await page.locator('#pg-lane-task-save').click();
  expect((await saved(page, '#pg-lane-task-export')).tasks[0].note).toBe('Continued review');
});
