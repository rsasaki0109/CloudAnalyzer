import { expect, test, type Download, type Page } from '@playwright/test';
import { writeFile } from 'node:fs/promises';

async function bytes(download: Download): Promise<Buffer> {
  const chunks: Buffer[] = [];
  for await (const chunk of await download.createReadStream()) chunks.push(chunk);
  return Buffer.concat(chunks);
}

async function snapshot(page: Page) {
  const saved = page.waitForEvent('download', d => d.suggestedFilename() === 'project.cloudanalyzer.json');
  await page.locator('#project-save').click();
  return JSON.parse((await bytes(await saved)).toString());
}

// OSM import may allocate different IDs. Compare geometry in the survey frame,
// allowing only the writer's decimal rounding, and compare the saved lane speeds.
function geometry(map: {boundaries: {geometry: number[][]}[]}) {
  return map.boundaries.map(b => JSON.stringify(b.geometry.map(p => p.map(v => Number(v.toFixed(6)))))).sort();
}

function travelAndTopology(map: any) {
  const boundaries = new Map<number, number[][]>(map.boundaries.map(b => [b.id, b.geometry]));
  const oriented = (ref: number | {id: number; reversed?: boolean}) => {
    const points = boundaries.get(typeof ref === 'number' ? ref : ref.id)!;
    const ordered = typeof ref !== 'number' && ref.reversed ? [...points].reverse() : points;
    return ordered.map(p => p.map(v => Number(v.toFixed(6))));
  };
  const lanes = new Map<number, string>(map.lanes.map(l => [l.id, JSON.stringify({left: oriented(l.left), right: oriented(l.right), speed: l.speed_limit?.kmh})]));
  const neighbor = (n: {lane: number; direction?: string} | undefined) => n ? {lane: lanes.get(n.lane), direction: n.direction ?? 'same'} : null;
  return map.lanes.map(l => {
    const t = map.topology?.find(t => t.lane === l.id);
    return JSON.stringify({lane: lanes.get(l.id), predecessors: (t?.predecessors ?? []).map(id => lanes.get(id)).sort(), successors: (t?.successors ?? []).map(id => lanes.get(id)).sort(), left: neighbor(t?.left), right: neighbor(t?.right)});
  }).sort();
}

test('NCLT corrected drive builds with default pieces, edits, checks and reopens its output', async ({ page }, info) => {
  test.skip(process.env.CLOUDANALYZER_REAL_DATA !== '1', 'Opt in with CLOUDANALYZER_REAL_DATA=1; NCLT attribution applies.');
  test.setTimeout(900_000);
  const started = Date.now(), errors: string[] = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.goto('/?demo=nclt');
  await expect(page.locator('#status')).toContainText('nclt-2012-04-29_map is colored', {timeout: 540_000});
  const importedMs = Date.now() - started;
  const posesReady = page.waitForEvent('download');
  await page.locator('#pg-save-kitti').click();
  const poses = await bytes(await posesReady);
  await page.locator('#file-input').setInputFiles({name: 'corrected-drive.kitti', mimeType: 'text/plain', buffer: poses});
  await expect(page.locator('#vm-trajectory option')).toHaveCount(1);
  await page.getByText('Build from a trajectory', {exact: true}).click();
  // Equipment proposals are a separate workflow; all road options stay default.
  await page.locator('#vm-discover-after-build').uncheck();
  await expect(page.locator('#vm-segment')).toHaveValue('50');
  const building = Date.now();
  await page.locator('#vm-build').click();
  // The nominal two-lane width folds an inner boundary on this sharp turn.
  // Reject that contradictory draft atomically, rather than export geometry
  // which Lanelet2 will interpret with a different boundary/travel direction.
  await expect(page.locator('#status')).toContainText('ambiguous travel directions', {timeout: 60_000});
  await expect(page.locator('#vm-status')).toContainText('No map yet');
  await expect(page.locator('#vm-build-failure')).toBeVisible();
  await expect(page.locator('#vm-failure-settings')).toContainText('1 forward / 1 backward');
  await expect(page.locator('#vm-failure-reason')).toContainText('Boundary travel direction');
  await page.locator('#vm-failure-focus').click();
  await page.screenshot({path: info.outputPath('rejected-boundaries.png')});
  await page.locator('#vm-failure-options').click();
  await page.locator('#vm-backward').fill('0');
  await expect(page.locator('#vm-failure-stale')).toBeVisible();
  await expect(page.locator('#vm-failure-settings')).toContainText('1 forward / 1 backward');
  await page.locator('#vm-failure-retry').click();
  await expect(page.locator('#status')).toContainText(/Draft roads added|Could not build draft roads/, {timeout: 60_000});
  await expect(page.locator('#status')).toContainText('Draft roads added', {timeout: 0});
  await expect(page.locator('#vm-status')).toContainText('Autoware check: 0 errors, 0 warnings.');
  await expect(page.locator('#vm-build-failure')).toBeHidden();
  const buildMs = Date.now() - building;
  const buildReport = await page.locator('#vm-build-report').textContent();
  const mapStatus = await page.locator('#vm-status').textContent();
  await page.locator('#vm-review-next').click();
  await page.locator('#vm-lane-speed').fill('25');
  await page.locator('#vm-lane-apply').click();
  await expect(page.locator('#status')).toContainText('Speed limit of lane');
  await page.locator('#vm-review-state').selectOption('deferred');
  await page.locator('#vm-review-notes').fill('Campus drive; verify lane geometry and permitted directions against the survey.');
  await page.locator('#vm-review-save').click();
  await page.locator('#vm-quality summary').click();
  await page.locator('#vm-quality-check').click();
  await expect(page.locator('#status')).toContainText('Source coverage checked', {timeout: 120_000});
  const quality = await page.locator('#vm-quality-report').textContent();
  const before = await snapshot(page), beforeMap = JSON.parse(before.vectorMap);
  expect(beforeMap.lanes.length).toBeGreaterThan(2);
  expect(beforeMap.lanes.filter(l => l.speed_limit?.kmh === 25)).toHaveLength(1);
  const osmReady = page.waitForEvent('download', d => d.suggestedFilename() === 'lanelet2_map.osm');
  const projectorReady = page.waitForEvent('download', d => d.suggestedFilename() === 'map_projector_info.yaml');
  await page.locator('#vm-export').click();
  const osm = await bytes(await osmReady), yaml = await bytes(await projectorReady);
  await expect(page.locator('#status')).toContainText('Saved lanelet2_map.osm');
  await expect(page.locator('#vm-export-issues [title="lanelet2.no_georeference"]')).toContainText('local_x/local_y');
  expect(yaml.toString()).toBe('projector_type: Local\n');
  const exportStatus = await page.locator('#status').textContent();
  const exportIssues = await page.locator('#vm-export-issues').textContent();
  await page.screenshot({path: info.outputPath('saved-draft.png')});
  await page.locator('#vm-clear').click();
  await expect(page.locator('#vm-export-report')).toBeHidden();
  await page.locator('#vm-file').setInputFiles({name: 'lanelet2_map.osm', mimeType: 'application/xml', buffer: osm});
  await expect(page.locator('#status')).toContainText(`Opened lanelet2_map.osm: ${beforeMap.lanes.length} lanes`);
  await expect(page.locator('#vm-status')).toContainText('Autoware check: 0 errors, 0 warnings.');
  // Chromium limits bursts of automatic downloads; this is the fifth file.
  await page.waitForTimeout(1100);
  const afterMap = JSON.parse((await snapshot(page)).vectorMap);
  await writeFile(info.outputPath('before-map.json'), JSON.stringify(beforeMap, null, 2));
  await writeFile(info.outputPath('reopened-map.json'), JSON.stringify(afterMap, null, 2));
  expect(geometry(afterMap)).toEqual(geometry(beforeMap));
  expect(travelAndTopology(afterMap)).toEqual(travelAndTopology(beforeMap));
  expect(afterMap.lanes.map(l => l.speed_limit?.kmh).sort()).toEqual(beforeMap.lanes.map(l => l.speed_limit?.kmh).sort());
  expect(errors).toEqual([]);
  for (const [name, data] of Object.entries({
    'workflow.json': JSON.stringify({importedMs, buildMs, totalMs: Date.now() - started, roadSettings: {forwardLanes: 1, backwardLanes: 0, pieceLength: 50, otherRoadOptions: 'default'}, defaultTwoLaneDraftRejected: true, poses: poses.toString().trim().split('\n').length, lanes: beforeMap.lanes.length, boundaries: beforeMap.boundaries.length, mapStatus, buildReport, quality, exportStatus, exportIssues, exportBytes: osm.length, errors}, null, 2),
    'lanelet2_map.osm': osm,
    'map_projector_info.yaml': yaml,
    'corrected-drive.kitti': poses,
  })) {
    const path = info.outputPath(name);
    await writeFile(path, data);
    await info.attach(name, {path});
  }
});
