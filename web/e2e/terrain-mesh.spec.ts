import { expect, test, type Page, type Download } from '@playwright/test';
import { writeFile } from 'node:fs/promises';

function ply(points: number[][]) {
  const header = Buffer.from(`ply\nformat binary_little_endian 1.0\nelement vertex ${points.length}\nproperty float x\nproperty float y\nproperty float z\nend_header\n`), body = Buffer.alloc(points.length * 12);
  points.forEach((p, i) => p.forEach((v, a) => body.writeFloatLE(v, i * 12 + a * 4)));
  return Buffer.concat([header, body]);
}
async function bytes(d: Download) { const chunks: Buffer[] = []; for await (const c of await d.createReadStream()) chunks.push(c); return Buffer.concat(chunks); }
async function save(page: Page, id: string) { const event = page.waitForEvent('download'); await page.locator(id).click(); return bytes(await event); }
async function json(page: Page, id = '#mesh-json') { return JSON.parse((await save(page, id)).toString()); }
const patches = () => [0, 10].flatMap(offset => Array.from({ length: 9 }, (_, i) => {
  const x = offset + i % 3, y = Math.floor(i / 3); return [1000 + x, 2000 + y, 10 + x * 0.2 + y * 0.1];
}));
async function load(page: Page, points = patches(), name = 'terrain.ply') {
  await page.locator('#file-input').setInputFiles({ name, mimeType: 'application/octet-stream', buffer: ply(points) });
  await expect(page.locator('#status')).toContainText(`Loaded ${name}`); await expect(page.locator('#task')).toBeHidden();
}
async function run(page: Page) {
  await page.locator('#mesh-run').click(); await expect(page.locator('#mesh-result')).toBeVisible(); await expect(page.locator('#task')).toBeHidden();
}
function obj(text: string) {
  const lines = text.split('\n');
  return { vertices: lines.filter(l => l.startsWith('v ')).map(l => l.trim().split(/\s+/).slice(1).map(Number)),
    faces: lines.filter(l => l.startsWith('f ')).map(l => l.trim().split(/\s+/).slice(1).map(v => Number(v.split('/')[0]) - 1)) };
}

test('terrain patches retain heights and separated faces; wireframe still exports faces, QA and one Undo', async ({ page }, info) => {
  const errors: string[] = []; page.on('pageerror', e => errors.push(e.message));
  await page.addInitScript(() => {
    (window as any).meshDrawModes = [];
    const original = WebGL2RenderingContext.prototype.drawElements;
    WebGL2RenderingContext.prototype.drawElements = function (mode, count, type, offset) {
      (window as any).meshDrawModes.push(mode); return original.call(this, mode, count, type, offset);
    };
  });
  await page.goto('/'); await load(page);
  await page.locator('#coverage-source').selectOption({ label: 'terrain.ply' });
  await page.locator('#coverage-mesh').click(); await expect(page.locator('#mesh-result')).toBeHidden();
  await expect(page.locator('#mesh-cloud')).toHaveValue(await page.locator('#coverage-source').inputValue());
  await page.locator('#mesh-max-edge').fill('1.5'); await run(page);
  const summary = await json(page);
  expect(summary.schema).toBe('cloudanalyzer.terrain_mesh.v1'); expect(summary.source.loadedPoints).toBe(18);
  expect(summary.output.vertices).toBe(18); expect(summary.output.triangles).toBe(16);
  expect(summary.settings).toEqual({ maxEdgeMode: 'length', requestedMaxEdge: 1.5, effectiveMaxEdge: 1.5 });
  expect(summary.processing.voxel).toBe(0); expect(summary.processing.removedTriangles).toBeGreaterThan(0);
  await page.evaluate(() => (window as any).meshDrawModes = []); await page.locator('#mesh-wireframe').check();
  await expect.poll(() => page.evaluate(() => (window as any).meshDrawModes.includes(1))).toBe(true);
  const exported = obj((await save(page, '#mesh-obj')).toString());
  expect(exported.vertices).toHaveLength(18); expect(exported.faces).toHaveLength(16);
  for (const vertex of exported.vertices) {
    const original = patches().find(p => p[0] === vertex[0] && p[1] === vertex[1]);
    expect(original).toBeDefined(); expect(vertex[2]).toBeCloseTo(original![2], 5);
  }
  const seen = [0, 0];
  for (const face of exported.faces) {
    const vertices = face.map(i => exported.vertices[i]); const side = vertices[0][0] < 1005 ? 0 : 1; seen[side]++;
    expect(vertices.every(v => (v[0] < 1005 ? 0 : 1) === side)).toBe(true);
    for (let i = 0; i < 3; i++) expect(Math.hypot(vertices[i][0] - vertices[(i + 1) % 3][0], vertices[i][1] - vertices[(i + 1) % 3][1])).toBeLessThanOrEqual(1.5);
  }
  expect(seen).toEqual([8, 8]);
  const plyBytes = await save(page, '#mesh-ply'), header = plyBytes.subarray(0, plyBytes.indexOf('end_header')).toString();
  expect(header).toContain('element vertex 18'); expect(header).toContain('element face 16');
  await page.evaluate(() => (window as any).meshDrawModes = []); await page.locator('#mesh-wireframe').uncheck();
  await expect.poll(() => page.evaluate(() => (window as any).meshDrawModes.includes(4))).toBe(true);
  const qa = await json(page, '#report-json'); expect(qa.sections['terrain-mesh'].metrics.triangles.value).toBe(16);
  await writeFile(info.outputPath('terrain-mesh-receipt.json'), JSON.stringify({ fixture: 'Synthetic separated sloped ground patches; workflow and geometry assertions, not survey accuracy', summary, facesPerPatch: seen, obj: exported, errors }, null, 2));
  await page.locator('#undo').click(); await expect(page.locator('#cloud-list > li')).toHaveCount(1);
  await expect(page.locator('#cloud-list > li input[title="Show / hide"]')).toBeChecked();
  await expect(page.locator('#mesh-result')).toBeHidden(); expect((await json(page, '#report-json')).sections['terrain-mesh']).toBeUndefined();
  await page.locator('#redo').click(); await expect(page.locator('#cloud-list > li')).toHaveCount(2); expect(errors).toEqual([]);
});

test('unlimited edges are recorded explicitly; invalid settings never publish geometry', async ({ page }) => {
  await page.goto('/'); await load(page);
  await page.locator('#mesh-max-edge').fill('-1'); await page.locator('#mesh-run').click();
  await expect(page.locator('#status')).toContainText('finite max edge'); await expect(page.locator('#cloud-list > li')).toHaveCount(1);
  await page.locator('#mesh-max-edge').fill('0'); await run(page);
  const result = await json(page); expect(result.settings).toEqual({ maxEdgeMode: 'unlimited', requestedMaxEdge: 0, effectiveMaxEdge: null });
  expect(result.output.triangles).toBeGreaterThan(16); expect(result.processing.removedTriangles).toBe(0);
  const exported = obj((await save(page, '#mesh-obj')).toString());
  expect(exported.faces.some(face => { const x = face.map(i => exported.vertices[i][0]); return Math.max(...x) - Math.min(...x) >= 8; })).toBe(true);
});

test('cancelled and changed in-flight triangulations release unpublished meshes', async ({ page }) => {
  await page.goto('/'); await load(page);
  await page.evaluate(() => {
    const original = Worker.prototype.postMessage; (window as any).meshAction = 'cancel'; (window as any).meshRemovals = 0;
    Worker.prototype.postMessage = function (message: any, ...rest: any[]) {
      if (message?.req?.kind === 'mesh') {
        if ((window as any).meshAction === 'cancel') document.getElementById('task-cancel')!.click();
        else if ((window as any).meshAction === 'change') { const el = document.getElementById('mesh-max-edge') as HTMLInputElement; el.value = '2'; el.dispatchEvent(new Event('input')); }
      }
      if (message?.req?.kind === 'remove') (window as any).meshRemovals++;
      return (original as any).call(this, message, ...rest);
    };
  });
  for (const [action, status] of [['cancel', 'Cancelled'], ['change', 'inputs/settings changed']]) {
    await page.evaluate(a => (window as any).meshAction = a, action); await page.locator('#mesh-run').click();
    await expect(page.locator('#status')).toContainText(status); await expect(page.locator('#task')).toBeHidden();
    await expect(page.locator('#cloud-list > li')).toHaveCount(1); await expect(page.locator('#mesh-result')).toBeHidden();
    await expect(page.locator('#cloud-list > li input[title="Show / hide"]')).toBeChecked();
  }
  expect(await page.evaluate(() => (window as any).meshRemovals)).toBe(2);
  expect((await json(page, '#report-json')).sections['terrain-mesh']).toBeUndefined();
  await page.evaluate(() => (window as any).meshAction = 'none'); await run(page);
});

test('removed source clears its receipt; a same-named reload has a fresh identity', async ({ page }) => {
  await page.goto('/'); await load(page); await run(page); const first = await json(page);
  await page.locator('#cloud-list > li').first().locator('button[title="Remove"]').click();
  await expect(page.locator('#mesh-result')).toBeHidden(); expect((await json(page, '#report-json')).sections['terrain-mesh']).toBeUndefined();
  await load(page, patches().map(p => [p[0], p[1], p[2] + 5]));
  await page.locator('#mesh-cloud').selectOption({ label: 'terrain.ply' }); await run(page); const second = await json(page);
  expect(second.source.name).toBe(first.source.name); expect(second.source.id).not.toBe(first.source.id);
  expect(Math.min(...obj((await save(page, '#mesh-obj')).toString()).vertices.map(v => v[2]))).toBeCloseTo(15, 5);
});

test('zero Undo capacity rejects publication without hiding or retaining the output', async ({ page }) => {
  await page.goto('/'); await load(page);
  await page.locator('#memory-panel summary').click(); await page.locator('#memory-steps').fill('0'); await page.locator('#memory-apply').click();
  await page.locator('#mesh-run').click(); await expect(page.locator('#status')).toContainText('do not fit one Undo step'); await expect(page.locator('#task')).toBeHidden();
  await expect(page.locator('#cloud-list > li')).toHaveCount(1); await expect(page.locator('#cloud-list > li input[title="Show / hide"]')).toBeChecked();
  await expect(page.locator('#mesh-result')).toBeHidden();
  await page.locator('#memory-steps').fill('20'); await page.locator('#memory-apply').click(); await run(page);
});

test('compared After cells become a mesh and feed the existing cut/fill calculation with provenance', async ({ page }) => {
  await page.goto('/');
  const before = Array.from({ length: 6 }, (_, i) => [1000 + i % 3, 2000 + Math.floor(i / 3), 10]);
  await load(page, before, 'before.ply'); await load(page, before.slice(0, 5).map((p, i) => [p[0], p[1], p[2] + [2, -1, 0, 0, 0][i]]), 'after.ply');
  await page.locator('#coverage-source').selectOption({ label: 'after.ply' }); await page.locator('#coverage-reference').selectOption({ label: 'before.ply' });
  await page.locator('#coverage-volume').click(); await page.locator('#volume-panel summary').click(); await page.locator('#volume-cell').fill('1');
  await page.locator('#volume-run').click(); await expect(page.locator('#status')).toContainText('fill 2 m³, cut 1 m³, net 1 m³');
  const volume = await json(page, '#volume-json'); await page.locator('#volume-mesh').click();
  await expect(page.locator('#mesh-cloud')).toHaveValue(String(volume.comparedCellCloud.id));
  await page.locator('#mesh-max-edge').fill('1.5'); await run(page); const mesh = await json(page);
  expect(mesh.source.id).toBe(volume.comparedCellCloud.id); expect(mesh.output.vertices).toBe(5); expect(mesh.output.triangles).toBe(3);
  const heights = obj((await save(page, '#mesh-obj')).toString()).vertices.map(v => v[2]).sort((a, b) => a - b); expect(heights).toEqual([9, 10, 10, 10, 12]);
  await page.locator('#mesh-volume-after').click(); await expect(page.locator('#volume-after')).toHaveValue(String(mesh.output.id));
  await expect(page.locator('#volume-before')).toHaveValue(String(volume.before.id)); await expect(page.locator('#volume-fill')).not.toBeChecked();
  // Selecting a different surface proposes a new grid size; explicitly review/reset it.
  await page.locator('#volume-cell').fill('1');
  await page.locator('#volume-run').click(); await expect(page.locator('#status')).toContainText('Volume: fill');
  const repeated = await json(page, '#volume-json'); expect(repeated.after.kind).toBe('mesh'); expect(repeated.result.net).toBeCloseTo(1, 6); expect(repeated.result.matchedCells).toBe(5);
});
