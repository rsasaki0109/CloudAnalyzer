import { expect, test, type Download, type Page } from '@playwright/test';

async function bytes(download: Download) {
  const chunks: Buffer[] = [];
  for await (const chunk of await download.createReadStream()) chunks.push(chunk);
  return Buffer.concat(chunks);
}
async function snapshot(page: Page) {
  const ready = page.waitForEvent('download', d => d.suggestedFilename() === 'project.cloudanalyzer.json');
  await page.locator('#project-save').click();
  return JSON.parse(JSON.parse((await bytes(await ready)).toString()).vectorMap);
}
function sharedMap() {
  return {format: 'vectormap-ir', version: 1,
    lanes: [{id: 4, kind: 'driving', left: 1, right: 2, speed_limit: {kmh: 40}},
      {id: 5, kind: 'driving', left: {boundary: 3, reversed: true}, right: {boundary: 2, reversed: true}, speed_limit: {kmh: 25}}],
    boundaries: [4, 0, -4].map((y, i) => ({id: i + 1, kind: {type: 'lane_marking', pattern: 'solid'},
      geometry: [0, 20, 40].map(x => [500000 + x, 4000000 + y, i === 1 && x === 20 ? 2.8 : 2 + x / 100])}))};
}

test('height repair retains XY and reversed shared lanes, improves source coverage, exports and undoes', async ({page}, info) => {
  const errors: string[] = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.goto('/');
  const points: number[][] = [];
  for (let x = -1; x <= 41; x += .25) for (let y = -5; y <= 5; y += .25) points.push([500000 + x, 4000000 + y, 2 + x / 100]);
  const header = `ply\nformat binary_little_endian 1.0\nelement vertex ${points.length}\nproperty double x\nproperty double y\nproperty double z\nend_header\n`;
  const data = Buffer.alloc(points.length * 24);
  points.forEach((p, i) => p.forEach((v, axis) => data.writeDoubleLE(v, i * 24 + axis * 8)));
  await page.locator('#file-input').setInputFiles({name: 'sloping-ground.ply', mimeType: 'application/octet-stream', buffer: Buffer.concat([Buffer.from(header), data])});
  await expect(page.locator('#status')).toContainText('Loaded sloping-ground.ply');
  await page.locator('#vm-file').setInputFiles({name: 'shared.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(sharedMap()))});
  await expect(page.locator('#status')).toContainText('Opened shared.json: 2 lanes');
  const before = await snapshot(page);
  await page.locator('#vm-quality summary').click();
  await page.locator('#vm-quality-check').click();
  await expect(page.locator('#vm-quality-report')).toContainText('2 need source review');
  const options = page.locator('#vm-quality-problem option');
  const problem = await options.filter({hasText: /Lane 5, right boundary:.*height disagreement/}).first().getAttribute('value');
  await page.locator('#vm-quality-problem').selectOption(problem!);
  await page.locator('#vm-quality-edit').click();
  await expect(page.locator('#vm-boundary-editor')).toHaveAttribute('open', '');
  await expect(page.locator('#vm-boundary')).toHaveValue('2');
  await expect(page.locator('#vm-boundary-vertex')).toHaveValue('1');
  await expect(page.locator('#vm-boundary-z')).toHaveValue('2.8');
  await expect(page.locator('#vm-boundary-position')).toContainText('X 500020 m, Y 4000000 m');
  await expect(page.locator('#vm-boundary-context')).toContainText('Used by lanes 4, 5');
  await expect(page.locator('#vm-boundary-apply')).toBeDisabled();
  await expect(page.locator('#vm-undo')).toBeDisabled();
  // Empty / overflowed input cannot become a silent zero or non-finite edit.
  for (const value of ['', '1e309']) {
    await page.locator('#vm-boundary-z').fill(value);
    await expect(page.locator('#vm-boundary-apply')).toBeDisabled();
    await expect(page.locator('#vm-boundary-edit-status')).toHaveText('Enter a finite height in metres.');
  }
  await page.locator('#vm-boundary-z').fill('2.8');
  await expect(page.locator('#vm-boundary-apply')).toBeDisabled();
  expect(await snapshot(page)).toEqual(before);
  await page.locator('#vm-boundary-z').fill('2.2');
  await expect(page.locator('#vm-boundary-edit-status')).toContainText('X and Y will stay fixed');
  await page.screenshot({path: info.outputPath('height-before.png')});
  await page.locator('#vm-boundary-apply').click();
  await expect(page.locator('#status')).toContainText('Boundary 2 vertex 2 height updated');
  await expect(page.locator('#vm-boundary-position')).toContainText('Current height Z 2.2 m');
  await expect(page.locator('#vm-boundary-apply')).toBeDisabled();
  await expect(page.locator('#vm-quality-locations')).toBeHidden();
  await expect(page.locator('#vm-quality-report')).toContainText('has not been checked');
  const edited = await snapshot(page);
  const expected = structuredClone(before);
  expected.boundaries.find(b => b.id === 2).geometry[1][2] = 2.2;
  expect(edited).toEqual(expected);
  await page.locator('#vm-quality-check').click();
  await expect(page.locator('#vm-quality-report')).toContainText('2 lanes checked; 0 need source review; 0 omitted; 0 malformed');
  await expect(options).toHaveCount(1);
  await page.screenshot({path: info.outputPath('height-rechecked.png')});
  const osmReady = page.waitForEvent('download', d => d.suggestedFilename() === 'lanelet2_map.osm');
  const yamlReady = page.waitForEvent('download', d => d.suggestedFilename() === 'map_projector_info.yaml');
  await page.locator('#vm-export').click();
  const osm = await bytes(await osmReady);
  expect((await bytes(await yamlReady)).toString()).toBe('projector_type: Local\n');
  // Verify the edited node is referenced by one boundary shared by two lanes.
  expect(await page.evaluate(xml => {
    const doc = new DOMParser().parseFromString(xml, 'application/xml');
    const value = (el: Element, key: string) => el.querySelector(`tag[k="${key}"]`)?.getAttribute('v');
    const node = [...doc.querySelectorAll('node')].find(n => value(n, 'local_x') === '500020' && value(n, 'local_y') === '4000000')!;
    const ways = [...doc.querySelectorAll('way')].filter(w => w.querySelector(`nd[ref="${node.id}"]`)).map(w => w.id);
    const lanes = [...doc.querySelectorAll('relation')].filter(r => [...r.querySelectorAll('member[type="way"]')].some(m => ways.includes(m.getAttribute('ref')!)));
    return {z: value(node, 'ele'), lanes: lanes.length};
  }, osm.toString())).toEqual({z: '2.2', lanes: 2});
  await page.locator('#vm-undo').click();
  await expect(page.locator('#status')).toContainText('Undone');
  await expect(page.locator('#vm-boundary-z')).toHaveValue('2.8');
  await page.waitForTimeout(1100);
  expect(await snapshot(page)).toEqual(before);
  await expect(page.locator('#vm-undo')).toBeDisabled();
  await page.locator('#vm-quality-check').click();
  await expect(page.locator('#vm-quality-report')).toContainText('2 need source review');
  await page.locator('#vm-file').setInputFiles({name: 'edited.osm', mimeType: 'application/xml', buffer: osm});
  await expect(page.locator('#status')).toContainText('Opened edited.osm: 2 lanes');
  await expect(page.locator('#vm-boundary')).toHaveValue('');
  await expect(page.locator('#vm-boundary-z')).toHaveValue('');
  const reopened = await snapshot(page);
  expect(reopened.boundaries.map(b => b.geometry)).toEqual(edited.boundaries.map(b => b.geometry));
  // The OSM writer supplies these standard tags; reopening retains them.
  expect(reopened.lanes).toEqual(edited.lanes.map(l => ({...l, attributes: {
    ...l.attributes, 'lanelet2:location': 'urban', 'lanelet2:participant:vehicle': 'yes',
  }})));
  await page.locator('#vm-quality-check').click();
  await expect(page.locator('#vm-quality-report')).toContainText('0 need source review');
  expect(errors).toEqual([]);
});

test('height selection works without a cloud and clears on replacing or clearing the map', async ({page}) => {
  await page.goto('/');
  await page.locator('#vm-file').setInputFiles({name: 'shared.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(sharedMap()))});
  await expect(page.locator('#status')).toContainText('Opened shared.json');
  await page.locator('#vm-boundary-editor summary').click();
  await page.locator('#vm-boundary').selectOption('2');
  await page.locator('#vm-boundary-vertex').selectOption('1');
  await page.locator('#vm-boundary-focus').click();
  // Picking a handle for a numeric height edit also works from a front view.
  await page.locator('#vm-boundary').selectOption('');
  await page.locator('[data-view="front"]').click();
  await page.locator('#vm-vertices').click();
  const canvas = (await page.locator('#viewport > canvas').boundingBox())!;
  await page.mouse.click(canvas.x + canvas.width / 2, canvas.y + canvas.height / 2);
  await expect(page.locator('#vm-boundary')).toHaveValue('2');
  await expect(page.locator('#vm-boundary-vertex')).toHaveValue('1');
  await expect(page.locator('#vm-boundary-z')).toHaveValue('2.8');
  await expect(page.locator('#vm-undo')).toBeDisabled();
  await page.locator('#vm-boundary-z').fill('-4.5');
  await page.locator('#vm-boundary-apply').click();
  await expect(page.locator('#status')).toContainText('Boundary 2 vertex 2 height updated');
  await expect(page.locator('#vm-boundary-z')).toHaveValue('-4.5');
  const negative = await snapshot(page);
  expect(negative.boundaries.find(b => b.id === 2).geometry[1]).toEqual([500020, 4000000, -4.5]);
  await page.locator('#vm-clear').click();
  await expect(page.locator('#status')).toContainText('Map cleared');
  await expect(page.locator('#vm-boundary')).toHaveValue('');
  await expect(page.locator('#vm-boundary-z')).toHaveValue('');
  await expect(page.locator('#vm-boundary-apply')).toBeDisabled();
  await page.locator('#vm-undo').click();
  await expect(page.locator('#status')).toContainText('Undone');
  await expect(page.locator('#vm-boundary')).toBeEnabled();
  await expect(page.locator('#vm-boundary')).toHaveValue('');
  await page.locator('#vm-boundary').selectOption('2');
  await page.locator('#vm-boundary-vertex').selectOption('1');
  await expect(page.locator('#vm-boundary-z')).toHaveValue('-4.5');
  await page.locator('#vm-file').setInputFiles({name: 'replacement.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(sharedMap()))});
  await expect(page.locator('#status')).toContainText('Opened replacement.json');
  await expect(page.locator('#vm-boundary')).toHaveValue('');
  await expect(page.locator('#vm-boundary-z')).toBeDisabled();
  await expect(page.locator('#vm-boundary-apply')).toBeDisabled();
});
