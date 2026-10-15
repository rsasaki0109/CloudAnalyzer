import { expect, test, type Download } from '@playwright/test';

async function bytes(download: Download) {
  const chunks: Buffer[] = [];
  for await (const chunk of await download.createReadStream()) chunks.push(chunk);
  return Buffer.concat(chunks).toString();
}

test('Lanelet2 export shows native writer warnings and clears stale results after map changes', async ({page}) => {
  await page.goto('/');
  const map = {format: 'vectormap-ir', version: 1, metadata: {},
    lanes: [{id: 3, kind: 'driving', left: 1, right: 2, speed_limit: {kmh: 40}}],
    boundaries: [
      {id: 1, kind: {type: 'virtual'}, geometry: [[0, 1.75, 2], [10, 1.75, 2]]},
      {id: 2, kind: {type: 'virtual'}, geometry: [[0, -1.75, 2], [10, -1.75, 2]]},
    ]};
  await page.locator('#vm-file').setInputFiles({name: 'local.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(map))});
  await expect(page.locator('#status')).toContainText('Opened local.json');
  const xmlReady = page.waitForEvent('download', d => d.suggestedFilename() === 'lanelet2_map.osm');
  const yamlReady = page.waitForEvent('download', d => d.suggestedFilename() === 'map_projector_info.yaml');
  await page.getByRole('button', {name: 'Save Lanelet2', exact: true}).click();
  const xml = await bytes(await xmlReady);
  expect(await bytes(await yamlReady)).toBe('projector_type: Local\n');
  expect(xml).toContain('<tag k="local_y" v="1.75"/>');
  await expect(page.locator('#vm-export-summary')).toHaveText('Last saved Lanelet2: 0 errors, 1 warning.');
  await expect(page.locator('#vm-export-issues [title="lanelet2.no_georeference"]')).toContainText('lat/lon were written as 0');
  await expect(page.locator('#status')).toContainText('Export: 0 errors, 1 warning');
  // The native writer can save a Local draft; its warning remains visible.
  await expect(page.locator('#vm-export')).toBeEnabled();
  await page.locator('#vm-review-next').click();
  await page.locator('#vm-lane-speed').fill('25');
  await page.locator('#vm-lane-apply').click();
  await expect(page.locator('#status')).toContainText('Speed limit of lane');
  await expect(page.locator('#vm-export-report')).toBeHidden();
  await expect(page.locator('#vm-export-issues')).toBeEmpty();
  // MGRS coordinates are inside the origin's 100 km square, not centred at 0.
  const georeferenced = {...map,
    boundaries: map.boundaries.map(b => ({...b, geometry: b.geometry.map(([x, y, z]) => [x + 50000, y + 70000, z])})),
    metadata: {georeference: {projection: 'mgrs', origin: {lat: 35.681236, lon: 139.767125}}}};
  await page.locator('#vm-file').setInputFiles({name: 'survey.json', mimeType: 'application/json', buffer: Buffer.from(JSON.stringify(georeferenced))});
  await expect(page.locator('#status')).toContainText('Opened survey.json');
  const mgrsReady = page.waitForEvent('download', d => d.suggestedFilename() === 'map_projector_info.yaml');
  await page.locator('#vm-export').click();
  expect(await bytes(await mgrsReady)).toContain('projector_type: MGRS');
  await expect(page.locator('#status')).toContainText('Export: 0 errors, 0 warnings');
  await expect(page.locator('#vm-export-report')).toBeHidden();
  await expect(page.locator('#vm-export-issues')).toBeEmpty();
});
