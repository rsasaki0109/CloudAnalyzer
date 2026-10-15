import { expect, test, type Page } from '@playwright/test';

async function snapshot(page: Page) {
  const ready = page.waitForEvent('download', d => d.suggestedFilename() === 'project.cloudanalyzer.json');
  await page.locator('#project-save').click();
  const chunks: Buffer[] = [];
  for await (const chunk of await (await ready).createReadStream()) chunks.push(chunk);
  return JSON.parse(JSON.parse(Buffer.concat(chunks).toString()).vectorMap);
}

test('source problems locate missing returns and wrong height, preserve shared editing and recheck', async ({page}, info) => {
  const errors: string[] = [];
  page.on('pageerror', e => errors.push(e.message));
  await page.goto('/');
  const points: number[][] = [];
  for (let x = -2; x <= 42; x += .25) for (let y = -6; y <= 6; y += .25) {
    if (x >= 30 && x <= 34) continue;
    points.push([500000+x, 4000000+y, x >= 14 && x <= 26 && Math.abs(y) <= 1 ? 4 : 2]);
  }
  const header = `ply\nformat binary_little_endian 1.0\nelement vertex ${points.length}\nproperty double x\nproperty double y\nproperty double z\nend_header\n`;
  const data = Buffer.alloc(points.length * 24);
  points.forEach((p,i) => p.forEach((v,a) => data.writeDoubleLE(v, i*24+a*8)));
  await page.locator('#file-input').setInputFiles({name:'source.ply',mimeType:'application/octet-stream',buffer:Buffer.concat([Buffer.from(header),data])});
  await expect(page.locator('#status')).toContainText('Loaded source.ply');
  const map = {format:'vectormap-ir',version:1,
    lanes:[{id:4,kind:'driving',left:1,right:2,speed_limit:{kmh:40}},
      {id:5,kind:'driving',left:{boundary:3,reversed:true},right:{boundary:2,reversed:true},speed_limit:{kmh:40}}],
    boundaries:[4,0,-4].map((y,i) => ({id:i+1,kind:{type:'lane_marking',pattern:'solid'},geometry:[0,20,40].map(x => [500000+x,4000000+y,2])}))};
  await page.locator('#vm-file').setInputFiles({name:'draft.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(map))});
  await expect(page.locator('#status')).toContainText('Opened draft.json');
  const before = await snapshot(page);
  await page.locator('#vm-quality summary').click();
  await page.locator('#vm-quality-check').click();
  await expect(page.locator('#status')).toContainText('Source coverage checked');
  await expect(page.locator('#vm-quality-location-summary')).toContainText('problem intervals shown');
  const options = page.locator('#vm-quality-problem option');
  await expect(options.filter({hasText:'insufficient returns'})).not.toHaveCount(0);
  await expect(options.filter({hasText:'height disagreement'})).not.toHaveCount(0);
  expect(await snapshot(page)).toEqual(before);
  await expect(page.locator('#vm-undo')).toBeDisabled();
  const chosen = await options.filter({hasText:/Lane 4, right boundary:.*height disagreement/}).first().getAttribute('value');
  await page.locator('#vm-quality-problem').selectOption(chosen!);
  await expect(page.locator('#vm-lane-title')).toHaveText('Lane 4');
  await page.locator('[data-view="top"]').click();
  await page.locator('#vm-quality-show').uncheck();
  await page.locator('#vm-quality-focus').click();
  await expect(page.locator('#vm-quality-show')).toBeChecked();
  await page.screenshot({path:info.outputPath('problem-interval.png')});
  await page.locator('#vm-quality-edit').click();
  await expect(page.locator('#vm-vertices')).toHaveAttribute('aria-pressed','true');
  const canvas = (await page.locator('#viewport > canvas').boundingBox())!;
  const x = canvas.x+canvas.width/2, y = canvas.y+canvas.height/2;
  await page.mouse.move(x,y); await page.mouse.down();
  await expect(page.locator('#vm-hint')).toContainText('Boundary 2, vertex 2');
  await page.mouse.move(x,y-60,{steps:6}); await page.mouse.up();
  await expect(page.locator('#status')).toContainText('Boundary 2 vertex moved');
  await expect(page.locator('#vm-quality-locations')).toBeHidden();
  await expect(page.locator('#vm-quality-report')).toContainText('has not been checked');
  const edited = await snapshot(page);
  expect(edited.lanes).toEqual(before.lanes);
  expect(edited.boundaries.filter(b => b.id !== 2)).toEqual(before.boundaries.filter(b => b.id !== 2));
  const boundary = edited.boundaries.find(b => b.id === 2);
  expect(boundary.geometry.map(p => p[2])).toEqual([2,2,2]);
  expect(boundary.geometry[1][1]).toBeGreaterThan(4000001);
  expect(boundary.geometry[1][1]).toBeLessThan(4000004);
  await page.locator('#vm-quality-check').click();
  await expect(page.locator('#status')).toContainText('Source coverage checked');
  await expect(options.filter({hasText:'height disagreement'})).toHaveCount(0);
  await expect(options.filter({hasText:'insufficient returns'})).not.toHaveCount(0);
  await page.screenshot({path:info.outputPath('rechecked-edit.png')});
  await page.locator('#vm-undo').click();
  await expect(page.locator('#status')).toContainText('Undone');
  // Project downloads are paced once a full burst reaches Chromium's limit.
  await page.waitForTimeout(1100);
  expect(await snapshot(page)).toEqual(before);
  await page.locator('#vm-quality-check').click();
  await expect(page.locator('#status')).toContainText('Source coverage checked');
  await expect(options.filter({hasText:'height disagreement'})).not.toHaveCount(0);
  await page.locator('#vm-quality-next').click();
  await expect(page.locator('#vm-quality-focus')).toBeEnabled();
  await page.locator('#file-input').setInputFiles({name:'other.xyz',mimeType:'text/plain',buffer:Buffer.from('500000 4000000 2\n')});
  await expect(page.locator('#vm-quality-locations')).toBeHidden();
  await expect(page.locator('#vm-quality-problem option')).toHaveCount(1);
  expect(errors).toEqual([]);
});

test('a limited location preview is explicit while coverage still checks the complete lane', async ({page}) => {
  await page.goto('/');
  await page.locator('#file-input').setInputFiles({name:'sparse.xyz',mimeType:'text/plain',buffer:Buffer.from('0 0 2\n0.1 0 2\n0 0.1 2\n')});
  await expect(page.locator('#status')).toContainText('Loaded sparse.xyz');
  const map = {format:'vectormap-ir',version:1,lanes:[{id:3,kind:'driving',left:1,right:2,speed_limit:{kmh:40}}],
    boundaries:[1.75,-1.75].map((y,i) => ({id:i+1,kind:{type:'virtual'},geometry:[[0,y,2],[2500,y,2]]}))};
  await page.locator('#vm-file').setInputFiles({name:'long.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(map))});
  await expect(page.locator('#status')).toContainText('Opened long.json');
  await page.locator('#vm-quality summary').click();
  await page.locator('#vm-quality-check').click();
  await expect(page.locator('#vm-quality-report')).toContainText('1 lanes checked; 1 need source review; 0 omitted; 0 malformed');
  await expect(page.locator('#vm-quality-report')).not.toContainText('Coverage check limited');
  await expect(page.locator('#vm-quality-location-summary')).toContainText('Location preview limited; further failed samples are not displayed');
  await expect(page.locator('#vm-quality-location-summary')).toContainText('Coverage figures include all checked samples');
  await page.locator('#vm-quality-next').click();
  await expect(page.locator('#vm-quality-focus')).toBeEnabled();
});
