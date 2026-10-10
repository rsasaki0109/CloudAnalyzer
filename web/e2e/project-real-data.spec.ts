import { expect, test, type Download, type Page } from '@playwright/test';
import { readFile, writeFile } from 'node:fs/promises';
import { fileURLToPath } from 'node:url';

const sample = (name: string) => fileURLToPath(new URL(`../public/samples/${name}`, import.meta.url));
const rellis = fileURLToPath(new URL('../../demo_assets/public/rellis3d-frame-000001/os1_cloud_node_kitti_bin/000001.bin', import.meta.url));
const draft = {format:'vectormap-ir',version:1,lanes:[{id:7,kind:'driving',left:1,right:2}],boundaries:[
  {id:1,kind:{type:'virtual'},geometry:[[0,1.75,2],[10,1.75,2]]},
  {id:2,kind:{type:'virtual'},geometry:[[0,-1.75,2],[10,-1.75,2]]},
]};
async function bytes(download: Download): Promise<Buffer> {
  const pieces: Buffer[] = [];
  for await (const piece of await download.createReadStream()) pieces.push(piece);
  return Buffer.concat(pieces);
}
async function save(page: Page): Promise<Buffer> {
  const saved = page.waitForEvent('download', d => d.suggestedFilename() === 'project.cloudanalyzer.json');
  await page.locator('#project-save').click();
  return bytes(await saved);
}
async function memory(page: Page) {
  const panel = page.locator('#memory-panel');
  if (await panel.getAttribute('open') !== null) await panel.locator('summary').click();
  // Wait for a fresh stats reply by emptying only the rendered report.
  await page.locator('#memory-report').evaluate(el => { el.textContent = ''; });
  await panel.locator('summary').click();
  await expect(page.locator('#memory-report')).toContainText('Main WASM');
  const report = (await page.locator('#memory-report').textContent())!;
  const value = (pattern: RegExp) => Number(pattern.exec(report)![1]);
  return {main:value(/Main WASM ([\d.]+)/),pool:value(/parallel WASM ([\d.]+)/),arrays:value(/retained arrays ([\d.]+)/),cloudSteps:value(/clouds (\d+)/),poseSteps:value(/poses (\d+)/),mapSteps:value(/map (\d+)/),report};
}
test.describe('real-data projects', () => {
  test.skip(process.env.CLOUDANALYZER_REAL_DATA !== '1', 'Opt in with CLOUDANALYZER_REAL_DATA=1; NCLT and RELLIS-3D attribution applies.');
  test('NCLT constraints, scans, map and review survive a browser restart', async ({page}, info) => {
    test.setTimeout(900_000);
    const started = Date.now(), errors: string[] = [];
    page.on('pageerror', e => errors.push(e.message));
    await page.goto('/?demo=nclt');
    await expect(page.locator('#status')).toContainText(/nclt-2012-04-29_map is colored by how far each point moved/,{timeout:540_000});
    await expect(page.locator('#pg-stats')).toContainText('Gravity');
    await expect(page.locator('#pg-loop-list li').first()).toBeVisible();
    const originalStats = (await page.locator('#pg-stats').textContent())!;
    await page.locator('#file-input').setInputFiles(sample('lidar_reference.pcd'));
    await expect(page.locator('#status')).toContainText('Loaded lidar_reference.pcd');
    await page.locator('#vm-file').setInputFiles({name:'draft.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(draft))});
    await expect(page.locator('#vm-status')).toContainText('1 lane');
    await page.locator('#vm-review-next').click();
    await page.locator('#vm-review-state').selectOption('deferred');
    await page.locator('#vm-review-notes').fill('Check the original survey before approving this draft.');
    await page.locator('#vm-review-save').click();
    const project = await save(page), before = JSON.parse(project.toString()), snapshot = JSON.parse(before.poseGraph.snapshot);
    expect(snapshot.graph.nodes.length).toBeGreaterThan(100);
    expect(snapshot.graph.gravity_edges.length).toBeGreaterThan(100);
    expect(snapshot.graph.edges.some(e => e.kind === 'Loop')).toBe(true);
    const initialMemory = await memory(page), openedMs = Date.now() - started;
    await page.goto('/');
    await page.locator('#file-input').setInputFiles({name:'project.cloudanalyzer.json',mimeType:'application/json',buffer:project});
    await expect(page.locator('#status')).toContainText('open matching source files');
    const restoring = Date.now();
    await page.locator('#file-input').setInputFiles([sample('lidar_reference.pcd'),sample('nclt-2012-04-29.mcap')]);
    await expect(page.locator('#status')).toContainText('Project restored',{timeout:540_000});
    await expect(page.locator('#pg-stats')).toHaveText(originalStats);
    await expect(page.locator('#cloud-list li')).toHaveCount(1); // Derived maps are external exports.
    await page.locator('#vm-review-filter').selectOption('all');
    await page.locator('#vm-review-next').click();
    await expect(page.locator('#vm-review-state')).toHaveValue('deferred');
    await expect(page.locator('#vm-review-notes')).toHaveValue('Check the original survey before approving this draft.');
    const after = JSON.parse((await save(page)).toString());
    expect(JSON.parse(after.poseGraph.snapshot)).toEqual(snapshot);
    expect(JSON.parse(after.vectorMap)).toEqual(JSON.parse(before.vectorMap));
    expect(after.reviews).toEqual(before.reviews);
    const restoredMemory = await memory(page);
    expect(restoredMemory.cloudSteps + restoredMemory.poseSteps + restoredMemory.mapSteps).toBe(0);
    expect(errors).toEqual([]);
    const metricsPath=info.outputPath('nclt-project-metrics.json');
    await writeFile(metricsPath,JSON.stringify({nodes:snapshot.graph.nodes.length,gravity:snapshot.graph.gravity_edges.length,loops:snapshot.graph.edges.filter(e => e.kind==='Loop').length,metadataBytes:project.length,openedMs,restoredMs:Date.now()-restoring,initialMemory,restoredMemory},null,2));
    await info.attach('nclt-project-metrics.json',{path:metricsPath,contentType:'application/json'});
    await page.screenshot({path:info.outputPath('restored-project.png')});
  });
  test('RELLIS stress data stays bounded across repeated edits and releases idle workers', async ({page}, info) => {
    test.setTimeout(600_000);
    const original = await readFile(rellis), count = original.length / 16, tiles = 16;
    // Translated copies provide a load case; they are not additional surveyed frames.
    const header = Buffer.from(`ply\nformat binary_little_endian 1.0\nelement vertex ${count*tiles}\nproperty float x\nproperty float y\nproperty float z\nproperty float intensity\nend_header\n`);
    const points = Buffer.alloc(original.length * tiles);
    for (let tile=0;tile<tiles;tile++) {
      original.copy(points,tile*original.length);
      for (let point=0;point<count;point++) {
        const at=tile*original.length + point*16;
        points.writeFloatLE(original.readFloatLE(point*16)+(tile%4)*200,at);
        points.writeFloatLE(original.readFloatLE(point*16+4)+Math.floor(tile/4)*200,at+4);
      }
    }
    const source = {name:'rellis-stress.ply',mimeType:'application/octet-stream',buffer:Buffer.concat([header,points])};
    const reference = {name:'lidar_reference.pcd',mimeType:'application/octet-stream',buffer:await readFile(sample('lidar_reference.pcd'))};
    const errors: string[] = []; page.on('pageerror', e => errors.push(e.message));
    await page.goto('/');
    await page.locator('#file-input').setInputFiles([reference,source]);
    await expect(page.locator('#status')).toContainText(`Loaded rellis-stress.ply: ${(count*tiles).toLocaleString('en-US')} points`,{timeout:180_000});
    await page.locator('#memory-panel summary').click();
    await page.locator('#memory-steps').fill('3');
    await page.locator('#memory-undo-budget').fill('2048');
    await page.locator('#memory-warning').fill('64');
    await page.locator('#memory-apply').click();
    await page.locator('#memory-panel summary').click();
    await page.locator('#align-panel summary').click();
    const measurements: unknown[] = [];
    let settled: Awaited<ReturnType<typeof memory>> | undefined;
    for (let edit=0;edit<24;edit++) {
      const shift = edit%2 ? -0.1 : 0.1;
      await page.locator('#align-matrix').fill(`1 0 0 ${shift}\n0 1 0 0\n0 0 1 0\n0 0 0 1`);
      await page.locator('#status').evaluate(el=>{el.textContent='';});
      await page.locator('#align-apply-matrix').click();
      await expect(page.locator('#status')).toContainText('Applied the matrix to rellis-stress.ply');
      if ([5,11,23].includes(edit)) {
        const current=await memory(page); measurements.push({edit:edit+1,...current});
        expect(current.cloudSteps).toBe(3);
        await expect(page.locator('#memory-warning-report')).toContainText('exceeds');
        if (!settled) settled=current;
        else {
          expect(current.main).toBeLessThanOrEqual(settled.main+64);
          expect(current.pool).toBeLessThanOrEqual(settled.pool+64);
          expect(current.arrays).toBeLessThanOrEqual(settled.arrays+16);
        }
        await page.locator('#memory-panel summary').click();
      }
    }
    const project=await save(page), before=JSON.parse(project.toString());
    expect(before.session.clouds.find(c=>c.name==='rellis-stress.ply').transforms).toHaveLength(24);
    const beforeRelease=await memory(page);
    await page.locator('#memory-release').click();
    await expect(page.locator('#status')).toContainText('Current clouds, map and poses retained');
    const released=await memory(page);
    expect(released.pool).toBe(0); expect(released.cloudSteps).toBe(0);
    await expect(page.locator('#cloud-list li')).toHaveCount(2);
    await expect(page.locator('#undo')).toBeDisabled();
    await page.reload();
    await page.locator('#file-input').setInputFiles([{name:'project.cloudanalyzer.json',mimeType:'application/json',buffer:project},reference,source]);
    await expect(page.locator('#status')).toContainText('Project restored',{timeout:180_000});
    const after=JSON.parse((await save(page)).toString());
    expect(after.session.clouds.map(c=>c.transforms)).toEqual(before.session.clouds.map(c=>c.transforms));
    expect(errors).toEqual([]);
    const metricsPath=info.outputPath('rellis-memory-metrics.json');
    await writeFile(metricsPath,JSON.stringify({originalFramePoints:count,tiles,stressPoints:count*tiles,sourceBytes:source.buffer.length,metadataBytes:project.length,edits:24,measurements,beforeRelease,released},null,2));
    await info.attach('rellis-memory-metrics.json',{path:metricsPath,contentType:'application/json'});
  });
});
