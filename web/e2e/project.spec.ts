import { expect, test, type Page, type Download } from '@playwright/test';

const cloud = (height = 2) => Buffer.from('ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\nproperty float z\nend_header\n' + `0 0 ${height}\n10 0 ${height}\n0 10 ${height}\n10 10 ${height}\n`);
const input = (name: string, buffer: Buffer) => ({ name, mimeType: 'application/octet-stream', buffer });
const map = { format: 'vectormap-ir', version: 1,
  lanes: [{id:7,kind:'driving',left:1,right:2}],
  boundaries: [{id:1,kind:{type:'virtual'},geometry:[[0,1.75,2],[10,1.75,2]]}, {id:2,kind:{type:'virtual'},geometry:[[0,-1.75,2],[10,-1.75,2]]}],
};
async function bytes(download: Download): Promise<Buffer> {
  const parts: Buffer[] = [];
  for await (const part of await download.createReadStream()) parts.push(part);
  return Buffer.concat(parts);
}
async function save(page: Page): Promise<Buffer> {
  const saved = page.waitForEvent('download', file => file.suggestedFilename() === 'project.cloudanalyzer.json');
  await page.locator('#project-save').click();
  return bytes(await saved);
}

test('project restores the edited map, graph constraints and loading settings from verified sources', async ({page}) => {
  await page.goto('/');
  const source = input('survey.ply', cloud());
  const poses = input('poses.kitti', Buffer.from('1 0 0 0 0 1 0 0 0 0 1 0\n1 0 0 1 0 1 0 0 0 0 1 0\n'));
  const scans = [input('000000.ply', cloud()), input('000001.ply', cloud())];
  await page.locator('#file-input').setInputFiles(source);
  await expect(page.locator('#status')).toContainText('Loaded survey.ply');
  await page.locator('#vm-file').setInputFiles(input('map.json', Buffer.from(JSON.stringify(map))));
  await expect(page.locator('#vm-status')).toContainText('1 lane');
  await page.locator('#vm-review-next').click();
  await page.locator('#vm-review-state').selectOption('reviewed');
  await page.locator('#vm-review-notes').fill('Surveyed curb, retain this note');
  await page.locator('#vm-review-save').click();
  await page.locator('#pg-files-input').setInputFiles([poses,...scans]);
  await expect(page.locator('#pg-stats')).toContainText('2 (2 with scans)');
  await page.locator('#pg-a').fill('1');
  await page.locator('#pg-fix').click();
  await expect(page.locator('#pg-fix')).toHaveText('Free A');
  await page.getByText('Road options', {exact:true}).click();
  await page.locator('#vm-width').fill('4.2');
  const project = await save(page), parsed = JSON.parse(project.toString());
  expect(parsed.app).toBe('CloudAnalyzer Project');
  expect(JSON.parse(parsed.poseGraph.snapshot).graph.nodes[1].fixed).toBe(true);
  expect(parsed.session.clouds[0].source.digest).toMatch(/^[0-9a-f]{64}$/);
  expect(parsed).not.toHaveProperty('pointData');
  await page.reload();
  await page.locator('#file-input').setInputFiles(input('project.cloudanalyzer.json', project));
  await expect(page.locator('#status')).toContainText('open matching source files');
  await expect(page.locator('#pg-body')).toBeHidden();
  await page.locator('#file-input').setInputFiles([source,poses,...scans]);
  await expect(page.locator('#status')).toContainText('Project restored');
  await expect(page.locator('#vm-status')).toContainText('1 lane');
  await expect(page.locator('#pg-stats')).toContainText('2 (2 with scans)');
  await expect(page.locator('#pg-fix')).toHaveText('Free A');
  await expect(page.locator('#vm-width')).toHaveValue('4.2');
  await page.locator('#vm-review-filter').selectOption('all');
  await page.locator('#vm-review-next').click();
  await expect(page.locator('#vm-review-notes')).toHaveValue('Surveyed curb, retain this note');
  await expect(page.locator('#vm-review-state')).toHaveValue('reviewed');
  const restored = JSON.parse((await save(page)).toString());
  expect(JSON.parse(restored.vectorMap)).toEqual(JSON.parse(parsed.vectorMap));
  expect(JSON.parse(restored.poseGraph.snapshot)).toEqual(JSON.parse(parsed.poseGraph.snapshot));
});

test('a same-name and same-size changed cloud cannot satisfy a saved project', async ({page}) => {
  await page.goto('/');
  await page.locator('#file-input').setInputFiles(input('survey.ply', cloud()));
  await expect(page.locator('#status')).toContainText('Loaded survey.ply');
  await page.locator('#vm-file').setInputFiles(input('map.json', Buffer.from(JSON.stringify(map))));
  const project = await save(page);
  await page.reload();
  await page.locator('#file-input').setInputFiles([input('project.cloudanalyzer.json', project),input('survey.ply',cloud(3))]);
  await expect(page.locator('#status')).toContainText('open matching source files');
  await expect(page.locator('#vm-status')).toContainText('No map yet');
  await page.locator('#file-input').setInputFiles(input('survey.ply',cloud()));
  await expect(page.locator('#status')).toContainText('Project restored');
  await expect(page.locator('#vm-status')).toContainText('1 lane');
});


test('lane edits require review again and memory release preserves current editing state', async ({page}) => {
  await page.goto('/');
  await page.locator('#file-input').setInputFiles(input('survey.ply', cloud()));
  await expect(page.locator('#status')).toContainText('Loaded survey.ply');
  await page.locator('#vm-file').setInputFiles(input('map.json', Buffer.from(JSON.stringify(map))));
  await expect(page.locator('#vm-status')).toContainText('1 lane');
  await page.locator('#vm-review-next').click();
  await page.locator('#vm-review-state').selectOption('reviewed');
  await page.locator('#vm-review-notes').fill('Measured width');
  await page.locator('#vm-review-save').click();
  await page.locator('#vm-lane-speed').fill('30');
  await page.locator('#vm-lane-apply').click();
  await expect(page.locator('#vm-review-state')).toHaveValue('unreviewed');
  await expect(page.locator('#vm-review-stale')).toContainText('changed');
  await expect(page.locator('#vm-review-notes')).toHaveValue('Measured width');
  await page.locator('#memory-panel summary').click();
  await expect(page.locator('#memory-report')).toContainText('map 1');
  await page.locator('#memory-steps').fill('0');
  await page.locator('#memory-apply').click();
  await expect(page.locator('#vm-undo')).toBeDisabled();
  await page.locator('#memory-release').click();
  await expect(page.locator('#status')).toContainText('Current clouds, map and poses retained');
  await expect(page.locator('#cloud-list')).toContainText('survey.ply');
  await expect(page.locator('#vm-status')).toContainText('1 lane');
  await expect(page.locator('#vm-lane-speed')).toHaveValue('30');
  await expect(page.locator('#memory-report')).toContainText('map 0');
});


test('reopening a project resets already-loaded moved clouds to their saved transforms and clears Undo', async ({page}) => {
  await page.goto('/');
  await page.locator('#file-input').setInputFiles([input('reference.ply',cloud()),input('moved.ply',cloud())]);
  await expect(page.locator('#status')).toContainText('Loaded moved.ply');
  await page.locator('#align-panel summary').click();
  await page.locator('#align-matrix').fill('1 0 0 5\n0 1 0 0\n0 0 1 0\n0 0 0 1');
  await page.locator('#align-apply-matrix').click();
  await expect(page.locator('#status')).toContainText('Applied the matrix');
  await page.locator('#c2c-run').click();
  await expect(page.locator('#status')).toContainText('C2C distance computed');
  const project = await save(page);
  await page.locator('#align-matrix').fill('1 0 0 0\n0 1 0 0\n0 0 1 3\n0 0 0 1');
  await page.locator('#align-apply-matrix').click();
  await expect(page.locator('#status')).toContainText('Applied the matrix');
  await page.locator('#file-input').setInputFiles(input('project.cloudanalyzer.json',project));
  await expect(page.locator('#status')).toContainText('Project restored');
  await expect(page.locator('#undo')).toBeDisabled();
  const restored = JSON.parse((await save(page)).toString());
  expect(restored.session.clouds.map(c => c.transforms)).toEqual(JSON.parse(project.toString()).session.clouds.map(c => c.transforms));
  const max = Number(await page.locator('#c2c-stats tr', {hasText:'Max'}).locator('td').textContent());
  expect(max).toBeCloseTo(5,5);
});

// Small loading caps exercise the same parser path as the production presets.
test('project retains each original loading cap after the global preset changes', async ({page}) => {
  await page.goto('/');
  await page.locator('#max-points').evaluate((el: HTMLSelectElement) => {
    el.add(new Option('Two test points','2')); el.value='2';
  });
  const source=input('survey.ply',cloud());
  await page.locator('#file-input').setInputFiles(source);
  await expect(page.locator('#status')).toContainText('Loaded survey.ply: 2 of 4 points');
  await page.locator('#max-points').selectOption('50000000');
  const project=await save(page);
  expect(JSON.parse(project.toString()).session.clouds[0].loadMaxPoints).toBe(2);
  await page.reload();
  await page.locator('#file-input').setInputFiles([input('project.cloudanalyzer.json',project),source]);
  await expect(page.locator('#status')).toContainText('Project restored');
  await expect(page.locator('#cloud-list')).toContainText('2 points');
  await expect(page.locator('#max-points')).toHaveValue('50000000');
  await page.reload();
  await page.locator('#file-input').setInputFiles(source);
  await expect(page.locator('#status')).toContainText('Loaded survey.ply: 4 points');
  await page.locator('#file-input').setInputFiles(input('project.cloudanalyzer.json',project));
  await expect(page.locator('#status')).toContainText('Project restored');
  await expect(page.locator('#cloud-list li')).toHaveCount(1);
  await expect(page.locator('#cloud-list')).toContainText('2 points');
});

test('a budget change while Undo awaits a worker reply clears history safely', async ({page}) => {
  await page.addInitScript(() => {
    const NativeWorker=window.Worker;
    window.Worker=class extends NativeWorker {
      delayed=new Set<number>();
      set onmessage(handler: ((this: Worker, event: MessageEvent) => any) | null) {
        super.onmessage=event => {
          if (this.delayed.has(event.data.seq) && event.data.response) {
            this.delayed.delete(event.data.seq);
            (window as any).__releaseUndoReply=()=>handler?.call(this,event);
          } else handler?.call(this,event);
        };
      }
      postMessage(message: any, options: any) {
        if ((window as any).__delayUndo && message.req?.kind==='transform') this.delayed.add(message.seq);
        super.postMessage(message,options);
      }
    };
  });
  const errors: string[]=[]; page.on('pageerror',e=>errors.push(e.message));
  await page.goto('/');
  await page.locator('#file-input').setInputFiles([input('reference.ply',cloud()),input('moved.ply',cloud())]);
  await expect(page.locator('#status')).toContainText('Loaded moved.ply');
  await page.locator('#align-panel summary').click();
  await page.locator('#align-matrix').fill('1 0 0 5\n0 1 0 0\n0 0 1 0\n0 0 0 1');
  await page.locator('#align-apply-matrix').click();
  await expect(page.locator('#status')).toContainText('Applied the matrix');
  await page.locator('#memory-panel summary').click();
  await page.evaluate(() => { (window as any).__delayUndo=true; });
  await page.locator('#undo').click();
  await expect(page.locator('#undo')).toBeDisabled();
  await expect.poll(()=>page.evaluate(()=>typeof (window as any).__releaseUndoReply)).toBe('function');
  await page.locator('#memory-steps').fill('0');
  await page.locator('#memory-apply').click();
  await page.evaluate(()=>{(window as any).__releaseUndoReply();});
  await expect(page.locator('#status')).toContainText('Undid');
  await expect(page.locator('#redo')).toBeDisabled();
  await expect(page.locator('#memory-report')).toContainText('clouds 0');
  const project=JSON.parse((await save(page)).toString());
  expect(project.session.clouds.find(c=>c.name==='moved.ply').transforms).toEqual([]);
  expect(errors).toEqual([]);
});
