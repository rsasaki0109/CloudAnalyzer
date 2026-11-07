import { createHash } from 'node:crypto';
import { expect, test } from '@playwright/test';
import { addPassingGlimPreflight, addGlimEvidence, packageFixture, packagePreview, previewFixture, cloud, map } from '../tests/mapping-review-fixture.mjs';
const input=(buffer:Buffer)=>({name:'generated-review.zip',mimeType:'application/zip',buffer});

test('opens the verified pair, focuses saved source failures and switches all four protocols', async ({page})=>{
  await page.goto('/');
  await page.locator('#mapping-review-file').setInputFiles(input(packageFixture()));
  await expect(page.locator('#status')).toContainText('Opened generated maps');
  await expect(page.locator('#cloud-list')).toContainText('map.ply');
  await expect(page.locator('#vm-status')).toContainText('1 lane');
  await expect(page.locator('#mapping-review-summary')).toContainText('10.0 / 20.0 m; requested extent unmet');
  await expect(page.locator('#vm-quality-report')).toContainText('Saved low quantile / editable IR');
  await expect(page.locator('#vm-quality-lanes')).toContainText('Lane 7');
  await page.locator('#vm-quality-next').click();
  await expect(page.locator('#vm-quality-problem-detail')).toContainText('Lane 7, left boundary');
  await page.locator('#vm-quality-focus').click();
  await page.locator('#mapping-review-audit').selectOption('1');
  await expect(page.locator('#vm-quality-report')).toContainText('low quantile / reopened OSM');
  await page.locator('#mapping-review-audit').selectOption('2');
  await expect(page.locator('#vm-quality-report')).toContainText('ground consensus / editable IR');
  await expect(page.locator('#vm-quality-lanes button')).toHaveCount(0);
  await page.locator('#mapping-review-audit').selectOption('3');
  await expect(page.locator('#vm-quality-report')).toContainText('ground consensus / reopened OSM');
  await page.locator('#vm-review-next').click();
  await page.locator('#vm-lane-speed').fill('30');
  await page.locator('#vm-lane-apply').click();
  await expect(page.locator('#mapping-review-audit')).toBeDisabled();
  await expect(page.locator('#mapping-review-state')).toContainText('HD map changed');
  await expect(page.locator('#vm-quality-report')).toContainText('has not been checked');
});

test('invalid evidence preserves existing map, cloud and saved lane review', async ({page})=>{
  await page.goto('/');
  await page.locator('#file-input').setInputFiles({name:'original.ply',mimeType:'application/octet-stream',buffer:cloud});
  await expect(page.locator('#status')).toContainText('Loaded original.ply');
  await page.locator('#vm-file').setInputFiles({name:'original.json',mimeType:'application/json',buffer:Buffer.from(JSON.stringify(map))});
  await page.locator('#vm-review-next').click();
  await page.locator('#vm-review-notes').fill('Preserve this review');
  await page.locator('#vm-review-save').click();
  const changed=packageFixture(d=>d.entries.find(([name]:[string,Buffer])=>name===d.manifest.roles.graph)[1]=Buffer.from('GRAPH'));
  await page.locator('#mapping-review-file').setInputFiles(input(changed));
  await expect(page.locator('#status')).toContainText('hash differs');
  await expect(page.locator('#cloud-list > li')).toHaveCount(1);
  await expect(page.locator('#cloud-list')).toContainText('original.ply');
  await expect(page.locator('#vm-review-notes')).toHaveValue('Preserve this review');
  await expect(page.locator('#vm-status')).toContainText('1 lane');
});

test('removing the imported point source invalidates saved audits', async ({page})=>{
  await page.goto('/');
  await page.locator('#mapping-review-file').setInputFiles(input(packageFixture()));
  await expect(page.locator('#status')).toContainText('Opened generated maps');
  await page.locator('#cloud-list button[title="Remove"]').click();
  await expect(page.locator('#mapping-review-audit')).toBeDisabled();
  await expect(page.locator('#mapping-review-state')).toContainText('removed');
});

test('preview shows full-source saved audits, blocks new source checks and persists the distinction in a project', async ({page})=>{
  await page.goto('/');
  await page.locator('#mapping-review-file').setInputFiles(input(packagePreview()));
  await expect(page.locator('#status')).toContainText('Opened generated maps');
  await expect(page.locator('#mapping-review-summary')).toContainText('2 loaded point records (display preview from 4)');
  await expect(page.locator('#mapping-review-state')).toContainText('original 4 points');
  await expect(page.locator('#vm-quality-check')).toBeDisabled();
  await expect(page.locator('#vm-quality-lanes button')).toHaveCount(1);
  await page.locator('#mapping-review-audit').selectOption('2');
  await expect(page.locator('#vm-quality-report')).toContainText('ground consensus / editable IR');
  await expect(page.locator('#vm-quality-lanes button')).toHaveCount(0);
  const download=page.waitForEvent('download',d=>d.suggestedFilename()==='project.cloudanalyzer.json');
  await page.locator('#project-save').click();
  const chunks:Buffer[]=[];for await(const part of await (await download).createReadStream()) chunks.push(part);
  const project=Buffer.concat(chunks);expect(JSON.parse(project.toString()).session.clouds[0].displayPreview).toBe(true);
  await page.reload();
  await page.locator('#file-input').setInputFiles({name:'project.cloudanalyzer.json',mimeType:'application/json',buffer:project});
  await expect(page.locator('#status')).toContainText('open matching source files');
  const preview=previewFixture().entries.find(([name]:[string,Buffer])=>name==='files/display-preview.ply')[1];
  await page.locator('#file-input').setInputFiles({name:'display-preview.ply',mimeType:'application/octet-stream',buffer:preview});
  await expect(page.locator('#status')).toContainText('Project restored');
  await expect(page.locator('#vm-quality-check')).toBeDisabled();
  await page.locator('#file-input').setInputFiles({name:'full-source.ply',mimeType:'application/octet-stream',buffer:cloud});
  await expect(page.locator('#status')).toContainText('Loaded full-source.ply');
  await page.locator('#vm-quality summary').click();
  const option=await page.locator('#vm-quality-cloud option').evaluateAll(nodes=>nodes.map(n=>({text:n.textContent,value:(n as HTMLOptionElement).value})).find(n=>n.text?.includes('full-source')));
  await page.locator('#vm-quality-cloud').selectOption(option!.value);
  await expect(page.locator('#vm-quality-check')).toBeEnabled();
});


test('verified generation history shows decisions as text and stays historical after map edits', async ({page}) => {
  const reason = '<img src=x onerror="window.historyInjected=true"> Review source';
  const buffer = packageFixture(d => {
    const path = d.manifest.roles.decision_history;
    const entry = d.entries.find(([p]) => p === path)!;
    const run = JSON.parse(entry[1].toString()); run.history[0].reason = reason;
    entry[1] = Buffer.from(JSON.stringify(run));
    Object.assign(d.manifest.files.find(f => f.path === path)!, {bytes:entry[1].length,sha256:createHash('sha256').update(entry[1]).digest('hex')});
  });
  await page.goto('/');
  await page.locator('#mapping-review-file').setInputFiles(input(buffer));
  await expect(page.locator('#status')).toContainText('Opened generated maps');
  await page.locator('#mapping-review-history > summary').click();
  await expect(page.locator('#mapping-review-history-summary')).toContainText('3 recorded actions');
  const steps = page.locator('#mapping-review-history-list > li');
  await expect(steps).toHaveCount(3);
  await steps.nth(0).locator('summary').click();
  await expect(steps.nth(0)).toContainText(reason);
  await expect(page.locator('#mapping-review-history-list img')).toHaveCount(0);
  await steps.nth(1).locator('summary').click();
  await expect(steps.nth(1)).toContainText('Candidate 1: defer 10–20 m');
  await expect(steps.nth(1)).toContainText('Source observations are insufficient');
  await page.locator('#vm-review-next').click();
  await page.locator('#vm-lane-speed').fill('30');
  await page.locator('#vm-lane-apply').click();
  await expect(page.locator('#mapping-review-state')).toContainText('HD map changed');
  await expect(steps).toHaveCount(3);
  await expect(page.locator('#mapping-review-history-summary')).toContainText('workspace edits are not included');
  expect(await page.evaluate(() => (window as any).historyInjected)).toBeUndefined();
});

test('saved trajectory toggles and frames without editing the map, then clears on map changes', async ({page}, info) => {
  const errors: string[]=[]; page.on('pageerror',e=>errors.push(e.message));
  const project = async () => {
    const waiting=page.waitForEvent('download'); await page.locator('#project-save').click();
    const parts:Buffer[]=[]; for await(const part of await (await waiting).createReadStream()) parts.push(part);
    return JSON.parse(Buffer.concat(parts).toString());
  };
  await page.goto('/');
  await page.locator('#mapping-review-file').setInputFiles(input(packageFixture()));
  await expect(page.locator('#status')).toContainText('Opened generated maps');
  const before=await project();
  await expect(page.locator('#mapping-trajectory-status')).toContainText('3 of 3 poses');
  await page.locator('#mapping-trajectory-show').check();
  await page.locator('#mapping-trajectory-fit').click();
  await page.locator('#viewport').screenshot({path:info.outputPath('saved-trajectory.png')});
  const after=await project();
  expect(after.vectorMap).toBe(before.vectorMap);
  expect(after.session.clouds).toEqual(before.session.clouds);
  await page.locator('#mapping-trajectory-show').uncheck();
  await page.locator('#mapping-trajectory-show').check();
  const bad=packageFixture(d=>{d.entries.find(([p])=>p===d.manifest.roles.trajectory)![1]=Buffer.from('corrupt');});
  await page.locator('#mapping-review-file').setInputFiles(input(bad));
  await expect(page.locator('#status')).toContainText('Could not open generated maps');
  await expect(page.locator('#mapping-trajectory-show')).toBeChecked();
  await page.locator('#vm-review-next').click();
  await page.locator('#vm-lane-speed').fill('30'); await page.locator('#vm-lane-apply').click();
  await expect(page.locator('#mapping-trajectory-show')).not.toBeChecked();
  await expect(page.locator('#mapping-trajectory-show')).toBeDisabled();
  await expect(page.locator('#mapping-trajectory-fit')).toBeDisabled();
  await expect(page.locator('#mapping-trajectory-status')).toContainText('map pair changed');
  expect(errors).toEqual([]);
});

test('unsupported trajectory leaves valid delivered maps available without an invented path', async ({page}) => {
  const buffer=packageFixture(d=>{
    const path=d.manifest.roles.trajectory, entry=d.entries.find(([p])=>p===path)!;
    entry[1]=Buffer.from('0 0 0 0 0 0 0 1\n1 1 0 0 0 0 0 1\n');
    Object.assign(d.manifest.files.find(f=>f.path===path)!,{bytes:entry[1].length,sha256:createHash('sha256').update(entry[1]).digest('hex')});
    Object.assign(d.manifest.review.artifacts.trajectory,d.manifest.files.find(f=>f.path===path));
  });
  await page.goto('/');
  await page.locator('#mapping-review-file').setInputFiles(input(buffer));
  await expect(page.locator('#status')).toContainText('Opened generated maps');
  await expect(page.locator('#mapping-trajectory-show')).toBeDisabled();
  await expect(page.locator('#mapping-trajectory-status')).toContainText('KITTI');
  await expect(page.locator('#vm-status')).toContainText('1 lane');
});

test('history source links frame verified intervals and disable after map edits', async ({page},info) => {
  const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));
  await page.goto('/');
  await page.locator('#mapping-review-file').setInputFiles(input(packageFixture()));
  await expect(page.locator('#status')).toContainText('Opened generated maps');
  await page.locator('#mapping-review-history > summary').click();
  const step=page.locator('#mapping-review-history-list > li').nth(1);
  await step.locator('summary').click();
  const focus=step.getByRole('button',{name:'Show source interval'});
  await expect(focus).toBeEnabled();await focus.click();
  await expect(page.locator('#mapping-location-status')).toContainText('candidate 1: defer 10–20 m');
  await page.locator('#viewport').screenshot({path:info.outputPath('source-interval.png')});
  await page.locator('#mapping-location-clear').click();
  await expect(page.locator('#mapping-location-status')).toBeEmpty();
  await focus.click();
  await page.locator('#vm-review-next').click();await page.locator('#vm-lane-speed').fill('30');await page.locator('#vm-lane-apply').click();
  await expect(focus).toBeDisabled();
  await expect(page.locator('#mapping-location-clear')).toBeHidden();
  await expect(page.locator('#mapping-location-status')).toBeEmpty();
  await expect(step).toContainText('Source observations are insufficient');
  expect(errors).toEqual([]);
});

test('reused source candidate IDs cannot link a different proposal revision', async ({page}) => {
  const buffer=packageFixture(d=>{
    const path=d.manifest.roles.decision_history,entry=d.entries.find(([p])=>p===path)!;
    const history=JSON.parse(entry[1].toString());history.history[1].proposal_sha256='c'.repeat(64);
    entry[1]=Buffer.from(JSON.stringify(history));
    Object.assign(d.manifest.files.find(f=>f.path===path)!,{bytes:entry[1].length,sha256:createHash('sha256').update(entry[1]).digest('hex')});
  });
  await page.goto('/');await page.locator('#mapping-review-file').setInputFiles(input(buffer));
  await expect(page.locator('#status')).toContainText('Opened generated maps');
  await page.locator('#mapping-review-history > summary').click();
  const step=page.locator('#mapping-review-history-list > li').nth(1);await step.locator('summary').click();
  await expect(step.getByRole('button',{name:'Show source interval'})).toBeDisabled();
  await expect(step).toContainText('different or unidentified source proposal');
});


test('shows original SLAM producer evidence as text and distinguishes an inconsistent report',async({page})=>{
  await page.goto('/');
  const version='<img src=x onerror="window.producerInjected=1">';
  await page.locator('#mapping-review-file').setInputFiles(input(packageFixture(d=>addGlimEvidence(d,{version}))));
  await expect(page.locator('#status')).toContainText('Opened generated maps');
  await page.locator('#mapping-review-provenance summary').click();
  await expect(page.locator('#mapping-review-producer')).toContainText(version);
  await expect(page.locator('#mapping-review-producer')).toContainText('source trajectory hashes match');
  await expect(page.locator('#mapping-review-producer img')).toHaveCount(0);
  await page.locator('#mapping-review-file').setInputFiles(input(packageFixture(d=>addGlimEvidence(d,{mutateRun:r=>r.status='output_invalid'}))));
  await expect(page.locator('#status')).toContainText('Opened generated maps');
  await expect(page.locator('#mapping-review-producer')).toContainText('No verified execution link');
  await expect(page.locator('#mapping-review-producer')).not.toContainText('Recorded GLIM image');
});


test('shows passed, unrequested and inconsistent input preflight without changing map availability',async({page})=>{
  await page.goto('/');
  const open=async(mutateRun)=>{
    await page.locator('#mapping-review-file').setInputFiles(input(packageFixture(d=>addGlimEvidence(d,{mutateRun}))));
    await expect(page.locator('#status')).toContainText('Opened generated maps');
  };
  await open(addPassingGlimPreflight);
  await page.locator('#mapping-review-provenance summary').click();
  const summary=page.locator('#mapping-review-producer');
  await expect(summary).toContainText('recorded basic checks passed');
  await expect(summary).toContainText('IMU gap limit: 50 ms');
  await open(r=>{r.input_preflight_status='not_requested';r.input_preflight=null;});
  await expect(summary).toContainText('not performed');
  await expect(summary).not.toContainText('recorded basic checks passed');
  await open(r=>{addPassingGlimPreflight(r);r.input_preflight.input.sha256='c'.repeat(64);});
  await expect(summary).toContainText('no diagnostic pass is shown');
  await expect(summary).not.toContainText('recorded basic checks passed');
});
