import { expect, test, type Download, type Page } from '@playwright/test';
import { readFile, writeFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
const ring=()=>Array.from({length:25},(_,i)=>[i%5+.25,Math.floor(i/5)+.25,0]).filter(([x,y])=>x===.25||x===4.25||y===.25||y===4.25).flatMap(p=>[p,p]);
const file=(name='ring.xyz')=>({name,mimeType:'text/plain',buffer:Buffer.from(ring().map(p=>p.join(' ')).join('\n'))});
async function downloaded(d:Download){const chunks:Buffer[]=[];for await(const c of await d.createReadStream())chunks.push(c);return JSON.parse(Buffer.concat(chunks).toString());}
async function analyze(page:Page){
  await expect(page.locator('#coverage-run')).toBeEnabled();
  await page.locator('#coverage-cell').fill('1');await page.locator('#coverage-min').fill('3');
  await page.locator('#coverage-run').click();await expect(page.locator('#coverage-summary')).toContainText('9 enclosed empty cells');
}
test('coverage finds enclosed blank cells, focuses a cell and exports bounded loaded-point metrics',async({page},info)=>{
  const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));await page.goto('/');
  await expect(page.locator('#coverage-run')).toBeDisabled();await page.locator('#file-input').setInputFiles(file());await analyze(page);
  await expect(page.locator('#coverage-summary')).toContainText('16 sparse');
  const canvas=page.locator('#coverage-grid');await canvas.click({position:{x:115,y:115}});
  await expect(page.locator('#coverage-cell-info')).toContainText('0 points; enclosed empty');
  const d=page.waitForEvent('download');await page.locator('#coverage-download').click();const metrics=await downloaded(await d);
  expect(metrics.schema).toBe('cloudanalyzer.map_coverage.v1');expect(metrics.occupied).toBe(16);expect(metrics.enclosedEmpty).toBe(9);
  expect(metrics.sparse).toBe(16);expect(metrics.counts).toBeUndefined();expect(metrics.loadedPoints).toBe(32);
  const report=page.waitForEvent('download');await page.locator('#report-json').click();
  const qualityReport=await downloaded(await report);expect(qualityReport.sections['map-coverage'].metrics.enclosed.value).toBe(9);
  await page.locator('#coverage-panel').evaluate(el=>el.scrollIntoView({block:'start'}));
  await page.screenshot({path:info.outputPath('coverage.png')});expect(errors).toEqual([]);
  await page.locator('#coverage-cell').fill('2');await expect(page.locator('#coverage-result')).toBeHidden();
  const stale=page.waitForEvent('download');await page.locator('#report-json').click();
  expect((await downloaded(await stale)).sections['map-coverage']).toBeUndefined();
});
test('cancelling the coverage worker publishes no partial metrics and leaves the source cloud',async({page})=>{
  await page.goto('/');await page.locator('#file-input').setInputFiles(file());
  await page.evaluate(()=>{
    const NativeWorker=window.Worker;
    window.Worker=class extends NativeWorker {
      postMessage(){document.getElementById('task-cancel')!.click();}
    };
  });
  await page.locator('#coverage-cell').fill('1');await page.locator('#coverage-run').click();
  await expect(page.locator('#status')).toContainText('Cancelled');
  await expect(page.locator('#coverage-result')).toBeHidden();await expect(page.locator('#cloud-list > li')).toHaveCount(1);
  await expect(page.locator('#coverage-run')).toBeEnabled();
});
test('thinned input reports loaded-only density instead of claiming full-source coverage',async({page})=>{
  await page.goto('/');await page.evaluate(()=>{
    const select=document.getElementById('max-points') as HTMLSelectElement;
    select.add(new Option('16 test points','16'));select.value='16';
  });
  const points=ring(),body=Buffer.alloc(points.length*12);
  points.forEach((p,i)=>p.forEach((v,k)=>body.writeFloatLE(v,i*12+k*4)));
  const header=Buffer.from(`ply\nformat binary_little_endian 1.0\nelement vertex ${points.length}\nproperty float x\nproperty float y\nproperty float z\nend_header\n`);
  await page.locator('#file-input').setInputFiles({name:'sampled.ply',mimeType:'application/octet-stream',buffer:Buffer.concat([header,body])});await analyze(page);
  await expect(page.locator('#coverage-scope')).toContainText('Sampled/partial data');
  const d=page.waitForEvent('download');await page.locator('#coverage-download').click();const report=await downloaded(await d);
  expect(report.loadedPoints).toBe(16);expect(report.sourceFilePoints).toBe(32);expect(report.keepEvery).toBe(2);
  expect(report.density.median).toBe(1);
});
test('quality comparison uses selected source/reference and never starts computation automatically',async({page})=>{
  await page.goto('/');await page.locator('#file-input').setInputFiles(file('first.xyz'));
  await expect(page.locator('#coverage-compare')).toBeDisabled();
  await page.locator('#file-input').setInputFiles(file('second.xyz'));
  await page.locator('#coverage-source').selectOption({label:'second.xyz'});
  await page.locator('#coverage-reference').selectOption({label:'first.xyz'});
  await page.locator('#coverage-compare').click();
  await expect(page.locator('#distance-method')).toHaveValue('quality');
  await expect(page.locator('#c2c-compared option:checked')).toHaveText('second.xyz');
  await expect(page.locator('#c2c-reference option:checked')).toHaveText('first.xyz');
  await expect(page.locator('#c2c-run')).toBeEnabled();await expect(page.locator('#c2c-result')).toBeHidden();
});
test('oversized grid fails without adding/removing clouds and source removal clears diagnostics',async({page})=>{
  await page.goto('/');await page.locator('#file-input').setInputFiles(file());await analyze(page);
  await page.locator('#coverage-cell').fill('.00001');await page.locator('#coverage-run').click();
  await expect(page.locator('#status')).toContainText('increase cell size');
  await expect(page.locator('#cloud-list > li')).toHaveCount(1);await expect(page.locator('#coverage-result')).toBeHidden();
  await page.locator('#cloud-list > li').locator('button[title="Remove"]').click();await expect(page.locator('#coverage-run')).toBeDisabled();
});
test('actual local map coverage stays read-only and records its loaded-point scope',async({page},info)=>{
  const path=process.env.MAP_COVERAGE_SOURCE;test.skip(!path,'Optional existing local map; no dataset download in CI');test.setTimeout(180000);
  const raw=await readFile(path!),hash=(bytes:Buffer)=>createHash('sha256').update(bytes).digest('hex');
  const errors:string[]=[];page.on('pageerror',e=>errors.push(e.message));await page.setViewportSize({width:1600,height:1000});await page.goto('/');
  await page.locator('#file-input').setInputFiles(path!);await expect(page.locator('#coverage-run')).toBeEnabled({timeout:120000});
  await page.locator('#coverage-cell').fill('5');await page.locator('#coverage-min').fill('10');await page.locator('#coverage-run').click();
  await expect(page.locator('#status')).toContainText('Coverage ready',{timeout:120000});
  const d=page.waitForEvent('download');await page.locator('#coverage-download').click();const report=await downloaded(await d);
  expect(report.loadedPoints).toBeGreaterThan(100000);expect(report.finitePoints+report.invalidPoints).toBe(report.loadedPoints);
  expect(report.occupied+report.empty).toBe(report.nx*report.ny);expect(report.enclosedEmpty).toBeLessThanOrEqual(report.empty);
  expect(report.keepEvery).toBe(1);expect(report.sparse).toBeLessThanOrEqual(report.occupied);expect(errors).toEqual([]);
  expect(hash(await readFile(path!))).toBe(hash(raw));await expect(page.locator('#cloud-list > li')).toHaveCount(1);
  await page.getByRole('button',{name:'Top',exact:true}).click();await page.locator('#coverage-panel').evaluate(el=>el.scrollIntoView({block:'start'}));
  await page.screenshot({path:info.outputPath('actual-map-coverage.png')});
  await writeFile(info.outputPath('coverage-receipt.json'),JSON.stringify({input_sha256:hash(raw),input_bytes:raw.length,
    report,source_unchanged:true,no_derived_cloud_added:true,browser_errors:errors,semantic_missing_data_accuracy_measured:false},null,2));
});
