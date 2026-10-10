import { expect, test, type Page, type Download } from '@playwright/test';
import { createHash } from 'node:crypto';
import { readFile, writeFile } from 'node:fs/promises';
import { readProjectSnapshot } from '../src/project-snapshot';
import { indexReviewZip, readReviewMember, REVIEW_LIMIT, WORKSPACE_LIMIT } from '../src/review-zip';

const signal = () => new AbortController().signal;
const hash = (data: Buffer) => createHash('sha256').update(data).digest('hex');
async function bytes(download: Download) {
  const chunks: Buffer[] = [];
  for await (const chunk of await download.createReadStream()) chunks.push(chunk);
  return Buffer.concat(chunks);
}
async function save(page: Page) {
  const event = page.waitForEvent('download', d => d.suggestedFilename() === 'project.cloudanalyzer.zip');
  await page.locator('#project-snapshot').click();
  return bytes(await event);
}
async function metadata(page: Page) {
  const event = page.waitForEvent('download', d => d.suggestedFilename() === 'project.cloudanalyzer.json');
  await page.locator('#project-save').click();
  return JSON.parse((await bytes(await event)).toString());
}
async function exported(page: Page, index: number) {
  const row = page.locator('#cloud-list > li').nth(index);
  await row.locator('button[title="Save as…"]').click();
  const event = page.waitForEvent('download');
  await row.locator('.save-formats button', { hasText: 'PLY' }).click();
  return bytes(await event);
}

test('manual workspace saves and restores native point records beyond the recovery budget', async ({ page }, info) => {
  test.setTimeout(180000);
  const count = 1_000_000, stride = 72;
  const header = Buffer.from(`ply\nformat binary_little_endian 1.0\nelement vertex ${count}\nproperty double x\nproperty double y\nproperty double z\n${Array.from({ length: 12 }, (_, i) => `property float survey_${i}\n`).join('')}end_header\n`);
  const records = Buffer.alloc(count * stride);
  for (let i = 0; i < count; i++) {
    records.writeDoubleLE(500000 + i % 1000 / 100, i * stride);
    records.writeDoubleLE(4000000 + Math.floor(i / 1000) / 100, i * stride + 8);
    records.writeDoubleLE(2.123456789, i * stride + 16);
    for (let j = 0; j < 12; j++) records.writeFloatLE((i % 97 + j) / 100, i * stride + 24 + j * 4);
  }
  const inputPath = info.outputPath('survey.ply');
  await writeFile(inputPath, Buffer.concat([header, records]));
  await page.goto('/');
  await page.locator('#file-input').setInputFiles(inputPath);
  await expect(page.locator('#status')).toContainText('Loaded survey.ply');
  await page.locator('#project-autosave-records').check();
  await expect(page.locator('#project-save-status')).toContainText('64 MiB');
  await page.locator('#project-autosave-records').uncheck();
  const before = await exported(page, 0), zip = await save(page);
  expect(zip.length).toBeGreaterThan(REVIEW_LIMIT);
  const packed = await readProjectSnapshot(new File([zip], 'project.cloudanalyzer.zip'), signal());
  expect(Buffer.from(await packed[1].arrayBuffer()).equals(before)).toBe(true);
  const outputPath = info.outputPath('project.cloudanalyzer.zip');
  await writeFile(outputPath, zip);
  await page.reload();
  await page.locator('#file-input').setInputFiles(outputPath);
  await expect(page.locator('#status')).toContainText('Project restored');
  expect((await exported(page, 0)).equals(before)).toBe(true);
  await expect(page.locator('#undo')).toBeDisabled();
});

test('actual April and June NCLT maps resume with their original graph and map evidence above 64 MiB', async ({ page }, info) => {
  const legacyPath = process.env.CLOUDANALYZER_LARGE_WORKSPACE_BASE;
  const candidatePath = process.env.CLOUDANALYZER_LARGE_WORKSPACE_CLOUD;
  test.skip(!legacyPath || !candidatePath, 'Opt in with a complete NCLT workspace and the second session point map');
  test.setTimeout(600000);
  const legacy = await readFile(legacyPath!), legacyFiles = await readProjectSnapshot(new File([legacy], 'project.cloudanalyzer.zip'), signal());
  const legacyInputPath = info.outputPath('legacy.cloudanalyzer.zip');
  await writeFile(legacyInputPath, legacy);
  await page.goto('/');
  await page.locator('#file-input').setInputFiles(legacyInputPath);
  await expect(page.locator('#status')).toContainText(/Project restored|Could not open/, { timeout: 300000 });
  await expect(page.locator('#status')).toContainText('Project restored');
  await page.locator('#file-input').setInputFiles(candidatePath!);
  await expect(page.locator('#cloud-list > li')).toHaveCount(3);
  await expect(page.locator('#status')).toContainText('Loaded');
  const before = await metadata(page), pointFiles: Buffer[] = [];
  for (let i = 0; i < 3; i++) pointFiles.push(await exported(page, i));
  const zip = await save(page);
  expect(zip.length).toBeGreaterThan(REVIEW_LIMIT);
  const file = new File([zip], 'project.cloudanalyzer.zip'), directory = await indexReviewZip(file, signal(), 2048, WORKSPACE_LIMIT);
  const manifest = JSON.parse(new TextDecoder().decode(await readReviewMember(file, directory.get('manifest.json')!, signal())));
  expect(manifest.schema).toBe('cloudanalyzer.project_snapshot.v3');
  const packed = await readProjectSnapshot(file, signal());
  const saved = JSON.parse(await packed[0].text());
  const assets = [];
  for (const asset of legacyFiles.slice(3)) {
    const restored = packed.find(f => f.name === asset.name)!;
    expect(Buffer.from(await restored.arrayBuffer()).equals(Buffer.from(await asset.arrayBuffer()))).toBe(true);
    assets.push({ name: asset.name, bytes: asset.size, sha256: hash(Buffer.from(await asset.arrayBuffer())) });
  }
  const output = info.outputPath('project.cloudanalyzer.zip');
  await writeFile(output, zip);
  await page.goto('/');
  const started = Date.now();
  await page.locator('#file-input').setInputFiles(output);
  await expect(page.locator('#status')).toContainText(/Project restored|Could not open/, { timeout: 300000 });
  await expect(page.locator('#status')).toContainText('Project restored');
  const restoredMs = Date.now() - started, after = await metadata(page);
  expect(JSON.parse(after.poseGraph.snapshot)).toEqual(JSON.parse(before.poseGraph.snapshot));
  expect(JSON.parse(after.vectorMap)).toEqual(JSON.parse(before.vectorMap));
  expect(after.reviews).toEqual(before.reviews);
  const clouds = [];
  for (let i = 0; i < 3; i++) {
    const actual = await exported(page, i);
    expect(actual.equals(pointFiles[i])).toBe(true);
    expect(actual.equals(Buffer.from(await packed[i + 1].arrayBuffer()))).toBe(true);
    clouds.push({ name: saved.session.clouds[i].name, bytes: actual.length, sha256: hash(actual) });
  }
  await expect(page.locator('#mapping-review-audit')).toBeDisabled();
  const originalArchive = legacyFiles.find(f => f.name === saved.reviewArchive.name)!;
  const event = page.waitForEvent('download');
  await page.locator('#mapping-review-download').click();
  expect((await bytes(await event)).equals(Buffer.from(await originalArchive.arrayBuffer()))).toBe(true);
  await expect(page.locator('#undo')).toBeDisabled();
  const receipt = {
    schema: 'cloudanalyzer.large_workspace_receipt.v1',
    legacyZipBytes: legacy.length, legacyZipSha256: hash(legacy),
    candidateInputBytes: (await readFile(candidatePath!)).length, candidateInputSha256: hash(await readFile(candidatePath!)),
    zipBytes: zip.length, zipSha256: hash(zip), restoredMs,
    nodes: JSON.parse(before.poseGraph.snapshot).graph.nodes.length,
    cloudExportsExact: true, originalAssetsExact: true, poseGraphExact: true,
    hdMapExact: true, reviewsExact: true, originalArchiveExact: true, savedAuditsDisabled: true,
    clouds, assets,
  };
  await writeFile(info.outputPath('large-workspace-receipt.json'), JSON.stringify(receipt, null, 2));
  await writeFile(info.outputPath('manifest.json'), JSON.stringify(manifest, null, 2));
  await writeFile(info.outputPath('project.json'), JSON.stringify(saved, null, 2));
});
