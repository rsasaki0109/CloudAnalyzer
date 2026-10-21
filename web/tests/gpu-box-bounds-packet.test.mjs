import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
const root = fileURLToPath(new URL('../../', import.meta.url)), directory = 'benchmarks/wasm-gpu-box-bounds';
const bytes = n => readFileSync(resolve(root, n));
const hash = b => createHash('sha256').update(b).digest('hex');
const normalizedHash = n => hash(bytes(n).toString('utf8').replaceAll('\r\n', '\n'));
const median = values => { const a = values.toSorted((x, y) => x - y); return (a[a.length / 2 - 1] + a[a.length / 2]) / 2; };

test('GPU box packet preserves disabled diagnostics, cold setup, memory and the failed adoption gate', () => {
  const manifest = JSON.parse(bytes(`${directory}/manifest.json`));
  const parents = JSON.parse(bytes("benchmarks/wasm-gpu-parent-traversal/manifest.json"));
  const compact = JSON.parse(bytes("benchmarks/wasm-gpu-compact-stack/manifest.json"));
  const aliases = { ...manifest.historicalSourceAliases, ...manifest.measuredSourceAliases, ...parents.historicalSourceAliases, ...compact.historicalSourceAliases };
  const source = (n, h) => {
    if (normalizedHash(n) === h) return;
    const a = aliases[h]; assert.equal(a?.original, n);
    assert.ok((a.archived.startsWith(`${directory}/executed-source/`) || a.archived.startsWith("benchmarks/wasm-gpu-parent-traversal/executed-source/") || a.archived.startsWith("benchmarks/wasm-gpu-compact-stack/executed-source/")) && !a.archived.includes('..'));
    assert.equal(normalizedHash(a.archived), h);
  };
  assert.equal(manifest.adopted, false); assert.equal(manifest.finalDefault, 'local');
  assert.equal(manifest.backendSelectionChanged, false); assert.equal(manifest.reports.length, 4);
  assert.equal(manifest.protocol.timingReportBudget, 4); assert.equal(manifest.protocol.coldSamplesPerReport, 2);
  assert.equal(manifest.protocol.diagnosticPairsPerReport, 2); assert.equal(manifest.protocol.diagnosticWarmupPairs, 2);
  const footprint = manifest.normalizedSourceFootprint;
  assert.equal(hash(Object.entries(footprint).sort(([a], [b]) => a < b ? -1 : 1).map(([p, h]) => `${p}\0${h}\n`).join('')), manifest.normalizedSourceFingerprintSha256);
  for (const [n, h] of Object.entries(footprint)) source(n, h);
  for (const [n, h] of Object.entries(manifest.finalBuild.normalizedInputs)) source(n, h);
  for (const [h, a] of Object.entries(aliases)) assert.equal(normalizedHash(a.archived), h);
  const priorBytes = bytes('benchmarks/wasm-gpu-diagnostics/manifest.json'), prior = JSON.parse(priorBytes);
  let winsTotal = 0; const gates = [], identities = new Map();
  for (const [index, entry] of manifest.reports.entries()) {
    const raw = bytes(`${directory}/${entry.file}`), r = JSON.parse(raw), m = r.measurement;
    assert.equal(hash(raw), entry.sha256); assert.equal(raw.length, entry.bytes);
    assert.equal(r.status, 'passed'); assert.equal(r.purpose, 'timing-comparison'); assert.equal(r.gpu, 'physical');
    assert.equal(r.gpuBoxBoundsComparison, true); assert.equal(r.threads, 4); assert.equal(r.warmups, 6); assert.equal(r.repeats, 12);
    assert.equal(r.filesUnchangedDuringMeasurement, true); assert.deepEqual(r.build, manifest.finalBuild);
    assert.equal(hash(JSON.stringify(r.build, null, 2) + '\n'), r.buildInfoSha256);
    for (const [n, h] of Object.entries({ ...r.sources, ...r.build.normalizedInputs })) source(n, h);
    const reused = r.build.wasmReuse;
    assert.equal(reused.priorMetadataSha256, hash(priorBytes)); assert.equal(Object.keys(reused.unchangedArtifacts).length, 4);
    assert.deepEqual(Object.keys(reused.unchangedRustInputs).sort(), Object.keys(prior.finalBuild.normalizedInputs).filter(n => n.startsWith('rust/')).sort());
    for (const [n, h] of Object.entries(reused.unchangedRustInputs)) { source(n, h); assert.equal(prior.finalBuild.normalizedInputs[n], h); }
    for (const [n, h] of Object.entries(reused.unchangedArtifacts)) { assert.equal(prior.finalBuild.artifacts[n], h); assert.equal(r.build.artifacts[n], h); }
    const spec = manifest.inputs[entry.dataset], inputBytes = bytes(`${directory}/${spec.file}`);
    assert.equal(inputBytes.length, spec.bytes); assert.equal(hash(inputBytes), spec.sha256); assert.equal(r.inputManifestSha256, spec.sha256);
    assert.equal(m.defaultBoxBounds, false); assert.equal(m.defaultValidation.compute.boxBounds, false);
    assert.equal(m.defaultValidation.compute.boxBoundsBytes, 0);
    for (const flag of ['queryBitsExact','queryRangeGuardsChecked','ownerSnapshotAndExportIndependenceChecked','distanceBitsExact','summaryFieldsExact','certificateCountsStablePerMode','sourceRemovalCompleted']) assert.equal(m[flag], true);
    assert.deepEqual(entry.certifiedByMode, m.certifiedByMode);
    const old = prior.reports.find(e => e.dataset === entry.dataset);
    assert.equal(m.queryBitsSha256, old.queryBitsSha256); assert.equal(m.displayedF32Sha256, old.displayedF32Sha256); assert.equal(m.certifiedByMode.local, old.certified);
    const identity = [m.queryBitsSha256, m.displayedF32Sha256, m.certifiedByMode];
    if (identities.has(entry.dataset)) assert.deepEqual(identity, identities.get(entry.dataset)); identities.set(entry.dataset, identity);
    const check = (result, mode, warm, diagnostic) => {
      const c = result.compute; assert.ok(result.elapsedMillis >= result.workerMillis);
      assert.equal(c.backend, 'webgpu'); assert.equal(c.fallback, undefined); assert.equal(c.verifyFallback, undefined);
      assert.equal(c.inheritedBounds, false); assert.equal(c.boxBounds, mode === 'boxes');
      assert.equal(c.querySliceCopy, false); assert.equal(c.queryOwnedCopy, false); assert.equal(c.verifyStagedQueries, false); assert.equal(c.verifyMainWorker, false);
      assert.equal(c.batchSize, 131072); assert.equal(c.workgroupSize, 64); assert.equal(c.verifyWorkers, 4);
      assert.equal(c.batches, Math.ceil(m.inputPoints.compared / 131072)); assert.equal(c.certified, m.certifiedByMode[mode]);
      for (const f of ['verifyBounds','verifySimd','prefetch','verifySnapshot']) assert.equal(c[f], true);
      if (warm) for (const f of ['referenceReused','deviceReused','verifyReferenceReused']) assert.equal(c[f], true);
      assert.ok(Number.isFinite(c.boxBoundsPrepareMillis) && c.boxBoundsPrepareMillis >= 0);
      if (mode === 'boxes') { assert.equal(c.boxBoundsBytes, entry.summary.boxBoundsBytes); if (warm) assert.equal(c.boxBoundsReused, true); }
      if (!diagnostic) { assert.equal(Object.hasOwn(c, 'diagnostics'), false); return; }
      const d = c.diagnostics; assert.equal(d.timestampQuery, 'enabled'); assert.equal(d.gpuBatchesOmitted, 0); assert.equal(d.verificationBatchesOmitted, 0);
      assert.equal(d.gpu.length, c.batches); assert.equal(d.gpu.reduce((n, b) => n + b.count, 0), m.inputPoints.compared);
      assert.equal(d.verification.reduce((n, b) => n + b.count, 0), m.inputPoints.compared);
      for (const b of d.gpu) assert.ok(Number.isFinite(b.shaderMillis) && b.shaderMillis >= 0);
    };
    assert.equal(m.warming.length, 6); assert.equal(m.trials.length, 12);
    for (const [i, t] of [...m.warming, ...m.trials].entries()) {
      assert.deepEqual(t.order, i % 2 ? ['boxes','local'] : ['local','boxes']);
      for (const mode of t.order) check(t.results[mode], mode, true, false);
    }
    assert.equal(r.coldFirst, index % 2 ? 'boxes' : 'local');
    assert.deepEqual(m.coldOrder, index % 2 ? ['boxes','local'] : ['local','boxes']);
    for (const mode of m.coldOrder) {
      const result = m.cold[mode], c = result.compute; check(result, mode, false, false);
      assert.equal(c.referenceReused, false); assert.equal(c.verifyReferenceReused, false); assert.equal(c.deviceReused, true); assert.equal(c.boxBoundsReused, false);
      if (mode === 'local') assert.equal(c.boxBoundsBytes, 0);
      assert.equal(entry.summary.coldMillis[mode], result.elapsedMillis);
    }
    assert.equal(m.diagnosticWarming.length, 2); assert.equal(m.diagnosticTrials.length, 2);
    for (const [i, t] of [...m.diagnosticWarming, ...m.diagnosticTrials].entries()) {
      assert.deepEqual(t.order, i % 2 ? ['boxes','local'] : ['local','boxes']);
      for (const mode of t.order) check(t.results[mode], mode, i > 0, true);
    }
    const s = entry.summary;
    for (const mode of ['local','boxes']) {
      assert.equal(s.medianMillis[mode], median(m.trials.map(t => t.results[mode].elapsedMillis)));
      assert.equal(s.maximumMillis[mode], Math.max(...m.trials.map(t => t.results[mode].elapsedMillis)));
      assert.equal(s.diagnosticGpuPassSumMedianMillis[mode], median(m.diagnosticTrials.map(t => t.results[mode].compute.diagnostics.gpu.reduce((n, b) => n + b.shaderMillis, 0))));
    }
    const paired = median(m.trials.map(t => (t.results.boxes.elapsedMillis / t.results.local.elapsedMillis - 1) * 100));
    const wins = m.trials.filter(t => t.results.boxes.elapsedMillis < t.results.local.elapsedMillis).length;
    const cold = (m.cold.boxes.elapsedMillis / m.cold.local.elapsedMillis - 1) * 100;
    assert.equal(s.boxesVsLocalPairedPercent, paired); assert.equal(s.fasterPairs, wins); winsTotal += wins;
    assert.equal(s.coldBoxesVsLocalPercent, cold); assert.equal(s.boxBoundsPrepareMillis, m.cold.boxes.compute.boxBoundsPrepareMillis);
    assert.equal(s.warmGatePassed, paired <= -1 && wins >= 7); assert.equal(s.coldGuardPassed, cold <= 5);
    assert.equal(s.memoryGuardPassed, s.boxBoundsBytes <= 16 * 1024 * 1024);
    assert.equal(s.adoptionGatePassed, s.warmGatePassed && s.coldGuardPassed && s.memoryGuardPassed); gates.push(s.adoptionGatePassed);
  }
  assert.equal(manifest.adoptionGatePassed, gates.every(Boolean)); assert.equal(manifest.fasterPairs, winsTotal); assert.equal(winsTotal, 0); assert.equal(manifest.totalPairs, 48);
});
