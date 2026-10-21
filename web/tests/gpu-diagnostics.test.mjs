import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';
import { stripTypeScriptTypes } from 'node:module';

const root = fileURLToPath(new URL('../../', import.meta.url)), directory = 'benchmarks/wasm-gpu-diagnostics';
const bytes = n => readFileSync(resolve(root, n));
const hash = b => createHash('sha256').update(b).digest('hex');
const normalizedHash = n => hash(bytes(n).toString('utf8').replaceAll('\r\n', '\n'));
const median = values => { const a = values.toSorted((x, y) => x - y); return (a[a.length / 2 - 1] + a[a.length / 2]) / 2; };

test('diagnostic verification frees its WASM result when distance export fails', async () => {
  const source = stripTypeScriptTypes(bytes('web/src/gpu-diagnostics.ts').toString('utf8'), { mode: 'transform' });
  const { timeGpuVerification } = await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'));
  let freed = 0;
  assert.throws(() => timeGpuVerification(() => ({ certified: 0, takeDistances() { throw new Error('export failure'); }, free() { freed++; } })), /export failure/);
  assert.equal(freed, 1);
});

test('GPU diagnostic packet retains identical outputs, complete traces and paired overhead', () => {
  const manifest = JSON.parse(bytes(`${directory}/manifest.json`));
  const boxes = JSON.parse(bytes("benchmarks/wasm-gpu-box-bounds/manifest.json"));
  const parents = JSON.parse(bytes("benchmarks/wasm-gpu-parent-traversal/manifest.json"));
  const aliases = { ...manifest.historicalSourceAliases, ...manifest.measuredSourceAliases, ...boxes.historicalSourceAliases, ...parents.historicalSourceAliases };
  const source = (name, expected) => {
    if (normalizedHash(name) === expected) return;
    const a = aliases[expected]; assert.equal(a?.original, name);
    assert.ok((a.archived.startsWith(`${directory}/executed-source/`) || a.archived.startsWith("benchmarks/wasm-gpu-box-bounds/executed-source/") || a.archived.startsWith("benchmarks/wasm-gpu-parent-traversal/executed-source/")) && !a.archived.includes('..'));
    assert.equal(normalizedHash(a.archived), expected);
  };
  assert.equal(manifest.diagnosticsDefault, false); assert.equal(manifest.backendSelectionChanged, false);
  assert.equal(manifest.performanceOptimizationAdopted, false); assert.equal(manifest.reports.length, 4);
  assert.equal(manifest.protocol.timingReportBudget, 4);
  const footprint = manifest.normalizedSourceFootprint;
  assert.equal(hash(Object.entries(footprint).sort(([a], [b]) => a < b ? -1 : 1).map(([p, h]) => `${p}\0${h}\n`).join('')), manifest.normalizedSourceFingerprintSha256);
  for (const [p, h] of Object.entries(footprint)) source(p, h);
  for (const [p, h] of Object.entries(manifest.finalBuild.normalizedInputs)) source(p, h);
  for (const [h, a] of Object.entries(aliases)) assert.equal(normalizedHash(a.archived), h);
  const priorBytes = bytes('benchmarks/wasm-gpu-inherited-bounds/manifest.json'), prior = JSON.parse(priorBytes);
  const identities = new Map();
  for (const entry of manifest.reports) {
    const raw = bytes(`${directory}/${entry.file}`), r = JSON.parse(raw), m = r.measurement;
    assert.equal(hash(raw), entry.sha256); assert.equal(raw.length, entry.bytes);
    assert.equal(r.status, 'passed'); assert.equal(r.purpose, 'timing-comparison'); assert.equal(r.gpu, 'physical');
    assert.equal(r.gpuDiagnosticsComparison, true); assert.equal(r.threads, 4); assert.equal(r.warmups, 6); assert.equal(r.repeats, 12);
    assert.equal(r.filesUnchangedDuringMeasurement, true); assert.deepEqual(r.build, manifest.finalBuild);
    assert.equal(hash(JSON.stringify(r.build, null, 2) + '\n'), r.buildInfoSha256);
    for (const [p, h] of Object.entries({ ...r.sources, ...r.build.normalizedInputs })) source(p, h);
    const reused = r.build.wasmReuse;
    assert.equal(reused.priorMetadataSha256, hash(priorBytes)); assert.equal(Object.keys(reused.unchangedArtifacts).length, 4);
    assert.deepEqual(Object.keys(reused.unchangedRustInputs).sort(), Object.keys(prior.finalBuild.normalizedInputs).filter(n => n.startsWith('rust/')).sort());
    for (const [p, h] of Object.entries(reused.unchangedRustInputs)) { source(p, h); assert.equal(prior.finalBuild.normalizedInputs[p], h); }
    for (const [p, h] of Object.entries(reused.unchangedArtifacts)) { assert.equal(prior.finalBuild.artifacts[p], h); assert.equal(r.build.artifacts[p], h); }
    const spec = manifest.inputs[entry.dataset], specBytes = bytes(`${directory}/${spec.file}`);
    assert.equal(specBytes.length, spec.bytes); assert.equal(hash(specBytes), spec.sha256); assert.equal(r.inputManifestSha256, spec.sha256);
    for (const f of ['queryBitsExact', 'queryRangeGuardsChecked', 'ownerSnapshotAndExportIndependenceChecked', 'distanceBitsExact', 'summaryFieldsExact', 'certificateCountsIdenticalAcrossControls', 'sourceRemovalCompleted', 'timestampQuerySupported']) assert.equal(m[f], true);
    assert.equal(m.defaultInheritedBounds, false); assert.equal(m.defaultValidation.compute.diagnostics, undefined);
    assert.equal(m.certifiedByMode.off, m.certifiedByMode.on); assert.equal(entry.certified, m.certifiedByMode.off);
    const identity = [m.queryBitsSha256, m.displayedF32Sha256, m.certifiedByMode.off];
    if (identities.has(entry.dataset)) assert.deepEqual(identity, identities.get(entry.dataset)); identities.set(entry.dataset, identity);
    assert.equal(entry.queryBitsSha256, m.queryBitsSha256); assert.equal(entry.displayedF32Sha256, m.displayedF32Sha256);
    const old = prior.reports.find(e => e.dataset === entry.dataset), oldReport = JSON.parse(bytes(`benchmarks/wasm-gpu-inherited-bounds/${old.file}`)).measurement;
    assert.deepEqual(identity, [oldReport.queryBitsSha256, oldReport.displayedF32Sha256, oldReport.certifiedByMode.local]);
    assert.equal(m.warming.length, 6); assert.equal(m.trials.length, 12);
    for (const [i, t] of [...m.warming, ...m.trials].entries()) {
      assert.deepEqual(t.order, i % 2 ? ['on', 'off'] : ['off', 'on']);
      for (const mode of t.order) {
        const result = t.results[mode], c = result.compute;
        assert.ok(result.elapsedMillis >= result.workerMillis); assert.equal(c.backend, 'webgpu');
        assert.equal(c.inheritedBounds, false); assert.equal(c.queryOwnedCopy, false); assert.equal(c.querySliceCopy, false);
        assert.equal(c.batchSize, 131072); assert.equal(c.workgroupSize, 64); assert.equal(c.verifyWorkers, 4); assert.equal(c.verifyStagedQueries, false);
        for (const f of ['verifyBounds', 'verifySimd', 'prefetch', 'verifySnapshot', 'referenceReused', 'deviceReused', 'verifyReferenceReused']) assert.equal(c[f], true);
        assert.equal(c.fallback, undefined); assert.equal(c.verifyFallback, undefined); assert.equal(c.certified, entry.certified);
        if (mode === 'off') { assert.equal(Object.hasOwn(c, 'diagnostics'), false); continue; }
        const d = c.diagnostics, nonnegative = v => Number.isFinite(v) && v >= 0;
        assert.equal(d.timestampQuery, 'enabled'); assert.equal(d.gpuBatchesOmitted, 0); assert.equal(d.verificationBatchesOmitted, 0);
        assert.equal(d.gpu.length, c.batches); assert.equal(c.batches, Math.ceil(m.inputPoints.compared / 131072));
        let first = 0;
        for (const b of d.gpu) { assert.equal(b.first, first); assert.ok(b.count > 0); first += b.count; assert.ok([b.queryExportMillis, b.uploadCallMillis, b.readbackMillis, b.shaderMillis].every(nonnegative)); }
        assert.equal(first, m.inputPoints.compared);
        first = 0;
        for (const l of d.verification.toSorted((a, b) => a.first - b.first)) {
          assert.equal(l.first, first); assert.ok(l.count > 0); first += l.count; assert.equal(l.resident, false); assert.ok(l.lane >= 0 && l.lane < 4);
          assert.ok([l.queryExportMillis, l.verifyCallMillis, l.distanceExportMillis, l.roundTripMillis].every(nonnegative));
        }
        assert.equal(first, m.inputPoints.compared);
      }
    }
    for (const mode of ['off', 'on']) {
      assert.equal(entry.summary.medianMillis[mode], median(m.trials.map(t => t.results[mode].elapsedMillis)));
      assert.equal(entry.summary.maximumMillis[mode], Math.max(...m.trials.map(t => t.results[mode].elapsedMillis)));
    }
    assert.equal(entry.summary.diagnosticsVsOffPairedPercent, median(m.trials.map(t => (t.results.on.elapsedMillis / t.results.off.elapsedMillis - 1) * 100)));
    assert.equal(entry.summary.diagnosticSlowerPairs, m.trials.filter(t => t.results.on.elapsedMillis > t.results.off.elapsedMillis).length);
    const on = m.trials.map(t => t.results.on.compute), d = entry.summary.diagnosticMedians;
    assert.equal(d.gpuPassSumMillis, median(on.map(c => c.diagnostics.gpu.reduce((n, b) => n + b.shaderMillis, 0))));
    assert.equal(d.hostVerificationWindowMillis, median(on.map(c => c.verifyMillis)));
    assert.equal(d.workerVerifyCallSumMillis, median(on.map(c => c.diagnostics.verification.reduce((n, l) => n + l.verifyCallMillis, 0))));
    assert.equal(entry.summary.timestampSamples, on.reduce((n, c) => n + c.diagnostics.gpu.length, 0));
    assert.equal(entry.summary.zeroTimestampIntervals, on.flatMap(c => c.diagnostics.gpu).filter(b => b.shaderMillis === 0).length);
  }
});
