import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
const root = fileURLToPath(new URL('../../', import.meta.url)), directory = 'benchmarks/wasm-gpu-inherited-bounds';
const bytes = name => readFileSync(resolve(root, name));
const hash = data => createHash('sha256').update(data).digest('hex');
const normalizedHash = name => hash(bytes(name).toString('utf8').replaceAll('\r\n', '\n'));
const median = values => { const a = values.toSorted((x, y) => x - y); return (a[a.length / 2 - 1] + a[a.length / 2]) / 2; };
test('inherited GPU bound packet retains exact outputs, reused WASM identity and the failed adoption gate', () => {
  const manifest = JSON.parse(bytes(`${directory}/manifest.json`));
  const diagnostics = JSON.parse(bytes("benchmarks/wasm-gpu-diagnostics/manifest.json"));
  const boxes = JSON.parse(bytes("benchmarks/wasm-gpu-box-bounds/manifest.json"));
  const parents = JSON.parse(bytes("benchmarks/wasm-gpu-parent-traversal/manifest.json"));
  const aliases = { ...manifest.historicalSourceAliases, ...manifest.measuredSourceAliases, ...diagnostics.historicalSourceAliases, ...boxes.historicalSourceAliases, ...parents.historicalSourceAliases };
  const source = (name, expected) => {
    if (normalizedHash(name) === expected) return;
    const alias = aliases[expected]; assert.equal(alias?.original, name);
    assert.ok((alias.archived.startsWith(`${directory}/executed-source/`) || alias.archived.startsWith("benchmarks/wasm-gpu-diagnostics/executed-source/") || alias.archived.startsWith("benchmarks/wasm-gpu-box-bounds/executed-source/") || alias.archived.startsWith("benchmarks/wasm-gpu-parent-traversal/executed-source/")) && !alias.archived.includes('..'));
    assert.equal(normalizedHash(alias.archived), expected);
  };
  assert.equal(manifest.adopted, false); assert.equal(manifest.finalDefault, 'local');
  assert.equal(manifest.backendSelectionChanged, false); assert.equal(manifest.reports.length, 4);
  assert.equal(manifest.protocol.timingReportBudget, 4);
  const footprint = manifest.normalizedSourceFootprint;
  assert.equal(hash(Object.entries(footprint).sort(([a], [b]) => a < b ? -1 : 1).map(([p, h]) => `${p}\0${h}\n`).join('')), manifest.normalizedSourceFingerprintSha256);
  for (const [name, h] of Object.entries(footprint)) source(name, h);
  for (const [name, h] of Object.entries(manifest.finalBuild.normalizedInputs)) source(name, h);
  for (const [h, alias] of Object.entries(aliases)) assert.equal(normalizedHash(alias.archived), h);
  const priorBytes = bytes('benchmarks/wasm-owned-query-export/manifest.json'), prior = JSON.parse(priorBytes);
  const wasmNames = ['ca_wasm.js', 'ca_wasm.d.ts', 'ca_wasm_bg.wasm', 'ca_wasm_bg.wasm.d.ts'].map(n => `web/src/wasm/${n}`);
  const identities = new Map(), gates = []; let winsTotal = 0;
  for (const entry of manifest.reports) {
    const raw = bytes(`${directory}/${entry.file}`), r = JSON.parse(raw), m = r.measurement;
    assert.equal(hash(raw), entry.sha256); assert.equal(raw.length, entry.bytes);
    assert.equal(r.status, 'passed'); assert.equal(r.purpose, 'timing-comparison'); assert.equal(r.gpu, 'physical');
    assert.equal(r.threads, 4); assert.equal(r.warmups, 6); assert.equal(r.repeats, 12); assert.equal(r.filesUnchangedDuringMeasurement, true);
    assert.equal(r.gpuInheritedBoundsComparison, true);
    const spec = manifest.inputs[entry.dataset], specBytes = bytes(`${directory}/${spec.file}`);
    assert.equal(hash(specBytes), spec.sha256); assert.equal(specBytes.length, spec.bytes); assert.equal(r.inputManifestSha256, spec.sha256);
    assert.deepEqual(r.build, manifest.finalBuild);
    assert.equal(hash(JSON.stringify(r.build, null, 2) + '\n'), r.buildInfoSha256);
    for (const [name, h] of Object.entries({ ...r.sources, ...r.build.normalizedInputs })) source(name, h);
    const reused = r.build.wasmReuse;
    assert.equal(reused.priorMetadataSha256, hash(priorBytes)); assert.equal(Object.keys(reused.unchangedArtifacts).length, 4);
    assert.deepEqual(Object.keys(reused.unchangedRustInputs).sort(), Object.keys(prior.finalBuild.normalizedInputs).filter(n => n.startsWith('rust/')).sort());
    for (const [name, h] of Object.entries(reused.unchangedRustInputs)) { source(name, h); assert.equal(prior.finalBuild.normalizedInputs[name], h); }
    for (const name of wasmNames) { assert.equal(reused.unchangedArtifacts[name], prior.finalBuild.artifacts[name]); assert.equal(r.build.artifacts[name], reused.unchangedArtifacts[name]); }
    for (const flag of ['queryBitsExact', 'queryRangeGuardsChecked', 'ownerSnapshotAndExportIndependenceChecked', 'distanceBitsExact', 'summaryFieldsExact', 'certificateCountsStablePerMode', 'sourceRemovalCompleted']) assert.equal(m[flag], true);
    assert.equal(m.defaultInheritedBounds, false); assert.equal(m.defaultValidation.compute.inheritedBounds, false);
    assert.equal(m.copiedQueryBytes, r.inputs.compared.points * 24);
    assert.deepEqual(m.inputPoints, { reference: r.inputs.reference.points, compared: r.inputs.compared.points });
    const identity = [m.queryBitsSha256, m.displayedF32Sha256, m.certifiedByMode];
    if (identities.has(entry.dataset)) assert.deepEqual(identity, identities.get(entry.dataset)); identities.set(entry.dataset, identity);
    const old = prior.reports.find(e => e.dataset === entry.dataset);
    assert.equal(m.queryBitsSha256, old.queryBitsSha256); assert.equal(m.displayedF32Sha256, old.displayedF32Sha256);
    assert.equal(entry.summary.baselineCertificatesEqualPriorPacket, m.certifiedByMode.local === old.certified);
    assert.deepEqual(entry.certifiedByMode, m.certifiedByMode);
    assert.equal(m.warming.length, 6); assert.equal(m.trials.length, 12);
    for (const [i, t] of [...m.warming, ...m.trials].entries()) {
      assert.deepEqual(t.order, i % 2 ? ['inherited', 'local'] : ['local', 'inherited']);
      for (const mode of t.order) {
        const result = t.results[mode], c = result.compute; assert.ok(result.elapsedMillis >= result.workerMillis);
        assert.equal(c.backend, 'webgpu'); assert.equal(c.inheritedBounds, mode === 'inherited');
        assert.equal(c.verifyWorkers, 4); assert.equal(c.verifyStagedQueries, false); assert.equal(c.leafOnly, false);
        assert.equal(c.workgroupSize, 64); assert.equal(c.batchSize, 131072); assert.equal(c.batches, Math.ceil(m.inputPoints.compared / 131072));
        assert.equal(c.querySliceCopy, false); assert.equal(c.queryOwnedCopy, false);
        for (const f of ['verifyBounds', 'verifySimd', 'prefetch', 'verifySnapshot', 'referenceReused', 'deviceReused', 'verifyReferenceReused']) assert.equal(c[f], true);
        assert.equal(c.fallback, undefined); assert.equal(c.verifyFallback, undefined); assert.equal(c.certified, m.certifiedByMode[mode]);
      }
    }
    for (const mode of ['local', 'inherited']) {
      assert.equal(entry.summary.medianMillis[mode], median(m.trials.map(t => t.results[mode].elapsedMillis)));
      assert.equal(entry.summary.maximumMillis[mode], Math.max(...m.trials.map(t => t.results[mode].elapsedMillis)));
    }
    const paired = median(m.trials.map(t => (t.results.inherited.elapsedMillis / t.results.local.elapsedMillis - 1) * 100));
    const wins = m.trials.filter(t => t.results.inherited.elapsedMillis < t.results.local.elapsedMillis).length;
    assert.equal(entry.summary.inheritedVsLocalPairedPercent, paired); assert.equal(entry.summary.fasterPairs, wins); winsTotal += wins; gates.push(paired <= -1 && wins >= 7);
  }
  assert.equal(manifest.adoptionGatePassed, gates.every(Boolean)); assert.equal(manifest.fasterPairs, winsTotal); assert.equal(manifest.totalPairs, 48);
});
