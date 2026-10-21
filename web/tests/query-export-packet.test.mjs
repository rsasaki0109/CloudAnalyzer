import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';

const root = fileURLToPath(new URL('../../', import.meta.url));
const directory = 'benchmarks/wasm-query-export';
const bytes = n => readFileSync(resolve(root, n));
const hash = b => createHash('sha256').update(b).digest('hex');
const normalizedHash = n => hash(bytes(n).toString('utf8').replaceAll('\r\n', '\n'));
const median = values => { const a = values.toSorted((x, y) => x - y); return (a[a.length / 2 - 1] + a[a.length / 2]) / 2; };

test('query-export packet retains measured candidate sources and the failed adoption gate', () => {
  const manifest = JSON.parse(bytes(`${directory}/manifest.json`));
  const later = JSON.parse(bytes('benchmarks/wasm-owned-query-export/manifest.json'));
  const bounds = JSON.parse(bytes('benchmarks/wasm-gpu-inherited-bounds/manifest.json'));
  const diagnostics = JSON.parse(bytes("benchmarks/wasm-gpu-diagnostics/manifest.json"));
  const aliases = { ...manifest.historicalSourceAliases, ...manifest.measuredSourceAliases, ...later.historicalSourceAliases, ...bounds.historicalSourceAliases, ...diagnostics.historicalSourceAliases };
  const source = (name, expected) => {
    if (normalizedHash(name) === expected) return;
    const alias = aliases[expected];
    assert.equal(alias?.original, name);
    assert.ok((alias.archived.startsWith(`${directory}/executed-source/`) || alias.archived.startsWith('benchmarks/wasm-owned-query-export/executed-source/') || alias.archived.startsWith('benchmarks/wasm-gpu-inherited-bounds/executed-source/') || alias.archived.startsWith('benchmarks/wasm-gpu-diagnostics/executed-source/')) && !alias.archived.includes('..'));
    assert.equal(normalizedHash(alias.archived), expected);
  };
  assert.equal(manifest.adopted, false);
  assert.equal(manifest.finalDefault, 'iterator');
  assert.equal(manifest.backendSelectionChanged, false);
  assert.equal(manifest.reports.length, 4);
  const footprint = manifest.normalizedSourceFootprint;
  assert.equal(hash(Object.entries(footprint).sort(([a], [b]) => a < b ? -1 : 1).map(([p, h]) => `${p}\0${h}\n`).join('')), manifest.normalizedSourceFingerprintSha256);
  for (const [name, h] of Object.entries(footprint)) source(name, h);
  for (const [name, h] of Object.entries(manifest.finalBuild.normalizedInputs)) source(name, h);
  for (const [h, alias] of Object.entries(aliases)) assert.equal(normalizedHash(alias.archived), h);
  let totalWins = 0;
  const gates = [], identities = new Map();
  for (const entry of manifest.reports) {
    const raw = bytes(`${directory}/${entry.file}`), r = JSON.parse(raw), m = r.measurement;
    assert.equal(hash(raw), entry.sha256);
    assert.equal(raw.length, entry.bytes);
    assert.equal(r.status, 'passed');
    assert.equal(r.purpose, 'timing-comparison');
    assert.equal(r.gpu, 'physical');
    assert.equal(r.threads, 4); assert.equal(r.warmups, 6); assert.equal(r.repeats, 12);
    assert.equal(r.filesUnchangedDuringMeasurement, true);
    const spec = manifest.inputs[entry.dataset], specBytes = bytes(`${directory}/${spec.file}`);
    assert.equal(hash(specBytes), spec.sha256); assert.equal(specBytes.length, spec.bytes);
    assert.equal(r.inputManifestSha256, spec.sha256);
    for (const [name, h] of Object.entries({ ...r.sources, ...r.build.normalizedInputs })) source(name, h);
    for (const flag of ['queryBitsExact', 'queryRangeGuardsChecked', 'ownerSnapshotAndExportIndependenceChecked',
      'distanceBitsExact', 'summaryFieldsExact', 'certificateCountsExact', 'sourceRemovalCompleted']) assert.equal(m[flag], true);
    assert.equal(m.defaultQuerySliceCopy, true); // Candidate was measured before restoring the iterator default.
    assert.equal(m.copiedQueryBytes, r.inputs.compared.points * 24);
    assert.deepEqual(m.inputPoints, { reference: r.inputs.reference.points, compared: r.inputs.compared.points });
    const identity = [m.queryBitsSha256, m.displayedF32Sha256, m.certified];
    if (identities.has(entry.dataset)) assert.deepEqual(identity, identities.get(entry.dataset));
    identities.set(entry.dataset, identity);
    assert.equal(m.warming.length, 6); assert.equal(m.trials.length, 12);
    for (const [i, trial] of [...m.warming, ...m.trials].entries()) {
      assert.deepEqual(trial.order, i % 2 ? ['slice', 'iterator'] : ['iterator', 'slice']);
      for (const mode of trial.order) {
        const t = trial.results[mode], c = t.compute;
        assert.ok(t.elapsedMillis >= t.workerMillis);
        assert.equal(c.querySliceCopy, mode === 'slice');
        assert.equal(c.backend, 'webgpu'); assert.equal(c.verifyWorkers, 4);
        assert.equal(c.verifyStagedQueries, false); assert.equal(c.batchSize, 131072);
        assert.equal(c.batches, Math.ceil(m.inputPoints.compared / 131072));
        for (const f of ['verifyBounds', 'verifySimd', 'prefetch', 'verifySnapshot', 'referenceReused', 'deviceReused', 'verifyReferenceReused']) assert.equal(c[f], true);
        assert.equal(c.fallback, undefined); assert.equal(c.verifyFallback, undefined);
        assert.equal(c.certified, m.certified);
      }
    }
    for (const mode of ['iterator', 'slice']) {
      assert.equal(entry.summary.medianMillis[mode], median(m.trials.map(t => t.results[mode].elapsedMillis)));
      assert.equal(entry.summary.maximumMillis[mode], Math.max(...m.trials.map(t => t.results[mode].elapsedMillis)));
    }
    const paired = median(m.trials.map(t => (t.results.slice.elapsedMillis / t.results.iterator.elapsedMillis - 1) * 100));
    const wins = m.trials.filter(t => t.results.slice.elapsedMillis < t.results.iterator.elapsedMillis).length;
    assert.equal(entry.summary.sliceVsIteratorPairedPercent, paired);
    assert.equal(entry.summary.fasterPairs, wins); totalWins += wins;
    gates.push(paired <= -1 && wins >= 7);
  }
  assert.equal(manifest.adoptionGatePassed, gates.every(Boolean));
  assert.equal(manifest.fasterPairs, totalWins);
  assert.equal(manifest.totalPairs, 48);
});
