import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';
import { verifierTrialOrder } from '../experiments/shared-gpu-c2c/trial-order.mjs';

const root = fileURLToPath(new URL('../../', import.meta.url));
const hash = bytes => createHash('sha256').update(bytes).digest('hex');
const bytes = name => readFileSync(resolve(root, name));
const json = name => JSON.parse(bytes(name));
const normalizedHash = name => hash(bytes(name).toString('utf8').replaceAll('\r\n', '\n'));
const previousDirectory = 'benchmarks/wasm-shared-gpu-c2c';
const freshDirectory = 'benchmarks/wasm-shared-gpu-fresh';

test('shared GPU measurement packets retain executed sources, raw reports and fresh-job boundaries', () => {
  const previous = json(`${previousDirectory}/manifest.json`), fresh = json(`${freshDirectory}/manifest.json`);
  const aliases = { ...previous.sourceAliases, ...fresh.historicalSourceAliases };
  const source = (name, expected) => {
    try { if (normalizedHash(name) === expected) return; } catch { /* Try a preserved source version. */ }
    const alias = aliases[expected];
    assert.equal(alias?.original, name, `Missing executed source: ${name}`);
    assert.ok(alias.archived.startsWith('benchmarks/') && !alias.archived.includes('..'));
    assert.equal(normalizedHash(alias.archived), expected, alias.archived);
  };
  const distanceHashes = new Map();
  for (const [directory, manifest, isFresh] of [[previousDirectory, previous, false], [freshDirectory, fresh, true]]) {
    assert.equal(manifest.productionDefaultChanged, false);
    for (const [name, expected] of Object.entries(manifest.normalizedSourceFootprint)) source(name, expected);
    assert.equal(manifest.reports.length, isFresh ? 4 : 8);
    for (const entry of manifest.reports) {
      const path = `${directory}/${entry.file}`, raw = bytes(path), report = JSON.parse(raw);
      assert.equal(raw.length, entry.bytes, path);
      assert.equal(hash(raw), entry.sha256, path);
      assert.equal(report.status, 'passed');
      assert.equal(report.purpose, 'timing-comparison');
      assert.equal(report.gpu, 'physical');
      for (const [name, expected] of Object.entries(report.sources)) source(name, expected);
      for (const build of Object.values(report.builds)) for (const [name, expected] of Object.entries(build.normalizedInputs)) {
        // Generated WASM/glue is identified by artifact hashes; it is not a tracked source.
        if (!name.startsWith('web/src/wasm/')) source(name, expected);
      }
      const result = report.measurement;
      for (const flag of ['isolated', 'distanceBitsExact', 'summaryFieldsExact', 'certificateCountsExact',
        'stableSerialBitsExact', 'releaseVerified']) assert.equal(result[flag], true, `${path}: ${flag}`);
      assert.equal(result.initialization.value.sharedInit.sharedMemory, true);
      assert.deepEqual(result.initialization.value.sharedInit.indices, [0, 1, 2, 3]);
      const identity = [result.exactF64Sha256, result.displayedF32Sha256, result.candidateSha256, result.certified];
      const previousHash = distanceHashes.get(entry.dataset);
      if (previousHash) assert.deepEqual(identity, previousHash);
      distanceHashes.set(entry.dataset, identity);
      if (!isFresh) continue;
      const inputSpec = fresh.inputs[entry.dataset];
      assert.equal(hash(bytes(`${freshDirectory}/${inputSpec.file}`)), inputSpec.sha256);
      assert.equal(report.inputManifestSha256, inputSpec.sha256);
      assert.equal(report.queryJobs, 'fresh');
      assert.equal(report.repeats, 12);
      assert.equal(report.orderScheme, 'williams-4');
      assert.equal(result.freshSetup.initialQueriesReleased, true);
      assert.equal(result.freshSetup.releasedQueryVerificationRejected, true);
      assert.equal(result.freshSetup.cachedReferenceGeneration, 1);
      let sequence = 0, generation = 0;
      for (const [name, pipeline] of Object.entries(result.measurements)) {
        assert.equal(pipeline.trials.length, 12);
        assert.equal(pipeline.warming.length, 5);
        for (const [i, trial] of pipeline.trials.entries()) assert.deepEqual(trial.order, verifierTrialOrder(5 + i));
        const median = values => {
          const sorted = values.toSorted((a, b) => a - b), middle = sorted.length / 2;
          return (sorted[middle - 1] + sorted[middle]) / 2;
        };
        for (const mode of ['pool', 'pool-staged', 'shared-serial', 'shared']) {
          assert.equal(entry.summary[name].medianMillis[mode], median(pipeline.trials.map(t => t.results[mode].elapsedMillis)));
        }
        for (const [baseline, field] of [['pool', 'sharedVsStreamingPairedPercent'], ['pool-staged', 'sharedVsStagedPairedPercent']]) {
          const paired = median(pipeline.trials.map(t => (t.results.shared.elapsedMillis / t.results[baseline].elapsedMillis - 1) * 100));
          assert.equal(entry.summary[name][field], paired);
        }
        const requests = [pipeline.cold, ...pipeline.warming.map(t => t.results), ...pipeline.trials.map(t => t.results)];
        for (const requestsInOrder of requests) for (const [mode, request] of Object.entries(requestsInOrder)) {
          assert.equal(request.freshJob, true);
          assert.equal(request.queryReleaseAcknowledged, true);
          assert.equal(request.poolReferenceReused, true);
          assert.ok(request.jobSequence > sequence); sequence = request.jobSequence;
          assert.ok(request.elapsedMillis >= request.workerMillis);
          assert.equal(request.stagedQueryBytes, mode === 'pool' ? 0 : report.inputs.compared.points * 24);
          if (mode.startsWith('shared')) {
            assert.equal(request.sharedReferenceGeneration, 1);
            assert.ok(request.sharedQueryGeneration > generation); generation = request.sharedQueryGeneration;
          }
        }
      }
    }
  }
});
