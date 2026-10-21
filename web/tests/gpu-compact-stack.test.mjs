import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { stripTypeScriptTypes } from 'node:module';

const shader = async path => {
  const ts = readFileSync(new URL(path, import.meta.url), 'utf8').replaceAll('\r\n', '\n');
  const source = stripTypeScriptTypes(ts + '\nexport { SHADER };', { mode: 'transform' });
  return (await import('data:text/javascript;base64,' + Buffer.from(source).toString('base64'))).SHADER;
};

test('compact GPU shader preserves the archived control and changes only three ordinary DFS capacity literals', async () => {
  const make = await shader('../src/gpu-candidates.ts');
  const prior = await shader('../../benchmarks/wasm-gpu-compact-stack/executed-source/pr344/web/src/gpu-candidates.ts');
  assert.equal(make(32), prior);
  const compact = prior.replace('var stack: array<u32, 32>;', 'var stack: array<u32, 20>;')
    .replace('var lower: array<f32, 32>;', 'var lower: array<f32, 20>;')
    .replace('pending + 2u > 32u', 'pending + 2u > 20u');
  assert.equal(make(20), compact);
  assert.equal(make(20).slice(make(20).indexOf('fn nearestBox')), prior.slice(prior.indexOf('fn nearestBox')));
});
