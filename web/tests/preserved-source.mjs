/** Resolve retained source bytes without rewriting any measured benchmark packet. */
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { fileURLToPath } from 'node:url';
import { resolve } from 'node:path';
const root=fileURLToPath(new URL('../../',import.meta.url));
const versions=JSON.parse(readFileSync(resolve(root,'benchmarks/source-history/manifest.json'),'utf8')).sources;
export function preservedSourcePath(original, expected){
  const version=versions[expected];
  if(!version || version.original!==original)return null;
  assert.ok(version.archived.startsWith('benchmarks/source-history/') && !version.archived.includes('..'));
  const source=readFileSync(resolve(root,version.archived),'utf8').replaceAll('\r\n','\n');
  assert.equal(createHash('sha256').update(source).digest('hex'),expected,`Retained source changed: ${original}`);
  return version.archived;
}
