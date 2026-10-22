import test from 'node:test';
import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {collectionSource,collectionBefore,collectionAfter,transformCollectionSource} from '../scripts/kdtree-collection-source.mjs';
const original=readFileSync(new URL('../../'+collectionSource,import.meta.url),'utf8').replaceAll('\r\n','\n');

test('collection experiment preserves the exact baseline and changes only final point/index ownership',()=>{
 assert.equal(transformCollectionSource(original,'baseline'),original);
 const changed=transformCollectionSource(original,'unzip');assert.equal(changed.replace(collectionAfter,collectionBefore),original);
 assert.equal(transformCollectionSource(original.replaceAll('\n','\r\n'),'unzip'),changed);
 assert.ok(changed.includes('items.into_iter().unzip()'));
});

test('collection experiment rejects ambiguous or changed source and unknown variants',()=>{
 assert.throws(()=>transformCollectionSource(original+collectionBefore,'unzip'),/source changed/);
 assert.throws(()=>transformCollectionSource(original.replace(collectionBefore,'new implementation'),'baseline'),/source changed/);
 assert.throws(()=>transformCollectionSource(original,'other'),/Unknown/);
});
