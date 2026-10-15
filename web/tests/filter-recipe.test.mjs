import assert from 'node:assert/strict';
import {test} from 'node:test';
import {parseFilterRecipe,parseProcessingRecord,recipeParameters} from '../src/filter-recipe.ts';
const recipe={app:'CloudAnalyzer Filter Recipe',version:1,name:'Road cleanup',steps:[{op:'sor',neighbors:8,stdRatio:1},{op:'voxel',size:0.1}]};
test('portable recipes preserve ordered parameters and do not round spatial tolerances',()=>{
 assert.deepEqual(parseFilterRecipe(recipe),recipe);
 assert.deepEqual(recipe.steps.map(recipeParameters),[{op:'sor',a:8,b:1},{op:'voxel',a:0.1,b:0}]);
});
test('recipes reject unsupported or ambiguous operations and bounded parameter violations',()=>{
 for(const step of [{op:'random',percent:10},{op:'ground'},{op:'voxel',size:0},{op:'spatial',spacing:Infinity},{op:'octree',level:22},{op:'octree',level:1.5},{op:'sor',neighbors:101,stdRatio:1},{op:'sor',neighbors:8,stdRatio:-1},{op:'splat',minOpacity:2,maxSize:1},{op:'voxel',size:1,seed:3}])
  assert.throws(()=>parseFilterRecipe({...recipe,steps:[step]}));
 for(const steps of [[],Array(9).fill(recipe.steps[0])])assert.throws(()=>parseFilterRecipe({...recipe,steps}));
 assert.throws(()=>parseFilterRecipe({...recipe,version:2}));
});
test('processing provenance requires complete input and output point-record hashes',()=>{
 const p={version:1,wasmSha256:'c'.repeat(64),recipe,input:{name:'scan.ply',points:600000,sha256:'a'.repeat(64)},output:{points:100000,sha256:'b'.repeat(64)}};
 assert.deepEqual(parseProcessingRecord(p),p);
 assert.throws(()=>parseProcessingRecord({...p,input:{...p.input,sha256:'filename-only'}}));
 assert.throws(()=>parseProcessingRecord({...p,output:{...p.output,points:1.5}}));
 assert.throws(()=>parseProcessingRecord({...p,output:{...p.output,points:600001}}));
 assert.throws(()=>parseProcessingRecord({...p,wasmSha256:'unknown build'}));
 assert.throws(()=>parseProcessingRecord({...p,trusted:true}));
});
