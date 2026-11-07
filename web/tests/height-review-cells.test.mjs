import { test } from 'node:test';
import assert from 'node:assert/strict';
import { screenedCellBounds, screenedCellsCsv } from '../src/height-review-cells.ts';
test('focus bounds use the full world-XY cell square and both height endpoints without f32 rounding',()=>{
 const c={index:8,x:1000000.25,y:2000000.5,before:10,after:8.000000000000002,difference:-1.9999999999999982};
 assert.deepEqual(screenedCellBounds(c,2),[999999.25,1999999.5,8.000000000000002,1000001.25,2000001.5,10]);
 assert.throws(()=>screenedCellBounds(c,0));assert.throws(()=>screenedCellBounds({...c,x:Infinity},1));
});
test('listed-cell CSV preserves original f64 values, direction, row-major identity and unlisted counts with safe labels',()=>{
 const review={threshold:1,riseCells:51,fallCells:0,withinCells:1,unlistedCells:1,cells:[{index:2,x:1,y:2,before:0,after:1.0000000000000002,difference:1.0000000000000002}]};
 const csv=screenedCellsCsv(review,1,'m','  =Survey, "A"');
 assert.ok(csv.includes('"\'  =Survey, ""A""","m",1,1,1,2,1,2,0,1.0000000000000002,1.0000000000000002,rise,1'));
 assert.equal(screenedCellsCsv({...review,cells:[]},1,'m','A').trimEnd().split('\n').length,1);
 assert.equal(review.cells[0].difference,1.0000000000000002);
});
