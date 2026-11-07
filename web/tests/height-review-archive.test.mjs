import { test } from 'node:test';
import assert from 'node:assert/strict';
import { startHeightCellReview, updateHeightCellReview, heightCellReviewSnapshot } from '../src/height-cell-review.ts';
import { parseHeightReviewArchive, heightArchiveCsv, heightArchiveHtml, HEIGHT_ARCHIVE_COMPARISONS } from '../src/height-review-archive.ts';
const time='2026-10-24T00:00:00.000Z';
function fixture() {
 const input={id:1,name:'before.ply',kind:'cloud',loadedPoints:3,filePoints:6,keepEvery:2,preview:true,copcLevels:null};
 const epsilon=1+Number.EPSILON;
 const review={threshold:1,riseCells:1,fallCells:1,withinCells:1,riseArea:4,fallArea:4,riseVolume:epsilon*4,fallVolume:12,cellLimit:50,unlistedCells:0,cells:[{index:1,x:1000002,y:2000000,before:0,after:-3,difference:-3},{index:0,x:1000000,y:2000000,before:0,after:epsilon,difference:epsilon}]};
 const session=startHeightCellReview(review,{label:'  =Survey <img src=x>',before:input,after:{...input,id:2,name:'after.ply'},settings:{cell:2,height:'mean',coordinateUnit:'m',fillEmpty:false,heightThreshold:1,region:[999999,1999999,1000005,2000001]},support:{mode:'common_observed',sha256:'a'.repeat(64),commonGridCells:3}},time);
 updateHeightCellReview(session,0,'deferred','  +Confirm\n"再確認" <script>alert(1)</script>',time);
 return heightCellReviewSnapshot([session]);
}
test('archive normalizes actual review producer output without modifying records or rounding original f64 cells',()=>{
 const value=fixture(), sessions=parseHeightReviewArchive(JSON.parse(JSON.stringify(value)));
 assert.deepEqual(heightCellReviewSnapshot(sessions),value);assert.equal(sessions[0].screening.cells[1].difference,1+Number.EPSILON);
 sessions[0].decisions[0].note='changed';sessions[0].context.settings.region[0]=0;assert.ok(value.comparisons[0].cells[0].note.startsWith('  +Confirm'));assert.equal(value.comparisons[0].settings.region[0],999999);
});
test('inconsistent metadata, geometry, ranks, decisions and declared coverage cannot be treated as a valid archive',()=>{
 const mutations=[
  v=>v.schema='other',v=>v.comparisons={},v=>v.comparisons=Array(HEIGHT_ARCHIVE_COMPARISONS+1).fill(v.comparisons[0]),
  v=>v.comparisons[0].openedAt='yesterday',v=>v.comparisons[0].before.preview='true',v=>v.comparisons[0].after.kind='object',v=>v.comparisons[0].before.filePoints=1,
  v=>v.comparisons[0].settings.cell=0,v=>v.comparisons[0].settings.heightThreshold=0,v=>v.comparisons[0].settings.height='median',v=>v.comparisons[0].settings.region=null,v=>v.comparisons[0].settings.region=[2,0,1,1],
  v=>v.comparisons[0].settings.fillEmpty=true,v=>v.comparisons[0].support.sha256='wrong',v=>v.comparisons[0].support.commonGridCells=1,v=>v.comparisons[0].support.mode='guessed',
  v=>v.comparisons[0].screening.riseArea=5,v=>v.comparisons[0].screening.riseVolume=1,v=>v.comparisons[0].screening.cellLimit=51,v=>v.comparisons[0].screening.unlistedCells=1,v=>v.comparisons[0].screening.withinCells=1000000,
  v=>v.comparisons[0].cells[0].difference=-2,v=>v.comparisons[0].cells[0].x=1000006,v=>v.comparisons[0].cells[0].index=1000000,v=>v.comparisons[0].cells[1].index=1,
  v=>v.comparisons[0].cells.reverse(),v=>v.comparisons[0].cells[0].listedRank=0,v=>v.comparisons[0].cells[0].after=Infinity,
  v=>v.comparisons[0].cells[0].updatedAt=null,v=>v.comparisons[0].cells[1].status='reviewed',v=>v.comparisons[0].cells[0].note='x'.repeat(2001),v=>v.comparisons[0].counts.deferred=0,
 ];
 for(const mutate of mutations){const value=fixture();mutate(value);assert.throws(()=>parseHeightReviewArchive(value),/inconsistent/);}
});
test('bounded lists retain unlisted support without pretending to verify unlisted cell geometry or source files',()=>{
 const v=fixture(),c=v.comparisons[0];c.support={mode:'independent'};delete c.settings.region;
 c.screening={threshold:1,riseCells:70,fallCells:0,withinCells:0,riseArea:280,fallArea:0,riseVolume:560,fallVolume:0,cellLimit:50,unlistedCells:20};
 c.cells=Array.from({length:50},(_,i)=>({listedRank:i+1,index:i,x:i*2,y:0,before:0,after:2,difference:2,status:'unreviewed',note:'',updatedAt:null}));c.counts={unreviewed:50,reviewed:0,deferred:0};
 const s=parseHeightReviewArchive(v);assert.equal(s[0].screening.unlistedCells,20);assert.equal(s[0].decisions.length,50);
 c.cells[2].index=1;assert.throws(()=>parseHeightReviewArchive(v));
});
test('independent constant planes, meshes and empty archives are allowed without inventing observed support',()=>{
 const value=fixture(),c=value.comparisons[0];c.support={mode:'independent'};c.before={constantHeight:0};c.after.kind='mesh';c.settings.fillEmpty=true;
 assert.deepEqual(parseHeightReviewArchive(value)[0].context.before,{constantHeight:0});
 c.support={mode:'common_observed',sha256:'a'.repeat(64),commonGridCells:3};assert.throws(()=>parseHeightReviewArchive(value));
 assert.deepEqual(parseHeightReviewArchive({schema:value.schema,comparisons:[]}),[]);
});
test('archive HTML is standalone escaped text with normalized JSON and CSV preserves safe notes and exact values',()=>{
 const s=parseHeightReviewArchive(fixture()),html=heightArchiveHtml(s),csv=heightArchiveCsv(s);
 assert.ok(html.includes('&lt;script&gt;alert(1)&lt;/script&gt;'));assert.ok(html.includes('Complete normalized JSON snapshot'));assert.ok(html.includes('Historical file only'));
 assert.ok(!/<script|<img|https?:\/\//i.test(html));assert.ok(csv.includes('"\'  +Confirm\n""再確認"" <script>alert(1)</script>"'));assert.ok(csv.includes('1.0000000000000002'));
});
