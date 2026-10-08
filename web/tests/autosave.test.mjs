import assert from 'node:assert/strict';
import { test } from 'node:test';
import { Autosave } from '../src/autosave.ts';

const deferred=()=>{let resolve,reject;const promise=new Promise((a,b)=>{resolve=a;reject=b;});return {promise,resolve,reject};};
function setup(options={}) {
  const writes=[];
  const saver=new Autosave({delay:60_000,capture:async()=>({value:1}),write:async value=>{writes.push(value);},idle:()=>true,render:()=>{},...options});
  return {saver,writes};
}
test('previous recovery stays protected, active operations defer capture and exports acknowledge only their revision', async () => {
  let idle=false;
  const {saver,writes}=setup({idle:()=>idle});
  try {
    saver.changed(); await saver.save(); assert.equal(writes.length,0);
    saver.paused=false; await saver.save(); assert.equal(writes.length,0);
    saver.exported(1); assert.equal(saver.unsaved,false);
    saver.changed(); saver.exported(1); assert.equal(saver.unsaved,true);
    idle=true; await saver.save(); assert.equal(writes.length,1); assert.equal(saver.unsaved,false);
  } finally { saver.dispose(); }
});
test('resuming does not acknowledge workspace changes absent from the recovered copy', async () => {
  const {saver,writes}=setup();
  try {
    saver.changed(); saver.recovered();
    assert.equal(saver.paused,false); assert.equal(saver.unsaved,true);
    await saver.save(); assert.equal(writes.length,1); assert.equal(saver.unsaved,false);
  } finally { saver.dispose(); }
});
test('an edit cancels source fingerprinting and prevents a stale capture from replacing the browser copy', async () => {
  const capture=deferred(); let signal;
  const {saver,writes}=setup({capture:s=>{signal=s;return capture.promise;}});
  try {
    saver.paused=false; saver.changed(); const saving=saver.save();
    saver.changed(); assert.equal(signal.aborted,true);
    capture.resolve({value:'stale'}); await saving;
    assert.equal(writes.length,0); assert.equal(saver.unsaved,true); assert.equal(saver.error,'');
  } finally { saver.dispose(); }
});
test('an edit during storage commit stays unsaved and writes are never overlapped', async () => {
  const commit=deferred(), written=[];
  const {saver}=setup({write:async value=>{written.push(value);await commit.promise;}});
  try {
    saver.paused=false; saver.changed(); const saving=saver.save();
    await Promise.resolve(); saver.changed(); await saver.save();
    assert.equal(written.length,1);
    commit.resolve(); await saving;
    assert.equal(saver.savedRevision,1); assert.equal(saver.unsaved,true);
    await saver.save(); assert.equal(written.length,2); assert.equal(saver.unsaved,false);
  } finally { saver.dispose(); }
});
test('a failed commit preserves the previous saved revision, keeps the unload warning and can retry', async () => {
  let fail=false;
  const {saver}=setup({write:async()=>{if(fail)throw new Error('Quota exceeded');}});
  try {
    saver.paused=false; saver.changed(); await saver.save();
    fail=true; saver.changed(); await saver.save();
    assert.equal(saver.savedRevision,1); assert.equal(saver.unsaved,true); assert.equal(saver.error,'Quota exceeded');
    fail=false; saver.retry(); await saver.save();
    assert.equal(saver.savedRevision,2); assert.equal(saver.unsaved,false);
  } finally { saver.dispose(); }
});
