/** Real-data workflow evidence, not a labeled change-detection accuracy benchmark. */
import { expect, test, type Page } from '@playwright/test';
import { writeFile, readFile } from 'node:fs/promises';
import { createHash } from 'node:crypto';
import { overlappingSurveyLanes, type SurveyLane } from '../src/lane-surface-overlap';

async function download(page: Page, button: string, filename?: string) {
  const ready = page.waitForEvent('download', d => !filename || d.suggestedFilename() === filename);
  await page.locator(button).click();
  const chunks: Buffer[] = [];
  for await (const chunk of await (await ready).createReadStream()) chunks.push(chunk as Buffer);
  return Buffer.concat(chunks);
}
async function project(page: Page) { return JSON.parse((await download(page, '#project-save', 'project.cloudanalyzer.json')).toString()); }
function lanes(map: any): SurveyLane[] {
  const curves = new Map<number, [number,number,number][]>(map.boundaries.map((b: any) => [b.id, b.geometry]));
  const curve = (ref: number | {id:number;reversed?:boolean}) => {
    const points = curves.get(typeof ref === 'number' ? ref : ref.id)!;
    return typeof ref !== 'number' && ref.reversed ? [...points].reverse() : points;
  };
  return map.lanes.map((lane: any) => ({id: lane.id, left: curve(lane.left), right: curve(lane.right)}));
}
test.use({actionTimeout:30_000});
const hash = (value: Buffer | string) => createHash('sha256').update(value).digest('hex');

test('NCLT April/June comparison reaches Lanelet2 review and a saved follow-up task', async ({page}, info) => {
  test.skip(process.env.CLOUDANALYZER_REAL_DATA !== '1', 'Opt in with CLOUDANALYZER_REAL_DATA=1; uses bundled NCLT excerpts.');
  test.setTimeout(900_000);
  const start = Date.now(), errors: string[] = [];
  const report: any = { schema:'cloudanalyzer.nclt_survey_workflow.v1', startedAt:new Date().toISOString(), stages:[],
    scope:'Automated real-data workflow only. Draft lane layout is a configured hypothesis, not an independently surveyed map. Physical change labels, false positives, recall, manual search times and usability improvements are not measured.',
    sources:[], settings:{mapLayout:{forwardLanes:1,backwardLanes:0,widthM:3.5,pieceLengthM:50,speedKmh:40,traffic:'right',semanticStatus:'unverified configured assumptions'},overlap:{xyMarginM:0,heightMarginM:1}},
    groundTruth:{independentLanelet2:false,labeledChanges:false,manualParticipant:false}};
  page.on('pageerror', e => errors.push(e.message));
  const save = async (name:string, data:Buffer|string) => { await writeFile(info.outputPath(name), data); };
  const stage = (name:string, data:object={}) => report.stages.push({name, elapsedMs:Date.now()-start, ...data});
  try {
    for (const name of ['nclt-2012-04-29.mcap','nclt-2012-06-15.mcap']) {
      const data=await readFile(new URL(`../public/samples/${name}`,import.meta.url));
      report.sources.push({name,bytes:data.length,sha256:hash(data),data:'University of Michigan NCLT, 190-second thinned excerpt; ODbL/DBCL'});
    }
    await page.addInitScript(() => localStorage.setItem('cloudanalyzer-autosave-enabled','off'));
    await page.goto('/?demo=nclt-seasons');
    await expect(page.locator('#pg-changes')).toBeVisible({timeout:720_000});
    await expect(page.locator('#pg-change-export')).toBeEnabled({timeout:60_000});
    const reviewBytes=await download(page,'#pg-change-export'), review=JSON.parse(reviewBytes.toString());
    await save('survey-change-review.initial.json',reviewBytes);
    expect(review.context.sessions).toHaveLength(2); expect(review.candidates.length).toBeGreaterThan(0);
    report.comparison={identity:review.identity,context:review.context,diagnostics:review.diagnostics,candidateCount:review.candidates.length,
      largestCandidates:review.candidates.slice(0,10),initialReviewSha256:hash(reviewBytes)};
    stage('two_pass_comparison_ready',{candidates:review.candidates.length});
    const resultRow=page.locator('#cloud-list > li').filter({hasText:review.sources.difference.name});
    await resultRow.locator('button[title="Save with distances…"]').click();
    const resultReady=page.waitForEvent('download');
    await resultRow.locator('.save-formats button',{hasText:'PLY'}).click();
    const resultChunks:Buffer[]=[];
    for await (const chunk of await (await resultReady).createReadStream()) resultChunks.push(chunk as Buffer);
    const resultPly=Buffer.concat(resultChunks); await save('m3c2-result.ply',resultPly);
    report.resultPly={bytes:resultPly.length,sha256:hash(resultPly)};
    await page.screenshot({path:info.outputPath('comparison-overview.png')});
    const poses=await download(page,'#pg-save-kitti'), rows=poses.toString().trim().split('\n');
    const aprilCount=review.context.sessions[0].count;
    const april=Buffer.from(rows.slice(0,aprilCount).join('\n')+'\n');
    await save('april-corrected.kitti',april);
    await page.locator('#file-input').setInputFiles({name:'april-corrected.kitti',mimeType:'text/plain',buffer:april});
    await expect(page.locator('#vm-trajectory option')).toHaveCount(1);
    await page.locator('#vm-road-options summary').click();
    await page.locator('#vm-forward').fill('1'); await page.locator('#vm-backward').fill('0');
    await page.locator('#vm-width').fill('3.5'); await page.locator('#vm-segment').fill('50');
    await page.locator('#vm-traffic').selectOption('right');
    await page.getByText('Build from a trajectory',{exact:true}).click();
    await page.locator('#vm-cloud').selectOption({label:review.sources.before.name});
    await page.locator('#vm-discover-after-build').uncheck();
    const building=Date.now(); await page.locator('#vm-build').click();
    await expect(page.locator('#status')).toContainText(/Draft roads added|Could not build draft roads/,{timeout:120_000});
    report.mapBuild={millis:Date.now()-building,status:await page.locator('#status').textContent(),report:await page.locator('#vm-build-report').textContent()};
    expect(report.mapBuild.status).toContain('Draft roads added');
    await page.locator('#vm-quality summary').click();
    await page.locator('#vm-quality-cloud').selectOption({label:review.sources.before.name});
    await page.locator('#vm-quality-check').click();
    await expect(page.locator('#status')).toContainText('Source coverage checked',{timeout:120_000});
    report.sourceQuality={summary:await page.locator('#vm-quality-report').textContent(),
      failedLanes:await page.locator('#vm-quality-lanes button').allTextContents()};
    const generated=await project(page), generatedMap=JSON.parse(generated.vectorMap);
    await save('generated-map.json',generated.vectorMap);
    report.mapBuild.lanes=generatedMap.lanes.length; stage('same_frame_map_draft_built');
    const yamlReady=page.waitForEvent('download',d=>d.suggestedFilename()==='map_projector_info.yaml');
    const osm=await download(page,'#vm-export','lanelet2_map.osm'), yamlChunks:Buffer[]=[];
    for await (const chunk of await (await yamlReady).createReadStream()) yamlChunks.push(chunk as Buffer);
    const yaml=Buffer.concat(yamlChunks); await save('lanelet2_map.osm',osm); await save('map_projector_info.yaml',yaml);
    report.lanelet2={bytes:osm.length,sha256:hash(osm),projector:yaml.toString(),warnings:await page.locator('#vm-export-issues').textContent()};
    await page.locator('#vm-file').setInputFiles({name:'lanelet2_map.osm',mimeType:'application/xml',buffer:osm});
    await expect(page.locator('#status')).toContainText('Opened lanelet2_map.osm');
    const reopened=await project(page), map=JSON.parse(reopened.vectorMap), roadLanes=lanes(map);
    await save('reopened-map.json',reopened.vectorMap);
    // Exhaustively enumerate box/surface associations. This measures geometric overlap, not correctness labels.
    const associations=review.candidates.map((candidate:any)=>({rank:candidate.rank,...overlappingSurveyLanes(roadLanes,candidate.bounds,0,1)}));
    await save('candidate-lane-overlaps.json',JSON.stringify(associations,null,2));
    report.associations={candidateCount:associations.length,withOverlap:associations.filter((a:any)=>a.ids.length>0).length,
      unchecked:associations.reduce((n:number,a:any)=>n+a.limited+a.invalid,0),selected:associations.find((a:any)=>a.ids.length>0)};
    expect(report.associations.selected).toBeTruthy();
    const selected=report.associations.selected, rank=selected.rank, laneId=selected.ids[0];
    for(let i=0;i<Math.floor((rank-1)/50);i++) await page.locator('#pg-change-next').click();
    await page.locator(`#pg-change-list button[data-rank="${rank}"]`).click();
    await page.locator('#pg-change-decision').selectOption('deferred');
    await page.locator('#pg-change-note').fill('Real NCLT April/June data. Candidate remains unverified: registration error, observation differences and physical change require source review. Lane map is a generated hypothesis.');
    for(const view of ['before','after','difference']) {
      await page.locator('#pg-change-view').selectOption(view);
      await page.screenshot({path:info.outputPath(`candidate-${rank}-${view}.png`)});
    }
    await page.locator('#pg-change-lanes summary').click();
    await page.locator('#pg-change-lane-frame').check();
    await expect(page.getByRole('button',{name:`Inspect lane ${laneId}`,exact:true})).toBeVisible();
    await page.getByRole('button',{name:`Inspect lane ${laneId}`,exact:true}).click();
    await page.getByRole('button',{name:`Record lane ${laneId}`,exact:true}).click();
    await expect(page.locator('#pg-lane-task-export')).toBeEnabled({timeout:60_000});
    await page.locator('#pg-lane-task-state').selectOption('deferred');
    await page.locator('#pg-lane-task-note').fill('Geometric overlap only. Verify source observations and registration before deciding whether this generated lane draft needs an update.');
    await page.locator('#pg-lane-task-save').click();
    const tasksBytes=await download(page,'#pg-lane-task-export'), tasks=JSON.parse(tasksBytes.toString());
    await save('survey-lane-tasks.json',tasksBytes);
    expect(tasks.tasks).toHaveLength(1); expect(tasks.tasks[0].lane.id).toBe(laneId); expect(tasks.tasks[0].candidate.rank).toBe(rank);
    expect(tasks.tasks[0].mapSha256).toBe(hash(reopened.vectorMap));
    expect(tasks.tasks[0].staleReason).toBeNull();
    expect(tasks.tasks[0].coreSupport).toEqual({protocol:'xy-lane-z-v1',status:'complete',total:655,supported:77,heightMarginM:1,reason:null});
    const finalReview=await download(page,'#pg-change-export'); await save('survey-change-review.final.json',finalReview);
    const after=await project(page); expect(after.vectorMap).toBe(reopened.vectorMap);
    expect(after.poseGraph).toEqual(reopened.poseGraph);
    report.task={bytes:tasksBytes.length,sha256:hash(tasksBytes),record:tasks.tasks[0]};
    report.selection={rank,laneId,candidate:review.candidates[rank-1]};
    await page.screenshot({path:info.outputPath('saved-follow-up.png')});
    stage('follow_up_saved_without_map_or_graph_mutation');
    await page.locator('#pg-lane-task-file').setInputFiles({name:'survey-lane-tasks.json',mimeType:'application/json',buffer:tasksBytes});
    await expect(page.locator('#pg-lane-task-archive-name')).toContainText('1 historical');
    await page.locator('#pg-lane-task-archive-list summary').first().click();
    await expect(page.locator('#pg-lane-task-archive-list')).toContainText('Saved core support: 77 / 655');
    stage('task_archive_reopened');
    const independent=JSON.parse(await readFile(new URL('../../benchmarks/vector-map/nclt-survey-change-review/point-support.json',import.meta.url),'utf8'));
    const browserPairs:any[]=[];let activePage=Math.floor((rank-1)/50);
    for(const candidateRank of [...new Set<number>(independent.pairs.map((pair:any)=>pair.candidate))]) {
      const wanted=Math.floor((candidateRank-1)/50);
      while(activePage<wanted){await page.locator('#pg-change-next').click();activePage++;}
      while(activePage>wanted){await page.locator('#pg-change-prev').click();activePage--;}
      await page.locator(`#pg-change-list button[data-rank="${candidateRank}"]`).click();
      for(const pair of independent.pairs.filter((pair:any)=>pair.candidate===candidateRank)) {
        const metric=page.locator(`[data-core-lane="${pair.lane}"]`);
        await expect(metric).toHaveAttribute('data-core-status','complete',{timeout:30_000});
        const supported=Number(await metric.getAttribute('data-core-supported'));
        const total=Number(await metric.getAttribute('data-core-total'));
        expect(supported).toBe(pair.coresInsideLaneAtHeightMargin1m);expect(total).toBe(pair.candidateCorePoints);
        browserPairs.push({candidate:candidateRank,lane:pair.lane,supported,total});
        if(candidateRank===rank&&pair.lane===laneId) {
          await page.getByRole('button',{name:`Show changed cores for lane ${laneId}`,exact:true}).click();
          await expect(page.locator('#pg-core-overlay-hint')).toContainText(`${supported} of ${total}`);
          await page.screenshot({path:info.outputPath('native-changed-core-support.png')});
        }
      }
    }
    report.browserCoreSupport={protocol:'xy-lane-z-v1',comparisonIdentity:review.identity,mapSha256:hash(reopened.vectorMap),pairCount:browserPairs.length,zeroSupportPairs:browserPairs.filter(pair=>pair.supported===0).length,
      allMatchIndependent:true,pairs:browserPairs};
    expect(browserPairs).toHaveLength(independent.boxLanePairs);
    expect(report.browserCoreSupport.zeroSupportPairs).toBe(independent.pairsWithNoCore);
    await save('browser-core-support.json',JSON.stringify(report.browserCoreSupport,null,2));
    stage('browser_support_matches_all_independent_pairs');
    expect(errors).toEqual([]); report.outcome='passed';
  } catch(error) { report.outcome='failed'; report.failure=String(error); throw error; }
  finally { report.browserErrors=errors; report.totalMs=Date.now()-start; await save('verification.json',JSON.stringify(report,null,2)); }
});


test('NCLT identical-recording control checks for above-threshold change candidates', async ({page},info) => {
  test.skip(process.env.CLOUDANALYZER_REAL_DATA !== '1','Opt in with CLOUDANALYZER_REAL_DATA=1; NCLT attribution applies.');
  test.setTimeout(900_000);
  const start=Date.now(), report:any={scope:'Identical observations processed twice: pipeline self-consistency only, not an independent accuracy or natural-change benchmark.'};
  try {
    await page.addInitScript(()=>localStorage.setItem('cloudanalyzer-autosave-enabled','off'));
    await page.goto('/');
    const path=new URL('../public/samples/nclt-2012-04-29.mcap',import.meta.url).pathname;
    await page.locator('#pg-files-input').setInputFiles(path);
    await expect(page.locator('#pg-stats')).toContainText('199 (199 with scans)',{timeout:540_000});
    await page.locator('#pose-graph-panel summary',{hasText:'Join another graph'}).click();
    await expect(page.locator('#pg-merge-files')).toBeEnabled({timeout:120_000});
    report.initialGravityStatus=await page.locator('#status').textContent();
    await page.locator('#pg-merge-files-input').setInputFiles(path);
    await expect(page.locator('#pg-stats')).toContainText('398 (398 with scans)',{timeout:180_000});
    await expect(page.locator('#pg-merge-files')).toBeEnabled({timeout:120_000});
    report.mergedGravityStatus=await page.locator('#status').textContent();
    await page.locator('#pose-graph-panel summary',{hasText:'Find loops automatically'}).click();
    await page.locator('#pg-find').click();
    await expect(page.locator('#pg-find')).toBeEnabled({timeout:180_000});
    report.loopStatus=await page.locator('#status').textContent();
    await page.locator('#pg-parts').click();
    await expect(page.locator('#pg-change-export')).toBeEnabled({timeout:180_000});
    const bytes=await download(page,'#pg-change-export'),review=JSON.parse(bytes.toString());
    await writeFile(info.outputPath('identical-recording-review.json'),bytes);
    report.candidates=review.candidates.length;report.diagnostics=review.diagnostics;report.context=review.context;
    report.reviewSha256=hash(bytes);report.outcome=review.candidates.length===0?'zero_candidates':'unexpected_candidates';
    expect(review.candidates).toHaveLength(0);
  } catch(error) {report.failure=String(error);throw error;}
  finally {report.totalMs=Date.now()-start;await writeFile(info.outputPath('control.json'),JSON.stringify(report,null,2));}
});
