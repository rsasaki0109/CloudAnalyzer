// Optional source-only runtime proof, with no surveyed-map input.
import {expect,test} from "@playwright/test";
import {createHash} from "node:crypto";
import {mkdirSync,readFileSync,writeFileSync} from "node:fs";
import {fileURLToPath} from "node:url";
const SOURCE=process.env.VECTOR_MAP_ANCHOR_SOURCE;
const PROOF=process.env.VECTOR_MAP_ANCHOR_PROOF;
const ALIGN=process.env.VECTOR_MAP_CURB_ALIGNMENT === "1";
const DIVIDER=process.env.VECTOR_MAP_PAINT_DIVIDER === "1";
const EDGES=process.env.VECTOR_MAP_LANE_EDGES === "1";
const PAINT=process.env.VECTOR_MAP_PAINT_CORRIDOR === "1";
const OUTPUT=fileURLToPath(new URL(EDGES ? "../media-frames/vector-map-lane-edges/" : DIVIDER ? "../media-frames/vector-map-paint-divider/" : PAINT ? "../media-frames/vector-map-paint-corridor/" : ALIGN ? "../media-frames/vector-map-curb-alignment/" : "../media-frames/vector-map-physical-anchors/",import.meta.url));
test.use({viewport:{width:1440,height:900},actionTimeout:30000});
test("actual planning points build a physical-anchor draft matching native geometry and Undo",async({page})=>{
  test.skip(!SOURCE||!PROOF,"Set cached source and frozen native proof paths");
  test.setTimeout(180000);mkdirSync(OUTPUT,{recursive:true});
  const native=JSON.parse(readFileSync(`${PROOF}/after-0.json`,"utf8"));
  const audit=JSON.parse(readFileSync(`${PROOF}/after-0-audit.json`,"utf8"));
  const errors:string[]=[];page.on("pageerror",e=>errors.push(String(e)));
  await page.goto("/");await page.locator("#file-input").setInputFiles(SOURCE!);
  await expect(page.locator("#status")).toContainText("Loaded",{timeout:120000});
  await expect(page.locator("#vm-status")).toContainText("No map yet");
  await page.locator("#file-input").setInputFiles(`${PROOF}/trajectory-0.csv`);
  await expect(page.locator("#status")).toContainText("trajectory of 2 poses");
  if(!await page.locator("#vm-segment").isVisible())await page.locator("#vector-map-panel").getByText("Road options",{exact:true}).click();
  await page.locator("#vm-segment").fill("0");
  if(!await page.locator("#vm-physical-anchors").isVisible())await page.locator("#vector-map-panel").getByText("Build from a trajectory",{exact:true}).click();
  await expect(page.locator("#vm-physical-anchors")).not.toBeChecked();
  await page.locator("#vm-physical-anchors").check();
  await expect(page.locator("#vm-align-curbs")).not.toBeChecked();
  if(ALIGN)await page.locator("#vm-align-curbs").check();
  await expect(page.locator("#vm-paint-corridor")).not.toBeChecked();
  if(PAINT)await page.locator("#vm-paint-corridor").check();
  await expect(page.locator("#vm-paint-divider")).not.toBeChecked();
  if(DIVIDER)await page.locator("#vm-paint-divider").check();
  await expect(page.locator("#vm-lane-edges")).not.toBeChecked();
  if(EDGES)await page.locator("#vm-lane-edges").check();
  await page.locator("#vm-source-surface").check();await page.locator("#vm-discover-after-build").uncheck();
  await page.locator("#vm-build").click();await expect(page.locator("#status")).toContainText("Draft roads added",{timeout:120000});
  await expect(page.locator("#vm-status")).toContainText(`${native.lanes.length} lanes`);
  if(ALIGN)await expect(page.locator("#vm-build-report")).toContainText(`Trace alignment ${audit.extraction.trace_alignment.applied ? "applied" : "held"}`);
  if(PAINT)await expect(page.locator("#vm-build-report")).toContainText(`White paint fit ${audit.extraction.paint_corridor.applied ? "applied" : "held"}`);
  if(DIVIDER)await expect(page.locator("#vm-build-report")).toContainText("Interior paint correction applied");
  if(EDGES)await expect(page.locator("#vm-build-report")).toContainText("Outer lane-edge inference applied");
  await expect(page.locator("#vm-build-report")).toContainText(`Coverage-edge anchor candidates ignored: ${audit.extraction.coverage_edge_anchor_candidates_ignored}.`);
  await page.locator("#vm-quality summary").click();await page.locator("#vm-quality-check").click();
  await expect(page.locator("#vm-quality-report")).toContainText(`${native.lanes.length} lanes checked; 0 need source review; 0 omitted`,{timeout:120000});
  const wait=page.waitForEvent("download",d=>d.suggestedFilename()==="lanelet2_map.osm");await page.locator("#vm-export").click();
  const xml=readFileSync((await(await wait).path())!,"utf8");
  const xyz=[...xml.matchAll(/<node id="[^"]+"[^>]*>([\s\S]*?)<\/node>/g)].map(m=>["local_x","local_y","ele"].map(k=>Number(m[1].match(new RegExp(`<tag k="${k}" v="([^"]+)"`))![1])));
  const positions=native.boundaries.flatMap((b:any)=>b.geometry);
  const maximum=Math.max(...positions.map((p:number[])=>Math.min(...xyz.map(q=>Math.hypot(...p.map((v,i)=>v-q[i]))))));
  expect(maximum).toBeLessThan(1e-6);
  expect(xml.match(/<tag k="subtype" v="road"\/>/g)).toHaveLength(native.lanes.length);
  await page.locator('[data-view="top"]').click();await page.locator("#fit").click();await page.locator("#edl").uncheck();
  if(PAINT||DIVIDER)await page.locator("#vm-fit").click();
  await page.screenshot({path:`${OUTPUT}/source-build.png`});
  await page.locator("#vm-undo").click();await expect(page.locator("#vm-status")).toContainText("No map yet");
  expect(errors).toEqual([]);
  writeFileSync(`${OUTPUT}/verification.json`,JSON.stringify({sourcePoints:1757841,generationReferenceInputs:[],physicalAnchorsOnly:true,
    laneEdgeInference:EDGES ? audit.extraction.lane_edge_inference : null,traceAlignment:ALIGN ? audit.extraction.trace_alignment : null,paintDivider:DIVIDER ? audit.extraction.paint_divider : null,paintCorridor:PAINT ? audit.extraction.paint_corridor : null,defaultOff:true,lanes:native.lanes.length,fullSourceAudit:true,nativeBoundaryVertices:positions.length,
    maximumNativeVertexToExportedNodeDistanceM:maximum,comparisonRole:"Boundary vertices to exported nodes, not complete topology or byte equality",
    nativeMapSha256:createHash("sha256").update(readFileSync(`${PROOF}/after-0.json`)).digest("hex"),
    exportedOsmSha256:createHash("sha256").update(xml).digest("hex"),oneStepUndo:true,pageErrors:errors},null,2)+"\n");
});
