// Actual production UI and original retained points, never reference geometry.
import {expect,test} from "@playwright/test";
import {mkdirSync,readFileSync,writeFileSync} from "node:fs";
import {fileURLToPath} from "node:url";
const SOURCE=process.env.VECTOR_MAP_SUPPORTED_SOURCE;
const PROOF=process.env.VECTOR_MAP_SUPPORTED_PROOF;
const FRAMES=fileURLToPath(new URL("../media-frames/vector-map-supported-intersection/",import.meta.url));
const PATH_PROOF=process.env.VECTOR_MAP_SUPPORTED_PATH_PROOF;
const PATH_INPUT=JSON.parse(readFileSync(fileURLToPath(new URL("vector-map-hard-intersection-inputs.json",import.meta.url)),"utf8"));
const INPUT=JSON.parse(readFileSync(fileURLToPath(new URL("vector-map-supported-intersection-inputs.json",import.meta.url)),"utf8"));

test.use({viewport:{width:1280,height:760},actionTimeout:30000});
test("source-supported intersection comparison, equipment, Undo and reload",async({page})=>{
  test.skip(!SOURCE||!PROOF||!PATH_PROOF,"Set prepared source and frozen native proof directories");
  test.setTimeout(900000);mkdirSync(FRAMES,{recursive:true});
  const proof=JSON.parse(readFileSync(`${PATH_PROOF}/roads-report.json`,"utf8"));
  const expected=JSON.parse(readFileSync(`${PROOF}/equipment-preview.json`,"utf8")).discovery;
  const native=JSON.parse(readFileSync(`${PROOF}/reviewed.json`,"utf8"));
  const additions=JSON.parse(readFileSync(`${PROOF}/reviewed-report.json`,"utf8")).additions;
  const manifest=JSON.parse(readFileSync(`${PROOF}/manifest.json`,"utf8"));
  expect(manifest.reference_inputs).toEqual([]);expect(manifest.operator_inputs).toEqual(INPUT);expect(native.lanes).toHaveLength(59);
  const errors:string[]=[];page.on("pageerror",e=>errors.push(String(e)));
  await page.goto("/");await page.locator("#file-input").setInputFiles(`${SOURCE}/geometry.las`);
  await expect(page.locator("#status")).toContainText("Loaded geometry.las",{timeout:120000});
  await expect(page.locator("#vm-status")).toContainText("No map yet");
  await page.locator('#cloud-list select[title="Color by"]').first().selectOption("intensity");
  await page.locator("#edl").uncheck();await page.locator("#point-size").fill("2");
  // Display clipping removes tall roof clutter; fitting/search still use the full
  // loaded source. This neither crops the cloud nor changes source coordinates.
  await page.locator("#clip-enabled").check();
  await page.locator('[aria-label="Z maximum"]').fill("140");
  const clipZ=await page.locator('.clip-axis[data-axis="2"] .clip-values').textContent();
  await page.locator('[data-view="top"]').click();await page.locator("#fit").click();
  await page.evaluate(()=>{
    const el=document.createElement("div");el.id="capture-caption";
    el.style.cssText="position:fixed;left:300px;top:92px;max-width:920px;padding:10px 16px;background:#111a29f0;color:white;font:600 19px sans-serif;border:1px solid #3a5975;border-radius:8px;z-index:1000;pointer-events:none";document.body.append(el);
  });
  const frames:{name:string;duration:number}[]=[];
  const shot=async(text:string,duration=1.8)=>{
    await page.locator("#capture-caption").evaluate((el,text)=>{el.textContent=text},text);await page.waitForTimeout(250);
    const name=`${String(frames.length).padStart(3,"0")}.png`;await page.screenshot({path:`${FRAMES}/${name}`});frames.push({name,duration});
  };
  let exports=0;
  const exportMap=async()=>{
    if(exports>0&&exports%4===0)await page.waitForTimeout(1100);exports++;
    const wait=page.waitForEvent("download",d=>d.suggestedFilename()==="lanelet2_map.osm");await page.locator("#vm-export").click();
    const path=await(await wait).path();if(!path)throw new Error("Missing export");return readFileSync(path,"utf8");
  };
  const nativeDifference=async(xml:string,native:any)=>page.evaluate(({xml,native})=>{
    const doc=new DOMParser().parseFromString(xml,"application/xml");
    const wayPoints=(id:number)=>[...doc.querySelector(`way[id="${id}"]`)!.querySelectorAll("nd")].map(nd=>{
      const node=doc.querySelector(`node[id="${nd.getAttribute("ref")}"]`)!;
      return ["local_x","local_y","ele"].map(k=>Number(node.querySelector(`tag[k="${k}"]`)!.getAttribute("v")));
    });
    const error=(expected:number[][],actual:number[][])=>{
      if(expected.length!==actual.length)throw new Error("Geometry vertex count differs");
      const compare=(p:number[][])=>Math.max(...expected.flatMap((q,i)=>q.map((v,j)=>Math.abs(v-p[i][j]))));
      return Math.min(compare(actual),compare([...actual].reverse()));
    };
    const differences=[...native.boundaries,...(native.stop_lines??[]),...(native.traffic_signals??[])].map(c=>error(c.geometry,wayPoints(c.id)));
    for(const c of native.crosswalks??[])for(const [role,geometry] of [["left",c.left_edge],["right",c.right_edge]]){
      const member=doc.querySelector(`relation[id="${c.id}"] member[role="${role}"]`)!;differences.push(error(geometry,wayPoints(Number(member.getAttribute("ref")))));
    }
    return Math.max(...differences);
  },{xml,native});
  const checkAssociations=async(xml:string)=>{
    const lanes=await page.evaluate(({xml,ids})=>{
      const doc=new DOMParser().parseFromString(xml,"application/xml");
      return ids.map(id=>[...doc.querySelectorAll(`relation > member[role="regulatory_element"][ref="${id}"]`)].map(m=>Number(m.parentElement!.getAttribute("id"))).sort((a,b)=>a-b));
    },{xml,ids:native.regulatory_elements.map((r:any)=>r.id)});
    expect(lanes).toEqual(native.regulatory_elements.map((r:any)=>[...r.lanes].sort((a:number,b:number)=>a-b)));
  };
  const audit=async(count:number,review:number,xml:string)=>{
    await page.locator("#vm-quality-check").click();
    await expect(page.locator("#vm-quality-report")).toContainText(`${count} lanes checked; ${review} need source review; 0 omitted; 0 malformed`,{timeout:120000});
    expect(await exportMap()).toBe(xml);
  };
  await shot("1. Real Tokyo points · no input vector map",1.6);
  await page.locator("#file-input").setInputFiles(proof.builds.map((b:any)=>`${PATH_PROOF}/${b.csv}`));
  await expect(page.locator("#vm-trajectory option")).toHaveCount(6);
  await page.locator("#vector-map-panel").getByText("Road options",{exact:true}).click();await page.locator("#vm-segment").fill("0");
  await page.locator("#vector-map-panel").getByText("Build from a trajectory",{exact:true}).click();await page.locator("#vm-discover-after-build").uncheck();
  const buildRoads=async()=>{for(let i=0;i<PATH_INPUT.paths.length;i++){
    const options=PATH_INPUT.paths[i].options;
    await page.locator("#vm-forward").fill(String(options.forward_lanes));await page.locator("#vm-backward").fill(String(options.backward_lanes));await page.locator("#vm-width").fill(String(options.lane_width));
    await page.locator("#vm-trajectory").selectOption({index:i});await page.locator("#vm-build").click();
    await expect(page.locator("#status")).toContainText("Draft roads added",{timeout:120000});
  }};
  await buildRoads();
  await expect(page.locator("#vm-status")).toContainText("26 lanes");
  await page.locator("#vector-map-panel").getByText("Map display",{exact:true}).click();await page.locator("#vm-context").fill("40");
  await page.locator("#vm-show-labels").uncheck();await page.locator("#vm-plan").click();await page.locator("#vm-fit").click();
  await page.locator("#vm-quality summary").click();
  const comparison=JSON.parse(readFileSync(`${PROOF}/road-comparison/comparison.json`,"utf8"));
  const before=await exportMap();
  const beforeNativeDifference=await nativeDifference(before,JSON.parse(readFileSync(`${PROOF}/road-comparison/before-road-5.json`,"utf8")));
  expect(beforeNativeDifference).toBeLessThan(1e-7);await audit(26,9,before);
  await shot("2. Before: 26 lanes, 9 need source review",2.5);
  for(let i=0;i<PATH_INPUT.paths.length;i++)await page.locator("#vm-undo").click();
  await expect(page.locator("#vm-status")).toContainText("No map yet");await expect(page.locator("#vm-undo")).toBeDisabled();
  await page.locator("#vm-source-surface").check();await buildRoads();
  await expect(page.locator("#vm-status")).toContainText("38 lanes");
  const sourceRoads=await exportMap();
  const sourceRoadNativeDifference=await nativeDifference(sourceRoads,JSON.parse(readFileSync(`${PROOF}/road-comparison/after-road-5.json`,"utf8")));
  expect(sourceRoadNativeDifference).toBeLessThan(1e-7);await audit(38,0,sourceRoads);
  await page.locator("#vm-plan").click();await page.locator("#vm-fit").click();
  await shot("After: 38 supported fragments, 0 source flags; 34.3 m deferred",3);
  await page.locator("#vector-map-panel").getByText("Draft junction connections",{exact:true}).click();await page.locator("#vm-junction-gap").fill(String(INPUT.junction_options.max_gap));
  const legacyJunctions=JSON.parse(readFileSync(`${PROOF}/junction-centre-preview.json`,"utf8")).junctions;
  await page.locator("#vm-junction-preview").click();
  await expect(page.locator("#vm-junction-report")).toContainText(`${legacyJunctions.candidates.length} ground-supported candidates`,{timeout:120000});
  await page.locator("#vm-junction-boundaries").check();await page.locator("#vm-junction-support").fill("100");
  await page.locator("#vm-junction-preview").click();
  const nativeJunctions=JSON.parse(readFileSync(`${PROOF}/junction-preview.json`,"utf8")).junctions;
  await expect(page.locator("#vm-junction-report")).toContainText(`${nativeJunctions.candidates.length} ground-supported candidates`,{timeout:120000});
  expect(nativeJunctions.candidates).toHaveLength(115);
  await expect(page.locator("#vm-junction-candidates")).toContainText("boundaries 100% / 100%");
  for(const pair of INPUT.deferred_requested_pairs)expect(nativeJunctions.candidates.find((c:any)=>c.from===pair[0]&&c.to===pair[1])).toBeUndefined();
  const beforeConnections=await exportMap();
  await page.locator("#vm-junction-none").click();
  for(const pair of INPUT.reviewed_pairs){
    const index=nativeJunctions.candidates.findIndex((c:any)=>c.from===pair[0]&&c.to===pair[1]);expect(index).toBeGreaterThanOrEqual(0);
    await page.locator(`#vm-junction-candidates input[value="${index}"]`).check();
  }
  await shot("3. Centre + both boundaries supported; review 21 connections",2);
  await page.locator("#vm-junction-apply").click();await expect(page.locator("#status")).toContainText("21 draft connections added",{timeout:120000});
  await expect(page.locator("#vm-status")).toContainText("59 lanes");
  const roads=await exportMap();expect(roads).not.toBe(beforeConnections);
  await page.locator("#vm-plan").click();await page.locator("#vm-discovery summary").click();await page.locator("#vm-discovery-scope").selectOption("ground_surface");await page.locator("#vm-discovery-search").click();
  await expect(page.locator("#status")).toContainText(`Found ${expected.candidates.length} unconfirmed equipment proposals`,{timeout:180000});
  await expect(page.locator("#vm-discovery-report")).toContainText(`${expected.detected_candidates} detected`);expect(await exportMap()).toBe(roads);
  await page.locator(`input[name="discovered-feature"][value="${INPUT.discarded_candidates[0]}"]`).check();
  await shot("4. Equipment search: 145 shown / 328 detected; review types",2);
  await page.locator("#vm-discovery-reject").click();expect(await exportMap()).toBe(roads);
  // Longitudinal lane paint is not a stop marking. Discard both reviewed bars
  // without adding rules, then use transverse branch markings instead.
  for(const id of INPUT.discarded_candidates.slice(1)){
    await page.locator(`input[name="discovered-feature"][value="${id}"]`).check();
    await page.locator("#vm-discovery-reject").click();
  }
  expect(await exportMap()).toBe(roads);
  for(let i=0;i<INPUT.confirmations.length;i++){
    const c=INPUT.confirmations[i];
    await page.locator(`input[name="discovered-feature"][value="${c.candidate}"]`).check();
    await expect(page.locator("#vm-discovery-kind")).toHaveValue("");await expect(page.locator("#vm-discovery-lanes")).toHaveValue("");
    if(i===0)await shot("5. Crosswalk proposal · fit edges to observed paint bands",2.2);
    if(c.candidate===62)await shot("6. Transverse branch marking · identify type and lane",1.8);
    if(c.candidate===129){
      await page.locator('[data-view="iso"]').click();await page.locator("#vm-discovery-inspect").click();
      await expect(page.locator("#status")).toContainText("original points isolated",{timeout:120000});await page.locator("#point-size").fill("5");
      await shot("7. Inspect housing points · choose object type and lanes",2.5);
    }
    await page.locator("#vm-discovery-kind").selectOption(c.classification);await page.locator("#vm-discovery-lanes").fill(c.lanes.join(","));await page.locator("#vm-discovery-add").click();
    await expect(page.locator("#status")).toContainText(`Reviewed ${c.classification.replaceAll("_"," ")}`,{timeout:180000});
  }
  // Restore the inspected source after all confirmations: cloud Undo clears
  // stale proposal controls, while retaining the separately edited vector map.
  await page.locator("#undo").click();await expect(page.locator("#status")).toContainText("Undid");await page.locator("#point-size").fill("2");
  const added=await exportMap();expect(added).toContain('v="observed_band_envelope_20cm_simplification"');expect(added).not.toContain('v="stop_sign"');
  const maximumNativeDifference=await nativeDifference(added,native);
  await checkAssociations(added);
  await audit(59,0,added);
  expect(maximumNativeDifference).toBeLessThan(1e-7);
  await expect(page.locator("#vm-status")).toContainText("0 errors, 19 warnings");
  await expect(page.locator('#vm-issues li[title="autoware.traffic_light_without_stop_line"]')).toHaveCount(4);
  await expect(page.locator('#vm-issues li[title="disconnected_topology"]')).toHaveCount(2);
  await expect(page.locator('#vm-issues li[title="isolated_lane"]')).toHaveCount(13);
  const generatedWarnings=(await page.locator("#vm-issues li").allTextContents()).sort();
  await page.locator("#vm-discovery-show").uncheck();
  await page.locator("#vm-feature-editor summary").click();await page.locator("#vm-feature").selectOption(`signal:${additions.at(-1).id}`);
  const z=Number(await page.locator("#vm-feature-z").inputValue());await page.locator("#vm-feature-z").fill(String(z+.08));await page.locator("#vm-feature-apply").click();
  await expect(page.locator("#status")).toContainText("edited");expect(await exportMap()).not.toBe(added);
  await shot("8. Edit measured geometry · original observations retained",1.5);
  await page.locator("#vm-undo").click();expect(await exportMap()).toBe(added);await shot("Undo restores the entire map exactly",1.3);
  // Display-only working-cloud crop removes the clipping-box wire. Source
  // fitting already finished on the full cloud; this copies original points
  // in browser memory, without a disk copy or changes to any map geometry.
  await page.locator("#clip-crop").click();await expect(page.locator("#status")).toContainText("Cropped:");
  await page.locator('#cloud-list select[title="Color by"]').last().selectOption("intensity");expect(await exportMap()).toBe(added);
  await page.locator("#vm-plan").click();await page.locator("#vm-fit").click();
  await shot("9. 59 road lanes, 7 crossings, 2 stops, 4 housings",2.8);
  await page.locator("#vm-iso").click();await shot("Draft: gaps retained; signal-stop links still need review",2.5);
  for(let i=0;i<INPUT.confirmations.length;i++)await page.locator("#vm-undo").click();expect(await exportMap()).toBe(roads);
  await page.locator("#vm-file").setInputFiles({name:"generated-intersection.osm",mimeType:"application/xml",buffer:Buffer.from(added)});
  await expect(page.locator("#status")).toContainText("Opened generated-intersection.osm: 59 lanes");await expect(page.locator("#vm-status")).toContainText("0 errors");
  const reloaded=await exportMap();
  expect(await nativeDifference(reloaded,native)).toBeLessThan(1e-7);await checkAssociations(reloaded);
  await expect(page.locator("#vm-status")).toContainText("0 errors, 19 warnings");
  expect((await page.locator("#vm-issues li").allTextContents()).sort()).toEqual(generatedWarnings);
  expect(reloaded).toContain('v="observed_band_envelope_20cm_simplification"');expect(reloaded).toContain('v="point_cloud_box_fit"');expect(errors).toEqual([]);
  await shot("10. Save and reopen Lanelet2 · editable measured provenance",2.2);
  writeFileSync(`${FRAMES}/verification.json`,JSON.stringify({inputMap:false,pointCount:1883866,recordedDriveIndex:1,operatorInputs:INPUT,clipDisplayZ:clipZ,roadLanes:59,proposals:expected.candidates.length,detected:expected.detected_candidates,unsupported:expected.unsupported_windows,crosswalks:7,stopMarkings:2,signalHousings:4,maximumNativeDifference,exactEditUndo:true,exactAllAdditionUndo:true,localOsmRoundtrip:true,exactRegulatoryLaneAssociations:true,beforeNativeDifference,sourceRoadNativeDifference,beforeSourceReviewLanes:9,afterSourceReviewLanes:0,approachFragments:38,selectedConnections:21,fullCurveCandidates:115,deferredRoadLengthM:comparison.after.reported_deferred_length_m,structuralErrors:0,connectivityWarnings:15,autowareWarnings:19,unreviewedSignalStopLinks:4,exactWarningRoundtrip:true,generatedWarnings,equipmentPreviewLimited:expected.limited,finalSourceReviewLanes:0,referenceInputs:manifest.reference_inputs,sourceSha256:manifest.input_cloud_sha256,sourceCommit:manifest.source_commit,nativeSha256:manifest.native_sha256,errors},null,2));
  writeFileSync(`${FRAMES}/before.osm`,before);writeFileSync(`${FRAMES}/supported-roads.osm`,sourceRoads);
  writeFileSync(`${FRAMES}/generated.osm`,added);writeFileSync(`${FRAMES}/frames.json`,JSON.stringify(frames,null,2));
  writeFileSync(`${FRAMES}/concat.txt`,frames.map(f=>`file '${f.name}'\nduration ${f.duration}\n`).join("")+`file '${frames.at(-1)!.name}'\n`);
});
