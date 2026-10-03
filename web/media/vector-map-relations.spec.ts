// Optional actual-source equipment-association proof, not a generation demo.
import {expect,test} from "@playwright/test";
import {mkdirSync,readFileSync,writeFileSync} from "node:fs";
import {fileURLToPath} from "node:url";
const SOURCE=process.env.VECTOR_MAP_RELATIONS_SOURCE;
const INPUT=process.env.VECTOR_MAP_RELATIONS_INPUT;
const PROOF=process.env.VECTOR_MAP_RELATIONS_PROOF;
const CONFIG=JSON.parse(readFileSync(fileURLToPath(new URL("../../benchmarks/vector-map/hard-intersection/equipment-relations/operator-inputs.json",import.meta.url)),"utf8"));
const OUTPUT=fileURLToPath(new URL("../media-frames/vector-map-relations/",import.meta.url));
test.use({viewport:{width:1440,height:900},actionTimeout:30000});
test("real generated map reviewed equipment associations preserve geometry, Undo and reload",async({page})=>{
  test.skip(!SOURCE||!INPUT||!PROOF,"Set original source, frozen generated map and native review proof");
  test.setTimeout(300000);mkdirSync(OUTPUT,{recursive:true});
  const report=JSON.parse(readFileSync(`${PROOF}/report.json`,"utf8"));
  expect(report.operator_inputs_sha256).toBe((await import("node:crypto")).createHash("sha256").update(readFileSync(fileURLToPath(new URL("../../benchmarks/vector-map/hard-intersection/equipment-relations/operator-inputs.json",import.meta.url)))).digest("hex"));
  expect(report.reference_inputs).toEqual([]);expect(report.physical_map_exact).toBe(true);
  const errors:string[]=[];page.on("pageerror",e=>errors.push(String(e)));
  await page.goto("/");await page.locator("#file-input").setInputFiles(SOURCE!);
  await expect(page.locator("#status")).toContainText("Loaded",{timeout:120000});
  await page.locator("#vm-file").setInputFiles(INPUT!);
  await expect(page.locator("#vm-status")).toContainText("59 lanes");await expect(page.locator("#vm-status")).toContainText("19 warnings");
  let exports=0;
  const exportMap=async()=>{if(exports++>0)await page.waitForTimeout(1100);const wait=page.waitForEvent("download",d=>d.suggestedFilename()==="lanelet2_map.osm");await page.locator("#vm-export").click();return readFileSync((await(await wait).path())!,"utf8");};
  const before=await exportMap();await expect(page.locator("#vm-undo")).toBeDisabled();
  await page.locator("#vm-relations-editor summary").click();
  for (const edit of CONFIG.edits) {
    await page.locator("#vm-relation").selectOption(String(edit.rule_id));
    if (edit.rule_id===173 || edit.rule_id===171) await page.locator("#vm-relation-crosswalks").fill(edit.controlled_crosswalks.join(","));
    else {await page.locator("#vm-relation-lanes").fill(edit.lanes.join(","));await page.locator("#vm-relation-stops").fill(edit.stop_lines.join(","));}
    await page.locator("#vm-relation-apply").click();await expect(page.locator("#status")).toContainText(`Rule ${edit.rule_id} associations updated`);
    await expect(page.locator("#vm-relation-current")).toContainText("user_reviewed");
  }
  await expect(page.locator("#vm-status")).toContainText("0 errors, 18 warnings");
  const reviewed=await exportMap();expect(reviewed).toBe(readFileSync(`${PROOF}/reviewed.osm`,"utf8"));
  const warningsBeforeReload=await page.locator("#vm-issues li").allTextContents();
  await page.locator("#vm-relation-apply").click();await expect(page.locator("#status")).toContainText("no Undo step added");expect(await exportMap()).toBe(reviewed);
  // This is a view-only audit over the complete original source, before display clipping.
  await page.locator("#vm-quality summary").click();await page.locator("#vm-quality-check").click();
  await expect(page.locator("#vm-quality-report")).toContainText("59 lanes checked; 0 need source review; 0 omitted",{timeout:120000});
  expect(await exportMap()).toBe(reviewed);
  for(let i=0;i<CONFIG.edits.length;i++)await page.locator("#vm-undo").click();
  expect(await exportMap()).toBe(before);await expect(page.locator("#vm-undo")).toBeDisabled();
  await page.locator("#vm-file").setInputFiles({name:"reviewed.osm",mimeType:"application/xml",buffer:Buffer.from(reviewed)});
  await expect(page.locator("#status")).toContainText("Opened reviewed.osm");await expect(page.locator("#vm-status")).toContainText("0 errors, 18 warnings");
  expect(await page.locator("#vm-issues li").allTextContents()).toEqual(warningsBeforeReload);
  for(const edit of CONFIG.edits){await page.locator("#vm-relation").selectOption(String(edit.rule_id));await expect(page.locator("#vm-relation-current")).toContainText(`crosswalks: ${edit.controlled_crosswalks.join(",")||"none"}; stops: ${edit.stop_lines.join(",")||"none"}`);}
  await page.locator("#vm-relation").selectOption("173");await page.locator("#vector-map-panel").getByText("Map display",{exact:true}).click();await page.locator("#vm-plan").click();await page.locator("#vm-relation-focus").click();await page.locator("#vector-map-panel").getByText("Map display",{exact:true}).click();
  await page.locator("#vm-quality summary").click();await page.locator("#vm-relations-editor").scrollIntoViewIfNeeded();
  await page.locator("#edl").uncheck();await page.locator("#point-size").fill("2");
  await page.screenshot({path:`${OUTPUT}/pedestrian-association.png`});
  writeFileSync(`${OUTPUT}/reviewed.osm`,reviewed);
  expect(errors).toEqual([]);
  writeFileSync(`${OUTPUT}/verification.json`,JSON.stringify({nativeOsmByteExact:true,physicalMapNativeExact:true,sourceQualityCheckedFullCloud:true,lanes:59,sourceReviewLanes:0,noOpUndoExact:true,allEditsUndoExact:true,controlTargetsReload:true,warningsReloadExact:true,warningsBefore:19,warningsAfter:18,unresolvedRules:report.unresolved_rules,pageErrors:errors},null,2)+"\n");
});
