// Optional actual-source proposal/adoption proof, not a generation demo.
import {expect,test} from "@playwright/test";
import {mkdirSync,readFileSync,writeFileSync} from "node:fs";
import {fileURLToPath} from "node:url";
const SOURCE=process.env.VECTOR_MAP_RELATIONS_SOURCE;
const INPUT=process.env.VECTOR_MAP_RELATIONS_INPUT;
const PROOF=process.env.VECTOR_MAP_RELATIONS_PROOF;
const CONFIG=JSON.parse(readFileSync(fileURLToPath(new URL("../../benchmarks/vector-map/hard-intersection/equipment-relations/operator-inputs.json",import.meta.url)),"utf8"));
const OUTPUT=fileURLToPath(new URL("../media-frames/vector-map-relation-proposals/",import.meta.url));
test.use({viewport:{width:1440,height:900},actionTimeout:30000});
test("real geometric target proposals hold nearest crossing and explicit adoption preserves source, Undo and reload",async({page})=>{
  test.skip(!SOURCE||!INPUT||!PROOF,"Set original source, frozen generated map and native review proof");
  test.setTimeout(300000);mkdirSync(OUTPUT,{recursive:true});
  const report=JSON.parse(readFileSync(`${PROOF}/report.json`,"utf8"));
  expect(report.operator_inputs_sha256).toBe((await import("node:crypto")).createHash("sha256").update(readFileSync(fileURLToPath(new URL("../../benchmarks/vector-map/hard-intersection/equipment-relations/operator-inputs.json",import.meta.url)))).digest("hex"));
  const proposals=JSON.parse(readFileSync(`${PROOF}/proposals.json`,"utf8"));
  expect(proposals.explicit_adoptions).toEqual(["stop_line:164","crosswalk:156"]);
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
  for(const rule of [167,171]) {
    await page.locator("#vm-relation").selectOption(String(rule));await page.locator("#vm-relation-preview").click();
    await expect(page.locator("#vm-relation-proposal-report")).toContainText("0 supported draft candidates");
    await expect(page.locator("#vm-relation-adopt")).toBeDisabled();await expect(page.locator("#vm-undo")).toBeDisabled();
    expect(await exportMap()).toBe(before);
  }
  for(const [rule,key] of [[169,"stop_line:164"],[173,"crosswalk:156"]] as const) {
    await page.locator("#vm-relation").selectOption(String(rule));await page.locator("#vm-relation-preview").click();
    await expect(page.locator("#vm-relation-proposal-report")).toContainText("1 supported draft candidate");
    await expect(page.locator("#vm-relation-adopt")).toBeDisabled();expect(await page.locator("#vm-relation-candidates input:checked").count()).toBe(0);
    if(rule===173) {
      await page.locator('#vm-relation-candidates input[value="crosswalk:152"]').check();await expect(page.locator("#vm-relation-adopt")).toBeDisabled();
      await expect(page.locator("#vm-relation-candidates")).toContainText("direction disagree");
      await page.locator("#vm-relations-editor").scrollIntoViewIfNeeded();await page.screenshot({path:`${OUTPUT}/held-nearest.png`});
    }
    await page.locator(`#vm-relation-candidates input[value="${key}"]`).check();await page.locator("#vm-relation-candidate-focus").click();
    await page.locator("#edl").uncheck();await page.locator("#point-size").fill("2");
    await page.locator("#vm-relations-editor").scrollIntoViewIfNeeded();await page.screenshot({path:`${OUTPUT}/candidate-${rule}.png`});
    await page.locator("#vm-relation-adopt").click();await expect(page.locator("#status")).toContainText(`Rule ${rule} reviewed candidate adopted`);
    await expect(page.locator("#vm-relation-current")).toContainText("user_reviewed");expect(await page.locator("#vm-relation-candidates input").count()).toBe(0);
  }
  await page.locator("#vm-relation").selectOption("171");await page.locator("#vm-relation-crosswalks").fill("");
  await page.locator("#vm-relation-apply").click();await expect(page.locator("#status")).toContainText("Rule 171 associations updated");
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
  writeFileSync(`${OUTPUT}/verification.json`,JSON.stringify({readonlyProposals:true,noAutomaticSelection:true,nearestWrongOrientationHeld:true,explicitAdoptions:["stop_line:164","crosswalk:156"],nativeOsmByteExact:true,physicalMapNativeExact:true,sourceQualityCheckedFullCloud:true,lanes:59,sourceReviewLanes:0,noOpUndoExact:true,allEditsUndoExact:true,controlTargetsReload:true,warningsReloadExact:true,warningsBefore:19,warningsAfter:18,unresolvedRules:report.unresolved_rules,pageErrors:errors},null,2)+"\n");
});
