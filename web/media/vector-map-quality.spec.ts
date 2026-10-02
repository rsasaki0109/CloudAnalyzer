// Optional actual-point workflow verification; no cloud or reference is bundled.
import {expect,test} from "@playwright/test";
import {mkdirSync,readFileSync,writeFileSync} from "node:fs";
import {fileURLToPath} from "node:url";
const SOURCE=process.env.VECTOR_MAP_QUALITY_SOURCE;
const PROOF=process.env.VECTOR_MAP_QUALITY_PROOF;
const AUDIT=process.env.VECTOR_MAP_QUALITY_AUDIT;
const OUTPUT=fileURLToPath(new URL("../media-frames/vector-map-quality/",import.meta.url));
test.use({viewport:{width:1280,height:760},actionTimeout:30000});
test("real generated map source coverage agrees with native and preserves edits",async({page})=>{
  test.skip(!SOURCE||!PROOF||!AUDIT,"Set source cloud, frozen generated-map proof and native audit");
  test.setTimeout(180000);mkdirSync(OUTPUT,{recursive:true});
  const audit=JSON.parse(readFileSync(AUDIT!,"utf8"));const quality=audit.native.quality;
  expect(audit.generation_reference_inputs).toEqual([]);
  await page.goto("/");await page.locator("#file-input").setInputFiles(SOURCE!);
  await expect(page.locator("#status")).toContainText("Loaded",{timeout:120000});
  await page.locator("#vm-file").setInputFiles(`${PROOF}/reviewed.osm`);
  await expect(page.locator("#vm-status")).toContainText(`${quality.lanes.length} lanes`);
  let exports=0;
  const exportMap=async()=>{if(exports++>0)await page.waitForTimeout(1100);const wait=page.waitForEvent("download",d=>d.suggestedFilename()==="lanelet2_map.osm");await page.locator("#vm-export").click();const path=await(await wait).path();return readFileSync(path!,"utf8");};
  const before=await exportMap();await expect(page.locator("#vm-undo")).toBeDisabled();
  await page.locator("#vm-quality summary").click();await page.locator("#vm-quality-check").click();
  await expect(page.locator("#vm-quality-report")).toContainText(`${quality.lanes.length} lanes checked; ${quality.low_support_lanes.length} need source review; ${quality.omitted_lanes.length} omitted`,{timeout:120000});
  const buttons=page.locator("#vm-quality-lanes button");await expect(buttons).toHaveCount(quality.low_support_lanes.length);
  const expected=quality.lanes.filter((l:any)=>l.needs_review).map((l:any)=>({id:l.lane,fractions:[l.center,l.left,l.right].map((s:any)=>Math.round(s.fraction*100))}));
  const percentage=(s:any)=>`${Math.round(s.fraction*100)}%${s.start_supported&&s.end_supported?"":" (end support missing)"}`;
  for(const lane of quality.lanes.filter((l:any)=>l.needs_review)){
    await expect(buttons.filter({hasText:`Lane ${lane.lane}:`})).toHaveText(
      `Lane ${lane.lane}: centre ${percentage(lane.center)}, left ${percentage(lane.left)}, right ${percentage(lane.right)}`,
    );
  }
  expect(await exportMap()).toBe(before);await expect(page.locator("#vm-undo")).toBeDisabled();
  await page.locator('#cloud-list select[title="Color by"]').first().selectOption("intensity");
  await page.locator("#edl").uncheck();await page.locator("#point-size").fill("2");
  await page.locator("#clip-enabled").check();await page.locator('[aria-label="Z maximum"]').fill("140");
  await page.locator("#vector-map-panel").getByText("Map display",{exact:true}).click();
  await page.locator("#vm-show-labels").uncheck();
  await page.locator("#vm-plan").click();await page.locator("#vm-fit").click();
  await page.locator("#vm-quality-report").scrollIntoViewIfNeeded();
  await page.screenshot({path:`${OUTPUT}/source-quality-overview.png`});
  await buttons.first().click();await expect(page.locator("#vm-lane-title")).toHaveText(`Lane ${expected[0].id}`);
  await page.locator("#vm-lane-speed").fill("20");await page.locator("#vm-lane-apply").click();
  await expect(page.locator("#vm-quality-report")).toContainText("has not been checked");
  await page.locator("#vm-undo").click();expect(await exportMap()).toBe(before);
  writeFileSync(`${OUTPUT}/verification.json`,JSON.stringify({generatedMapSha256:audit.generated_osm_sha256,auditSourceCommit:audit.audit_source_commit,checkedLanes:quality.lanes.length,lowSupportLaneIds:quality.low_support_lanes,roundedNativeWebSupportAgrees:true,auditLeavesMapAndUndoUnchanged:true,editInvalidatesQuality:true,editUndoExact:true},null,2));
});
