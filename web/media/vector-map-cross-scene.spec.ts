// Optional surveyed-context component audit, not source-only generation media.
import {expect,test} from "@playwright/test";
import {mkdirSync,readFileSync,writeFileSync} from "node:fs";
import {fileURLToPath} from "node:url";
const SOURCE=process.env.VECTOR_MAP_CROSS_SCENE_SOURCE;
const REFERENCE=process.env.VECTOR_MAP_CROSS_SCENE_REFERENCE;
const OUTPUT=fileURLToPath(new URL("../media-frames/vector-map-cross-scene/",import.meta.url));
test.use({viewport:{width:1440,height:900},actionTimeout:30000});
test("real second intersection holds opposing stop without changing surveyed movements",async({page})=>{
  test.skip(!SOURCE||!REFERENCE,"Set cached planning source and reference map");
  test.setTimeout(180000);mkdirSync(OUTPUT,{recursive:true});
  const errors:string[]=[];page.on("pageerror",e=>errors.push(String(e)));
  await page.goto("/");await page.locator("#file-input").setInputFiles(SOURCE!);
  await expect(page.locator("#status")).toContainText("Loaded",{timeout:120000});
  await page.locator("#vm-file").setInputFiles(REFERENCE!);
  await expect(page.locator("#vm-status")).toContainText("186 lanes");
  let exports=0;
  const exported=async()=>{if(exports++>0)await page.waitForTimeout(1100);const wait=page.waitForEvent("download",d=>d.suggestedFilename()==="lanelet2_map.osm");await page.locator("#vm-export").click();return readFileSync((await(await wait).path())!,"utf8");};
  const before=await exported();
  await page.locator("#vm-relations-editor summary").click();
  for(const [rule,wrong,expected] of [[1007,351,null],[1008,348,351],[1024,414,null],[1025,411,null]] as const) {
    await page.locator("#vm-relation").selectOption(String(rule));await page.locator("#vm-relation-preview").click();
    await expect(page.locator("#vm-relation-proposal-report")).toContainText(`${expected===null?0:1} supported draft candidate`);
    expect(await page.locator("#vm-relation-candidates input:checked").count()).toBe(0);
    await page.locator(`#vm-relation-candidates input[value="stop_line:${wrong}"]`).check();
    await expect(page.locator("#vm-relation-candidates")).toContainText("every reviewed vehicle lane");
    await expect(page.locator("#vm-relation-adopt")).toBeDisabled();await expect(page.locator("#vm-undo")).toBeDisabled();
    if(expected!==null){await page.locator(`#vm-relation-candidates input[value="stop_line:${expected}"]`).check();await expect(page.locator("#vm-relation-adopt")).toBeEnabled();}
    expect(await exported()).toBe(before);
    if(rule===1007){await page.locator("#vm-relation-candidate-focus").click();await page.locator("#edl").uncheck();await page.locator("#point-size").fill("2");await page.locator("#vm-relations-editor").scrollIntoViewIfNeeded();await page.screenshot({path:`${OUTPUT}/held-opposing-stop.png`});}
  }
  expect(errors).toEqual([]);
  writeFileSync(`${OUTPUT}/verification.json`,JSON.stringify({sourcePoints:1757841,role:"surveyed-context component audit",opposingTargetsHeld:4,readonlyExport:true,undoUnchanged:true,explicitSelectionRequired:true,expectedTarget351Available:true,pageErrors:errors},null,2)+"\n");
});
