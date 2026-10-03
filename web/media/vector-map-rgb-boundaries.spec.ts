// Real original planning points; source-only build, no reference map opened.
import {expect,test} from "@playwright/test";
import {mkdirSync,readFileSync,writeFileSync} from "node:fs";
import {fileURLToPath} from "node:url";
const SOURCE=process.env.VECTOR_MAP_RGB_SOURCE;
const TRAJECTORY=process.env.VECTOR_MAP_RGB_TRAJECTORY;
const PROOF=process.env.VECTOR_MAP_RGB_PROOF;
const OUTPUT=fileURLToPath(new URL("../media-frames/vector-map-rgb-boundaries/",import.meta.url));
test.use({viewport:{width:1440,height:900},actionTimeout:30000});
test("real planning source RGB road drafting matches native geometry and undoes exactly",async({page})=>{
  test.skip(!SOURCE||!TRAJECTORY||!PROOF,"Set cached planning source, trajectory and native proof");
  test.setTimeout(240000);
  mkdirSync(OUTPUT,{recursive:true});
  const errors:string[]=[];page.on("pageerror",e=>errors.push(String(e)));
  const expected=JSON.parse(readFileSync(`${PROOF}/rgb-0.json`,"utf8"));
  const report=JSON.parse(readFileSync(`${PROOF}/evaluation.json`,"utf8"));
  const extraction=report.cases.rgb[0].extraction;
  await page.goto("/");
  await page.locator("#file-input").setInputFiles([SOURCE!,TRAJECTORY!]);
  await expect(page.locator("#status")).toContainText("trajectory of 2 poses",{timeout:120000});
  await page.locator("#vector-map-panel").getByText("Build from a trajectory",{exact:true}).click();
  await expect(page.locator("#vm-rgb-boundaries")).not.toBeChecked();
  await page.locator("#vector-map-panel").getByText("Road options",{exact:true}).click();
  await page.locator("#vm-segment").fill("0");
  await page.locator("#vm-discover-after-build").uncheck();
  await page.locator("#vm-source-surface").check();await page.locator("#vm-rgb-boundaries").check();
  await page.locator("#vm-build").click();
  await expect(page.locator("#status")).toContainText("Draft roads added",{timeout:180000});
  await expect(page.locator("#vm-build-report")).toContainText(`RGB white-paint sources: ${extraction.rgb_paint_vertices}`);
  const exported=page.waitForEvent("download",d=>d.suggestedFilename()==="lanelet2_map.osm");
  await page.locator("#vm-export").click();
  const xml=readFileSync((await(await exported).path())!,"utf8");
  const nodes=[...xml.matchAll(/<node\b[^>]*>([\s\S]*?)<\/node>/g)].map(match=>{
    const value=(key:string)=>Number(match[1].match(new RegExp(`k="${key}" v="([^"]+)"`))![1]);
    return [value("local_x"),value("local_y"),value("ele")];
  });
  const nativePoints: number[][]=expected.boundaries.flatMap((b:{geometry:number[][]})=>b.geometry);
  let maximum=0;
  for(const p of nativePoints){const distance=Math.min(...nodes.map(q=>Math.hypot(...p.map((v,i)=>v-q[i]))));maximum=Math.max(maximum,distance);}
  expect(maximum).toBeLessThan(1e-7);
  await page.locator("#vm-quality summary").click();await page.locator("#vm-quality-check").click();
  await expect(page.locator("#vm-quality-report")).toContainText(`${expected.lanes.length} lanes checked; 0 need source review; 0 omitted`,{timeout:120000});
  await page.locator("#vector-map-panel").getByText("Map display",{exact:true}).click();await page.locator("#vm-plan").click();
  await page.locator("#vm-build-report").scrollIntoViewIfNeeded();await page.screenshot({path:`${OUTPUT}/source-build.png`});
  await page.locator("#vm-undo").click();await expect(page.locator("#vm-status")).toContainText("No map yet");
  await expect(page.locator("#vm-undo")).toBeDisabled();expect(errors).toEqual([]);
  writeFileSync(`${OUTPUT}/verification.json`,JSON.stringify({sourceOnly:true,sourcePoints:1757841,rgbSourceVertices:extraction.rgb_paint_vertices,
    lanes:expected.lanes.length,sourceQualityFullCloud:true,maximumNativeBoundaryVertexDistanceM:maximum,
    geometryComparison:"Every native boundary vertex to exported UI nodes; tolerance, not OSM byte identity or full topology equality.",
    explicitOptIn:true,oneStepUndo:true,pageErrors:errors},null,2)+"\n");
});
