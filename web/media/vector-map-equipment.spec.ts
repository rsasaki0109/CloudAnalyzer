// Source-only production UI. Paths/type/lane reviews are explicit operator inputs.
// No surveyed map, feature ROI or reference geometry is loaded into the generator.
import {expect,test} from "@playwright/test";
import {mkdirSync,readFileSync,writeFileSync} from "node:fs";
import {fileURLToPath} from "node:url";
const MAP=process.env.VECTOR_MAP_PLANNING_DIR;
const PROOF=process.env.VECTOR_MAP_DISCOVERY_DIR;
const FRAMES=fileURLToPath(new URL("../media-frames/vector-map-equipment/",import.meta.url));
const PATHS=JSON.parse(readFileSync(fileURLToPath(new URL("vector-map-operator-paths.json",import.meta.url)),"utf8")) as number[][][];
test.use({viewport:{width:1280,height:760}});
test("source-only equipment README GIF on actual planning points",async({page})=>{
  test.skip(!MAP||!PROOF,"Set planning cloud and source-only native evaluation directories");
  test.setTimeout(360000);mkdirSync(FRAMES,{recursive:true});
  const expected=JSON.parse(readFileSync(`${PROOF}/ground-preview.json`,"utf8")).report.discovery;
  const errors:string[]=[];page.on("pageerror",e=>errors.push(String(e)));
  await page.goto("/");await page.locator("#file-input").setInputFiles(`${MAP}/pointcloud_map.pcd`);
  await expect(page.locator("#status")).toContainText("Loaded pointcloud_map.pcd",{timeout:120000});
  await expect(page.locator("#vm-status")).toContainText("No map yet");
  await page.locator("#file-input").setInputFiles(PATHS.map((p,i)=>({name:`operator-path-${i}.csv`,mimeType:"text/csv",buffer:Buffer.from("timestamp,x,y,z\n"+p.map((q,j)=>`${j},${q.join(",")}`).join("\n"))})));
  await expect(page.locator("#vm-trajectory option")).toHaveCount(3);
  await page.locator("#edl").uncheck();await page.locator("#point-size").fill("2");
  await page.locator('[data-view="top"]').click();
  await page.evaluate(()=>{const el=document.createElement("div");el.id="capture-caption";el.style.cssText="position:fixed;left:300px;top:92px;padding:10px 16px;background:#111a29f0;color:white;font:600 19px sans-serif;border:1px solid #3a5975;border-radius:8px;z-index:1000;pointer-events:none";document.body.append(el);});
  const frames:{name:string;duration:number}[]=[];
  const shot=async(text:string,duration=1.8)=>{await page.locator("#capture-caption").evaluate((el,text)=>{el.textContent=text},text);await page.waitForTimeout(200);const name=`${String(frames.length).padStart(3,"0")}.png`;await page.screenshot({path:`${FRAMES}/${name}`});frames.push({name,duration});};
  let exports=0;
  const exportMap=async()=>{if(exports>0&&exports%4===0)await page.waitForTimeout(1100);exports++;const wait=page.waitForEvent("download",d=>d.suggestedFilename()==="lanelet2_map.osm");await page.locator("#vm-export").click();const p=await(await wait).path();if(!p)throw new Error("Export missing");return readFileSync(p,"utf8");};
  await shot("1. Real points + operator paths · no input map",1.8);
  await page.locator("#vector-map-panel").getByText("Road options",{exact:true}).click();await page.locator("#vm-segment").fill("0");
  await page.locator("#vector-map-panel").getByText("Build from a trajectory",{exact:true}).click();await page.locator("#vm-discover-after-build").uncheck();
  for(let i=0;i<3;i++){
    await page.locator("#vm-trajectory").selectOption({index:i});await page.locator("#vm-build").click();
    await expect(page.locator("#status")).toContainText("Draft roads added");
  }
  await expect(page.locator("#vm-status")).toContainText("6 lanes");
  await page.locator("#vector-map-panel").getByText("Map display",{exact:true}).click();await page.locator("#vm-context").fill("45");await page.locator("#vm-plan").click();await page.locator("#vm-fit").click();
  await shot("2. Fit road heights & boundaries · widths remain priors",2);
  await page.locator("#vector-map-panel").getByText("Draft junction connections",{exact:true}).click();await page.locator("#vm-junction-preview").click();
  await expect(page.locator("#vm-junction-report")).toContainText("5 ground-supported candidates");
  const pairs=(await page.locator("#vm-junction-candidates label").allTextContents()).map(s=>s.match(/(\d+) → (\d+)/)!.slice(1).map(Number));
  expect(pairs).toEqual([[4,9],[4,15],[10,5],[14,5],[14,9]]);
  await shot("3. Review five point-supported junction connections",2);
  await page.locator("#vm-junction-apply").click();await expect(page.locator("#status")).toContainText("5 draft connections added");
  const roads=await exportMap();expect(roads.match(/k="subtype" v="road"/g)).toHaveLength(11);
  await page.locator("#vm-discovery summary").click();await page.locator("#vm-discovery-scope").selectOption("ground_surface");await page.locator("#vm-discovery-search").click();
  await expect(page.locator("#status")).toContainText("Found 129 unconfirmed equipment proposals",{timeout:120000});
  await expect(page.locator("#vm-discovery-report")).toContainText("806 detected");
  await expect(page.locator("#vm-discovery-report")).toContainText("906 windows (71 unsupported)");
  expect(expected.candidates).toHaveLength(129);expect(await exportMap()).toBe(roads);
  // The repeated-paint proposal is a false crossing on this source. Show review
  // and rejection instead of decorating the generated map with a fake crosswalk.
  await page.locator('input[name="discovered-feature"][value="94"]').check();
  await shot("4. Automatic proposals · inspect false paint patterns",2);
  await page.locator("#vm-discovery-reject").click();expect(await exportMap()).toBe(roads);
  await page.locator('input[name="discovered-feature"][value="93"]').check();
  await expect(page.locator("#vm-discovery-kind")).toHaveValue("");await expect(page.locator("#vm-discovery-lanes")).toHaveValue("");
  await shot("5. Observed stop marking · type and lane need review",2);
  await page.locator("#vm-discovery-kind").selectOption("stop_line");await page.locator("#vm-discovery-lanes").fill("14");await page.locator("#vm-discovery-add").click();
  await expect(page.locator("#status")).toContainText("Reviewed stop line",{timeout:120000});
  const stopped=await exportMap();expect(stopped).toContain('v="point_cloud_brightness_bar"');expect(stopped).not.toContain('v="stop_sign"');
  await page.locator('input[name="discovered-feature"][value="96"]').check();await page.locator('[data-view="iso"]').click();await page.locator("#vm-discovery-inspect").click();
  await expect(page.locator("#status")).toContainText("original points isolated");
  await page.locator("#point-size").fill("5");
  await shot("6. Inspect automatically located panel · original points",2.5);
  await page.locator("#vm-discovery-kind").selectOption("vehicle_signal");await page.locator("#vm-discovery-lanes").fill("14");await page.locator("#vm-discovery-add").click();
  await expect(page.locator("#status")).toContainText("Reviewed vehicle signal",{timeout:120000});
  const added=await exportMap();expect(added).toContain('v="user_confirmed_automatic_proposal"');expect(added).toContain('v="point_cloud_box_fit"');
  const measured=await page.evaluate(xml=>{
    const doc=new DOMParser().parseFromString(xml,"application/xml");
    return [31,33].map(id=>{
      const way=doc.querySelector(`way[id="${id}"]`)!;
      return [...way.querySelectorAll("nd")].map(nd=>{
        const node=doc.querySelector(`node[id="${nd.getAttribute("ref")}"]`)!;
        return ["local_x","local_y","ele"].map(key=>Number(node.querySelector(`tag[k="${key}"]`)!.getAttribute("v")));
      });
    });
  },added);
  const native=JSON.parse(readFileSync(`${PROOF}/reviewed.json`,"utf8"));
  const nativeGeometry=[native.stop_lines.find((s:any)=>s.id===31).geometry,native.traffic_signals.find((s:any)=>s.id===33).geometry];
  const maximumNativeDifference=Math.max(...measured.flatMap((g,i)=>g.flatMap((p,j)=>p.map((v,k)=>Math.abs(v-nativeGeometry[i][j][k])))));
  expect(maximumNativeDifference).toBeLessThan(1e-7);
  await page.locator("#vm-discovery-show").uncheck();
  await shot("7. Add reviewed housing geometry · no invented lamps",2);
  await page.locator("#vm-feature-editor summary").click();const z=Number(await page.locator("#vm-feature-z").inputValue());
  await page.locator("#vm-feature-z").fill(String(z+.02));await page.locator("#vm-feature-apply").click();await expect(page.locator("#status")).toContainText("edited");
  expect(await exportMap()).not.toBe(added);await shot("8. Refine measured geometry · retain lane assignments",1.5);
  await page.locator("#vm-undo").click();expect(await exportMap()).toBe(added);await shot("Undo restores the complete map exactly",1.4);
  // Cloud Undo restores the original source; it is separate from map Undo.
  await page.locator("#undo").click();await expect(page.locator("#status")).toContainText("Undid");
  await page.locator("#point-size").fill("2");await page.locator("#vm-plan").click();await page.locator("#vm-fit").click();
  await shot("9. Generated roads + reviewed stop and signal · save Lanelet2",2.5);
  await page.locator("#vm-iso").click();await shot("Source-only draft · operator paths, types & lanes",2.3);
  const final=await exportMap();expect(final).toBe(added);
  await page.locator("#vm-undo").click();expect(await exportMap()).toBe(stopped);await page.locator("#vm-undo").click();expect(await exportMap()).toBe(roads);
  await page.locator("#vm-file").setInputFiles({name:"generated.osm",mimeType:"application/xml",buffer:Buffer.from(final)});
  await expect(page.locator("#status")).toContainText("Opened generated.osm: 11 lanes");
  const reloaded=await exportMap();expect(reloaded).toContain('v="point_cloud_brightness_bar"');expect(reloaded).toContain('v="point_cloud_box_fit"');expect(errors).toEqual([]);
  writeFileSync(`${FRAMES}/verification.json`,JSON.stringify({inputMap:false,pointCount:1757841,pathSource:"operator-traced, not recorded drive",roadLanes:6,connections:pairs,proposals:129,detected:806,unsupported:71,discardedPaint:94,confirmedStop:93,confirmedPanel:96,explicitLane:14,maximumNativeDifference,exactEditUndo:true,exactAdditionUndo:true,localOsmRoundtrip:true,errors},null,2));
  writeFileSync(`${FRAMES}/frames.json`,JSON.stringify(frames,null,2));writeFileSync(`${FRAMES}/concat.txt`,frames.map(f=>`file '${f.name}'\nduration ${f.duration}\n`).join("")+`file '${frames.at(-1)!.name}'\n`);
});
