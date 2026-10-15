/** Reusable filter sequences with one Undo step and staged native outputs. */
import { exportCloud, filterBatch, discardCloud } from "../api";
import { parseFilterRecipe, parseRecipeStep, recipeStepLabel, type FilterRecipe, type ProcessingRecord, type RecipeStep } from "../filter-recipe";
import { nativeCloudEstimate } from "../memory-budget";
import { onProjectChanged } from "../project-change";
import type { LoadedCloud } from "../protocol";
import { $, download, errorText, setStatus } from "./dom";
import { addEntry, renderList } from "./entries";
import { cloudHistoryReady, record, requireDerivedUndo } from "./history";
import { clouds, entries, listChanged } from "./state";
import { endTask, showProgress, startTask, taskActive } from "./tasks";
import { graphProjectReady } from "./posegraph";
import { mapProjectReady } from "./vectormap";
import wasmUrl from "../wasm/ca_wasm_bg.wasm?url";

const picker=$("recipe-clouds"),stepsView=$("recipe-steps"),run=$<HTMLButtonElement>("recipe-run");
const name=$<HTMLInputElement>("recipe-name");
let steps:RecipeStep[]=[],running=false,revision=0;
let engineHash:string|undefined;
onProjectChanged(()=>revision++);
const selected=()=>[...picker.querySelectorAll<HTMLInputElement>("input:checked")].map(el=>entries.get(Number(el.value))).filter(e=>e!==undefined);
function renderSources():void {
  const chosen=new Set(selected().map(e=>e.cloud.id));picker.replaceChildren();
  for(const entry of clouds()){
    const label=document.createElement("label"),input=document.createElement("input");
    input.type="checkbox";input.value=String(entry.cloud.id);input.checked=chosen.has(entry.cloud.id);
    input.onchange=renderRun;
    label.append(input,document.createTextNode(entry.cloud.name));picker.append(label);
  }
  renderRun();
}
function renderRun():void {run.disabled=running||!steps.length||!selected().length;}
function renderSteps():void {
  stepsView.replaceChildren();
  steps.forEach((step,i)=>{
    const li=document.createElement("li"),button=document.createElement("button");
    li.append(document.createTextNode(recipeStepLabel(step)+" "));
    button.textContent="Remove";button.title="Remove recipe step "+(i+1);
    button.onclick=()=>{steps.splice(i,1);renderSteps();};li.append(button);stepsView.append(li);
  });
  renderRun();
}
function currentStep():RecipeStep {
  const op=$<HTMLSelectElement>("filter-op").value;
  const value=(id:string)=>Number($<HTMLInputElement>(id).value);
  switch(op){
    case "voxel":return parseRecipeStep({op,size:value("filter-voxel")});
    case "spatial":return parseRecipeStep({op,spacing:value("filter-spacing")});
    case "octree":return parseRecipeStep({op,level:value("filter-level")});
    case "sor":return parseRecipeStep({op,neighbors:value("filter-k"),stdRatio:value("filter-ratio")});
    case "splat":return parseRecipeStep({op,minOpacity:value("filter-opacity"),maxSize:value("filter-size")});
    default:throw new Error("Recipes support voxel, spatial, octree, SOR and splat cleanup");
  }
}
const recipe=():FilterRecipe=>parseFilterRecipe({app:"CloudAnalyzer Filter Recipe",version:1,name:name.value,steps});
$("recipe-add").onclick=()=>{
  try {if(steps.length>=8)throw new Error("A recipe can contain at most 8 steps");steps.push(currentStep());renderSteps();}
  catch(error){setStatus(errorText(error),true);}
};
$("recipe-visible").onclick=()=>{
  for(const input of picker.querySelectorAll<HTMLInputElement>("input"))input.checked=!!entries.get(Number(input.value))?.visible;
  renderRun();
};
$("recipe-export").onclick=()=>{
  try {download(new Blob([JSON.stringify(recipe(),null,2)],{type:"application/json"}),"filter-recipe.json");}
  catch(error){setStatus(errorText(error),true);}
};
const fileInput=$<HTMLInputElement>("recipe-file");
$("recipe-import").onclick=()=>fileInput.click();
fileInput.onchange=async()=>{
  const file=fileInput.files?.[0];fileInput.value="";if(!file)return;
  try {
    if(file.size>16384)throw new Error("Recipe files must fit in 16 KiB");
    const loaded=parseFilterRecipe(JSON.parse(await file.text()));steps=loaded.steps;name.value=loaded.name;renderSteps();
    setStatus("Recipe opened. Choose point clouds and press Run recipe to apply.");
  } catch(error){setStatus("Could not open recipe: "+errorText(error),true);}
};
async function pointHash(id:number,signal:AbortSignal):Promise<string>{
  signal.throwIfAborted();const bytes=await exportCloud(id,"ply");signal.throwIfAborted();
  if(bytes.length>64*1024*1024)throw new Error("A recipe point-record export exceeds 64 MiB; choose a smaller source");
  const hash=await crypto.subtle.digest("SHA-256",bytes as Uint8Array<ArrayBuffer>);signal.throwIfAborted();
  return [...new Uint8Array(hash)].map(b=>b.toString(16).padStart(2,"0")).join("");
}
run.onclick=async()=>{
  if(running)return;
  if(taskActive()||!cloudHistoryReady()||!mapProjectReady()||!graphProjectReady()){setStatus("Finish the current operation before running a recipe",true);return;}
  let plan:FilterRecipe;
  try {plan=recipe();} catch(error){setStatus(errorText(error),true);return;}
  const sources=selected();
  if(sources.length<1||sources.length>16){setStatus("Choose 1–16 point clouds",true);return;}
  if(sources.reduce((sum,e)=>sum+nativeCloudEstimate(e.cloud,e.fields.size),0)>256*1024*1024){setStatus("Selected clouds exceed the 256 MiB recipe estimate; run smaller batches",true);return;}
  running=true;renderRun();const signal=startTask(),originalRevision=revision;
  $("recipe-status").textContent="Preparing "+plan.name+"…";
  let results:LoadedCloud[]=[],committed=false;
  try {
    if(!engineHash){
      const response=await fetch(wasmUrl,{signal});if(!response.ok)throw new Error("Could not record the filter engine build");
      const hash=await crypto.subtle.digest("SHA-256",await response.arrayBuffer());signal.throwIfAborted();
      engineHash=[...new Uint8Array(hash)].map(b=>b.toString(16).padStart(2,"0")).join("");
    }
    const inputs:ProcessingRecord["input"][]=[];
    for(const source of sources){
      setStatus("Recording recipe input "+source.cloud.name+"…");
      inputs.push({name:source.cloud.name,points:source.cloud.count,sha256:await pointHash(source.cloud.id,signal)});
    }
    if(revision!==originalRevision)throw new Error("Current work changed during recipe preparation; original work is retained");
    const names=new Set([...entries.values()].map(e=>e.cloud.name)),outputs=sources.map(source=>{
      const base=source.cloud.name.replace(/\.[^.]+$/,"")+"_recipe";let result=base,index=2;
      while(names.has(result))result=base+index++;names.add(result);return {id:source.cloud.id,name:result};
    });
    results=await filterBatch(outputs,plan,p=>{showProgress(p);setStatus("Preparing recipe: "+p.note+"…");},signal);
    requireDerivedUndo(results,sources);
    const records:ProcessingRecord[]=[];
    for(const [i,cloud] of results.entries()){
      setStatus("Recording recipe output "+cloud.name+"…");
      records.push({version:1,wasmSha256:engineHash,recipe:plan,input:inputs[i],output:{points:cloud.count,sha256:await pointHash(cloud.id,signal)}});
    }
    signal.throwIfAborted();
    if(revision!==originalRevision||!cloudHistoryReady())throw new Error("Current work changed during recipe preparation; original work is retained");
    const added=results.map((cloud,i)=>{
      const entry=addEntry(cloud,{kind:"derived",displayPreview:sources[i].origin.displayPreview});entry.processing=records[i];return entry;
    });
    record({label:"recipe "+plan.name,added,hide:sources});committed=true;renderList();
    const message="Recipe complete: "+results.length+" clouds, "+plan.steps.length+" steps. Undo restores the entire batch.";
    $("recipe-status").textContent=message;setStatus(message);
  } catch(error){const message="Recipe failed: "+(signal.aborted?"Cancelled":errorText(error));$("recipe-status").textContent=message;setStatus(message,true);}
  finally {
    if(!committed)await Promise.all(results.map(cloud=>discardCloud(cloud.id)));
    endTask(signal);running=false;renderRun();
  }
};
listChanged.add(renderSources);renderSources();renderSteps();
