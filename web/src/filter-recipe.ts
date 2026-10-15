/** Portable, bounded instructions for the five filters supported by version 1. */
export type RecipeStep =
  | {op: "voxel"; size: number}
  | {op: "spatial"; spacing: number}
  | {op: "octree"; level: number}
  | {op: "sor"; neighbors: number; stdRatio: number}
  | {op: "splat"; minOpacity: number; maxSize: number};
export interface FilterRecipe {
  app: "CloudAnalyzer Filter Recipe";
  version: 1;
  name: string;
  steps: RecipeStep[];
}
export interface ProcessingRecord {
  version: 1;
  wasmSha256: string;
  recipe: FilterRecipe;
  input: {name: string; points: number; sha256: string};
  output: {points: number; sha256: string};
}
const object = (v: unknown): Record<string, unknown> => {
  if (!v || typeof v !== "object" || Array.isArray(v)) throw new Error("Invalid filter recipe object");
  return v as Record<string, unknown>;
};
function keys(v: Record<string, unknown>, names: string[]): void {
  if (Object.keys(v).some(k => !names.includes(k))) throw new Error("Unknown filter recipe field");
}
function number(v: unknown, name: string, min: number, max = Number.MAX_VALUE, integer = false): number {
  if (typeof v !== "number" || !Number.isFinite(v) || v < min || v > max || (integer && !Number.isSafeInteger(v)))
    throw new Error("Invalid filter recipe " + name);
  return v;
}
export function parseRecipeStep(value: unknown): RecipeStep {
  const s = object(value);
  switch(s.op) {
    case "voxel":
      keys(s,["op","size"]);return {op:s.op,size:number(s.size,"voxel size",Number.MIN_VALUE)};
    case "spatial":
      keys(s,["op","spacing"]);return {op:s.op,spacing:number(s.spacing,"spacing",Number.MIN_VALUE)};
    case "octree":
      keys(s,["op","level"]);return {op:s.op,level:number(s.level,"octree level",1,21,true)};
    case "sor":
      keys(s,["op","neighbors","stdRatio"]);return {op:s.op,neighbors:number(s.neighbors,"neighbors",1,100,true),stdRatio:number(s.stdRatio,"standard deviation ratio",0)};
    case "splat":
      keys(s,["op","minOpacity","maxSize"]);return {op:s.op,minOpacity:number(s.minOpacity,"opacity",0,1),maxSize:number(s.maxSize,"maximum size",Number.MIN_VALUE)};
    default:throw new Error("Unsupported filter recipe operation; use voxel, spatial, octree, SOR or splat cleanup");
  }
}
export function parseFilterRecipe(value: unknown): FilterRecipe {
  const r=object(value);keys(r,["app","version","name","steps"]);
  if(r.app!=="CloudAnalyzer Filter Recipe"||r.version!==1)throw new Error("Unsupported filter recipe format");
  if(typeof r.name!=="string"||!r.name.trim()||r.name.length>80)throw new Error("Enter a recipe name of 1–80 characters");
  if(!Array.isArray(r.steps)||r.steps.length<1||r.steps.length>8)throw new Error("A recipe needs 1–8 steps");
  return {app:r.app,version:1,name:r.name.trim(),steps:r.steps.map(parseRecipeStep)};
}
export function recipeParameters(step: RecipeStep): {op: RecipeStep["op"]; a: number; b: number} {
  switch(step.op) {
    case "voxel":return {op:step.op,a:step.size,b:0};
    case "spatial":return {op:step.op,a:step.spacing,b:0};
    case "octree":return {op:step.op,a:step.level,b:0};
    case "sor":return {op:step.op,a:step.neighbors,b:step.stdRatio};
    case "splat":return {op:step.op,a:step.minOpacity,b:step.maxSize};
  }
}
export function recipeStepLabel(step: RecipeStep): string {
  switch(step.op) {
    case "voxel":return "Voxel size " + step.size;
    case "spatial":return "Minimum distance " + step.spacing;
    case "octree":return "Octree level " + step.level;
    case "sor":return "SOR: " + step.neighbors + " neighbors, ratio " + step.stdRatio;
    case "splat":return "Splat opacity ≥ " + step.minOpacity + ", size ≤ " + step.maxSize;
  }
}
export function parseProcessingRecord(value: unknown): ProcessingRecord {
  const p=object(value),i=object(p.input),o=object(p.output);keys(p,["version","wasmSha256","recipe","input","output"]);keys(i,["name","points","sha256"]);keys(o,["points","sha256"]);
  if(p.version!==1||typeof i.name!=="string"||!i.name||i.name.length>1024)throw new Error("Invalid processing provenance");
  const hash=(v:unknown)=>{if(typeof v!=="string"||!/^[0-9a-f]{64}$/.test(v))throw new Error("Invalid processing record hash");return v;};
  const inputPoints=number(i.points,"input points",0,Number.MAX_SAFE_INTEGER,true),outputPoints=number(o.points,"output points",0,inputPoints,true);
  return {version:1,wasmSha256:hash(p.wasmSha256),recipe:parseFilterRecipe(p.recipe),input:{name:i.name,points:inputPoints,sha256:hash(i.sha256)},output:{points:outputPoints,sha256:hash(o.sha256)}};
}
