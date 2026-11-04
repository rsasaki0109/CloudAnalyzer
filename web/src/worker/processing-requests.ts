// Processing requests that derive clouds or meshes: filters, normals, merge/split/crop, ground, segmentation and meshing.

import { Cloud, CloudMerger } from "../wasm/ca_wasm.js";
import type { LoadedCloud, MeshOutput, Request } from "../protocol";
import { parseFilterRecipe, recipeParameters } from "../filter-recipe";
import { describe, getCloud, items, newId, splitItem } from "./items";
import { buildIndex } from "./build-index";
import { parallelNormals, parallelSorMeans } from "./knn";
import { shapes } from "./shapes";

export const PROCESSING_KINDS = [
  "filter",
  "filter-batch",
  "normals",
  "merge",
  "split",
  "shapes",
  "ground",
  "crop",
  "segment",
  "filter-field",
  "mesh",
] as const;
export type ProcessingRequest = Extract<Request, { kind: (typeof PROCESSING_KINDS)[number] }>;
const kinds: ReadonlySet<string> = new Set(PROCESSING_KINDS);
export const isProcessingRequest = (req: Request): req is ProcessingRequest => kinds.has(req.kind);

/** Larger clouds are voxel-thinned before meshing, to bound time and memory. */
const MESH_MAX_POINTS = 5_000_000;

/** The current native voxel key packs 21 bits per relative axis. */
function requireVoxelRange(cloud:Cloud,size:number):void {
  const bounds=cloud.bounds(),extent=Math.max(bounds[3]-bounds[0],bounds[4]-bounds[1],bounds[5]-bounds[2]);
  const minimum=extent/((1<<21)-2);
  if(size<minimum)throw new Error("Voxel size is too small for this cloud's extent. Use at least "+(minimum*1.00001).toPrecision(6)+" in the cloud's units");
}

export async function processingRequest(
  req: ProcessingRequest,
  progress: (note: string, fraction?: number) => void,
  check: () => void,
): Promise<{ value: unknown; transfer: Transferable[] }> {
  switch (req.kind) {
    case "filter": {
      const source = items.get(req.id);
      const t = performance.now();
      const cloud = getCloud(req.id);
      if(req.op==="voxel")requireVoxelRange(cloud,req.a);
      const k = Math.max(1, Math.floor(req.a));
      const means = req.op === "sor" ? await parallelSorMeans(cloud, k, req.knnCache, req.knnBounds, req.knnHint) : null;
      const filtered = means ? cloud.filterSor(means, req.b) : cloud.filter(req.op, req.a, req.b);
      await buildIndex(filtered);
      const id = newId();
      const base = source!.name.replace(/\.[^.]+$/, "");
      const suffix = {
        voxel: `voxel${req.a}`,
        random: `random${req.a}`,
        spatial: `space${req.a}`,
        octree: `octree${req.a}`,
        sor: "sor",
        splat: "clean",
      }[req.op];
      items.set(id, { kind: "cloud", cloud: filtered, name: `${base}_${suffix}` });
      return describe(id, { parse: 0, index: performance.now() - t });
    }
    case "filter-batch": {
      const recipe=parseFilterRecipe(req.recipe);
      if(req.sources.length<1||req.sources.length>16||new Set(req.sources.map(s=>s.id)).size!==req.sources.length)
        throw new Error("Choose 1–16 different point clouds");
      if(req.sources.some(s=>!s.name||s.name.length>1024)||new Set(req.sources.map(s=>s.name)).size!==req.sources.length)
        throw new Error("Batch output names must be unique");
      if(req.sources.reduce((sum,source)=>sum+getCloud(source.id).length,0)>4000000)
        throw new Error("Recipe batches support at most 4 million loaded points");
      const owned=new Set<Cloud>(),registered:number[]=[],finals:{cloud:Cloud;name:string;millis:number;sources:string[]|null}[]=[];
      try {
        for(const [sourceIndex,source] of req.sources.entries()){
          let current:Cloud|null=null;const start=performance.now(),sourceItem=items.get(source.id);
          const sourceNames=sourceItem?.kind==="cloud"?sourceItem.sources?.slice()??null:null;
          for(const [stepIndex,step] of recipe.steps.entries()){
            progress(`Cloud ${sourceIndex+1}/${req.sources.length}, step ${stepIndex+1}/${recipe.steps.length}: ${step.op}`,(sourceIndex*recipe.steps.length+stepIndex)/(req.sources.length*recipe.steps.length));
            await new Promise(resolve=>setTimeout(resolve,0));check();
            const input:Cloud=current??getCloud(source.id),{op,a,b}=recipeParameters(step);
            if(op==="voxel")requireVoxelRange(input,a);
            const means:Float64Array|null=op==="sor"?await parallelSorMeans(input,a):null;check();
            const filtered:Cloud=means?input.filterSor(means,b):input.filter(op,a,b);
            owned.add(filtered);
            await buildIndex(filtered);check();
            if(current){owned.delete(current);current.free();}
            current=filtered;
          }
          finals.push({cloud:current!,name:source.name,millis:performance.now()-start,sources:sourceNames});
        }
        check();const value:LoadedCloud[]=[],transfer:Transferable[]=[];
        for(const final of finals){
          const id=newId();registered.push(id);items.set(id,{kind:"cloud",cloud:final.cloud,name:final.name,sources:final.sources??undefined});
          const output=describe(id,{parse:0,index:final.millis});value.push(output.value);transfer.push(...output.transfer);
        }
        owned.clear();
        return {value,transfer};
      } catch(error){
        for(const id of registered)items.delete(id);
        for(const cloud of owned)cloud.free();
        throw error;
      }
    }
    case "normals": {
      const cloud = getCloud(req.id);
      const k = Math.max(3, Math.floor(req.k));
      if (!(await parallelNormals(cloud, k, req.orientation, req.knnCache, req.knnBounds, req.knnHint))) cloud.estimateNormals(k, req.orientation);
      const normals = cloud.normals()!;
      return { value: normals, transfer: [normals.buffer] };
    }
    case "merge": {
      const t = performance.now();
      const merger = new CloudMerger();
      const names: string[] = [];
      req.ids.forEach((id, k) => {
        merger.add(getCloud(id), ...req.fills[k]);
        names.push(items.get(id)!.name);
      });
      const merged = merger.finish();
      await buildIndex(merged);
      const id = newId();
      items.set(id, { kind: "cloud", cloud: merged, name: `merged_${names.length}`, sources: names });
      return describe(id, { parse: 0, index: performance.now() - t });
    }
    case "split": {
      const item = items.get(req.id);
      if (item?.kind !== "cloud") throw new Error("not a point cloud");
      return splitItem(item, req.by);
    }
    case "shapes":
      return shapes(req);
    case "ground": {
      const source = items.get(req.id);
      const t = performance.now();
      const cloud = getCloud(req.id).extractGround(req.clothResolution, req.classThreshold, req.rigidness, req.output);
      const id = newId();
      const base = source!.name.replace(/\.[^.]+$/, "");
      const suffix = { classified: "csf", ground: "ground", objects: "objects" }[req.output];
      items.set(id, { kind: "cloud", cloud, name: `${base}_${suffix}` });
      return describe(id, { parse: 0, index: performance.now() - t });
    }
    case "crop": {
      const source = items.get(req.id);
      const cropped = getCloud(req.id).crop(new Float64Array(req.min), new Float64Array(req.max), req.inside);
      const id = newId();
      const base = source!.name.replace(/\.[^.]+$/, "");
      items.set(id, { kind: "cloud", cloud: cropped, name: `${base}_${req.inside ? "crop" : "rest"}` });
      return describe(id);
    }
    case "segment": {
      const source = items.get(req.id);
      const t = performance.now();
      const cloud = getCloud(req.id);
      const clip = req.clip ? new Float64Array([...req.clip.min, ...req.clip.max]) : new Float64Array();
      const mask = cloud.lassoMask(
        new Float64Array(req.matrix),
        new Float64Array(req.polygon),
        clip,
        new Uint8Array(req.hiddenClasses),
      );
      const selected = mask.reduce((n, v) => n + v, 0);
      if (req.keep !== "outside" && selected === 0) throw new Error("the lasso selected no points");
      if (req.keep !== "inside" && selected === mask.length) throw new Error("the lasso selected every point");
      const base = source!.name.replace(/\.[^.]+$/, "");
      const parts: [number, string][] = [];
      if (req.keep !== "outside") parts.push([1, "segmented"]);
      if (req.keep !== "inside") parts.push([0, "remaining"]);
      const out: LoadedCloud[] = [];
      const transfer: Transferable[] = [];
      for (const [value, suffix] of parts) {
        const part = cloud.selectMask(mask, value);
        await buildIndex(part);
        const id = newId();
        items.set(id, { kind: "cloud", cloud: part, name: `${base}_${suffix}` });
        const described = describe(id, { parse: 0, index: performance.now() - t });
        out.push(described.value);
        transfer.push(...described.transfer);
      }
      return { value: out, transfer };
    }
    case "filter-field": {
      const source = items.get(req.id);
      const t = performance.now();
      const { values, lo, hi, inside } = req;
      const mask = new Uint8Array(values.length);
      for (let i = 0; i < values.length; i++) mask[i] = +(values[i] >= lo && values[i] <= hi);
      const part = getCloud(req.id).selectMask(mask, inside ? 1 : 0);
      await buildIndex(part);
      const id = newId();
      const base = source!.name.replace(/\.[^.]+$/, "");
      items.set(id, { kind: "cloud", cloud: part, name: `${base}_${inside ? "in" : "out"}` });
      return describe(id, { parse: 0, index: performance.now() - t });
    }
    case "mesh": {
      const start = performance.now();
      const out = getCloud(req.id).meshDelaunay(req.maxEdge ?? undefined, MESH_MAX_POINTS);
      const id = newId();
      const base = items.get(req.id)!.name.replace(/\.[^.]+$/, "");
      items.set(id, { kind: "mesh", mesh: out.takeMesh(), name: `${base}_mesh` });
      const described = describe(id);
      const value: MeshOutput = {
        mesh: described.value,
        points: out.points,
        voxel: out.voxel,
        maxEdge: out.maxEdge,
        removed: out.removed,
        millis: performance.now() - start,
      };
      out.free();
      return { value, transfer: described.transfer };
    }
  }
}
