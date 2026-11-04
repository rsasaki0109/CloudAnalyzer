// Runs the Rust/WASM core off the UI thread. All clouds and meshes live here
// so that analyses can use full f64 coordinates without copying them back and
// forth.

import init, { VectorMapSession, warmUp } from "./wasm/ca_wasm.js";
import {
  CANCELLED,
  type LoadedCloud,
  type Progress,
  type UiMessage,
  type Request,
  type Response,
  type WorkerMessage,
} from "./protocol";
import { poolMemory, releasePool, warmUpPool } from "./pool";
import { getCloud, items } from "./worker/items";
import { type NativeGraph, poseGraph, prepareGraphProject, setPoseGraph } from "./worker/pose-graph";
import { isPoseGraphRequest, poseGraphRequest } from "./worker/pose-graph-requests";
import { isCloudRequest, cloudRequest } from "./worker/cloud-requests";
import { isProcessingRequest, processingRequest } from "./worker/processing-requests";
import { isAnalysisRequest, analysisRequest } from "./worker/analysis-requests";

let poolPreparation: Promise<unknown> = Promise.resolve();
const ready = init().then((wasm) => {
  // Optimize the heavy kernels here and on the pool before the first file
  // arrives (see `warmUp`); requests wait for this one.
  warmUp();
  poolPreparation = warmUpPool();
  void poolPreparation.catch(error => console.warn("Could not warm the worker pool", error));
  return wasm;
});

/** The vector map, made on first use. */
let vectorMap: VectorMapSession | null = null;

let workspacePreparing = false;
let workspaceSequence = 0;
let preparedWorkspace: {token:number; map:VectorMapSession; graph:NativeGraph | null; ids:number[]} | null = null;
function releasePreparedWorkspace(): void {
  if (!preparedWorkspace) return;
  for (const id of preparedWorkspace.ids) {
    const item = items.get(id);
    if (item?.kind === "cloud") item.cloud.free();
    if (item?.kind === "mesh") item.mesh.free();
    items.delete(id);
  }
  preparedWorkspace.map.free();
  preparedWorkspace.graph?.session.free();
  preparedWorkspace = null;
}
/** A "vm" request's answer: JSON text, or for edits `{result, view}` with the view after the edit. */
function vectorMapRequest(req: Extract<Request, { kind: "vm" }>): string {
  const map = (vectorMap ??= new VectorMapSession());
  const withView = (result: string) => `{"result":${result},"view":${map.view()},"undo":${map.undoDepth}}`;
  switch (req.op) {
    case "open":
      return withView(map.open(req.name ?? "", req.text ?? ""));
    case "apply":
      return withView(map.apply(req.text ?? "[]"));
    case "relations-preview":
      return map.previewRelations(BigInt(req.id ?? 0));
    case "relations-adopt":
      return withView(map.adoptRelation(req.text ?? "{}"));
    case "relations-edit":
      return withView(map.editRelations(req.text ?? "{}"));
    case "feature-edit":
      return withView(map.editFeatureGeometry(req.text ?? "{}"));
    case "feature-discover":
    case "feature-confirm": {
      if(req.id === undefined) throw new Error("Choose a point cloud.");
      const request=JSON.parse(req.text ?? "{}");
      return req.op === "feature-discover" ? map.discoverFeatures(getCloud(req.id), JSON.stringify(request.options ?? {})) : withView(map.confirmFeatures(getCloud(req.id),JSON.stringify(request.options ?? {}),JSON.stringify(request.confirmations ?? [])));
    }
    case "build":
      if (req.id === undefined || !req.positions) throw new Error("Choose a point cloud and trajectory.");
      return withView(map.buildFromTrajectory(getCloud(req.id), req.positions, req.text ?? "{}"));
    case "quality":
      if (req.id === undefined) throw new Error("Choose a point cloud.");
      return map.auditQuality(getCloud(req.id));
    case "junction-preview":
      if (req.id === undefined) throw new Error("Choose a point cloud.");
      return map.previewJunctions(getCloud(req.id), req.text ?? "{}");
    case "signal-preview":
    case "signal-add":
      if (req.id === undefined) throw new Error("Choose a point cloud.");
      return withView(map.measureSignal(getCloud(req.id), req.text ?? "{}", req.op === "signal-preview"));
    case "crosswalk-preview":
    case "crosswalk-add":
      if (req.id === undefined) throw new Error("Choose a point cloud.");
      return withView(map.measureCrosswalk(getCloud(req.id), req.text ?? "{}", req.op === "crosswalk-preview"));
    case "junction-connect":
      if (req.id === undefined) throw new Error("Choose a point cloud.");
      return withView(map.connectJunctions(getCloud(req.id), req.text ?? "{}", JSON.stringify(req.pairs ?? null)));
    case "undo":
      return withView(String(map.undo()));
    case "clear":
      map.clear();
      return withView("null");
    case "history-budget": {
      const policy = JSON.parse(req.text ?? "{}");
      map.setHistoryBudget(policy.steps, policy.bytes);
      return withView("null");
    }
    case "history-clear":
      map.clearHistory(); return withView("null");
    case "view":
      return withView("null");
    case "validate":
      return map.validate(req.autoware ?? true);
    case "export":
      return map.exportLanelet2(req.autoware ?? true);
    case "nearest":
      return map.nearestLane(req.x ?? 0, req.y ?? 0);
    case "json":
      return map.toJson();
    case "check-project": {
      const candidate = new VectorMapSession();
      try { candidate.open("project-map.json", req.text ?? ""); }
      finally { candidate.free(); }
      return "null";
    }
  }
}

async function handle(
  req: Request,
  progress: (note: string, fraction?: number) => void,
  check: () => void,
  signal: AbortSignal,
): Promise<{ value: unknown; transfer: Transferable[] }> {
  await ready;
  if (isPoseGraphRequest(req)) return poseGraphRequest(req, progress, check);
  if (isCloudRequest(req)) return cloudRequest(req, progress, check, signal);
  if (isProcessingRequest(req)) return processingRequest(req, progress, check);
  if (isAnalysisRequest(req)) return analysisRequest(req);
  switch (req.kind) {
    case "workspace-prepare": {
      if (workspacePreparing || preparedWorkspace) throw new Error("A workspace import is already staged");
      if (req.clouds.length > 127) throw new Error("Workspace import exceeds cloud limit");
      workspacePreparing = true;
      try {
        const staged = {token:++workspaceSequence, map:new VectorMapSession(), graph:null as NativeGraph | null, ids:[] as number[]};
        preparedWorkspace = staged;
        const result = staged.map.open("project-map.json",req.mapText);
        const map = `{"result":${result},"view":${staged.map.view()},"undo":0}`;
        const graph = req.graph ? await prepareGraphProject(req.graph,req.graphName,progress,check) : null;
        staged.graph = graph?.native ?? null;
        const clouds:LoadedCloud[] = [], transfer:Transferable[] = [...(graph?.transfer ?? [])];
        for (const source of req.clouds) {
          check();
          progress(`Staging ${source.name}`);
          const loaded = await handle({kind:"load",file:source.file,displayName:source.name,maxPoints:Infinity},(note,fraction)=>progress(`${source.name}: ${note}`,fraction),check,signal);
          const cloud = loaded.value as LoadedCloud;
          staged.ids.push(cloud.id); clouds.push(cloud);transfer.push(...loaded.transfer);
        }
        check();
        return {value:{token:staged.token,clouds,graph:graph?.value ?? null,map},transfer};
      } catch(error) { releasePreparedWorkspace(); throw error; }
      finally {workspacePreparing = false;}
    }
    case "workspace-commit": {
      if (workspacePreparing || preparedWorkspace?.token !== req.token) throw new Error("Workspace stage is unavailable");
      const staged = preparedWorkspace;
      vectorMap?.free(); poseGraph?.session.free();
      vectorMap = staged.map; setPoseGraph(staged.graph);
      preparedWorkspace = null;
      return {value:null,transfer:[]};
    }
    case "workspace-discard": {
      if (workspacePreparing) throw new Error("Workspace preparation is still running");
      if (preparedWorkspace?.token === req.token) releasePreparedWorkspace();
      return {value:null,transfer:[]};
    }
    case "memory-stats":
      return {value: {main: (await ready).memory.buffer.byteLength, pool: poolMemory(), mapHistory: vectorMap?.historyBytes() ?? 0, mapSteps: vectorMap?.undoDepth ?? 0}, transfer: []};
    case "release-pool":
      if (controllers.size > 1) throw new Error("Finish the active operation before releasing memory");
      releasePool(); return {value: null, transfer: []};
    case "vm":
      return { value: vectorMapRequest(req), transfer: [] };
    case "cpu-prepare":
      await poolPreparation;
      return { value: undefined, transfer: [] };
  }
}
/** Requests asked to stop; they check between steps. */
const cancelled = new Set<number>();
const controllers = new Map<number, AbortController>();
self.onmessage = async (event: MessageEvent<UiMessage>) => {
  if ("cancel" in event.data) {
    if (controllers.has(event.data.cancel)) {
      cancelled.add(event.data.cancel);
      controllers.get(event.data.cancel)!.abort();
    }
    return;
  }
  const { seq, req } = event.data;
  const controller = new AbortController();
  controllers.set(seq, controller);
  let response: Response;
  let transfer: Transferable[] = [];
  const progress = (note: string, fraction?: number) => {
    const update: Progress = { note, fraction };
    const message: WorkerMessage = { seq, progress: update };
    self.postMessage(message);
  };
  const check = () => {
    if (cancelled.has(seq)) throw new Error(CANCELLED);
  };
  try {
    const out = await handle(req, progress, check, controller.signal);
    response = { ok: true, value: out.value };
    transfer = out.transfer;
  } catch (err) {
    let error = err instanceof Error ? err.message : String(err);
    if (cancelled.has(seq)) error = CANCELLED;
    if (/memory|allocation|unreachable/i.test(error)) {
      error = `out of memory (${error}); lower "Max points" and load the file again`;
    }
    response = { ok: false, error };
  }
  cancelled.delete(seq);
  controllers.delete(seq);
  const memory = (await ready).memory.buffer.byteLength;
  const message: WorkerMessage = { seq, response, memory, poolMemory: poolMemory() };
  self.postMessage(message, { transfer });
};
