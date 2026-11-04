// Cloud requests: loading, inspecting, editing fields of, exporting and removing items.

import { invalidateGpuReference } from "../gpu-c2c";
import { invalidateCpuC2c } from "../cpu-c2c";
import { invalidateM3c2Preparation } from "../m3c2-preparation";
import { invalidateM3c2CoreSelection } from "../m3c2-core-selection";
import { invalidateIcpReference } from "../icp-preparation";
import { invalidateVoxelScores } from "../map-voxel-cache";
import { invalidateMeshSources } from "../mesh-preparation";
import { invalidateKnnPreparation } from "../knn-preparation";
import type { Request } from "../protocol";
import { describe, dropDetail, getCloud, items, newId, readDetail } from "./items";
import { buildIndex } from "./build-index";
import { readFullDensityBox, readPoints } from "./readers";

export const CLOUD_KINDS = [
  "copc-box",
  "load",
  "load-url",
  "detail",
  "point",
  "transform",
  "field",
  "set-field",
  "export",
  "discard-cloud",
  "remove",
] as const;
export type CloudRequest = Extract<Request, { kind: (typeof CLOUD_KINDS)[number] }>;
const kinds: ReadonlySet<string> = new Set(CLOUD_KINDS);
export const isCloudRequest = (req: Request): req is CloudRequest => kinds.has(req.kind);

export async function cloudRequest(
  req: CloudRequest,
  progress: (note: string, fraction?: number) => void,
  check: () => void,
  signal: AbortSignal,
): Promise<{ value: unknown; transfer: Transferable[] }> {
  switch (req.kind) {
    case "copc-box":
      return readFullDensityBox(req, progress, check, signal);
    case "load":
    case "load-url": {
      const { maxPoints } = req;
      const name = req.kind === "load" ? req.file.name : req.name;
      const displayName = req.kind === "load" ? req.displayName ?? name : name;
      let t = performance.now();
      const loaded =
        req.kind === "load"
          ? await readPoints({ file: req.file }, name, req.file.size, maxPoints, progress, check)
          : await readPoints({ url: req.url, total: req.size, etag: req.etag }, name, req.size, maxPoints, progress, check);
      const parse = performance.now() - t;
      try {
        check();
      } catch (err) {
        (loaded.kind === "mesh" ? loaded.mesh : loaded.cloud).free();
        throw err;
      }
      if (loaded.kind === "mesh") {
        const id = newId();
        items.set(id, { kind: "mesh", mesh: loaded.mesh, name: displayName });
        try { return describe(id, { parse, index: 0 }); }
        catch(error) {loaded.mesh.free();items.delete(id);throw error;}
      }
      const { cloud } = loaded;
      progress(`indexing ${cloud.length.toLocaleString()} points`);
      t = performance.now();
      let workers: number;
      try { workers = await buildIndex(cloud); check(); }
      catch(error) { cloud.free(); throw error; }
      const index = performance.now() - t;
      const id = newId();
      items.set(id, {
        kind: "cloud",
        cloud,
        name: displayName,
        keepEvery: loaded.keepEvery,
        filePoints: loaded.filePoints,
        copcLevels: loaded.copcLevels,
        sources: loaded.sources,
        detail: loaded.detail,
        copcSource: loaded.copcSource,
      });
      progress("preparing for display");
      try { return describe(id, { parse, index, workers }); }
      catch(error) {cloud.free();items.delete(id);throw error;}
    }
    case "detail":
      return readDetail(req);
    case "point": {
      const item = items.get(req.id);
      const xyz = item?.kind === "cloud" ? item.cloud.point(req.index) : undefined;
      if (!xyz) throw new Error("point not found");
      return { value: Array.from(xyz), transfer: [] };
    }
    case "transform": {
      getCloud(req.id).transform(new Float64Array(req.matrix));
      dropDetail(req.id);
      return describe(req.id);
    }
    case "field": {
      const cloud = getCloud(req.id);
      const values = req.name === "Z" ? cloud.heights() : cloud.attribute(req.name);
      if (!values) throw new Error(`no field "${req.name}"`);
      return { value: values, transfer: [values.buffer] };
    }
    case "set-field": {
      getCloud(req.id).setAttribute(req.name, req.values);
      return { value: null, transfer: [] };
    }
    case "export": {
      const item = items.get(req.id);
      const bytes =
        item?.kind === "mesh"
          ? item.mesh.export(req.format)
          : getCloud(req.id).export(req.format, req.scalar?.name, req.scalar?.values);
      return { value: bytes, transfer: [bytes.buffer] };
    }
    case "discard-cloud":
    case "remove": {
      const item = items.get(req.id);
      if (item?.kind === "cloud") {
        invalidateCpuC2c(item.cloud);
        invalidateM3c2Preparation(item.cloud);
        invalidateM3c2CoreSelection(item.cloud);
        invalidateGpuReference(item.cloud);
        invalidateIcpReference(item.cloud);
        invalidateVoxelScores(item.cloud);
        invalidateKnnPreparation(item.cloud);
        invalidateMeshSources(item.cloud);
        item.cloud.free();
      }
      if (item?.kind === "mesh") { invalidateMeshSources(item.mesh); item.mesh.free(); }
      items.delete(req.id);
      return { value: null, transfer: [] };
    }
  }
}
