// Shape and cluster detection.

import type { Segment, SegmentOutput, Request, Vec3 } from "../protocol";
import { describe, getCloud, items, newId, splitItem } from "./items";
import { buildIndex } from "./build-index";
import { parallelNormals } from "./knn";

const SEGMENT_KINDS: Segment["kind"][] = ["plane", "sphere", "cylinder", "cluster", "other clusters", "rest"];
/**
 * Shapes (RANSAC) or clusters of a cloud: one cloud whose `source` is the
 * segment of each point, or split into a cloud per segment.
 */
export async function shapes(
  req: Extract<Request, { kind: "shapes" }>,
): Promise<{ value: SegmentOutput; transfer: Transferable[] }> {
  const start = performance.now();
  const source = items.get(req.id)!;
  const cloud = getCloud(req.id);
  let normals: Float32Array | null = null;
  if (req.method !== "cluster" && !cloud.normals()) {
    // Shapes are sampled with normals: estimate them on the pool and keep them.
    if (!(await parallelNormals(cloud, 12, "up"))) cloud.estimateNormals(12, "up");
    normals = cloud.normals()!;
  }
  const result =
    req.method === "cluster"
      ? cloud.clusters(req.distance, req.minPoints, !req.split)
      : cloud.detectShapes(req.method, req.distance, req.minPoints, req.maxShapes, !req.split);
  const kinds = result.kinds();
  const counts = result.counts();
  const params = result.params();
  const rms = result.rms();
  const colors = result.colors();
  const segments: Segment[] = Array.from(kinds, (k, i) => ({
    kind: SEGMENT_KINDS[k],
    count: counts[i],
    params: Array.from(params.subarray(i * 8, i * 8 + 8)),
    rms: rms[i],
    color: Array.from(colors.subarray(i * 3, i * 3 + 3)) as Vec3,
  }));
  const found = result.found;
  const segmented = result.takeCloud();
  result.free();
  const base = source.name.replace(/\.[^.]+$/, "");
  const names = segments.map((s, i) =>
    s.kind === "rest"
      ? `${base}_${req.method === "cluster" ? "noise" : "rest"}`
      : s.kind === "other clusters"
        ? `${base}_clusters${i + 1}+`
        : `${base}_${s.kind}${i + 1}`,
  );
  const value: SegmentOutput = { clouds: [], segments, found, normals, millis: 0 };
  const transfer: Transferable[] = normals ? [normals.buffer] : [];
  if (found === 0) {
    segmented.free();
  } else {
    const item = { kind: "cloud" as const, cloud: segmented, name: `${base}_${req.method}s`, sources: names };
    if (req.split) {
      const parts = await splitItem(item, "source");
      segmented.free();
      value.clouds = parts.value;
      transfer.push(...parts.transfer);
    } else {
      await buildIndex(segmented);
      const id = newId();
      items.set(id, item);
      const described = describe(id);
      value.clouds = [described.value];
      transfer.push(...described.transfer);
    }
  }
  value.millis = performance.now() - start;
  return { value, transfer };
}
