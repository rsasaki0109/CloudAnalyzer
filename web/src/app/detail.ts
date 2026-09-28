/**
 * Full detail for thinned LAS/LAZ clouds: the viewer asks for the chunks of
 * the file it wants drawn at full density (see `Viewer.onDetailWanted`) and
 * they are decoded from the file on the worker pool, a few at a time. Only
 * the newest list counts, so chunks that left the view before their turn
 * are never read.
 */

import * as THREE from "three";
import { readDetail } from "../api";
import { poolSize } from "../pool";
import { detailColors, detailShown } from "./colors";
import { errorText, setStatus } from "./dom";
import { type Entry, entries, toRender, viewer } from "./state";

let queue: { id: number; chunk: number }[] = [];
const inFlight = new Set<string>();
const MAX_IN_FLIGHT = Math.max(2, poolSize());

/** Chunks asked for and not drawable yet. */
export function detailLoading(): number {
  return queue.length + inFlight.size;
}

viewer.onDetailWanted = (wanted) => {
  queue = wanted.filter((w) => !inFlight.has(`${w.id}:${w.chunk}`));
  pump();
};

function pump(): void {
  while (inFlight.size < MAX_IN_FLIGHT && queue.length > 0) {
    const { id, chunk } = queue.shift()!;
    const entry = entries.get(id);
    if (!entry) continue;
    const { cloud } = entry;
    const key = `${id}:${chunk}`;
    inFlight.add(key);
    readDetail(id, chunk, cloud.shift)
      .then((points) => {
        // The cloud may have been removed or replaced meanwhile.
        if (entries.get(id)?.cloud !== cloud) return;
        viewer.addDetail(id, chunk, points.positions, detailColors(entry, points), points.pieces, points);
      })
      .catch((err: unknown) => {
        // E.g. the file changed on disk: keep showing the loaded points.
        viewer.setDetail(id, [], 1);
        setStatus(`${cloud.name}: full detail is not available (${errorText(err)})`, true);
      })
      .finally(() => {
        inFlight.delete(key);
        pump();
      });
  }
}

/** Tell the viewer about the chunks behind a thinned cloud, if it has any. */
export function drawDetail(entry: Entry): void {
  const { cloud } = entry;
  const table = cloud.detailChunks;
  if (!table) return;
  const chunks = [];
  for (let k = 0; k < table.length; k += 7) {
    const box = new THREE.Box3(toRender([table[k], table[k + 1], table[k + 2]]), toRender([table[k + 3], table[k + 4], table[k + 5]]));
    chunks.push({ box, count: table[k + 6] });
  }
  viewer.setDetail(cloud.id, chunks, cloud.keepEvery);
  viewer.setDetailActive(cloud.id, detailShown(entry));
}
