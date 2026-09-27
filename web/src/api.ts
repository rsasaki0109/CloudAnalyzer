// Promise-based client for the WASM worker.

import type { C2cOutput, IcpOutput, LoadedCloud, Request, WorkerMessage } from "./protocol";

const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
const pending = new Map<
  number,
  { resolve: (v: unknown) => void; reject: (e: Error) => void; progress?: (note: string) => void }
>();
let seq = 0;

worker.onmessage = (event: MessageEvent<WorkerMessage>) => {
  const message = event.data;
  const entry = pending.get(message.seq);
  if (!entry) return;
  if ("progress" in message) {
    entry.progress?.(message.progress);
    return;
  }
  pending.delete(message.seq);
  if (message.response.ok) entry.resolve(message.response.value);
  else entry.reject(new Error(message.response.error));
};

function call<T>(req: Request, transfer: Transferable[] = [], progress?: (note: string) => void): Promise<T> {
  const id = ++seq;
  return new Promise<T>((resolve, reject) => {
    pending.set(id, { resolve: resolve as (v: unknown) => void, reject, progress });
    worker.postMessage({ seq: id, req }, { transfer });
  });
}

export function loadCloud(
  name: string,
  bytes: ArrayBuffer,
  progress?: (note: string) => void,
): Promise<LoadedCloud> {
  return call({ kind: "load", name, bytes }, [bytes], progress);
}

/** C2C, or C2M when `reference` is a mesh (`signed` then applies). */
export function cloudToCloud(compared: number, reference: number, signed = true): Promise<C2cOutput> {
  return call({ kind: "c2c", compared, reference, signed });
}

export function removeCloud(id: number): Promise<void> {
  return call({ kind: "remove", id });
}

/** Exact (unshifted, f64) coordinates of a point in octree order. */
export function pointAt(id: number, index: number): Promise<[number, number, number]> {
  return call({ kind: "point", id, index });
}

export function registerIcp(params: Omit<Extract<Request, { kind: "icp" }>, "kind">): Promise<IcpOutput> {
  return call({ kind: "icp", ...params });
}

/** Apply a row-major 4x4 rigid transform to a cloud; returns it re-described. */
export function transformCloud(id: number, matrix: number[]): Promise<LoadedCloud> {
  return call({ kind: "transform", id, matrix });
}
