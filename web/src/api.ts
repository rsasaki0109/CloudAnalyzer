// Promise-based client for the WASM worker.

import type { C2cOutput, LoadedCloud, Request, Response } from "./protocol";

const worker = new Worker(new URL("./worker.ts", import.meta.url), { type: "module" });
const pending = new Map<number, { resolve: (v: unknown) => void; reject: (e: Error) => void }>();
let seq = 0;

worker.onmessage = (event: MessageEvent<{ seq: number; response: Response }>) => {
  const { seq, response } = event.data;
  const entry = pending.get(seq);
  if (!entry) return;
  pending.delete(seq);
  if (response.ok) entry.resolve(response.value);
  else entry.reject(new Error(response.error));
};

function call<T>(req: Request, transfer: Transferable[] = []): Promise<T> {
  const id = ++seq;
  return new Promise<T>((resolve, reject) => {
    pending.set(id, { resolve: resolve as (v: unknown) => void, reject });
    worker.postMessage({ seq: id, req }, { transfer });
  });
}

export function loadCloud(name: string, bytes: ArrayBuffer): Promise<LoadedCloud> {
  return call({ kind: "load", name, bytes }, [bytes]);
}

export function cloudToCloud(compared: number, reference: number): Promise<C2cOutput> {
  return call({ kind: "c2c", compared, reference });
}

export function removeCloud(id: number): Promise<void> {
  return call({ kind: "remove", id });
}
