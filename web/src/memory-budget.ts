import type { LoadedCloud } from "./protocol";
/** Account each backing buffer once, even when fields and geometry share it. */
export function retainedBuffers(value: unknown, buffers = new Set<ArrayBufferLike>(), seen = new WeakSet<object>()): Set<ArrayBufferLike> {
  if (!value || typeof value !== "object" || seen.has(value)) return buffers;
  seen.add(value);
  if (ArrayBuffer.isView(value)) { buffers.add(value.buffer); return buffers; }
  if (value instanceof ArrayBuffer) { buffers.add(value); return buffers; }
  // Files/Blobs are external data, not retained typed-array allocations.
  if (value instanceof Blob) return buffers;
  const children = value instanceof Map || value instanceof Set ? [...value.values()] : Object.values(value);
  for (const child of children) retainedBuffers(child, buffers, seen);
  return buffers;
}
/** GPU uploads use the view length; interleaved attributes share their owner. */
export function geometryBytes(attributes: {owner: object; array: ArrayBufferView}[]): number {
  const owners = new Set<object>();
  let bytes = 0;
  for (const attribute of attributes) if (!owners.has(attribute.owner)) { owners.add(attribute.owner); bytes += attribute.array.byteLength; }
  return bytes;
}
/** Conservative proxy for retained native points/attributes/indexes, beyond UI arrays. */
export function nativeCloudEstimate(cloud: LoadedCloud, computedFields = 0): number {
  const pointBytes = 32 + (cloud.colors ? 3 : 0) + (cloud.normals ? 12 : 0) + (cloud.classification ? 1 : 0) + 4 * (cloud.scalarNames.length + computedFields);
  return 2 * (cloud.count * pointBytes + cloud.triangles * 12 + cloud.lodNodes.byteLength);
}
export function bufferBytes(value: unknown): number { return [...retainedBuffers(value)].reduce((sum,b) => sum + b.byteLength,0); }
/** Evict oldest entries; never retain a single entry larger than the budget. */
export function trimHistory<T>(list: T[], maxSteps: number, maxBytes: number, bytes: (list: T[]) => number): T[] {
  const removed: T[] = [];
  while (list.length && (list.length > maxSteps || bytes(list) > maxBytes)) removed.push(list.shift()!);
  return removed;
}
export interface HistoryPolicy { steps: number; bytes: number }
let policy: HistoryPolicy = {steps: 20, bytes: 128 * 1024 * 1024};
const listeners = new Set<(policy: HistoryPolicy) => void>();
export const historyPolicy = () => policy;
export function onHistoryPolicy(listener: (policy: HistoryPolicy) => void): void { listeners.add(listener); }
export function configureHistory(steps: number, bytes: number): void {
  if (!Number.isFinite(steps) || !Number.isFinite(bytes)) throw new Error("Invalid history budget");
  policy = {steps: Math.max(0, Math.min(100, Math.floor(steps))), bytes: Math.max(0, Math.min(2048 * 1024 * 1024, bytes))};
  for (const listener of listeners) listener(policy);
}
