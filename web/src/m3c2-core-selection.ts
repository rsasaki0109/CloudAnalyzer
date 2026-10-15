import { M3c2Plan, type Cloud } from "./wasm/ca_wasm.js";

const USE_M3C2_CORE_SELECTION = true;
const CORE_SELECTION_BUDGET = 8 * 1024 * 1024;
let current: { compared: Cloud; spacing: number; indices: Uint32Array } | undefined;

/** One bounded index list. Every plan still owns fresh coordinates/attributes
 * before the first yield, so edits/removal cannot change in-flight output. */
export function planM3c2Cores(compared: Cloud, spacing: number, reuse = true): {
  plan: M3c2Plan; hit: boolean; bytes: number;
} {
  if (!USE_M3C2_CORE_SELECTION || !reuse || !Number.isFinite(spacing) || spacing <= 0 ||
    compared.length > CORE_SELECTION_BUDGET / Uint32Array.BYTES_PER_ELEMENT) {
    current = undefined;
    return { plan: new M3c2Plan(compared, spacing), hit: false, bytes: 0 };
  }
  if (current?.compared === compared && current.spacing === spacing) {
    return { plan: M3c2Plan.fromIndices(compared, current.indices), hit: true, bytes: current.indices.byteLength };
  }
  current = undefined;
  const indices = M3c2Plan.selectedIndices(compared, spacing);
  if (indices.byteLength > CORE_SELECTION_BUDGET) throw new Error("M3C2 core selection exceeded its budget");
  const plan = M3c2Plan.fromIndices(compared, indices);
  current = { compared, spacing, indices };
  return { plan, hit: false, bytes: indices.byteLength };
}

export function invalidateM3c2CoreSelection(cloud: Cloud): void {
  if (current?.compared === cloud) current = undefined;
}
