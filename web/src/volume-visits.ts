/** Portable repeated-visit summaries. Rows are always relative to one baseline. */
import type { AreaVolumeRow } from "./named-areas.ts";
import { areaVolumesCsv } from "./named-areas.ts";
export const MAX_VOLUME_VISITS = 8;
export interface VisitVolumeRow extends AreaVolumeRow { visit: string; visitIndex: number }
export const visitVolumeScope = "Each visit is compared independently with the fixed Before baseline on one shared union grid. Visit order is user-controlled, initially loaded order; no acquisition dates are inferred. Whole cells selected by center; no fractional clipping. Missing heights are excluded, never zero change. Common grid does not imply common observed cells: support may differ even when counts match. Do not subtract totals to infer between-visit earthwork. Overlapping areas must not be summed. Optional interpolation uses full loaded surfaces and can fill unobserved cells. Loaded/decimated data only; no registration, unit conversion, closed-solid volume or survey-accuracy guarantee. Results are 2.5D.";
export function visitVolumesCsv(rows: VisitVolumeRow[], unit: string, support?: VisitSupport): string {
  const screening = rows.some(r => !!r.heightReview);
  const label = (s: string) => '"' + (/^[=+\-@]/.test(s) ? "'" + s : s).replaceAll('"', '""') + '"';
  const header = "visit_index,visit_name," + areaVolumesCsv([], unit, screening).trim() + (support ? ",support_mode,common_mask_sha256" : "");
  return header + "\n" + rows.map(row => `${row.visitIndex},${label(row.visit)},${areaVolumesCsv([row], unit, screening).trimEnd().split("\n")[1]}${support ? "," + support.mode + "," + (support.sha256 ?? "") : ""}\n`).join("");
}

export type VisitSupportMode = "independent" | "common_observed";
export interface VisitSupport { mode: VisitSupportMode; sha256?: string; commonGridCells?: number; mask?: Uint8Array }
export const commonVolumeScope = "Same row-major support mask for baseline and every selected visit: each included cell has a finite raw height from finite XYZ points in every input, before interpolation. Point clouds only; interpolation disabled. Shared union grid and full cell-center-selected square areas. Per-area uncompared cells include cells absent in any selected input. Totals are baseline-relative 2.5D height volumes. Net differences use identical cells, but differences of gross fill/cut totals are not between-visit fill/cut; compute those directly. Visit order is user-controlled, initially loaded order; no acquisition dates are inferred. Overlapping areas must not be summed. Support is loaded/possibly decimated points, not true full-density coverage, alignment or field-accuracy proof.";

/** Preserve explicit order when selection or unrelated cloud-list entries change. */
export function reconcileVisitOrder(previous: readonly number[], selected: readonly number[]): number[] {
  if ([...previous, ...selected].some(id => !Number.isSafeInteger(id) || id < 0) || new Set(selected).size !== selected.length || new Set(previous).size !== previous.length) throw new Error("Visit order needs unique runtime cloud IDs");
  const wanted = new Set(selected), retained = previous.filter(id => wanted.has(id)), known = new Set(retained);
  return [...retained, ...selected.filter(id => !known.has(id))];
}
export interface IntervalVolumeRow extends AreaVolumeRow {
  intervalIndex: number; fromVisitIndex: number; toVisitIndex: number; fromId: number; toId: number; fromVisit: string; toVisit: string;
}
export const intervalVolumeScope = "Direct after-minus-before height comparison for adjacent visits in the explicit order, not subtraction of baseline fill/cut totals. Same raw observed support mask across baseline and every selected visit, shared grid, height mode and named bounds. Point clouds only; interpolation off; full cell areas selected by closed centers. Missing-in-any-input cells are excluded. Empty support is unavailable, not zero change. Ordering is user-controlled, initially loaded order; no acquisition dates are inferred. Loaded/possibly decimated 2.5D surfaces, not closed-solid volumes, registration, physical-ground matching or field-accuracy proof. Overlapping areas and interval gross fill/cut must not be summed as unique earthwork.";
export function intervalVolumesCsv(rows: readonly IntervalVolumeRow[], unit: string, support: VisitSupport): string {
  const screening = rows.some(r => !!r.heightReview);
  const label = (s: string) => '"' + (/^\s*[=+\-@]/.test(s) ? "'" + s : s).replaceAll('"', '""') + '"';
  const header = "interval_index,from_visit_index,from_cloud_id,from_visit_name,to_visit_index,to_cloud_id,to_visit_name," + areaVolumesCsv([], unit, screening).trim() + ",support_mode,common_mask_sha256";
  return header + "\n" + rows.map(row => [row.intervalIndex, row.fromVisitIndex, row.fromId, label(row.fromVisit), row.toVisitIndex, row.toId, label(row.toVisit),
    areaVolumesCsv([row], unit, screening).trimEnd().split("\n")[1], support.mode, support.sha256 ?? ""].join(",") + "\n").join("");
}
