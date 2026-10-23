/** Portable repeated-visit summaries. Rows are always relative to one baseline. */
import type { AreaVolumeRow } from "./named-areas.ts";
import { areaVolumesCsv } from "./named-areas.ts";
export const MAX_VOLUME_VISITS = 8;
export interface VisitVolumeRow extends AreaVolumeRow { visit: string; visitIndex: number }
export const visitVolumeScope = "Each visit is compared independently with the fixed Before baseline on one shared union grid. Loaded order is not an acquisition date. Whole cells selected by center; no fractional clipping. Missing heights are excluded, never zero change. Common grid does not imply common observed cells: support may differ even when counts match. Do not subtract totals to infer between-visit earthwork. Overlapping areas must not be summed. Optional interpolation uses full loaded surfaces and can fill unobserved cells. Loaded/decimated data only; no registration, unit conversion, closed-solid volume or survey-accuracy guarantee. Results are 2.5D.";
export function visitVolumesCsv(rows: VisitVolumeRow[], unit: string, support?: VisitSupport): string {
  const label = (s: string) => '"' + (/^[=+\-@]/.test(s) ? "'" + s : s).replaceAll('"', '""') + '"';
  const header = "visit_index,visit_name," + areaVolumesCsv([], unit).trim() + (support ? ",support_mode,common_mask_sha256" : "");
  return header + "\n" + rows.map(row => `${row.visitIndex},${label(row.visit)},${areaVolumesCsv([row], unit).trimEnd().split("\n")[1]}${support ? "," + support.mode + "," + (support.sha256 ?? "") : ""}\n`).join("");
}

export type VisitSupportMode = "independent" | "common_observed";
export interface VisitSupport { mode: VisitSupportMode; sha256?: string; commonGridCells?: number; mask?: Uint8Array }
export const commonVolumeScope = "Same row-major support mask for baseline and every selected visit: each included cell has a finite raw height from finite XYZ points in every input, before interpolation. Point clouds only; interpolation disabled. Shared union grid and full cell-center-selected square areas. Per-area uncompared cells include cells absent in any selected input. Totals are baseline-relative 2.5D height volumes. Net differences use identical cells, but differences of gross fill/cut totals are not between-visit fill/cut; compute those directly. Selection/order is loaded order, not an inferred survey date. Overlapping areas must not be summed. Support is loaded/possibly decimated points, not true full-density coverage, alignment or field-accuracy proof.";
