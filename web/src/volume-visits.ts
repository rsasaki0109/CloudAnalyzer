/** Portable repeated-visit summaries. Rows are always relative to one baseline. */
import type { AreaVolumeRow } from "./named-areas.ts";
import { areaVolumesCsv } from "./named-areas.ts";
export const MAX_VOLUME_VISITS = 8;
export interface VisitVolumeRow extends AreaVolumeRow { visit: string; visitIndex: number }
export const visitVolumeScope = "Each visit is compared independently with the fixed Before baseline on one shared union grid. Loaded order is not an acquisition date. Whole cells selected by center; no fractional clipping. Missing heights are excluded, never zero change. Common grid does not imply common observed cells: support may differ even when counts match. Do not subtract totals to infer between-visit earthwork. Overlapping areas must not be summed. Optional interpolation uses full loaded surfaces and can fill unobserved cells. Loaded/decimated data only; no registration, unit conversion, closed-solid volume or survey-accuracy guarantee. Results are 2.5D.";
export function visitVolumesCsv(rows: VisitVolumeRow[], unit: string): string {
  const label = (s: string) => '"' + (/^[=+\-@]/.test(s) ? "'" + s : s).replaceAll('"', '""') + '"';
  const header = "visit_index,visit_name," + areaVolumesCsv([], unit).trim();
  return header + "\n" + rows.map(row => `${row.visitIndex},${label(row.visit)},${areaVolumesCsv([row], unit).trimEnd().split("\n")[1]}\n`).join("");
}
