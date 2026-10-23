/** Small portable world-XY area definitions; no clouds or calculated results. */
import { validateExpectedOverlap, type ExpectedOverlap } from "./distance-regions.ts";
import { projectChanged } from "./project-change.ts";

export const MAX_NAMED_AREAS = 32;
export interface NamedArea { name: string; bounds: ExpectedOverlap }
export function parseNamedAreas(value: unknown): NamedArea[] {
  if (!Array.isArray(value) || value.length > MAX_NAMED_AREAS) throw new Error("Use at most 32 named areas");
  const names = new Set<string>();
  return value.map(item => {
    if (!item || typeof item !== "object" || Array.isArray(item)) throw new Error("Invalid named area");
    const { name, bounds } = item as Record<string, unknown>;
    if (!Array.isArray(bounds)) throw new Error("Named areas need four XY bounds");
    if (typeof name !== "string" || !name.trim() || name.length > 80 || /[\x00-\x1f\x7f]/.test(name)) throw new Error("Area names need 1–80 characters without control characters");
    const label = name.trim();
    if (names.has(label)) throw new Error("Area names must be unique");
    names.add(label);
    return { name: label, bounds: validateExpectedOverlap(bounds as unknown as ExpectedOverlap)! };
  });
}
export function parseAreaFile(value: unknown): NamedArea[] {
  if (!value || typeof value !== "object" || Array.isArray(value) ||
      (value as Record<string, unknown>).schema !== "cloudanalyzer.named_review_areas.v1") throw new Error("Unsupported named area file");
  return parseNamedAreas((value as Record<string, unknown>).areas);
}
let saved: NamedArea[] = [];
const listeners = new Set<() => void>();
export function captureNamedAreas(): NamedArea[] { return structuredClone(saved); }
export function replaceNamedAreas(value: unknown): void {
  saved = parseNamedAreas(value);
  for (const listener of listeners) listener();
  projectChanged();
}
export function onNamedAreasChanged(listener: () => void): void { listeners.add(listener); }

export interface AreaVolumeRow extends NamedArea {
  added: number; removed: number; addedArea: number; removedArea: number;
  matchedCells: number; totalCells: number; outsideRegionCells: number;
}
export function areaVolumesCsv(rows: AreaVolumeRow[], unit: string): string {
  const label = (value: string) => '"' + (/^[=+\-@]/.test(value) ? "'" + value : value).replaceAll('"', '""') + '"';
  const output = ["area_name,min_x,min_y,max_x,max_y,coordinate_unit,comparison_available,fill,cut,net,fill_area,cut_area,compared_cells,uncompared_cells,outside_cells"];
  for (const r of rows) {
    const quantities = r.matchedCells ? [r.added, r.removed, r.added - r.removed, r.addedArea, r.removedArea] : ["", "", "", "", ""];
    output.push([label(r.name), ...r.bounds, label(unit), r.matchedCells > 0, ...quantities,
      r.matchedCells, r.totalCells - r.matchedCells, r.outsideRegionCells].join(","));
  }
  return output.join("\n") + "\n";
}
