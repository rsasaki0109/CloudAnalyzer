/** Bounded volume UI/export helpers; the existing WASM engine computes totals. */
import type { HeightReview } from "./protocol.ts";
export const heightReviewScope = "User-defined strict height screening on the original f64 raster heights: after-minus-before > threshold is rise; < -threshold is fall; equality is within the band. Screened volumes use full height differences, not only excess above the threshold. Missing/outside cells remain excluded. Within-band does not mean unchanged or statistically insignificant. This is not calibrated survey uncertainty or an M3C2 detection limit. Full-cell 2.5D loaded-data estimates; interpolation, when enabled, includes estimated heights. Total cut/fill and the displayed compared-cell cloud remain unchanged. Screening is opt-in separately for single-pair and saved-area/visit comparisons. Up to 50 largest absolute exceedance cells are retained per row, ties by ascending row-major grid index; full aggregates include unlisted cells. Each row uses its own bounds and comparison support; overlapping areas or gross interval volumes must not be summed as unique earthwork.";
export function readHeightThreshold(enabled: boolean, text: string): number | undefined {
  if (!enabled) return undefined;
  const value = Number(text);
  if (!text.trim() || !Number.isFinite(value) || value < 0) throw new Error("Enter a finite nonnegative height screening threshold");
  return value;
}
export function heightReviewCsv(review: HeightReview, comparedCells: number, unit: string): string {
  const quoted = '"' + (/^\s*[=+\-@]/.test(unit) ? "'" + unit : unit).replaceAll('"', '""') + '"';
  return "height_threshold,coordinate_unit,comparison_available,rise_cells,fall_cells,within_cells,rise_area,fall_area,rise_volume,fall_volume,screened_net\n" +
    [review.threshold, quoted, comparedCells > 0, review.riseCells, review.fallCells, review.withinCells,
      ...(comparedCells ? [review.riseArea, review.fallArea, review.riseVolume, review.fallVolume, review.riseVolume - review.fallVolume] : ["", "", "", "", ""])].join(",") + "\n";
}
export const VOLUME_GRID_LIMIT = 1_000_000;
export const VOLUME_CSV_CELL_LIMIT = 100_000;

/** Lower-left cell boundary X/Y and dimensions; cell size is supplied separately. */
export type VolumeGrid = readonly [number, number, number, number];
export function sharedVolumeGrid(bounds: readonly (readonly number[])[], cell: number): VolumeGrid {
  if (!Number.isFinite(cell) || cell <= 0 || !Number.isFinite(cell * cell) || cell * cell === 0) {
    throw new Error("Enter a finite positive cell size with representable area");
  }
  if (!bounds.length || bounds.some(b => b.length !== 6 || !b.every(Number.isFinite) || b[0] > b[3] || b[1] > b[4])) {
    throw new Error("Volume needs a surface with finite XY bounds");
  }
  const minX = Math.min(...bounds.map(b => b[0])), minY = Math.min(...bounds.map(b => b[1]));
  const maxX = Math.max(...bounds.map(b => b[3])), maxY = Math.max(...bounds.map(b => b[4]));
  // Match the existing engine: the lowest data corner is the first cell center.
  const nx = Math.floor((maxX - minX) / cell + .5) + 1, ny = Math.floor((maxY - minY) / cell + .5) + 1;
  if (!Number.isSafeInteger(nx) || !Number.isSafeInteger(ny) || nx * ny > VOLUME_GRID_LIMIT) {
    throw new Error("Volume grid exceeds 1,000,000 cells; increase cell size or crop the surfaces");
  }
  const grid: VolumeGrid = [minX - .5 * cell, minY - .5 * cell, nx, ny];
  if (!grid.every(Number.isFinite)) throw new Error("Volume grid origin cannot be represented");
  return grid;
}
export function validateVolumeGrid(bounds: readonly (readonly number[])[], cell: number): void {
  sharedVolumeGrid(bounds, cell);
}

export function volumeCellsCsv(positions: Float32Array, shift: readonly number[], difference: Float32Array): string {
  if (positions.length !== difference.length * 3 || shift.length !== 3 || !shift.every(Number.isFinite)) {
    throw new Error("Volume CSV needs matching XYZ/height differences and a finite coordinate shift");
  }
  if (difference.length > VOLUME_CSV_CELL_LIMIT) throw new Error("CSV export exceeds 100,000 compared cells; increase cell size or crop the surfaces");
  const rows = ["cell_index,x,y,after_height,after_minus_before"];
  for (let i = 0; i < difference.length; i++) {
    const point = [positions[i * 3] + shift[0], positions[i * 3 + 1] + shift[1], positions[i * 3 + 2] + shift[2]];
    if (!point.every(Number.isFinite) || !Number.isFinite(difference[i])) throw new Error("Volume CSV contains an undefined compared cell");
    rows.push([i, ...point, difference[i]].join(","));
  }
  return rows.join("\n") + "\n";
}
