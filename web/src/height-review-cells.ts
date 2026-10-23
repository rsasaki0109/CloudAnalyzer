/** Bounded f64 screened grid-cell evidence; no point-cloud correspondence. */
import type { HeightReview, HeightReviewCell } from "./protocol.ts";
export function screenedCellBounds(c: HeightReviewCell, cell: number): readonly [number, number, number, number, number, number] {
  const bounds = [c.x - cell / 2, c.y - cell / 2, Math.min(c.before, c.after), c.x + cell / 2, c.y + cell / 2, Math.max(c.before, c.after)] as const;
  if (!Number.isFinite(cell) || cell <= 0 || !bounds.every(Number.isFinite)) throw new Error("Screened cell bounds cannot be represented");
  return bounds;
}
export function screenedCellsCsv(review: HeightReview, cell: number, unit: string, comparison: string): string {
  const quote = (s: string) => '"' + (/^\s*[=+\-@]/.test(s) ? "'" + s : s).replaceAll('"', '""') + '"';
  const lines = ["comparison,coordinate_unit,height_threshold,cell_size,listed_rank,grid_index,x,y,before_height,after_height,difference,kind,unlisted_cells"];
  for (const [i, c] of (review.cells ?? []).entries()) lines.push([quote(comparison), quote(unit), review.threshold, cell, i + 1, c.index, c.x, c.y, c.before, c.after, c.difference, c.difference > 0 ? "rise" : "fall", review.unlistedCells ?? ""].join(","));
  return lines.join("\n") + "\n";
}
