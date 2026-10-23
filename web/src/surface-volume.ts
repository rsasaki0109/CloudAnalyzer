/** Bounded volume UI/export helpers; the existing WASM engine computes totals. */
export const VOLUME_GRID_LIMIT = 1_000_000;
export const VOLUME_CSV_CELL_LIMIT = 100_000;

export function validateVolumeGrid(bounds: readonly (readonly number[])[], cell: number): void {
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
