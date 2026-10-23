/** Bounded XY coverage diagnostics over every loaded point, not a terrain model. */
export const MAX_COVERAGE_CELLS = 262_144;
export const MAX_COVERAGE_INPUT_BYTES = 64 * 1024 * 1024;
export interface CoverageGrid {
  cell: number; minPoints: number; nx: number; ny: number;
  minX: number; minY: number; minZ: number; maxZ: number;
  points: number; finitePoints: number; invalidPoints: number;
  occupied: number; sparse: number; empty: number; enclosedEmpty: number;
  density: { min: number; median: number; max: number; mean: number };
  counts: Uint32Array;
  /** 0 exterior-connected empty, 1 occupied, 2 sparse, 3 enclosed empty. */
  categories: Uint8Array;
}
export function analyzeCoverage(positions: Float32Array, shift: readonly number[], cell: number, minPoints: number): CoverageGrid {
  if (!positions.length || positions.length % 3 || positions.byteLength > MAX_COVERAGE_INPUT_BYTES) {
    throw new Error("Coverage requires XYZ records up to 64 MiB; crop a smaller area for analysis");
  }
  if (!Number.isFinite(cell) || cell <= 0 || !Number.isFinite(cell * cell) || cell * cell === 0 ||
      !Number.isSafeInteger(minPoints) || minPoints < 1 || minPoints > 1_000_000 ||
      shift.length !== 3 || !shift.every(Number.isFinite)) throw new Error("Enter a positive cell size and an integer minimum point count (1–1,000,000)");
  let loX = Infinity, loY = Infinity, loZ = Infinity, hiX = -Infinity, hiY = -Infinity, hiZ = -Infinity, finitePoints = 0;
  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i], y = positions[i + 1], z = positions[i + 2];
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    finitePoints++; loX = Math.min(loX, x); loY = Math.min(loY, y); loZ = Math.min(loZ, z);
    hiX = Math.max(hiX, x); hiY = Math.max(hiY, y); hiZ = Math.max(hiZ, z);
  }
  if (!finitePoints) throw new Error("Coverage needs at least one finite XYZ point");
  const nx = Math.floor((hiX - loX) / cell) + 1, ny = Math.floor((hiY - loY) / cell) + 1;
  if (!Number.isSafeInteger(nx * ny) || nx * ny > MAX_COVERAGE_CELLS || nx > 2048 || ny > 2048) {
    throw new Error("Coverage grid exceeds 262,144 cells or 2,048 per axis; increase cell size or crop a smaller area");
  }
  const minX = loX + shift[0], minY = loY + shift[1], minZ = loZ + shift[2], maxZ = hiZ + shift[2];
  if (![minX, minY, minZ, maxZ, minX + nx * cell, minY + ny * cell].every(Number.isFinite) || minX + cell === minX || minY + cell === minY) {
    throw new Error("Coverage cell size cannot be represented in this coordinate frame");
  }
  const counts = new Uint32Array(nx * ny), categories = new Uint8Array(counts.length);
  for (let i = 0; i < positions.length; i += 3) {
    const x = positions[i], y = positions[i + 1], z = positions[i + 2];
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    // Half-open bins anchored at the finite minimum; an exact boundary starts
    // the next cell, including the maximum source coordinate.
    const col = Math.min(nx - 1, Math.floor((x - loX) / cell)), row = Math.min(ny - 1, Math.floor((y - loY) / cell));
    counts[row * nx + col]++;
  }
  let occupied = 0, sparse = 0;
  for (let i = 0; i < counts.length; i++) {
    if (counts[i]) { occupied++; categories[i] = counts[i] < minPoints ? 2 : 1; if (categories[i] === 2) sparse++; }
  }
  // Empty cells connected to the rectangular boundary through ANY of eight
  // neighbors are not enclosed gaps. No interpolation or expected survey mask.
  const queue = new Uint32Array(counts.length), exterior = new Uint8Array(counts.length);
  let tail = 0;
  const enqueue = (i: number) => { if (!counts[i] && !exterior[i]) { exterior[i] = 1; queue[tail++] = i; } };
  for (let x = 0; x < nx; x++) { enqueue(x); enqueue((ny - 1) * nx + x); }
  for (let y = 0; y < ny; y++) { enqueue(y * nx); enqueue(y * nx + nx - 1); }
  for (let head = 0; head < tail; head++) {
    const i = queue[head], x = i % nx, y = Math.floor(i / nx);
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      if (x + dx >= 0 && x + dx < nx && y + dy >= 0 && y + dy < ny) enqueue((y + dy) * nx + x + dx);
    }
  }
  let enclosedEmpty = 0;
  for (let i = 0; i < counts.length; i++) if (!counts[i] && !exterior[i]) { categories[i] = 3; enclosedEmpty++; }
  const sorted = counts.filter(v => v > 0).sort(), mid = Math.floor(sorted.length / 2), area = cell * cell;
  if (!Number.isFinite(finitePoints / area)) throw new Error("Density cannot be represented at this cell size");
  return { cell, minPoints, nx, ny, minX, minY, minZ, maxZ, points: positions.length / 3, finitePoints,
    invalidPoints: positions.length / 3 - finitePoints, occupied, sparse, empty: counts.length - occupied, enclosedEmpty,
    density: { min: sorted[0] / area, median: (sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2) / area,
      max: sorted[sorted.length - 1] / area, mean: finitePoints / (occupied * area) }, counts, categories };
}
