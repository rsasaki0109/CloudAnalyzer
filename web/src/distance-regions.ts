/** Group unsigned source→reference discrepancies; this does not diagnose causes. */
export const REGION_INPUT_BYTES = 64 * 1024 * 1024;
export const REGION_CELL_LIMIT = 50_000;
export const REGION_LIST_LIMIT = 50;
export interface DistanceRegion {
  rank: number; points: number; maxDistance: number; meanDistance: number;
  bounds: [number, number, number, number, number, number];
}
export interface DistanceRegions {
  threshold: number; effectiveThreshold: number; cell: number; minPoints: number;
  points: number; measured: number; invalidPositions: number; unmeasured: number;
  exceeding: number; occupiedCells: number; components: number; eligibleRegions: number;
  smallRegionPoints: number; unlistedEligiblePoints: number;
  regions: DistanceRegion[];
}
interface Bucket { x: number; y: number; points: number; sum: number; max: number; bounds: number[] }
export interface DistanceCells {
  regionRank: number; regionPoints: number; threshold: number; effectiveThreshold: number;
  parentCell: number; cell: number; gridOrigin: [number, number]; occupiedCells: number; unlistedPoints: number;
  cells: (DistanceRegion & { gridX: number; gridY: number })[];
}
export function validateRegionSettings(threshold: number, cell: number, minPoints: number): number {
  const cutoff = Math.fround(threshold);
  if (!Number.isFinite(threshold) || threshold < 0 || !Number.isFinite(cutoff) || !Number.isFinite(cell) || cell <= 0 ||
      !Number.isSafeInteger(minPoints) || minPoints < 1 || minPoints > 1_000_000) {
    throw new Error("Enter a nonnegative distance threshold, positive grouping cell and integer minimum points (1–1,000,000)");
  }
  return cutoff;
}
export function locateDistanceRegions(positions: Float32Array, distances: Float32Array, shift: readonly number[],
  threshold: number, cell: number, minPoints: number): DistanceRegions {
  return analyze(positions, distances, shift, threshold, cell, minPoints).summary;
}
export function inspectDistanceCells(positions: Float32Array, distances: Float32Array, shift: readonly number[],
  threshold: number, cell: number, minPoints: number, regionRank: number, fineCell: number): DistanceCells {
  if (!Number.isSafeInteger(regionRank) || regionRank < 1 || regionRank > REGION_LIST_LIMIT ||
      !Number.isFinite(fineCell) || fineCell <= 0 || fineCell > cell) {
    throw new Error("Choose a listed region and a positive inspection cell no larger than the grouping cell");
  }
  return analyze(positions, distances, shift, threshold, cell, minPoints, { regionRank, fineCell }).inspection!;
}
function analyze(positions: Float32Array, distances: Float32Array, shift: readonly number[],
  threshold: number, cell: number, minPoints: number, inspect?: { regionRank: number; fineCell: number }):
  { summary: DistanceRegions; inspection?: DistanceCells } {
  if (!distances.length || positions.length !== distances.length * 3 || positions.byteLength + distances.byteLength > REGION_INPUT_BYTES) {
    throw new Error("Discrepancy input must have matching XYZ/distances up to 64 MiB; crop a smaller area");
  }
  const cutoff = validateRegionSettings(threshold, cell, minPoints);
  if (shift.length !== 3 || !shift.every(Number.isFinite)) throw new Error("Finite source coordinate shift required");
  let minX = Infinity, minY = Infinity, invalidPositions = 0, unmeasured = 0, measured = 0, exceeding = 0;
  for (let i = 0; i < distances.length; i++) {
    const x = positions[i * 3], y = positions[i * 3 + 1], z = positions[i * 3 + 2];
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) { invalidPositions++; continue; }
    minX = Math.min(minX, x); minY = Math.min(minY, y);
    const distance = distances[i];
    if (!Number.isFinite(distance)) { unmeasured++; continue; }
    if (distance < 0) throw new Error("Unsigned nearest-neighbor distances required; signed/M3C2 results are unsupported");
    measured++; if (distance > cutoff) exceeding++;
  }
  const buckets = new Map<string, Bucket>();
  const key = (x: number, y: number) => `${x},${y}`;
  for (let i = 0; i < distances.length; i++) {
    const distance = distances[i], px = positions[i * 3], py = positions[i * 3 + 1], pz = positions[i * 3 + 2];
    if (!Number.isFinite(distance) || distance <= cutoff || !Number.isFinite(px) || !Number.isFinite(py) || !Number.isFinite(pz)) continue;
    const x = Math.floor((px - minX) / cell), y = Math.floor((py - minY) / cell);
    if (!Number.isSafeInteger(x) || !Number.isSafeInteger(y) || x >= Number.MAX_SAFE_INTEGER - 1 || y >= Number.MAX_SAFE_INTEGER - 1) {
      throw new Error("Grouping cell cannot represent this extent; increase cell size or crop a smaller area");
    }
    const world = [px + shift[0], py + shift[1], pz + shift[2]];
    if (!world.every(Number.isFinite) || world[0] + cell === world[0] || world[1] + cell === world[1]) {
      throw new Error("Grouping cell cannot be represented in the coordinate frame");
    }
    let bucket = buckets.get(key(x, y));
    if (!bucket) {
      if (buckets.size === REGION_CELL_LIMIT) throw new Error("Discrepancies exceed 50,000 occupied cells; use a larger grouping cell or crop");
      bucket = { x, y, points: 0, sum: 0, max: 0, bounds: [...world, ...world] }; buckets.set(key(x, y), bucket);
    }
    bucket.points++; bucket.sum += distance; bucket.max = Math.max(bucket.max, distance);
    for (let axis = 0; axis < 3; axis++) { bucket.bounds[axis] = Math.min(bucket.bounds[axis], world[axis]); bucket.bounds[axis + 3] = Math.max(bucket.bounds[axis + 3], world[axis]); }
  }
  const occupiedCells = buckets.size, groups: (Bucket & { members: Bucket[] })[] = [], queue: Bucket[] = [];
  while (buckets.size) {
    const seed = buckets.values().next().value!; buckets.delete(key(seed.x, seed.y));
    const group: Bucket = { ...seed, points: 0, sum: 0, max: 0, bounds: [...seed.bounds] };
    queue.length = 0; queue.push(seed);
    for (let head = 0; head < queue.length; head++) {
      const item = queue[head]; group.points += item.points; group.sum += item.sum; group.max = Math.max(group.max, item.max);
      if (item.x < group.x || item.x === group.x && item.y < group.y) { group.x = item.x; group.y = item.y; }
      for (let axis = 0; axis < 3; axis++) { group.bounds[axis] = Math.min(group.bounds[axis], item.bounds[axis]); group.bounds[axis + 3] = Math.max(group.bounds[axis + 3], item.bounds[axis + 3]); }
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const id = key(item.x + dx, item.y + dy), neighbor = buckets.get(id);
        if (neighbor) { buckets.delete(id); queue.push(neighbor); }
      }
    }
    groups.push({ ...group, members: inspect ? queue.slice() : [] });
  }
  const eligible = groups.filter(g => g.points >= minPoints).sort((a, b) => b.max - a.max || b.points - a.points || a.x - b.x || a.y - b.y);
  const summary: DistanceRegions = { threshold, effectiveThreshold: cutoff, cell, minPoints, points: distances.length, measured, invalidPositions, unmeasured,
    exceeding, occupiedCells, components: groups.length, eligibleRegions: eligible.length,
    smallRegionPoints: groups.filter(g => g.points < minPoints).reduce((n, g) => n + g.points, 0),
    unlistedEligiblePoints: eligible.slice(REGION_LIST_LIMIT).reduce((n, g) => n + g.points, 0),
    regions: eligible.slice(0, REGION_LIST_LIMIT).map((g, i) => ({ rank: i + 1, points: g.points, maxDistance: g.max,
      meanDistance: g.sum / g.points, bounds: g.bounds as DistanceRegion["bounds"] })) };
  if (!inspect) return { summary };
  const selected = eligible[inspect.regionRank - 1];
  if (!selected) throw new Error("Selected region is no longer available");
  const members = new Set(selected.members.map(b => key(b.x, b.y))), fine = new Map<string, Bucket>();
  for (let i = 0; i < distances.length; i++) {
    const d = distances[i], px = positions[i * 3], py = positions[i * 3 + 1], pz = positions[i * 3 + 2];
    if (!Number.isFinite(d) || d <= cutoff || !Number.isFinite(px) || !Number.isFinite(py) || !Number.isFinite(pz) ||
        !members.has(key(Math.floor((px - minX) / cell), Math.floor((py - minY) / cell)))) continue;
    const x = Math.floor((px - minX) / inspect.fineCell), y = Math.floor((py - minY) / inspect.fineCell);
    const world = [px + shift[0], py + shift[1], pz + shift[2]];
    if (!Number.isSafeInteger(x) || !Number.isSafeInteger(y) || world[0] + inspect.fineCell === world[0] || world[1] + inspect.fineCell === world[1]) {
      throw new Error("Inspection cell cannot represent this extent; increase cell size or crop");
    }
    let b = fine.get(key(x, y));
    if (!b) {
      if (fine.size === REGION_CELL_LIMIT) throw new Error("Inspection exceeds 50,000 occupied cells; increase inspection cell size or crop");
      b = { x, y, points: 0, sum: 0, max: 0, bounds: [...world, ...world] }; fine.set(key(x, y), b);
    }
    b.points++; b.sum += d; b.max = Math.max(b.max, d);
    for (let axis = 0; axis < 3; axis++) { b.bounds[axis] = Math.min(b.bounds[axis], world[axis]); b.bounds[axis + 3] = Math.max(b.bounds[axis + 3], world[axis]); }
  }
  const ranked = [...fine.values()].sort((a, b) => b.max - a.max || b.points - a.points || a.x - b.x || a.y - b.y);
  return { summary, inspection: { regionRank: inspect.regionRank, regionPoints: selected.points, threshold, effectiveThreshold: cutoff,
    parentCell: cell, cell: inspect.fineCell, gridOrigin: [minX + shift[0], minY + shift[1]], occupiedCells: fine.size,
    unlistedPoints: ranked.slice(REGION_LIST_LIMIT).reduce((n, b) => n + b.points, 0),
    cells: ranked.slice(0, REGION_LIST_LIMIT).map((b, i) => ({ rank: i + 1, gridX: b.x, gridY: b.y, points: b.points,
      maxDistance: b.max, meanDistance: b.sum / b.points, bounds: b.bounds as DistanceRegion["bounds"] })) } };
}
export function distanceCellsCsv(result: DistanceCells): string {
  return ["region_rank,cell_rank,grid_x,grid_y,exceeding_points,max_distance,mean_distance,min_x,min_y,min_z,max_x,max_y,max_z",
    ...result.cells.map(c => [result.regionRank, c.rank, c.gridX, c.gridY, c.points, c.maxDistance, c.meanDistance, ...c.bounds].join(","))].join("\n") + "\n";
}
export function distanceRegionsCsv(result: DistanceRegions): string {
  return ["rank,exceeding_points,max_distance,mean_distance,min_x,min_y,min_z,max_x,max_y,max_z",
    ...result.regions.map(r => [r.rank, r.points, r.maxDistance, r.meanDistance, ...r.bounds].join(","))].join("\n") + "\n";
}
