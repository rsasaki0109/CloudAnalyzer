/** Verify a native single-scan result before loading any records into the viewer.
 * The mask indexes original KITTI records, never the viewer's octree order.
 * Hash agreement binds the files together; it does not establish AI accuracy.
 */
export const MAX_SCAN_POINTS = 150_000;
export const SEMANTIC_CLASSES = [
  "unlabeled", "car", "bicycle", "motorcycle", "truck", "other-vehicle", "person",
  "bicyclist", "motorcyclist", "road", "parking", "sidewalk", "other-ground",
  "building", "fence", "vegetation", "trunk", "terrain", "pole", "traffic-sign",
] as const;
const PIXELS = 64 * 2048;
const CHECKPOINT = "1757b8917e81b39058feb78dd17d690d74a878cab272a2c7b47bd8e5869ee811";
const ONNX = "8d61577f0d9582adfa28baea4246f915c29d4c5d5e3b1d27adac83ffef4815f5";
type ObjectValue = Record<string, unknown>;
function object(value: unknown): ObjectValue {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Invalid AI report object");
  return value as ObjectValue;
}
function count(value: unknown): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0 || value > MAX_SCAN_POINTS) {
    throw new Error("Invalid AI report point count");
  }
  return value;
}
function hash(value: unknown): string {
  if (typeof value !== "string" || !/^[a-f0-9]{64}$/.test(value)) throw new Error("Invalid AI report SHA256");
  return value;
}
export interface AiScanReport {
  points: number;
  projected: number;
  unclassified: number;
  removed: number;
  retained: number;
  threshold: number;
  targets: string[];
  sourceSha256: string;
  candidateSha256: string;
  retainedSha256: string;
  projection: Record<"projected" | "occluded" | "invalid" | "outside_view" | "outside_range", number>;
}
export function parseAiScanReport(value: unknown): AiScanReport {
  const r = object(value), p = object(r.projection), s = object(r.selection), profile = object(r.sensor_profile);
  if (r.experimental !== true || r.model !== "SalsaNext SemanticKITTI20" || r.backend !== "wgpu" ||
      r.device_selection !== "discrete_gpu_0" || r.source_modified !== false ||
      r.source_weight_sha256 !== CHECKPOINT || r.onnx_sha256 !== ONNX) {
    throw new Error("Unsupported AI model report; use the native SalsaNext/wgpu experiment");
  }
  hash(r.weights_sha256);
  if (profile.height !== 64 || profile.width !== 2048 || profile.fov_up_deg !== 3 ||
      profile.fov_down_deg !== -25 || profile.max_range_m !== 100 || profile.remission !== "unchanged normalized0..1") {
    throw new Error("Unsupported AI sensor profile; one HDL-64 sensor-frame scan is required");
  }
  if (JSON.stringify(r.class_names) !== JSON.stringify(SEMANTIC_CLASSES)) throw new Error("Unsupported AI class mapping");
  const points = count(p.points), projected = count(p.projected), removed = count(s.removed), retained = count(s.retained);
  const projection = { projected, occluded: count(p.occluded), invalid: count(p.invalid),
    outside_view: count(p.outside_view), outside_range: count(p.outside_range) };
  const unclassified = count(s.unclassified_retained);
  if (!points || !projected || count(r.mask_bytes) !== points || removed + retained !== points ||
      Object.values(projection).reduce((a, b) => a + b, 0) !== points || projected + unclassified !== points || removed > projected) {
    throw new Error("AI report counts do not balance");
  }
  const threshold = s.threshold;
  if (typeof threshold !== "number" || !Number.isFinite(threshold) || threshold < 0 || threshold > 1) {
    throw new Error("Invalid AI score threshold");
  }
  const targets = s.target_classes;
  if (!Array.isArray(targets) || !targets.length || targets.length > 19 ||
      targets.some(c => typeof c !== "string" || !(SEMANTIC_CLASSES as readonly string[]).slice(1).includes(c))) {
    throw new Error("Invalid AI target classes");
  }
  const counts = s.class_counts;
  if (!Array.isArray(counts) || counts.length !== 20 || counts.map(count).reduce((a, b) => a + b, 0) !== projected ||
      targets.reduce((n, c) => n + count(counts[SEMANTIC_CLASSES.indexOf(c)]), 0) < removed) {
    throw new Error("AI class counts do not balance");
  }
  return { points, projected, unclassified, removed, retained, threshold, targets,
    sourceSha256: hash(r.input_sha256), candidateSha256: hash(r.candidate_sha256), retainedSha256: hash(r.retained_sha256), projection };
}
async function sha256(bytes: Uint8Array): Promise<string> {
  const result = await crypto.subtle.digest("SHA-256", bytes as Uint8Array<ArrayBuffer>);
  return [...new Uint8Array(result)].map(v => v.toString(16).padStart(2, "0")).join("");
}

/** Recompute nearest-only ownership independently from the raw sensor records. */
function verifyOwnership(raw: Uint8Array, mask: Uint8Array, report: AiScanReport): void {
  const owners = new Int32Array(PIXELS).fill(-1), ranges = new Float64Array(PIXELS).fill(Infinity);
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const actual = { projected: 0, occluded: 0, invalid: 0, outside_view: 0, outside_range: 0 };
  const upper = 3 * Math.PI / 180, lower = -25 * Math.PI / 180;
  let eligible = 0;
  for (let i = 0; i < mask.length; i++) {
    const x = view.getFloat32(i * 16, true), y = view.getFloat32(i * 16 + 4, true);
    const z = view.getFloat32(i * 16 + 8, true), intensity = view.getFloat32(i * 16 + 12, true);
    if (![x, y, z, intensity].every(Number.isFinite)) { actual.invalid++; continue; }
    if (intensity < 0 || intensity > 1) throw new Error("Source remission must already be in 0..1");
    const range = Math.hypot(Math.hypot(x, y), z);
    if (range < 1e-6 || range > 100) { actual.outside_range++; continue; }
    const pitch = Math.asin(z / range);
    if (pitch < lower || pitch > upper) { actual.outside_view++; continue; }
    const col = Math.min(2047, Math.floor(0.5 * (-Math.atan2(y, x) / Math.PI + 1) * 2048));
    const row = Math.min(63, Math.floor((upper - pitch) / (upper - lower) * 64));
    const pixel = row * 2048 + col;
    eligible++;
    if (range < ranges[pixel]) { ranges[pixel] = range; owners[pixel] = i; }
  }
  const classified = new Uint8Array(mask.length);
  for (const owner of owners) if (owner >= 0) { classified[owner] = 1; actual.projected++; }
  actual.occluded = eligible - actual.projected;
  for (const key of Object.keys(actual) as (keyof typeof actual)[]) {
    if (actual[key] !== report.projection[key]) throw new Error("Source projection counts differ from the AI report");
  }
  for (let i = 0; i < mask.length; i++) {
    if (mask[i] && !classified[i]) throw new Error("AI mask selects an unclassified or occluded return");
  }
}
export interface AiScanReview {
  source: File;
  reportFile: File;
  maskFile: File;
  retained: File;
  candidates: File;
  report: AiScanReport;
}

/** All reads are bounded; no inference, download, scene edits or source writes. */
export async function readAiScanReview(files: File[], signal: AbortSignal): Promise<AiScanReview> {
  signal.throwIfAborted();
  const reportFile = files.find(f => f.name === "report.json"), maskFile = files.find(f => f.name === "candidate-mask.u8");
  const sources = files.filter(f => /\.bin$/i.test(f.name));
  if (files.length !== 3 || !reportFile || !maskFile || sources.length !== 1) {
    throw new Error("Choose exactly the original scan.bin, report.json and candidate-mask.u8 together");
  }
  const source = sources[0];
  if (reportFile.size > 65536 || maskFile.size > MAX_SCAN_POINTS || source.size > MAX_SCAN_POINTS * 16 ||
      !source.size || source.size % 16) throw new Error("AI review exceeds single-scan file limits or has malformed XYZI records");
  const report = parseAiScanReport(JSON.parse(await reportFile.text()));
  if (source.size !== report.points * 16 || maskFile.size !== report.points) throw new Error("Source or mask length differs from AI report");
  const [raw, mask] = await Promise.all([source.arrayBuffer(), maskFile.arrayBuffer()].map(async p => new Uint8Array(await p)));
  signal.throwIfAborted();
  if (await sha256(raw) !== report.sourceSha256) throw new Error("Source SHA256 differs from AI report; choose the original scan");
  signal.throwIfAborted();
  if (mask.some(v => v !== 0 && v !== 1)) throw new Error("AI mask must contain only 0/1 bytes");
  if (mask.reduce((n, v) => n + v, 0) !== report.removed) throw new Error("AI mask count differs from report");
  verifyOwnership(raw, mask, report);
  const kept = new Uint8Array(report.retained * 16), candidates = new Uint8Array(report.removed * 16);
  let k = 0, c = 0;
  for (let i = 0; i < mask.length; i++) {
    const record = raw.subarray(i * 16, (i + 1) * 16);
    if (mask[i]) { candidates.set(record, c); c += 16; } else { kept.set(record, k); k += 16; }
  }
  const [keptSha, candidateSha] = await Promise.all([sha256(kept), sha256(candidates)]);
  signal.throwIfAborted();
  if (keptSha !== report.retainedSha256 || candidateSha !== report.candidateSha256) {
    throw new Error("AI mask record order or output SHA256 differs from report");
  }
  return { source, reportFile, maskFile, report,
    retained: new File([kept], "retained.bin"), candidates: new File([candidates], "candidates.bin") };
}
