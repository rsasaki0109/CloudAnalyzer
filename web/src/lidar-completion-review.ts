/** File/projection verification for a separate experimental completion result.
 * Cached probability tensors are intentionally not loaded here: this is review,
 * not re-execution of the native two-seed/class-veto completion policy.
 */
import { MAX_SCAN_POINTS, readAiScanReview, verifyOwnership, type AiScanReview } from "./lidar-ai-review.ts";

type Value = Record<string, unknown>;
function object(value: unknown): Value {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Invalid completion report object");
  return value as Value;
}
function count(value: unknown): number {
  if (typeof value !== "number" || !Number.isSafeInteger(value) || value < 0 || value > MAX_SCAN_POINTS) {
    throw new Error("Invalid completion point count");
  }
  return value;
}
function hash(value: unknown): string {
  if (typeof value !== "string" || !/^[a-f0-9]{64}$/.test(value)) throw new Error("Invalid completion SHA256");
  return value;
}
async function sha(bytes: Uint8Array): Promise<string> {
  const result = await crypto.subtle.digest("SHA-256", bytes as Uint8Array<ArrayBuffer>);
  return [...new Uint8Array(result)].map(v => v.toString(16).padStart(2, "0")).join("");
}

/** Exactly six bounded files; the normal three-file native reader stays strict. */
export async function readCompletionScanReview(files: File[], signal: AbortSignal): Promise<AiScanReview> {
  signal.throwIfAborted();
  const required = ["report.json", "candidate-mask.u8", "completion-report.json", "completed-mask.u8", "added-mask.u8"];
  const sources = files.filter(f => /\.bin$/i.test(f.name));
  if (files.length !== 6 || sources.length !== 1 || required.some(name => files.filter(f => f.name === name).length !== 1)) {
    throw new Error("Choose exactly the original scan.bin, report.json, candidate-mask.u8, completion-report.json, completed-mask.u8 and added-mask.u8");
  }
  const get = (name: string) => files.find(f => f.name === name)!;
  if (get("completion-report.json").size > 65536 || ["completed-mask.u8", "added-mask.u8"].some(n => get(n).size > MAX_SCAN_POINTS)) {
    throw new Error("Completion review exceeds single-scan file limits");
  }
  const review = await readAiScanReview([sources[0], get("report.json"), get("candidate-mask.u8")], signal);
  const metadata = object(JSON.parse(await get("completion-report.json").text()));
  const base = object(metadata.base), policy = object(metadata.policy), counts = object(metadata.counts);
  if (metadata.app !== "CloudAnalyzer LiDAR surface completion experiment" || metadata.version !== 1 || metadata.source_modified !== false ||
      policy.id !== "near-surface-two-seeds-v1" || policy.radius_m !== .15 || policy.min_seed_separation_m !== .02 ||
      policy.seed_score !== Math.fround(.99) || policy.conflict_score !== Math.fround(.9) || policy.max_neighborhood_owners !== 256) {
    throw new Error("Unsupported experimental completion policy");
  }
  const baselineMetadata = object(JSON.parse(await review.reportFile.text()));
  if (metadata.input_sha256 !== review.report.sourceSha256 || base.onnx_sha256 !== baselineMetadata.onnx_sha256 ||
      base.weights_sha256 !== baselineMetadata.weights_sha256 || base.threshold !== review.report.threshold ||
      JSON.stringify(base.target_classes) !== JSON.stringify(review.report.targets)) {
    throw new Error("Completion source/model/selection differs from the native baseline");
  }
  hash(base.input_tensor_sha256); hash(base.probabilities_sha256);
  const addedCount = count(counts.added), removed = count(counts.total_removed), retained = count(counts.retained);
  const rejected = [counts.distance_rejected, counts.crowded_rejected, counts.conflict_rejected, counts.support_rejected].map(count);
  if (count(counts.original_removed) !== review.report.removed || removed !== review.report.removed + addedCount ||
      removed + retained !== review.report.points || addedCount + rejected.reduce((a, b) => a + b, 0) > review.report.projection.occluded) {
    throw new Error("Completion counts do not balance with the native baseline");
  }
  if (["completed-mask.u8", "added-mask.u8"].some(n => get(n).size !== review.report.points)) throw new Error("Completion mask length differs");
  const [raw, baseline, completed, added, reportBytes] = await Promise.all([
    review.source, review.maskFile, get("completed-mask.u8"), get("added-mask.u8"), review.reportFile,
  ].map(async f => new Uint8Array(await f.arrayBuffer())));
  signal.throwIfAborted();
  const hashes = await Promise.all([reportBytes, baseline, completed, added].map(sha));
  signal.throwIfAborted();
  if (hash(base.report_sha256) !== hashes[0] || hash(base.candidate_mask_sha256) !== hashes[1] ||
      hash(metadata.completed_mask_sha256) !== hashes[2] || hash(metadata.added_mask_sha256) !== hashes[3]) {
    throw new Error("Completion report or mask SHA256 differs");
  }
  if (completed.reduce((sum, value) => sum + value, 0) !== removed || added.reduce((sum, value) => sum + value, 0) !== addedCount) {
    throw new Error("Completion mask counts differ");
  }
  const { owners, classified, pixels } = verifyOwnership(raw, baseline, review.report);
  const view = new DataView(raw.buffer, raw.byteOffset, raw.byteLength);
  const keptBytes = new Uint8Array(retained * 16), candidateBytes = new Uint8Array(removed * 16), addedBytes = new Uint8Array(addedCount * 16);
  let k = 0, c = 0, a = 0;
  for (let i = 0; i < baseline.length; i++) {
    if ((completed[i] !== 0 && completed[i] !== 1) || (added[i] !== 0 && added[i] !== 1) ||
        completed[i] !== (baseline[i] | added[i]) || (baseline[i] & added[i])) {
      throw new Error("Completion must preserve baseline decisions and add disjoint 0/1 selections");
    }
    if (added[i]) {
      const owner = pixels[i] < 0 ? -1 : owners[pixels[i]];
      if (classified[i] || owner < 0 || !baseline[owner]) throw new Error("Completion addition must be an eligible collision with a selected original owner");
      let distance = 0;
      for (let axis = 0; axis < 3; axis++) distance += (view.getFloat32(i * 16 + axis * 4, true) - view.getFloat32(owner * 16 + axis * 4, true)) ** 2;
      if (distance > .15 ** 2) throw new Error("Completion addition exceeds the direct-owner radius");
      addedBytes.set(raw.subarray(i * 16, i * 16 + 16), a); a += 16;
    }
    const record = raw.subarray(i * 16, i * 16 + 16);
    if (completed[i]) { candidateBytes.set(record, c); c += 16; } else { keptBytes.set(record, k); k += 16; }
  }
  if (a !== addedBytes.length || c !== candidateBytes.length || k !== keptBytes.length) throw new Error("Completion mask counts differ");
  const [keptHash, candidateHash] = await Promise.all([sha(keptBytes), sha(candidateBytes)]);
  signal.throwIfAborted();
  if (hash(metadata.retained_sha256) !== keptHash || hash(metadata.candidate_sha256) !== candidateHash) {
    throw new Error("Completion record order or output SHA256 differs");
  }
  return { ...review, retained: new File([keptBytes], "retained.bin"), candidates: new File([candidateBytes], "candidates.bin"),
    completion: { baselineCandidates: review.candidates, added: new File([addedBytes], "added.bin"), addedCount, removed, retained } };
}
