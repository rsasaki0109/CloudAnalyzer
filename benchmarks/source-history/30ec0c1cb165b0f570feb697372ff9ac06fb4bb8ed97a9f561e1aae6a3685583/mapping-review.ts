/** Verify a portable generated-map review ZIP before any browser map changes. */
import type { SourceQualityReport } from "./app/vectormap";
import { indexReviewZip, readReviewMember } from "./review-zip.ts";
export { REVIEW_LIMIT } from "./review-zip.ts";
const SCHEMA = "cloudanalyzer.mapping_review_bundle.v1";
const PREVIEW_SCHEMA = "cloudanalyzer.mapping_review_bundle.v2";
const REQUIRED = ["map", "graph", "trajectory", "hd_map", "hd_editable_map", "hd_projector", "hd_source_audits", "layout_hypothesis", "source_proposal", "decision_history"];
type ObjectValue = Record<string, unknown>;
interface Descriptor { path: string; sha256: string; bytes: number }
export interface MappingReview {
  preview: { sourceCount: number; previewCount: number; stride: number } | null;
  roles: Record<string, string>;
  members: Map<string, File>;
  attribution: string;
  review: ObjectValue;
  verifiedFiles: number;
  uncompressedBytes: number;
}

export interface SavedAudit { label: string; report: SourceQualityReport }
/** Bound and validate saved reports before using their locations in the viewer. */
export function parseSavedAudits(value: unknown): SavedAudit[] {
  const audits = object(value), consensus = object(audits.ground_consensus);
  const sources = [["low quantile / editable IR", audits.editable], ["low quantile / reopened OSM", audits.reopened_osm],
    ["ground consensus / editable IR", consensus.editable], ["ground consensus / reopened OSM", consensus.reopened_osm]] as const;
  return sources.map(([label, audit]) => {
    const q = object(object(audit).quality);
    const ids = (v: unknown) => Array.isArray(v) && v.length <= 1024 && v.every(i => Number.isSafeInteger(i) && i > 0);
    if (!Array.isArray(q.lanes) || q.lanes.length > 256 || !Array.isArray(q.problems) || q.problems.length > 4096 ||
      !ids(q.low_support_lanes) || !ids(q.omitted_lanes) || !ids(q.malformed_lanes) || typeof q.limited !== "boolean" ||
      typeof q.problems_limited !== "boolean" || !Array.isArray(q.warnings) || q.warnings.length > 1024 || q.warnings.some(w => typeof w !== "string")) throw new Error("Incomplete or excessive saved source audit");
    const lanes = new Set<number>();
    for (const value of q.lanes) {
      const lane = object(value);
      if (!Number.isSafeInteger(lane.lane) || (lane.lane as number) <= 0 || lanes.has(lane.lane as number) || typeof lane.needs_review !== "boolean") throw new Error("Invalid saved audit lane");
      lanes.add(lane.lane as number);
      for (const curve of [lane.left, lane.right, lane.center]) {
        const support = object(curve);
        if (typeof support.fraction !== "number" || !Number.isFinite(support.fraction) || support.fraction < 0 || support.fraction > 1 ||
          typeof support.start_supported !== "boolean" || typeof support.end_supported !== "boolean" ||
          ![support.insufficient_returns, support.height_mismatches].every(n => Number.isSafeInteger(n) && (n as number) >= 0)) throw new Error("Invalid saved curve support");
      }
    }
    let points = 0;
    for (const value of q.problems) {
      const p = object(value);
      if (!lanes.has(p.lane as number) || !["center", "left", "right"].includes(p.curve as string) || !["insufficient_returns", "height_mismatch"].includes(p.reason as string) ||
        typeof p.from_m !== "number" || !Number.isFinite(p.from_m) || typeof p.to_m !== "number" || !Number.isFinite(p.to_m) || p.to_m < p.from_m ||
        !Array.isArray(p.points) || !p.points.length || p.points.some(x => !Array.isArray(x) || x.length !== 3 || x.some(n => typeof n !== "number" || !Number.isFinite(n)))) throw new Error("Invalid saved audit location");
      points += p.points.length;
      if (points > 100000) throw new Error("Saved audit location budget exceeded");
    }
    return { label, report: q as unknown as SourceQualityReport };
  });
}

function object(value: unknown): ObjectValue {
  if (!value || typeof value !== "object" || Array.isArray(value)) throw new Error("Invalid review metadata");
  return value as ObjectValue;
}
function descriptor(value: unknown): Descriptor {
  const file = object(value);
  const name = file.path;
  if (typeof name !== "string" || !name || name.startsWith("/") || name.includes("\\") || name.split("/").some(p => !p || p === "." || p === "..")) throw new Error("Unsafe review member path");
  if (!name.startsWith("files/") || typeof file.sha256 !== "string" || !/^[0-9a-f]{64}$/.test(file.sha256) || !Number.isSafeInteger(file.bytes) || (file.bytes as number) < 0) throw new Error("Invalid review artifact identity");
  return { path: name, sha256: file.sha256, bytes: file.bytes as number };
}
const decoder = new TextDecoder("utf-8", { fatal: true });
/** All members are checked, even when only the map and audits are kept for display. */
export async function readMappingReview(file: File, signal: AbortSignal, progress: (done: number, total: number) => void = () => {}): Promise<MappingReview> {
  const directory = await indexReviewZip(file, signal);
  const header = await readReviewMember(file, directory.get("manifest.json")!, signal);
  const manifest = object(JSON.parse(decoder.decode(header)));
  if (![SCHEMA, PREVIEW_SCHEMA].includes(manifest.schema as string) || !Array.isArray(manifest.files) || manifest.files.length > 128) throw new Error("Unsupported generated-map review schema");
  const rawPreview = manifest.schema === PREVIEW_SCHEMA ? object(manifest.preview_pointcloud) : null;
  const files = manifest.files.map(descriptor), paths = new Set(files.map(f => f.path));
  if (paths.size !== files.length || directory.size !== paths.size + 1 || [...directory.keys()].some(k => k !== "manifest.json" && !paths.has(k))) throw new Error("Review members differ from manifest");
  const rawRoles = object(manifest.roles), roles: Record<string, string> = {};
  for (const [role, value] of Object.entries(rawRoles)) {
    if (typeof value !== "string" || !paths.has(value)) throw new Error("Review role references a missing member");
    roles[role] = value;
  }
  if (REQUIRED.some(role => !roles[rawPreview && role === "map" ? "preview_map" : role])) throw new Error("Review package is missing maps or evidence");
  const review = object(manifest.review), artifacts = object(review.artifacts);
  if (REQUIRED.slice(0, 7).some(role => !artifacts[role])) throw new Error("Delivered output is missing map roles");
  const byPath = new Map(files.map(f => [f.path, f]));
  let preview: MappingReview["preview"] = null;
  if (rawPreview) {
    const p = rawPreview, source = object(p.source), delivered = object(artifacts.map), packed = descriptor(p.file);
    const count = p.source_count as number, kept = p.preview_count as number, stride = p.every_nth_record as number, cap = p.max_preview_points as number;
    const member = byPath.get(roles.preview_map);
    if (![count, kept, stride, cap].every(n => Number.isSafeInteger(n) && n > 0) || count > 1e9 || cap > 1e6 || cap >= count ||
      stride !== Math.ceil(count / cap) || kept !== Math.ceil(count / stride) || object(manifest.pointcloud_summary).map_points !== count) throw new Error("Invalid preview sampling counts");
    if (typeof source.path !== "string" || !source.path || typeof source.sha256 !== "string" || !/^[0-9a-f]{64}$/.test(source.sha256) || !Number.isSafeInteger(source.bytes) || (source.bytes as number) < packed.bytes ||
      ["path", "bytes", "sha256"].some(k => source[k] !== delivered[k]) || packed.path !== roles.preview_map || !member || packed.bytes !== member.bytes || packed.sha256 !== member.sha256) throw new Error("Preview differs from original or packaged identity");
    if (p.purpose !== "display_only" || p.first_record !== 0 || p.source_for_saved_audits !== "original_full_point_map" || p.full_point_map_included !== false ||
      p.coordinate_frame_changed !== false || p.coordinate_or_attribute_quantization !== false || p.original_record_bytes_preserved !== true) throw new Error("Invalid display preview provenance");
    if (roles.map) throw new Error("Preview package must distinguish its original full point map");
    preview = { sourceCount: count, previewCount: kept, stride };
  }
  for (const [role, value] of Object.entries(artifacts)) {
    if (preview && role === "map") continue;
    const artifact = descriptor(value), original = byPath.get(artifact.path);
    if (roles[role] !== artifact.path || !original || original.bytes !== artifact.bytes || original.sha256 !== artifact.sha256) throw new Error("Delivered output differs from member identity");
  }
  const retain = new Set([preview ? roles.preview_map : roles.map, roles.hd_editable_map, roles.hd_source_audits]);
  const members = new Map<string, File>();
  let done = 0, total = header.byteLength;
  for (const artifact of files) {
    signal.throwIfAborted();
    const member = directory.get(artifact.path);
    if (!member || member.bytes !== artifact.bytes) throw new Error("Review artifact size differs from ZIP");
    const data = await readReviewMember(file, member, signal);
    const hash = [...new Uint8Array(await crypto.subtle.digest("SHA-256", data))].map(n => n.toString(16).padStart(2, "0")).join("");
    if (hash !== artifact.sha256) throw new Error(`Review artifact hash differs: ${artifact.path}`);
    if (retain.has(artifact.path)) members.set(artifact.path, new File([data], artifact.path.split("/").at(-1)!));
    total += data.byteLength;
    progress(++done, files.length);
  }
  signal.throwIfAborted();
  if (preview) {
    const file = members.get(roles.preview_map)!, bytes = new Uint8Array(await file.slice(0, 16384).arrayBuffer());
    const end = new TextEncoder().encode("end_header\n");
    let size = -1;
    for (let i = 0; i <= bytes.length - end.length; i++) if (end.every((v, j) => bytes[i + j] === v)) { size = i + end.length; break; }
    if (size < 0) throw new Error("Incomplete preview PLY header");
    const lines = decoder.decode(bytes.slice(0, size)).split("\n");
    if (lines.slice(0, 6).join("\n") !== `ply\nformat binary_little_endian 1.0\nelement vertex ${preview.previewCount}\nproperty double x\nproperty double y\nproperty double z`) throw new Error("Preview PLY differs from sampling metadata");
    const attributes = lines.slice(6, -2).map(line => /^property float ([A-Za-z_][A-Za-z_0-9]*)$/.exec(line)?.[1]);
    if (attributes.length > 16 || attributes.some(n => !n) || new Set(["x", "y", "z", ...attributes]).size !== attributes.length + 3) throw new Error("Invalid preview PLY fields");
    const width = 24 + 4 * attributes.length;
    if (rawPreview!.record_size_bytes !== width || file.size !== size + preview.previewCount * width) throw new Error("Preview record bytes differ from manifest");
  }
  return { preview, roles, members, attribution: typeof manifest.attribution === "string" ? manifest.attribution : "", review, verifiedFiles: files.length, uncompressedBytes: total };
}
