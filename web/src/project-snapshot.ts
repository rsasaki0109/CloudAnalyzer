/** Exact current point/mesh PLY records alongside normal project metadata. */
import type { Project } from "./project";
import { matchesFile, parseSource, referenceFile, type SourceReference } from "./source-reference.ts";
import {
  indexReviewZip,
  readReviewMember,
  writeReviewZip,
  REVIEW_LIMIT,
  MANIFEST_LIMIT,
  WORKSPACE_MEMBER_LIMIT,
  WORKSPACE_LIMIT,
} from "./review-zip.ts";

const SCHEMA = "cloudanalyzer.project_snapshot.v1";
const ASSET_SCHEMA = "cloudanalyzer.project_snapshot.v2";
const LARGE_SCHEMA = "cloudanalyzer.project_snapshot.v3";
const hex = (bytes: ArrayBuffer) =>
  [...new Uint8Array(bytes)]
    .map((n) => n.toString(16).padStart(2, "0"))
    .join("");
interface Descriptor {
  path: string;
  name: string;
  bytes: number;
  sha256?: string;
  identity?: SourceReference;
}

async function checkSources(
  project: Project,
  files: File[],
  signal: AbortSignal,
): Promise<void> {
  if (
    !project?.session ||
    !Array.isArray(project.session.clouds) ||
    project.session.clouds.length !== files.length ||
    new Set(files.map((f) => f.name)).size !== files.length
  )
    throw new Error("Snapshot sources differ from project clouds");
  const used = new Set<string>();
  for (const cloud of project.session.clouds) {
    const source = cloud.source,
      file = files.find((f) => f.name === source?.name);
    if (
      !source ||
      source.kind !== "file" ||
      !file ||
      used.has(file.name) ||
      cloud.transforms?.length !== 0 ||
      cloud.loadMaxPoints !== 0 ||
      !(await matchesFile(file, source, signal))
    )
      throw new Error(
        "Snapshot cloud identity/frame differs from project metadata",
      );
    used.add(file.name);
  }
}

/** Original scan names must survive: pose graph node association uses them. */
async function checkAssets(project: Project, assets: File[], signal: AbortSignal): Promise<void> {
  const refs = [
    ...(project.poseGraph?.sources.flatMap(s => [...(s.graph ? [s.graph] : []), ...s.scans]) ?? []),
    ...(project.reviewArchive ? [project.reviewArchive] : []),
  ];
  const used = new Set<File>();
  for (const ref of refs) {
    signal.throwIfAborted();
    // Equal basenames across sessions may have different contents.
    let matched = false;
    for (const candidate of assets) {
      if (candidate.name === ref.name.split("/").at(-1) && await matchesFile(candidate, ref, signal)) {
        used.add(candidate); matched = true; break;
      }
    }
    if (!matched) throw new Error(`Snapshot is missing verified input ${ref.name}`);
  }
  if (assets.some(f => !used.has(f))) throw new Error("Snapshot contains unreferenced input assets");
}

export async function writeProjectSnapshot(
  project: Project,
  files: File[],
  signal: AbortSignal,
  assets: File[] = [],
  maximumBytes = WORKSPACE_LIMIT,
): Promise<Blob> {
  if (!Number.isSafeInteger(maximumBytes) || maximumBytes < 1 || maximumBytes > WORKSPACE_LIMIT)
    throw new Error("Invalid workspace content limit");
  if (files.length > 127)
    throw new Error("Snapshot supports at most 127 clouds/meshes");
  const metadata = new File(
    [JSON.stringify(project)],
    "project.cloudanalyzer.json",
    { type: "application/json" },
  );
  if (
    metadata.size > MANIFEST_LIMIT ||
    metadata.size + files.reduce((n, f) => n + f.size, 0) > maximumBytes
  )
    throw new Error(
      `Snapshot exceeds the ${maximumBytes / 1024 / 1024} MiB content or 10 MiB project metadata limit`,
    );
  const includeAssets = assets.length > 0;
  if (files.length + assets.length + 2 > WORKSPACE_MEMBER_LIMIT)
    throw new Error("Snapshot exceeds the original input member limit");
  if (metadata.size + [...files, ...assets].reduce((n, f) => n + f.size, 0) > maximumBytes)
    throw new Error(`Snapshot exceeds the ${maximumBytes / 1024 / 1024} MiB content limit including original inputs`);
  if (includeAssets) await checkAssets(project, assets, signal);
  await checkSources(project, files, signal);
  const all = [metadata, ...files, ...assets],
    descriptors: Descriptor[] = [],
    entries: [string, Blob][] = [];
  for (const [i, file] of all.entries()) {
    signal.throwIfAborted();
    const path =
      i === 0 ? "project.json" : i <= files.length
        ? `clouds/${String(i - 1).padStart(3, "0")}.ply`
        : `assets/${String(i - files.length - 1).padStart(4, "0")}`;
    descriptors.push({
      path,
      name: file.name,
      bytes: file.size,
      identity: { ...await referenceFile(file, signal), name: file.name },
    });
    entries.push([path, file]);
  }
  const manifest = {
    schema: LARGE_SCHEMA,
    files: descriptors,
    point_data: "current_loaded_records",
    coordinate_frame: "current_transformed_coordinates",
    unloaded_original_density_included: false,
    pose_graph_sources_included: includeAssets,
    cloud_count: files.length,
    review_archive_included: includeAssets && !!project.reviewArchive,
  };
  return writeReviewZip(
    [["manifest.json", new Blob([JSON.stringify(manifest)])], ...entries],
    signal,
    includeAssets ? WORKSPACE_MEMBER_LIMIT : 129,
    maximumBytes,
  );
}

/** Verify all members and source identities before any workspace loading. */
export async function readProjectSnapshot(
  file: File,
  signal: AbortSignal,
): Promise<File[]> {
  const directory = await indexReviewZip(file, signal, WORKSPACE_MEMBER_LIMIT, WORKSPACE_LIMIT);
  const manifest = JSON.parse(
    new TextDecoder("utf-8", { fatal: true }).decode(
      await readReviewMember(file, directory.get("manifest.json")!, signal),
    ),
  );
  if (
    ![SCHEMA, ASSET_SCHEMA, LARGE_SCHEMA].includes(manifest?.schema) ||
    !Array.isArray(manifest.files) ||
    manifest.files.length < 1 ||
    manifest.files.length > (manifest.schema === SCHEMA ? 128 : WORKSPACE_MEMBER_LIMIT - 1) ||
    manifest.point_data !== "current_loaded_records" ||
    manifest.coordinate_frame !== "current_transformed_coordinates" ||
    manifest.unloaded_original_density_included !== false ||
    (manifest.schema !== LARGE_SCHEMA && manifest.pose_graph_sources_included !== (manifest.schema === ASSET_SCHEMA)) ||
    (manifest.schema === LARGE_SCHEMA && typeof manifest.pose_graph_sources_included !== "boolean")
  )
    throw new Error("Unsupported workspace snapshot schema/provenance");
  const large = manifest.schema === LARGE_SCHEMA;
  if (!large && [...directory.values()].reduce((n, m) => n + m.bytes, 0) > REVIEW_LIMIT)
    throw new Error("Legacy workspace exceeds the 64 MiB content limit");
  const descriptors = manifest.files as Descriptor[];
  const cloudCount = manifest.schema === SCHEMA ? descriptors.length - 1 : manifest.cloud_count;
  if (!Number.isSafeInteger(cloudCount) || cloudCount < 0 || cloudCount > 127 || cloudCount >= descriptors.length)
    throw new Error("Invalid snapshot cloud count");
  if (
    directory.size !== descriptors.length + 1 ||
    new Set(descriptors.map((f) => f?.path)).size !== descriptors.length ||
    new Set(descriptors.slice(0, cloudCount + 1).map((f) => f?.name)).size !== cloudCount + 1 ||
    descriptors[0]?.path !== "project.json" ||
    descriptors[0]?.name !== "project.cloudanalyzer.json"
  )
    throw new Error("Snapshot members differ from manifest");
  const files: File[] = [];
  for (const [i, d] of descriptors.entries()) {
    const member = directory.get(d?.path);
    if (
      !member ||
      member.path === "manifest.json" ||
      !Number.isSafeInteger(d.bytes) ||
      member.bytes !== d.bytes ||
      (!large && (typeof d.sha256 !== "string" || !/^[0-9a-f]{64}$/.test(d.sha256))) ||
      typeof d.name !== "string" ||
      d.name.includes("/") ||
      d.name.includes("\\") ||
      !d.name ||
      (i === 0
        ? d.bytes > MANIFEST_LIMIT
        : i <= cloudCount
          ? !/^clouds\/[0-9]{3}\.ply$/.test(d.path) || !d.name.endsWith(".ply")
          : !/^assets\/[0-9]{4}$/.test(d.path))
    )
      throw new Error("Invalid snapshot member identity");
    signal.throwIfAborted();
    if (large) {
      const identity = parseSource(d.identity);
      if (member.method !== 0 || member.compressed !== member.bytes || identity.kind !== "file" || identity.name !== d.name || identity.size !== d.bytes)
        throw new Error("Invalid stored workspace member identity");
      // Blob slices retain the stored file without materializing a whole member.
      const restored = new File([file.slice(member.offset, member.offset + member.bytes)], d.name);
      if (!await matchesFile(restored, identity, signal))
        throw new Error(`Snapshot member hash differs: ${d.path}`);
      files.push(restored);
    } else {
      const bytes = await readReviewMember(file, member, signal);
      if (hex(await crypto.subtle.digest("SHA-256", bytes)) !== d.sha256)
        throw new Error(`Snapshot member hash differs: ${d.path}`);
      files.push(new File([bytes], d.name));
    }
  }
  const project = JSON.parse(await files[0].text()) as Project;
  await checkSources(project, files.slice(1, cloudCount + 1), signal);
  if (large && manifest.pose_graph_sources_included !== (files.length > cloudCount + 1))
    throw new Error("Snapshot asset provenance differs from members");
  if (manifest.schema === ASSET_SCHEMA || (large && manifest.pose_graph_sources_included)) {
    if (manifest.review_archive_included !== !!project.reviewArchive)
      throw new Error("Snapshot archive provenance differs from project");
    await checkAssets(project, files.slice(cloudCount + 1), signal);
  } else if (large && manifest.review_archive_included !== false) {
    throw new Error("Snapshot archive provenance differs from external inputs");
  }
  signal.throwIfAborted();
  return files;
}
