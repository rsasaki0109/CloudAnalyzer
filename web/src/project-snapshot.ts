/** Exact current point/mesh PLY records alongside normal project metadata. */
import type { Project } from "./project";
import { matchesFile } from "./source-reference.ts";
import {
  indexReviewZip,
  readReviewMember,
  writeReviewZip,
  REVIEW_LIMIT,
  MANIFEST_LIMIT,
} from "./review-zip.ts";

const SCHEMA = "cloudanalyzer.project_snapshot.v1";
const hex = (bytes: ArrayBuffer) =>
  [...new Uint8Array(bytes)]
    .map((n) => n.toString(16).padStart(2, "0"))
    .join("");
const hash = async (file: Blob) =>
  hex(await crypto.subtle.digest("SHA-256", await file.arrayBuffer()));
interface Descriptor {
  path: string;
  name: string;
  bytes: number;
  sha256: string;
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

export async function writeProjectSnapshot(
  project: Project,
  files: File[],
  signal: AbortSignal,
): Promise<Blob> {
  if (files.length > 127)
    throw new Error("Snapshot supports at most 127 clouds/meshes");
  const metadata = new File(
    [JSON.stringify(project)],
    "project.cloudanalyzer.json",
    { type: "application/json" },
  );
  if (
    metadata.size > MANIFEST_LIMIT ||
    metadata.size + files.reduce((n, f) => n + f.size, 0) > REVIEW_LIMIT
  )
    throw new Error(
      "Snapshot exceeds the 64 MiB content or 10 MiB project metadata limit",
    );
  await checkSources(project, files, signal);
  const all = [metadata, ...files],
    descriptors: Descriptor[] = [],
    entries: [string, Blob][] = [];
  for (const [i, file] of all.entries()) {
    signal.throwIfAborted();
    const path =
      i === 0 ? "project.json" : `clouds/${String(i - 1).padStart(3, "0")}.ply`;
    descriptors.push({
      path,
      name: file.name,
      bytes: file.size,
      sha256: await hash(file),
    });
    entries.push([path, file]);
  }
  const manifest = {
    schema: SCHEMA,
    files: descriptors,
    point_data: "current_loaded_records",
    coordinate_frame: "current_transformed_coordinates",
    unloaded_original_density_included: false,
    pose_graph_sources_included: false,
  };
  return writeReviewZip(
    [["manifest.json", new Blob([JSON.stringify(manifest)])], ...entries],
    signal,
  );
}

/** Verify all members and source identities before any workspace loading. */
export async function readProjectSnapshot(
  file: File,
  signal: AbortSignal,
): Promise<File[]> {
  const directory = await indexReviewZip(file, signal);
  const manifest = JSON.parse(
    new TextDecoder("utf-8", { fatal: true }).decode(
      await readReviewMember(file, directory.get("manifest.json")!, signal),
    ),
  );
  if (
    manifest?.schema !== SCHEMA ||
    !Array.isArray(manifest.files) ||
    manifest.files.length < 1 ||
    manifest.files.length > 128 ||
    manifest.point_data !== "current_loaded_records" ||
    manifest.coordinate_frame !== "current_transformed_coordinates" ||
    manifest.unloaded_original_density_included !== false ||
    manifest.pose_graph_sources_included !== false
  )
    throw new Error("Unsupported workspace snapshot schema/provenance");
  const descriptors = manifest.files as Descriptor[];
  if (
    directory.size !== descriptors.length + 1 ||
    new Set(descriptors.map((f) => f?.path)).size !== descriptors.length ||
    new Set(descriptors.map((f) => f?.name)).size !== descriptors.length ||
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
      typeof d.sha256 !== "string" ||
      !/^[0-9a-f]{64}$/.test(d.sha256) ||
      typeof d.name !== "string" ||
      d.name.includes("/") ||
      d.name.includes("\\") ||
      !d.name ||
      (i === 0
        ? d.bytes > MANIFEST_LIMIT
        : !/^clouds\/[0-9]{3}\.ply$/.test(d.path) || !d.name.endsWith(".ply"))
    )
      throw new Error("Invalid snapshot member identity");
    const bytes = await readReviewMember(file, member, signal);
    if (hex(await crypto.subtle.digest("SHA-256", bytes)) !== d.sha256)
      throw new Error(`Snapshot member hash differs: ${d.path}`);
    files.push(new File([bytes], d.name));
  }
  const project = JSON.parse(await files[0].text()) as Project;
  await checkSources(project, files.slice(1), signal);
  signal.throwIfAborted();
  return files;
}
