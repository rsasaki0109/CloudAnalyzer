/** Portable editing projects; source files stay external and are verified. */
import { parseProject, projectSources, type Project, type ControlSetting, type SavedPoseGraph } from "../project";
import { matchesFile, referenceFile, sourceName, type SourceReference } from "../source-reference";
import { vectorMap } from "../api";
import type { PoseGraphProject } from "../protocol";
import { applySession, captureSession } from "./session";
import { captureGraphProject, restoreGraphProject } from "./posegraph";
import { captureMapProject, openVectorMap, captureLaneReviews, restoreLaneReviews } from "./vectormap";
import { entries, type Entry } from "./state";
import { loadFiles, loadUrls } from "./loading";
import { removeEntry } from "./entries";
import { $, download, errorText, setStatus } from "./dom";
import { endTask, showProgress, startTask } from "./tasks";
import { setTool } from "./tools";
import { clearCloudHistory } from "./history";

const MAX_PROJECT_BYTES = 64 * 1024 * 1024;
const available = new Set<File>();
let pending: Project | null = null;
let completing = false;
let projectRevision = 0;

function controls(): (HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement)[] {
  return [...document.querySelectorAll<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>("input[id], select[id], textarea[id]")]
    .filter(c => ((/^(vm-|pg-|memory-)/.test(c.id) && !c.id.startsWith("vm-review-")) || c.id === "max-points") && !(c instanceof HTMLInputElement && c.type === "file"));
}

function captureSettings(): Record<string, ControlSetting> {
  return Object.fromEntries(controls().map(c => {
    const setting: ControlSetting = { value: c instanceof HTMLInputElement && c.type === "checkbox" ? c.checked : c.value };
    if (c instanceof HTMLSelectElement && c.id.endsWith("-cloud")) setting.cloud = entries.get(Number(c.value))?.cloud.name;
    return [c.id, setting];
  }));
}

function restoreSettings(settings: Project["settings"]): void {
  for (const control of controls()) {
    const setting = settings[control.id];
    if (!setting) continue;
    if (control instanceof HTMLInputElement && control.type === "checkbox") control.checked = setting.value === true;
    else {
      let value = String(setting.value);
      if (setting.cloud && control instanceof HTMLSelectElement) value = [...control.options].find(o => entries.get(Number(o.value))?.cloud.name === setting.cloud)?.value ?? "";
      if (control instanceof HTMLSelectElement && ![...control.options].some(o => o.value === value)) continue;
      control.value = value;
    }
    control.dispatchEvent(new Event("input", { bubbles: true }));
    control.dispatchEvent(new Event("change", { bubbles: true }));
  }
}

async function cloudReference(id: number, signal: AbortSignal): Promise<SourceReference> {
  const entry = entries.get(id)!;
  const origin = entry.origin;
  if (origin.kind !== "derived" && origin.file) return referenceFile(origin.file, signal);
  if (origin.kind === "url" && origin.size !== undefined && origin.etag && /^"[^"\r\n]*"$/.test(origin.etag)) {
    return { kind: "http", name: entry.cloud.name, size: origin.size, url: origin.url, etag: origin.etag };
  }
  throw new Error(`Cannot verify ${entry.cloud.name}; open a local file or a URL with a strong ETag`);
}

export async function captureProject(signal: AbortSignal): Promise<Project> {
  const session = structuredClone(captureSession()), settings = captureSettings();
  if (new Set(session.clouds.map(c => c.name)).size !== session.clouds.length) throw new Error("Rename duplicate cloud files before saving a project; source names must be unique");
  const reviews = captureLaneReviews();
  const map = await captureMapProject(), captured = await captureGraphProject();
  const metadata = captured ? structuredClone({ name: captured.name, sessions: captured.sessions, imu: captured.imu }) : null;
  for (const saved of session.clouds) {
    const entry = [...entries.values()].find(e => e.cloud.name === saved.name && e.origin.kind !== "derived");
    if (!entry) throw new Error(`Source ${saved.name} was removed while saving`);
    showProgress({ note: `Checking ${saved.name}` });
    saved.source = await cloudReference(entry.cloud.id, signal);
    saved.loadMaxPoints = entry.origin.loadMaxPoints;
  }
  let poseGraph: SavedPoseGraph | null = null;
  if (captured && metadata) {
    const sources: SavedPoseGraph["sources"] = [];
    for (const source of captured.project.sources) {
      const { graph, scans, ...options } = source.files;
      const references: SourceReference[] = [];
      for (const file of scans) {
        showProgress({ note: `Checking ${file.name}` });
        references.push(await referenceFile(file, signal));
      }
      sources.push({ graph: graph ? await referenceFile(graph, signal) : null, scans: references, options, first: source.first, nodeIds: source.nodeIds });
    }
    poseGraph = { ...metadata, snapshot: captured.project.snapshot, sources };
  }
  return { app: "CloudAnalyzer Project", version: 1, session, vectorMap: map, poseGraph, settings, reviews };
}

/** Stage projects before loading; graph inputs go to the graph reader. */
export async function prepareProjectFiles(files: File[]): Promise<Set<File>> {
  const consumed = new Set<File>();
  for (const file of files) {
    if (!/\.json$/i.test(file.name)) continue;
    if (file.size > MAX_PROJECT_BYTES) {
      if (/project\.cloudanalyzer\.json$/i.test(file.name)) throw new Error("Project exceeds the 64 MiB metadata limit");
      continue;
    }
    let value: unknown;
    try { value = JSON.parse(await file.text()); } catch { continue; }
    if (value && typeof value === "object" && (value as { app?: string }).app === "CloudAnalyzer Project") {
      const project = parseProject(value);
      const maxPoints = project.settings["max-points"]?.value;
      const control = $<HTMLSelectElement>("max-points");
      if (typeof maxPoints === "string" && [...control.options].some(o => o.value === maxPoints)) control.value = maxPoints;
      pending = project;
      projectRevision++;
      available.clear();
      consumed.add(file);
    }
  }
  if (pending) for (const file of files) if (!consumed.has(file)) available.add(file);
  return consumed;
}

const sameName = (file: File, ref: SourceReference) => sourceName(file) === ref.name || file.name === ref.name.split("/").at(-1);

export function isProjectGraphSource(file: File): boolean {
  if (!pending?.poseGraph) return false;
  if (pending.session.clouds.some(c => c.source && sameName(file, c.source))) return false;
  return pending.poseGraph.sources.some(s => [...(s.graph ? [s.graph] : []), ...s.scans].some(ref => sameName(file, ref)));
}

/** Each source keeps the loading limit used when its cloud was created. */
export function projectCloudLoadLimit(file: File | {name: string; url: string}): number | undefined {
  return pending?.session.clouds.find(c => file instanceof File ? c.source && sameName(file,c.source) : c.url === file.url)?.loadMaxPoints;
}

export async function projectCloudFileAllowed(file: File, signal: AbortSignal): Promise<boolean> {
  const refs = pending?.session.clouds.flatMap(c => c.source && sameName(file, c.source) ? [c.source] : []) ?? [];
  if (!refs.length) return true;
  setStatus(`Verifying project source ${file.name}…`);
  for (const ref of refs) if (await matchesFile(file, ref, signal)) return true;
  return false;
}

async function findSource(ref: SourceReference, signal: AbortSignal): Promise<File | undefined> {
  for (const file of available) if (sameName(file, ref) && await matchesFile(file, ref, signal)) return file;
  for (const entry of entries.values()) {
    const origin = entry.origin;
    if (origin.kind !== "derived" && origin.file && sameName(origin.file, ref) && await matchesFile(origin.file, ref, signal)) return origin.file;
  }
  return undefined;
}

export async function completePendingProject(): Promise<void> {
  if (!pending || completing) return;
  completing = true;
  const project = pending, revision = projectRevision;
  let signal: AbortSignal | undefined;
  try {
    const urls = project.session.clouds.filter(c => c.url && ![...entries.values()].some(e => e.origin.kind === "url" && e.origin.url === c.url)).map(c => c.url!);
    if (urls.length) await loadUrls(urls);
    signal = startTask();
    const resolved = new Map<SourceReference, File>();
    const missing: string[] = [];
    for (const source of projectSources(project)) {
      signal.throwIfAborted();
      setStatus(`Verifying project source ${source.name}…`);
      if (source.kind === "file") {
        const file = await findSource(source, signal);
        if (file) resolved.set(source, file);
        else missing.push(source.name);
      } else if (![...entries.values()].some(e => e.origin.kind === "url" && e.origin.url === source.url && e.origin.etag === source.etag && e.origin.size === source.size)) {
        throw new Error(`Remote source ${source.name} changed or no longer exposes its saved ETag`);
      }
    }
    if (revision !== projectRevision) return;
    if (missing.length) {
      setStatus(`Project: open matching source files to continue: ${[...new Set(missing)].join(", ")}. Same-name files with different contents are not used.`);
      return;
    }
    await vectorMap("check-project", { name: "project-map.json", text: project.vectorMap });
    // Reopen a verified source when the listed cloud uses a different sampling limit.
    const stale: number[] = [];
    for (const saved of project.session.clouds) {
      if (saved.loadMaxPoints === undefined || !saved.source) continue;
      const source = saved.source;
      const matching: Entry[] = [];
      for (const entry of entries.values()) {
        if (entry.cloud.name !== saved.name || entry.origin.kind === "derived") continue;
        const origin = entry.origin;
        if (source.kind === "file" && origin.file && await matchesFile(origin.file,source,signal)) matching.push(entry);
        if (source.kind === "http" && origin.kind === "url" && origin.url === source.url && origin.etag === source.etag && origin.size === source.size) matching.push(entry);
      }
      if (!matching.some(e => e.origin.loadMaxPoints === saved.loadMaxPoints)) {
        if (source.kind === "file") await loadFiles([resolved.get(source)!], matching[0] ? [matching[0].origin] : undefined);
        else await loadUrls([source.url]);
        let reopened = false;
        for (const entry of entries.values()) {
          if (entry.cloud.name !== saved.name || entry.origin.loadMaxPoints !== saved.loadMaxPoints || entry.origin.kind === "derived" || matching.includes(entry)) continue;
          const origin = entry.origin;
          if (source.kind === "file" && origin.file && await matchesFile(origin.file, source, signal)) reopened = true;
          if (source.kind === "http" && origin.kind === "url" && origin.url === source.url && origin.etag === source.etag && origin.size === source.size) reopened = true;
        }
        if (!reopened) throw new Error(`Could not reopen ${saved.name} at its saved loading limit with verified contents`);
      }
      stale.push(...matching.filter(e => e.origin.loadMaxPoints !== saved.loadMaxPoints).map(e => e.cloud.id));
    }
    for (const id of stale) await removeEntry(id);
    if (revision !== projectRevision) return;
    let native: PoseGraphProject | null = null;
    if (project.poseGraph) {
      native = { snapshot: project.poseGraph.snapshot, sources: project.poseGraph.sources.map(s => ({
        files: { ...s.options, graph: s.graph ? resolved.get(s.graph)! : null, scans: s.scans.map(ref => resolved.get(ref)!) }, first: s.first, nodeIds: s.nodeIds,
      })) };
    }
    setTool(null);
    await restoreGraphProject(native, project.poseGraph ?? undefined);
    await openVectorMap("project-map.json", project.vectorMap);
    restoreSettings(project.settings);
    $("memory-apply").click();
    if (!clearCloudHistory()) throw new Error("Finish Undo or Redo before opening a project");
    await applySession(project.session, true);
    restoreLaneReviews(project.reviews);
    pending = null;
    available.clear();
    setStatus("Project restored: source files verified, map and pose graph ready to edit. Undo starts from this saved state.");
  } finally { if (signal) endTask(signal); completing = false; }
}

$("project-save").onclick = async () => {
  const button = $<HTMLButtonElement>("project-save");
  if (button.disabled) return;
  button.disabled = true;
  const signal = startTask();
  try {
    setStatus("Saving the project and verifying its source files…");
    const project = await captureProject(signal);
    const json = JSON.stringify(project);
    if (new Blob([json]).size > MAX_PROJECT_BYTES) throw new Error("Project exceeds the 64 MiB metadata limit");
    download(new Blob([json], { type: "application/json" }), "project.cloudanalyzer.json");
    setStatus("Saved project: map, pose graph and settings. Keep the original source files to resume editing.");
  } catch (error) { setStatus(`Could not save project: ${errorText(error)}`, true); }
  finally { button.disabled = false; endTask(signal); }
};
$("project-open").onclick = () => $<HTMLInputElement>("file-input").click();
