/** Portable editing projects; source files stay external and are verified. */
import { parseProject, projectSources, type Project, type ControlSetting, type SavedPoseGraph } from "../project";
import { matchesFile, referenceFile, sourceName, type SourceReference } from "../source-reference";
import { vectorMap, workerBusy, exportCloud, exportMesh } from "../api";
import { writeProjectSnapshot, readProjectSnapshot } from "../project-snapshot";
import { REVIEW_LIMIT } from "../review-zip";
import { Autosave } from "../autosave";
import { onProjectChanged, projectChanged } from "../project-change";
import { readRecovery, writeRecovery, MAX_RECOVERY_BYTES, type Recovery, type ReviewDraft } from "../recovery-store";
import type { PoseGraphProject } from "../protocol";
import { applySession, captureSession } from "./session";
import { captureGraphProject, restoreGraphProject, graphProjectReady } from "./posegraph";
import { captureMapProject, openVectorMap, captureLaneReviews, restoreLaneReviews, captureReviewDraft, restoreReviewDraft, mapProjectReady } from "./vectormap";
import { entries, listChanged, distanceChanged, type Entry } from "./state";
import { loadFiles, loadUrls } from "./loading";
import { removeEntry } from "./entries";
import { $, download, errorText, setStatus } from "./dom";
import { endTask, showProgress, startTask, taskActive } from "./tasks";
import { setTool } from "./tools";
import { clearCloudHistory } from "./history";

const MAX_PROJECT_BYTES = MAX_RECOVERY_BYTES;
const available = new Set<File>();
let pending: Project | null = null;
let completing = false;
let projectRevision = 0;
let pendingDraft: ReviewDraft | undefined;
let restoringBrowser = false;
let manualSaving = false;
let snapshotRevision = -1;
let checkingRecovery = true;
let recovery: Recovery | null = null;
let recoveryToken: string | null = null;
let offerRecovery = false;
let recoveryAction = false;

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

export async function captureProject(signal: AbortSignal, progress = true, snapshots?: Map<number, File>): Promise<Project> {
  const session = structuredClone(captureSession(!!snapshots)), settings = captureSettings();
  if (new Set(session.clouds.map(c => c.name)).size !== session.clouds.length) throw new Error("Rename duplicate cloud files before saving a project; source names must be unique");
  const reviews = captureLaneReviews();
  const map = await captureMapProject(), captured = await captureGraphProject();
  const metadata = captured ? structuredClone({ name: captured.name, sessions: captured.sessions, imu: captured.imu }) : null;
  for (const saved of session.clouds) {
    const entry = [...entries.values()].find(e => e.cloud.name === saved.name && (snapshots || e.origin.kind !== "derived"));
    if (!entry) throw new Error(`Source ${saved.name} was removed while saving`);
    if (progress) showProgress({ note: `Checking ${saved.name}` });
    const snapshot = snapshots?.get(entry.cloud.id);
    if (snapshots && !snapshot) throw new Error(`Snapshot ${saved.name} was removed while saving`);
    saved.source = snapshot ? await referenceFile(snapshot, signal) : await cloudReference(entry.cloud.id, signal);
    saved.loadMaxPoints = snapshot ? 0 : entry.origin.loadMaxPoints;
    if (snapshot) { saved.transforms = []; saved.url = undefined; }
  }
  let poseGraph: SavedPoseGraph | null = null;
  if (captured && metadata) {
    const sources: SavedPoseGraph["sources"] = [];
    for (const source of captured.project.sources) {
      const { graph, scans, ...options } = source.files;
      const references: SourceReference[] = [];
      for (const file of scans) {
        if (progress) showProgress({ note: `Checking ${file.name}` });
        references.push(await referenceFile(file, signal));
      }
      sources.push({ graph: graph ? await referenceFile(graph, signal) : null, scans: references, options, first: source.first, nodeIds: source.nodeIds });
    }
    poseGraph = { ...metadata, snapshot: captured.project.snapshot, sources };
  }
  return { app: "CloudAnalyzer Project", version: 1, session, vectorMap: map, poseGraph, settings, reviews };
}

/** Stage projects before loading; graph inputs go to the graph reader. */
function stageProject(project: Project): void {
  const maxPoints = project.settings["max-points"]?.value;
  const control = $<HTMLSelectElement>("max-points");
  if (typeof maxPoints === "string" && [...control.options].some(o => o.value === maxPoints)) control.value = maxPoints;
  pending = project;
  projectRevision++;
  available.clear();
}
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
      stageProject(project);
      pendingDraft = undefined; restoringBrowser = false;
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

/** Snapshot PLY source names differ from their saved display names. */
export function projectCloudDisplayName(file: File): string | undefined {
  return pending?.session.clouds.find(c => c.source && sameName(file, c.source))?.name;
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
    restoreReviewDraft(pendingDraft);
    pendingDraft = undefined;
    pending = null;
    available.clear();
    if (restoringBrowser) { restoringBrowser = false; offerRecovery = false; autosave.recovered(); }
    setStatus("Project restored: source files verified, map and pose graph ready to edit. Undo starts from this saved state.");
  } finally { if (signal) endTask(signal); completing = false; renderSaveState(); autosave.schedule(); }
}

$("project-save").onclick = async () => {
  const button = $<HTMLButtonElement>("project-save");
  if (button.disabled) return;
  if (taskActive() || workerBusy() || !mapProjectReady() || !graphProjectReady()) return setStatus("Finish the current operation before saving a project.", true);
  button.disabled = true;
  manualSaving = true;
  const revision = autosave.revision;
  const signal = startTask();
  try {
    setStatus("Saving the project and verifying its source files…");
    const project = await captureProject(signal);
    const json = JSON.stringify(project);
    if (new Blob([json]).size > MAX_PROJECT_BYTES) throw new Error("Project exceeds the 64 MiB metadata limit");
    download(new Blob([json], { type: "application/json" }), "project.cloudanalyzer.json");
    // A portable export protects only the exact revision captured above.
    if (!captureReviewDraft()) autosave.exported(revision);
    setStatus("Saved project: map, pose graph and settings. Keep the original source files to resume editing.");
  } catch (error) { setStatus(`Could not save project: ${errorText(error)}`, true); }
  finally { button.disabled = false; manualSaving = false; endTask(signal); autosave.schedule(); }
};

async function captureWorkspace(signal: AbortSignal, progress: boolean): Promise<{project: Project; zip: Blob}> {
  const savedEntries = [...entries.values()];
  if (savedEntries.length > 127) throw new Error("Snapshot supports at most 127 clouds/meshes");
  if (new Set(savedEntries.map(e => e.cloud.name)).size !== savedEntries.length) throw new Error("Rename duplicate cloud files before saving a snapshot");
  const snapshots = new Map<number, File>(), prefix = `workspace-${crypto.randomUUID()}`;
  let total = 0;
  for (const [i, entry] of savedEntries.entries()) {
    signal.throwIfAborted();
    if (progress) setStatus(`Saving current records for ${entry.cloud.name}…`);
    const bytes = entry.cloud.kind === "mesh" ? await exportMesh(entry.cloud.id, "ply") : await exportCloud(entry.cloud.id, "ply");
    total += bytes.byteLength;
    if (total > REVIEW_LIMIT) throw new Error("Snapshot exceeds the 64 MiB content limit; export large results separately");
    snapshots.set(entry.cloud.id, new File([new Uint8Array(bytes)], `${prefix}-${i}.ply`));
  }
  const project = await captureProject(signal, progress, snapshots);
  const zip = await writeProjectSnapshot(project, [...snapshots.values()], signal);
  signal.throwIfAborted();
  return {project,zip};
}

$("project-snapshot").onclick = async () => {
  if (taskActive() || workerBusy() || !mapProjectReady() || !graphProjectReady()) return setStatus("Finish the current operation before saving a workspace snapshot.", true);
  const button = $<HTMLButtonElement>("project-snapshot"), signal = startTask(), revision = autosave.revision;
  button.disabled = true; manualSaving = true;
  try {
    const {project,zip} = await captureWorkspace(signal,true);
    if (autosave.revision !== revision) throw new Error("The workspace changed while saving; retry the snapshot");
    download(zip, "project.cloudanalyzer.zip");
    snapshotRevision = revision;
    if (!captureReviewDraft()) autosave.exported(revision);
    setStatus(`Saved workspace snapshot: ${project.session.clouds.length} current clouds/meshes with project metadata. Unloaded original detail and pose-graph input files remain external.`);
  } catch (error) { setStatus(`Could not save workspace snapshot: ${errorText(error)}`, true); }
  finally { button.disabled = false; manualSaving = false; endTask(signal); autosave.schedule(); }
};
$("project-open").onclick = () => $<HTMLInputElement>("file-input").click();

const autosave = new Autosave<Recovery>({
  idle: () => !checkingRecovery && !pending && !completing && !manualSaving && !taskActive() && !workerBusy() && mapProjectReady() && graphProjectReady(),
  capture: async signal => {
    if ($<HTMLInputElement>("project-autosave-records").checked) {
      const {project,zip} = await captureWorkspace(signal,false);
      return {version:1,token:crypto.randomUUID(),savedAt:new Date().toISOString(),project,draft:captureReviewDraft(),snapshot:zip};
    }
    const project = await captureProject(signal,false), draft = captureReviewDraft();
    return {version:1,token:crypto.randomUUID(),savedAt:new Date().toISOString(),project,draft};
  },
  write: async value => {
    await writeRecovery(value,recoveryToken);
    recovery = value; recoveryToken = value.token;
  },
  render: renderSaveState,
});
function renderSaveState(): void {
  const busy = completing || manualSaving || autosave.saving || checkingRecovery || recoveryAction;
  $("project-recovery").hidden = !offerRecovery;
  $<HTMLButtonElement>("project-resume").disabled = busy || !!pending || !recovery;
  $<HTMLButtonElement>("project-forget").disabled = busy || !recovery;
  $<HTMLButtonElement>("project-download-recovery").disabled = busy || !recovery;
  $("project-autosave-retry").hidden = !autosave.error || checkingRecovery;
  $("project-recovery-summary").textContent = recovery ? `Browser copy from ${new Date(recovery.savedAt).toLocaleString()}. ${recovery.snapshot ? "Includes current point/mesh records. Pose-graph inputs remain external. " : ""}${pending && restoringBrowser ? "Open its matching original files to finish resuming." : "Resume it or discard it before automatic saves replace this copy."}` : "";
  $("project-save-status").textContent = checkingRecovery ? "Checking browser recovery…" : autosave.error ? `Browser save failed: ${autosave.error}. Use Save project to keep your work.` : autosave.saving ? "Saving editing state in this browser…" : autosave.paused ? "Automatic saving paused; the previous browser copy is protected." : !autosave.enabled ? `Automatic saving is off.${autosave.unsaved ? " Changes are not saved." : ""}` : autosave.revision > autosave.savedRevision ? "Changes waiting for browser save…" : recovery ? `Editing state saved in this browser at ${new Date(recovery.savedAt).toLocaleTimeString()}.` : "Automatic saving ready. Original source files remain external.";
  const derived = [...entries.values()].filter(e => e.origin.kind === "derived").length;
  if (derived && !(recovery?.snapshot && autosave.revision <= autosave.savedRevision)) $("project-save-status").textContent += ` ${derived} processed clouds/meshes are outside the browser metadata copy; use Save workspace snapshot to keep their records.`;
  if (recovery?.snapshot && autosave.revision <= autosave.savedRevision) $("project-save-status").textContent += " Current point/mesh records are included in this browser copy.";
}
onProjectChanged(() => autosave.changed());
listChanged.add(() => { if (entries.size || autosave.revision) projectChanged(); });
distanceChanged.add(() => { if (entries.size) projectChanged(); });
for (const event of ["input","change"]) document.addEventListener(event,e => {
  const target = e.target;
  if (!(target instanceof HTMLInputElement || target instanceof HTMLSelectElement || target instanceof HTMLTextAreaElement) || target.closest("#project-recovery-controls") || target instanceof HTMLInputElement && (target.type === "file" || target.readOnly)) return;
  if (["vm-review-filter","vm-quality-cloud"].includes(target.id)) return;
  projectChanged();
});
window.addEventListener("beforeunload",event => {
  const recordsSavedRevision = recovery?.snapshot ? autosave.savedRevision : -1;
  const derivedNeedsSnapshot = [...entries.values()].some(e => e.origin.kind === "derived") && autosave.revision > Math.max(snapshotRevision,recordsSavedRevision);
  const currentRecordsNeedSave = $<HTMLInputElement>("project-autosave-records").checked && entries.size > 0 && autosave.revision > Math.max(snapshotRevision,recordsSavedRevision);
  if (!autosave.unsaved && !derivedNeedsSnapshot && !currentRecordsNeedSave) return;
  event.preventDefault(); event.returnValue = "";
});
document.addEventListener("visibilitychange",() => { if (document.hidden) void autosave.save(); });
$("project-autosave").onchange = () => {
  autosave.enabled = $<HTMLInputElement>("project-autosave").checked;
  try { localStorage.setItem("cloudanalyzer-autosave-enabled",autosave.enabled ? "on" : "off"); }
  catch { /* Saving failures are reported independently by the recovery store. */ }
  renderSaveState(); autosave.schedule();
};
$("project-autosave-records").onchange = () => {
  try { localStorage.setItem("cloudanalyzer-autosave-records",$<HTMLInputElement>("project-autosave-records").checked ? "on" : "off"); } catch { /* Storage failures are reported by the recovery commit. */ }
  projectChanged();
};
$("project-autosave-retry").onclick = () => {
  if (autosave.paused && !offerRecovery) { checkingRecovery = true; renderSaveState(); void initializeRecovery(); }
  else autosave.retry();
};
$("project-resume").onclick = async () => {
  if (!recovery || completing || autosave.saving || pending) return;
  if (taskActive() || !mapProjectReady() || !graphProjectReady()) return setStatus("Finish the current operation before resuming browser work.",true);
  recoveryAction = true; renderSaveState();
  try {
    const current = await readRecovery();
    if (!current || current.token !== recoveryToken) throw new Error("The browser copy changed in another tab. Reload to choose which work to resume.");
    let files: File[] = [];
    if (current?.snapshot) {
      files = await readProjectSnapshot(new File([current.snapshot],"project.cloudanalyzer.zip"),new AbortController().signal);
      if (JSON.stringify(parseProject(JSON.parse(await files[0].text()))) !== JSON.stringify(current.project)) throw new Error("Browser snapshot metadata differs from the recovery project");
      if (current.project.session.clouds.some(saved => [...entries.values()].some(e => e.cloud.name === saved.name))) throw new Error("A current cloud has the same name as saved work. Export and close it before resuming the browser snapshot");
    }
    recovery = current;
    stageProject(current.project); pendingDraft = current.draft; restoringBrowser = true;
    for (const file of files.slice(1)) available.add(file);
    renderSaveState(); await completePendingProject();
  } catch (error) { setStatus(`Could not resume browser copy: ${errorText(error)}`,true); }
  finally { recoveryAction = false; renderSaveState(); }
};
$("project-download-recovery").onclick = () => {
  if (!recovery) return;
  if (recovery.snapshot) { download(recovery.snapshot,"project.cloudanalyzer.zip"); setStatus("Downloaded the saved point/mesh workspace. Unsaved review text stays in browser recovery; save the lane review to include it in a portable project."); return; }
  download(new Blob([JSON.stringify(recovery.project)],{type:"application/json"}),"project.cloudanalyzer.json");
  setStatus("Downloaded the saved browser project. Unsaved review text stays in browser recovery; save a lane review before exporting it.");
};
$("project-forget").onclick = async () => {
  if (!recovery || completing || autosave.saving) return;
  recoveryAction = true; renderSaveState();
  try {
    await writeRecovery(null,recoveryToken);
    recovery = null; recoveryToken = null; offerRecovery = false;
    if (restoringBrowser) { pending = null; pendingDraft = undefined; restoringBrowser = false; available.clear(); projectRevision++; }
    autosave.paused = false; autosave.error = "";
    renderSaveState(); autosave.schedule();
  } catch (error) { autosave.error = errorText(error); renderSaveState(); }
  finally { recoveryAction = false; renderSaveState(); }
};
async function initializeRecovery(): Promise<void> {
  try {
    try { autosave.enabled = localStorage.getItem("cloudanalyzer-autosave-enabled") !== "off"; } catch { /* IDB reports unavailable browser storage below. */ }
    try { $<HTMLInputElement>("project-autosave-records").checked = localStorage.getItem("cloudanalyzer-autosave-records") === "on"; } catch { /* IDB reports unavailable browser storage below. */ }
    $<HTMLInputElement>("project-autosave").checked = autosave.enabled;
    recovery = await readRecovery(); recoveryToken = recovery?.token ?? null;
    offerRecovery = !!recovery; autosave.paused = offerRecovery;
    autosave.error = "";
  } catch (error) { autosave.error = errorText(error); }
  finally { checkingRecovery = false; renderSaveState(); autosave.schedule(); }
}
void initializeRecovery();
