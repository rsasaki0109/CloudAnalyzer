/** Saving, sharing and restoring sessions. */

import { transformCloud } from "../api";
import { RAMPS, type RampName } from "../colormap";
import { encodeSession, type Session } from "../session";
import { matchesFile, type SourceReference } from "../source-reference";
import { setHiddenClasses } from "./classes";
import { clipBox, setClipBox } from "./clip";
import { availableModes, refreshColors } from "./colors";
import { applyDisplay, captureDisplay, currentView, goToView } from "./display";
import { applyRange, runNearest } from "./distance";
import { $, download, setStatus } from "./dom";
import { renderList, replaceCloud } from "./entries";
import { loadUrls } from "./loading";
import { invertRigid } from "./history";
import { savedNotes, setNotes } from "./picking";
import { savedGates, setGates } from "./report";
import { colorByField, fieldNames } from "./scalars";
import { restoreProfile, savedProfile } from "./profile";
import { display, entries, hiddenClasses, viewer } from "./state";

/** A session waiting for some of its clouds to be opened. */
let pendingSession: Session | null = null;
/** Clouds of the pending session already restored. */
const restored = new Set<number>();
let applyingSession = false;
let restoreExactTransforms = false;

export function captureSession(): Session {
  const { views, ...settings } = captureDisplay();
  return {
    app: "CloudAnalyzer Web",
    version: 1,
    camera: entries.size ? currentView() : undefined,
    ...settings,
    ramp: display.ramp,
    range: display.range,
    hiddenClasses: [...hiddenClasses],
    clip: clipBox(),
    profile: savedProfile(),
    views,
    labels: savedNotes(),
    gates: savedGates(),
    clouds: [...entries.values()]
      .filter((e) => e.origin.kind !== "derived")
      .map((e) => ({
        name: e.cloud.name,
        displayPreview: e.origin.displayPreview || undefined,
        url: e.origin.kind === "url" ? e.origin.url : undefined,
        visible: e.visible,
        mode: e.mode,
        field: e.mode === "scalar" ? e.field?.name : undefined,
        solid: e.solid,
        transforms: e.transforms,
        distance:
          e.c2c && (e.c2c.kind === "c2c" || e.c2c.kind === "c2m")
            ? { reference: e.c2c.referenceName, signed: e.c2c.signed }
            : undefined,
      })),
  };
}

/** Apply a session: settings now, URL clouds after downloading them, file clouds as they are opened. */
export async function applySession(session: Session, exactTransforms = false): Promise<void> {
  restoreExactTransforms = exactTransforms;
  pendingSession = session;
  restored.clear();
  applyDisplay(session);
  const loaded = new Set([...entries.values()].map((e) => e.cloud.name));
  const urls = session.clouds.filter((c) => c.url && !loaded.has(c.name)).map((c) => c.url!);
  if (urls.length) {
    applyingSession = true;
    try {
      await loadUrls(urls);
    } finally {
      applyingSession = false;
    }
  }
  await restorePending();
}

/** After files were opened: restore what a pending session was waiting for. */
export async function restorePendingAfterLoad(): Promise<void> {
  if (pendingSession && !applyingSession) await restorePending();
}

/** Restore what the pending session can with the clouds open now. */
async function restorePending(): Promise<void> {
  const session = pendingSession;
  if (!session) return;
  const byName = async (name: string, sources = true, identity?: SourceReference, loadMaxPoints?: number) => {
    for (const entry of entries.values()) {
      if (entry.cloud.name !== name || (sources && entry.origin.kind === "derived")) continue;
      if (restoreExactTransforms && loadMaxPoints !== undefined && entry.origin.loadMaxPoints !== loadMaxPoints) continue;
      if (!identity) return entry;
      const origin = entry.origin;
      if (identity.kind === "file" && origin.kind !== "derived" && origin.file && await matchesFile(origin.file, identity)) return entry;
      if (identity.kind === "http" && origin.kind === "url" && origin.url === identity.url && origin.size === identity.size && origin.etag === identity.etag) return entry;
    }
    return undefined;
  };
  const missing: string[] = [];
  for (const saved of session.clouds) {
    const entry = await byName(saved.name, true, saved.source, saved.loadMaxPoints);
    if (!entry) {
      missing.push(saved.name);
      continue;
    }
    if (restored.has(entry.cloud.id)) continue;
    entry.origin.displayPreview = entry.origin.displayPreview || saved.displayPreview;
    if (restoreExactTransforms && JSON.stringify(entry.transforms) !== JSON.stringify(saved.transforms)) {
      // Existing sources may already be moved; return them to their source frame.
      while (entry.transforms.length) {
        replaceCloud(entry, await transformCloud(entry.cloud.id, invertRigid(entry.transforms.at(-1)!)));
        entry.transforms.pop();
      }
    }
    if (entry.transforms.length === 0) {
      for (const matrix of saved.transforms) {
        replaceCloud(entry, await transformCloud(entry.cloud.id, matrix));
        entry.transforms.push(matrix);
      }
    }
    entry.visible = saved.visible;
    viewer.setVisible(entry.cloud.id, saved.visible);
    entry.solid = saved.solid;
    // Distances are recomputed below, which also colors by them; so are fields.
    if (saved.mode !== "c2c" && saved.mode !== "scalar" && availableModes(entry)[saved.mode]) entry.mode = saved.mode;
    refreshColors(entry);
    restored.add(entry.cloud.id);
  }
  for (const saved of session.clouds) {
    const entry = await byName(saved.name, true, saved.source, saved.loadMaxPoints);
    const reference = saved.distance ? await byName(saved.distance.reference, false, session.clouds.find(c => c.name === saved.distance?.reference)?.source) : undefined;
    if (entry && reference && saved.distance && (restoreExactTransforms || !entry.c2c)) await runNearest(entry, reference, saved.distance.signed);
    if (entry && saved.mode === "scalar" && saved.field && fieldNames(entry).includes(saved.field)) {
      await colorByField(entry, saved.field).catch(() => {});
    }
  }
  if (session.ramp in RAMPS) display.ramp = session.ramp as RampName;
  display.range = session.range;
  applyRange();
  setHiddenClasses(session.hiddenClasses);
  for (const entry of entries.values()) refreshColors(entry);
  if (session.clip && entries.size) setClipBox(session.clip);
  if (session.camera && entries.size) goToView(session.camera);
  if (session.labels && entries.size) setNotes(session.labels);
  if (session.gates) setGates(session.gates);
  if (session.profile && entries.size) await restoreProfile(session.profile);
  renderList();
  if (missing.length) {
    setStatus(`Session: open ${missing.join(", ")} to finish restoring it`);
  } else {
    pendingSession = null;
    restored.clear();
    setStatus(`Session restored (${session.clouds.length} ${session.clouds.length === 1 ? "cloud" : "clouds"})`);
  }
}

$<HTMLButtonElement>("session-save").onclick = () => {
  const json = JSON.stringify(captureSession(), null, 2);
  download(new Blob([json], { type: "application/json" }), "session.cloudanalyzer.json");
  setStatus("Saved the session; open it together with the same files to restore it");
};

$<HTMLButtonElement>("share").onclick = async () => {
  const session = captureSession();
  const link = `${location.origin}${location.pathname}#session=${encodeSession(session)}`;
  const field = $<HTMLInputElement>("share-link");
  field.value = link;
  field.hidden = false;
  field.select();
  let copied = false;
  try {
    await navigator.clipboard.writeText(link);
    copied = true;
  } catch {
    // Not allowed here: the link stays selected in the field.
  }
  const local = session.clouds.filter((c) => !c.url).map((c) => c.name);
  setStatus(
    `${copied ? "Link copied" : "Link ready"}` +
      (local.length
        ? `; ${local.join(", ")} ${local.length === 1 ? "is a local file" : "are local files"}, not in the link — ` +
          "whoever opens it is asked to open them"
        : ""),
  );
};
