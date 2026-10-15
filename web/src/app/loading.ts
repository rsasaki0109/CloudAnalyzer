/** Opening clouds from files, drops and URLs. */

import { loadCloud, loadTrajectory, loadUrl } from "../api";
import { readRangeResponse } from "../bytes";
import { CANCELLED, type Progress } from "../protocol";
import { nameFromUrl, parseSession } from "../session";
import { $, errorText, setStatus } from "./dom";
import { addEntry, renderList } from "./entries";
import { applySession, restorePendingAfterLoad } from "./session";
import { entries, globalShift, type Origin, viewer } from "./state";
import { endTask, showProgress, startTask, taskActive } from "./tasks";
import { addTrajectory } from "./trajectory";
import { openVectorMap } from "./vectormap";
import { completePendingProject, isProjectGraphSource, prepareProjectFiles, projectCloudFileAllowed, projectCloudLoadLimit, projectCloudDisplayName, restoreWorkspaceFiles } from "./project";
import { readProjectSnapshot } from "../project-snapshot";

/** A LAS/LAZ or COPC file on a server, read with range requests instead of downloaded. */
interface RemoteFile {
  url: string;
  name: string;
  size: number;
  etag?: string;
}

/** Extensions a trajectory can have; `.txt` and `.csv` may also be point clouds (the worker tells). */
const TRAJECTORY_FILE = /\.(tum|kitti|txt|csv)$/i;

const seconds = (ms: number) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`);

/**
 * Load point clouds and meshes; session files (.json) among them are applied
 * once the others are in. `origins` tells where each file came from.
 */
export async function loadFiles(files: (File | RemoteFile)[], origins?: Origin[]): Promise<void> {
  if (files.some(file => file instanceof File && /\.cloudanalyzer\.zip$/i.test(file.name))) {
    if (taskActive()) {setStatus("Finish the current operation before opening a workspace snapshot",true);return;}
    const snapshots=files.filter((f):f is File=>f instanceof File&&/\.cloudanalyzer\.zip$/i.test(f.name));
    if(snapshots.length!==1||files.some(f=>!(f instanceof File))) {setStatus("Open one workspace snapshot at a time",true);return;}
    const extras=files.filter((f):f is File=>f instanceof File&&!snapshots.includes(f));
    let expanded:File[];
    const signal = startTask();
    try {
      expanded = await readProjectSnapshot(snapshots[0],signal);
    } catch (error) { setStatus(`Could not open workspace snapshot: ${errorText(error)}`, true); return; }
    finally { endTask(signal); }
    try {
      await restoreWorkspaceFiles(expanded,extras);
    } catch(error) {setStatus(`Could not open workspace snapshot: ${errorText(error)}`,true);}
    return;
  }
  let projects: Set<File>;
  try { projects = await prepareProjectFiles(files.filter((f): f is File => f instanceof File)); }
  catch (error) { setStatus(`Could not open project: ${errorText(error)}`, true); return; }
  const sessions: File[] = [];
  const signal = startTask();
  for (const [i, file] of files.entries()) {
    if (signal.aborted) break;
    if (file instanceof File && (projects.has(file) || isProjectGraphSource(file))) continue;
    if (file instanceof File && /\.json$/i.test(file.name)) {
      sessions.push(file);
      continue;
    }
    if (file instanceof File && /\.osm$/i.test(file.name)) {
      try {
        await openVectorMap(file.name, await file.text());
      } catch (err) {
        setStatus(`Could not open ${file.name}: ${errorText(err)}`);
      }
      continue;
    }
    const size = file.size;
    const mb = `${(size / 1e6).toFixed(size >= 1e7 ? 0 : 1)} MB${file instanceof File ? "" : " on the server"}`;
    setStatus(`Loading ${file.name} (${mb}): reading…`);
    const start = performance.now();
    try {
      if (file instanceof File && !await projectCloudFileAllowed(file, signal)) continue;
      const loadLimit = projectCloudLoadLimit(file) ?? Number($<HTMLSelectElement>("max-points").value);
      const maxPoints = loadLimit || Number.POSITIVE_INFINITY;
      if (file instanceof File && TRAJECTORY_FILE.test(file.name)) {
        const poses = await loadTrajectory(file);
        if (poses) {
          addTrajectory(file.name, poses);
          const n = poses.timestamps.length;
          setStatus(`Loaded ${file.name}: trajectory of ${n.toLocaleString()} poses (${poses.format.toUpperCase()})`);
          continue;
        }
      }
      const onProgress = (p: Progress) => {
        showProgress(p);
        setStatus(`Loading ${file.name} (${mb}): ${p.note}…`);
      };
      const cloud =
        file instanceof File
          ? await loadCloud(file, maxPoints, onProgress, signal, projectCloudDisplayName(file))
          : await loadUrl(file.url, file.name, file.size, maxPoints, onProgress, signal, file.etag);
      const origin = origins?.[i] ?? { kind: "file" as const };
      addEntry(cloud, origin.kind === "derived" ? origin : {
        ...origin,
        loadMaxPoints: loadLimit,
        ...(file instanceof File ? { file } : { size: file.size, etag: file.etag }),
      });
      if (entries.size === 1) viewer.fit();
      const [sx, sy, sz] = globalShift();
      $("shift").textContent = sx || sy || sz ? `Global shift: (${-sx}, ${-sy}, ${-sz})` : "";
      const { parse, index, prepare, workers } = cloud.timings;
      const size =
        cloud.kind === "mesh"
          ? `${cloud.triangles.toLocaleString()} triangles`
          : cloud.copcLevels !== null
            ? `${cloud.count.toLocaleString()} of ${cloud.filePoints.toLocaleString()} points ` +
              `(COPC levels 0–${cloud.copcLevels - 1})`
            : cloud.keepEvery > 1
              ? `${cloud.count.toLocaleString()} of ${cloud.filePoints.toLocaleString()} points (1 in ${cloud.keepEvery})`
              : `${cloud.count.toLocaleString()} points`;
      const splats = cloud.opacity ? " (Gaussian splat centers)" : "";
      setStatus(
        `Loaded ${file.name}: ${size}${splats} in ${seconds(performance.now() - start)} ` +
          `(read ${seconds(parse)} · index ${seconds(index)}` +
          `${workers && workers > 1 ? ` on ${workers} workers` : ""} · prepare ${seconds(prepare)})`,
      );
    } catch (err) {
      if (err instanceof Error && err.message === CANCELLED) {
        const rest = files.length - i - 1;
        setStatus(`Stopped loading ${file.name}${rest > 0 ? ` (and ${rest} more)` : ""}`);
      } else {
        setStatus(`${file.name}: ${errorText(err)}`, true);
      }
    }
    renderList();
  }
  endTask(signal);
  if (signal.aborted) return;
  for (const file of sessions) {
    try {
      await applySession(parseSession(JSON.parse(await file.text())));
    } catch (err) {
      setStatus(`${file.name}: ${errorText(err)}`, true);
    }
  }
  // Files a pending session was waiting for.
  if (sessions.length === 0) await restorePendingAfterLoad();
  try { await completePendingProject(); }
  catch (error) { setStatus(`Could not restore project: ${errorText(error)}`, true); }
}

/** Download clouds from URLs (the server must allow cross-origin requests) and load them. */
export async function loadUrls(urls: string[]): Promise<void> {
  const files: (File | RemoteFile)[] = [];
  const origins: Origin[] = [];
  const failed: string[] = [];
  const signal = startTask();
  for (const raw of urls) {
    const url = new URL(raw, location.href).href;
    const name = nameFromUrl(url);
    setStatus(`Downloading ${name}…`);
    try {
      // LAS/COPC must stay on the range path, even for extensionless URLs.
      const probe = await fetch(url, { signal, cache: "no-store", headers: { Range: "bytes=0-1023" } });
      if (!probe.ok) throw new Error(`HTTP ${probe.status}`);
      if (probe.status === 206) {
        const { bytes: head, total: size, etag } = await readRangeResponse(probe, 0, 1024);
        if (isLasHead(head) && size > 0) {
          files.push({ url, name, size, etag });
        } else {
          const response = await fetch(url, { signal });
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          files.push(new File([await readWithProgress(response, name)], name));
        }
      } else {
        if (/\.(las|laz)$/i.test(name)) {
          await probe.body?.cancel();
          throw new Error("LAS/LAZ URLs require HTTP byte range support (206)");
        }
        files.push(new File([await readWithProgress(probe, name)], name));
      }
      origins.push({ kind: "url", url });
    } catch (err) {
      if (signal.aborted) {
        endTask(signal);
        setStatus(`Stopped downloading ${name}`);
        return;
      }
      failed.push(`${name} (${errorText(err)})`);
    }
  }
  endTask(signal);
  await loadFiles(files, origins);
  if (failed.length) {
    setStatus(`Could not download ${failed.join(", ")}; the server must allow cross-origin requests`, true);
  }
}

/** Whether these first bytes are LAS/LAZ (COPC included), which the worker reads in pieces. */
function isLasHead(head: Uint8Array): boolean {
  return String.fromCharCode(...head.subarray(0, 4)) === "LASF";
}

/** The body of a download, reporting progress when its size is known. */
async function readWithProgress(response: Response, name: string): Promise<Blob> {
  const total = Number(response.headers.get("content-length")) || 0;
  if (!response.body) throw new Error("missing download body");
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
    if (received >= 4) {
      const signature = new Uint8Array(4);
      let at = 0;
      for (const chunk of chunks) {
        const piece = chunk.subarray(0, 4 - at);
        signature.set(piece, at);
        at += piece.length;
        if (at === 4) break;
      }
      if (isLasHead(signature)) {
        await reader.cancel();
        throw new Error("LAS/LAZ URLs require HTTP byte range support (206)");
      }
    }
    if (!total) continue;
    showProgress({ note: "", fraction: Math.min(1, received / total) });
    setStatus(`Downloading ${name}: ${Math.round((received / total) * 100)}% of ${(total / 1e6).toFixed(1)} MB…`);
  }
  return new Blob(chunks as BlobPart[]);
}

$<HTMLButtonElement>("url-open").onclick = () => {
  const url = $<HTMLInputElement>("url-input").value.trim();
  if (url) void loadUrls([url]);
};

$<HTMLButtonElement>("open").onclick = () => $<HTMLInputElement>("file-input").click();
$<HTMLButtonElement>("session-open").onclick = () => $<HTMLInputElement>("file-input").click();
$<HTMLInputElement>("file-input").onchange = (e) => {
  const input = e.target as HTMLInputElement;
  if (input.files) void loadFiles([...input.files]);
  input.value = "";
};

const overlay = $("drop-overlay");
let dragDepth = 0;
window.addEventListener("dragenter", (e) => {
  if (!e.dataTransfer?.types.includes("Files")) return;
  dragDepth++;
  overlay.hidden = false;
});
window.addEventListener("dragleave", () => {
  dragDepth = Math.max(0, dragDepth - 1);
  if (dragDepth === 0) overlay.hidden = true;
});
window.addEventListener("dragover", (e) => e.preventDefault());
window.addEventListener("drop", (e) => {
  e.preventDefault();
  dragDepth = 0;
  overlay.hidden = true;
  if (e.dataTransfer?.files.length) void loadFiles([...e.dataTransfer.files]);
});
