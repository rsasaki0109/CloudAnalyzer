/** Opening clouds from files, drops and URLs. */

import { loadCloud, loadCopcUrl, loadTrajectory } from "../api";
import { CANCELLED, type Progress } from "../protocol";
import { nameFromUrl, parseSession } from "../session";
import { $, errorText, setStatus } from "./dom";
import { addEntry, renderList } from "./entries";
import { applySession, restorePendingAfterLoad } from "./session";
import { entries, type Origin, viewer } from "./state";
import { endTask, showProgress, startTask } from "./tasks";
import { addTrajectory } from "./trajectory";

/** A COPC file on a server, read node by node instead of downloaded. */
interface RemoteCopc {
  url: string;
  name: string;
}

/** Extensions a trajectory can have; `.txt` and `.csv` may also be point clouds (the worker tells). */
const TRAJECTORY_FILE = /\.(tum|kitti|txt|csv)$/i;

const seconds = (ms: number) => (ms >= 1000 ? `${(ms / 1000).toFixed(1)} s` : `${Math.round(ms)} ms`);

/**
 * Load point clouds and meshes; session files (.json) among them are applied
 * once the others are in. `origins` tells where each file came from.
 */
export async function loadFiles(files: (File | RemoteCopc)[], origins?: Origin[]): Promise<void> {
  const sessions: File[] = [];
  const signal = startTask();
  for (const [i, file] of files.entries()) {
    if (signal.aborted) break;
    if (file instanceof File && /\.json$/i.test(file.name)) {
      sessions.push(file);
      continue;
    }
    const mb = file instanceof File ? `${(file.size / 1e6).toFixed(file.size >= 1e7 ? 0 : 1)} MB` : "COPC";
    setStatus(`Loading ${file.name} (${mb}): reading…`);
    const start = performance.now();
    try {
      const maxPoints = Number($<HTMLSelectElement>("max-points").value) || Number.POSITIVE_INFINITY;
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
          ? await loadCloud(file, maxPoints, onProgress, signal)
          : await loadCopcUrl(file.url, file.name, maxPoints, onProgress, signal);
      addEntry(cloud, origins?.[i] ?? { kind: "file" });
      if (entries.size === 1) viewer.fit();
      const [sx, sy, sz] = cloud.shift;
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
}

/** Download clouds from URLs (the server must allow cross-origin requests) and load them. */
export async function loadUrls(urls: string[]): Promise<void> {
  const files: (File | RemoteCopc)[] = [];
  const origins: Origin[] = [];
  const failed: string[] = [];
  const signal = startTask();
  for (const raw of urls) {
    const url = new URL(raw, location.href).href;
    const name = nameFromUrl(url);
    setStatus(`Downloading ${name}…`);
    try {
      // Ask for the first bytes: a COPC file is then read node by node; a
      // server that ignores the range sends the whole file right away. Not
      // cached: Chrome can otherwise splice this partial response into a
      // later full download of the same URL.
      const probe = await fetch(url, { signal, cache: "no-store", headers: { Range: "bytes=0-1023" } });
      if (!probe.ok) throw new Error(`HTTP ${probe.status}`);
      if (probe.status === 206) {
        const head = new Uint8Array(await probe.arrayBuffer());
        if (isCopcHead(head)) {
          files.push({ url, name });
        } else {
          const response = await fetch(url, { signal });
          if (!response.ok) throw new Error(`HTTP ${response.status}`);
          files.push(new File([await readWithProgress(response, name)], name));
        }
      } else {
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

/** Whether these first bytes are a COPC file: LAS 1.4 whose first VLR is "copc". */
function isCopcHead(head: Uint8Array): boolean {
  const text = (at: number, n: number) => String.fromCharCode(...head.subarray(at, at + n));
  return head.length >= 400 && text(0, 4) === "LASF" && head[24] === 1 && head[25] === 4 && text(377, 5) === "copc\0";
}

/** The body of a download, reporting progress when its size is known. */
async function readWithProgress(response: Response, name: string): Promise<Blob> {
  const total = Number(response.headers.get("content-length")) || 0;
  if (!response.body || !total) return response.blob();
  const reader = response.body.getReader();
  const chunks: Uint8Array[] = [];
  let received = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    received += value.length;
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
