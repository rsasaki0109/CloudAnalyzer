/** Transient original-core evidence and a capped, two-colour inspection overlay. */
import * as THREE from "three";
import { changeCoreSupport } from "../api";
import type { ChangeCandidate } from "../change-review";
import type { LaneCoreEvidence, CoreSupportRow } from "../lane-core-support";
import type { LaneCoreSnapshot } from "../survey-lane-tasks";
import type { SurveyLane } from "../lane-surface-overlap";
import type { LoadedCloud } from "../protocol";
import { $, errorText } from "./dom";
import { entries, globalShift, viewer } from "./state";

interface Binding { lane: number; label: HTMLSpanElement; button: HTMLButtonElement }
interface Context {
  key: string; cloud: LoadedCloud; candidate: ChangeCandidate; lanes: SurveyLane[];
  heightMargin: number; current: () => boolean; frame: () => void;
}
let generation = 0, running = false, pending: (() => void) | null = null;
let cached: { key: string; cloud: LoadedCloud; evidence: LaneCoreEvidence } | null = null;
const group = new THREE.Group(); group.name = "survey-lane-core-evidence"; group.visible = false;
viewer.overlay.add(group);
const materials = [0xf2ad57, 0x47eddf].map((color, i) => {
  const material = new THREE.PointsMaterial({ color, size: i ? 7 : 5, sizeAttenuation: false, depthTest: false, depthWrite: false });
  viewer.registerPointMaterial(material); return material;
});
function clearOverlay(): void {
  for (const child of [...group.children]) { group.remove(child); (child as THREE.Points).geometry.dispose(); }
  group.visible = false;
  $("pg-core-overlay-hint").textContent = "";
  $("pg-core-overlay-clear").hidden = true;
  viewer.requestRender();
}
export function clearLaneCoreView(dropCache = false): void {
  generation++; pending = null; clearOverlay(); if (dropCache) cached = null;
}
$("pg-core-overlay-clear").onclick = clearOverlay;
function showCores(evidence: LaneCoreEvidence, row: CoreSupportRow): void {
  clearOverlay();
  if (!row.mask) return;
  const shift = globalShift(); let shown = 0;
  for (const kind of [0, 1]) {
    const count = kind ? row.supported! : row.total - row.supported!;
    const stride = Math.max(1, Math.ceil(count / 10_000));
    const coordinates: number[] = []; let index = 0;
    for (let i = 0; i < row.total; i++) if (row.mask[i] === kind) {
      if (index++ % stride) continue;
      coordinates.push(evidence.positions[i*3]-shift[0], evidence.positions[i*3+1]-shift[1], evidence.positions[i*3+2]-shift[2]);
    }
    shown += coordinates.length / 3;
    if (coordinates.length) {
      const geometry = new THREE.BufferGeometry();
      geometry.setAttribute("position", new THREE.Float32BufferAttribute(coordinates, 3));
      const points = new THREE.Points(geometry, materials[kind]); points.frustumCulled = false; group.add(points);
    }
  }
  group.visible = true;
  $("pg-core-overlay-hint").textContent = `Lane ${row.lane}: ${row.supported} of ${row.total} changed cores supported. ` +
    `Cyan: lane support; amber: other candidate cores. Showing ${shown} of ${row.total}; markers draw over surfaces. Counts use original float64 coordinates.`;
  $("pg-core-overlay-clear").hidden = false;
  viewer.requestRender();
}
export function requestLaneCoreView(context: Context, bindings: Binding[]): void {
  const token = generation;
  const current = () => token === generation && context.current() && entries.get(context.cloud.id)?.cloud === context.cloud;
  const failed = (reason: string) => {
    if (!current()) return;
    for (const binding of bindings) {
      binding.label.textContent = `Core support unavailable: ${reason}`;
      binding.label.dataset.coreStatus = "unavailable"; delete binding.label.dataset.coreSupported;
      binding.button.disabled = true; binding.button.dataset.coreReady = "false";
    }
  };
  const apply = (evidence: LaneCoreEvidence) => {
    if (!current()) return;
    if (evidence.sourceCount !== context.cloud.count) { failed("Source count changed"); return; }
    for (const binding of bindings) {
      const row = evidence.rows.find(row => row.lane === binding.lane);
      if (!row) { failed("Missing lane evidence"); return; }
      binding.label.dataset.coreStatus = row.status; binding.label.dataset.coreTotal = String(row.total);
      if (row.status !== "complete") {
        binding.label.textContent = `Core support unavailable: ${row.reason}`;
        delete binding.label.dataset.coreSupported; binding.button.disabled = true; binding.button.dataset.coreReady = "false"; continue;
      }
      binding.label.dataset.coreSupported = String(row.supported);
      binding.label.textContent = `${row.supported} / ${row.total} changed cores · ${(100*row.supported!/Math.max(1,row.total)).toFixed(1)}%` +
        (row.supported === 0 ? " · box overlap only" : "");
      binding.button.disabled = false; binding.button.dataset.coreReady = "true";
      binding.button.onclick = () => { if (current()) { context.frame(); showCores(evidence,row); } };
    }
  };
  const run = () => {
    if (!current()) return;
    if (cached?.key === context.key && cached.cloud === context.cloud) { apply(cached.evidence); return; }
    running = true;
    void changeCoreSupport(context.cloud.id, context.candidate.rank, context.candidate.points, context.candidate.bounds, context.lanes, context.heightMargin)
      .then(evidence => { if (current()) { cached = {key:context.key,cloud:context.cloud,evidence}; apply(evidence); } })
      .catch(error => failed(errorText(error)))
      .finally(() => { running = false; const next = pending; pending = null; next?.(); });
  };
  // Rapid selection/margin changes replace the queued request instead of flooding the worker.
  if (running) pending = run; else run();
}

/** Re-read one lane for a durable snapshot; preview sampling/cache never substitutes for evidence. */
export async function captureLaneCoreSnapshot(cloud: LoadedCloud | undefined, candidate: ChangeCandidate,
  lane: SurveyLane, heightMarginM: number): Promise<LaneCoreSnapshot> {
  const base = { protocol: "xy-lane-z-v1" as const, total: candidate.points, heightMarginM };
  try {
    if (!cloud || entries.get(cloud.id)?.cloud !== cloud) throw new Error("Original comparison cloud unavailable");
    const evidence = await changeCoreSupport(cloud.id, candidate.rank, candidate.points, candidate.bounds, [lane], heightMarginM);
    if (entries.get(cloud.id)?.cloud !== cloud || evidence.sourceCount !== cloud.count) throw new Error("Original comparison cloud changed");
    const row = evidence.rows.find(row => row.lane === lane.id);
    if (!row || row.total !== candidate.points) throw new Error("Missing or inconsistent lane core evidence");
    return { ...base, status: row.status, supported: row.supported, reason: row.reason };
  } catch (error) {
    const reason = errorText(error).replace(/[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]/g, " ").slice(0, 1000) || "Core evidence unavailable";
    return { ...base, status: "unavailable", supported: null, reason };
  }
}
