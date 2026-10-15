/** Cross-section profiles along a polyline. */

import { projectChanged } from "../project-change";
import * as THREE from "three";
import { profileCloud } from "../api";
import { $, download, errorText, fmt, setStatus } from "./dom";
import { clouds, globalShift, viewer } from "./state";
import { activeTool, setTool, type Tool } from "./tools";

/** Profile polyline in original XY coordinates. */
let profileLine: [number, number][] = [];
/** Render-space height the line is drawn (and clicked) at. */
let profileZ = 0;
interface ProfileSeries {
  name: string;
  color: string;
  along: Float64Array;
  positions: Float64Array;
  total: number;
}
let profileSeries: ProfileSeries[] = [];
const PROFILE_MAX_POINTS = 200_000;
const profileWidth = $<HTMLInputElement>("profile-width");
const profileDraw = $<HTMLButtonElement>("profile-draw");
export const profilePlot = $("profile-plot");
export const profileCanvas = profilePlot.querySelector("canvas")!;

const profileHalfWidth = () => Math.max(0, Number(profileWidth.value) || 0) / 2;
const drawing = () => activeTool() === profileTool;

function profileLength(): number {
  let length = 0;
  for (let i = 1; i < profileLine.length; i++) {
    length += Math.hypot(profileLine[i][0] - profileLine[i - 1][0], profileLine[i][1] - profileLine[i - 1][1]);
  }
  return length;
}

/** `v` rounded to one significant digit, e.g. 0.37 -> 0.4. */
function roundNicely(v: number): number {
  if (!(v > 0)) return 1;
  const p = 10 ** Math.floor(Math.log10(v));
  return Math.round(v / p) * p;
}

function drawProfileLine(): void {
  const shift = globalShift();
  viewer.setProfile(
    profileLine.map(([x, y]) => new THREE.Vector3(x - shift[0], y - shift[1], profileZ)),
    profileHalfWidth(),
  );
}

function renderProfileHint(): void {
  $("profile-hint").textContent = drawing()
    ? "Click points on the view; double-click or Enter to finish, Esc to cancel."
    : profileLine.length >= 2
      ? `${profileLine.length} vertices, ${fmt(profileLength())} long.`
      : "Draw a line across the clouds; looking from above works best.";
}

function renderDrawButton(): void {
  profileDraw.setAttribute("aria-pressed", String(drawing()));
  profileDraw.textContent = drawing() ? "Finish line" : "Draw line";
  renderProfileHint();
}

const profileTool: Tool = {
  click(x, y) {
    const p = viewer.groundPoint(x, y, profileZ);
    if (!p) return;
    const shift = globalShift();
    profileLine.push([p.x + shift[0], p.y + shift[1]]);
    projectChanged();
    drawProfileLine();
    renderProfileHint();
  },
  doubleClick() {
    // The second click of the double click added a duplicate vertex.
    profileLine.pop();
    projectChanged();
    void finishProfile();
  },
  key(e) {
    if (e.key === "Enter") void finishProfile();
    else if (e.key === "Escape") clearProfile();
    else return false;
    return true;
  },
  enter: renderDrawButton,
  exit: renderDrawButton,
};

function clearProfile(): void {
  profileLine = [];
  projectChanged();
  profileSeries = [];
  setTool(null);
  drawProfileLine();
  renderProfilePlot();
  renderProfileHint();
}

profileDraw.onclick = () => {
  if (drawing()) {
    void finishProfile();
    return;
  }
  profileLine = [];
  profileSeries = [];
  renderProfilePlot();
  projectChanged();
  profileZ = viewer.getCamera().target.z;
  if (!(Number(profileWidth.value) > 0)) {
    const size = viewer.contentBounds().getSize(new THREE.Vector3());
    profileWidth.value = String(roundNicely(Math.hypot(size.x, size.y) / 100));
  }
  drawProfileLine();
  setTool(profileTool);
};
$<HTMLButtonElement>("profile-clear").onclick = clearProfile;
$<HTMLButtonElement>("profile-close").onclick = clearProfile;
profileWidth.onchange = () => {
  drawProfileLine();
  if (!drawing() && profileLine.length >= 2) void computeProfile();
};

async function finishProfile(): Promise<void> {
  setTool(null);
  if (profileLine.length < 2) {
    clearProfile();
    return;
  }
  await computeProfile();
}

/** Cut every visible cloud along the line and plot the result. */
async function computeProfile(): Promise<void> {
  const halfWidth = profileHalfWidth();
  const sources = clouds().filter((e) => e.visible);
  if (profileLine.length < 2 || sources.length === 0) return;
  if (!(halfWidth > 0)) {
    setStatus("Profile: enter a positive width", true);
    return;
  }
  setStatus("Computing the profile…");
  try {
    const series: ProfileSeries[] = [];
    for (const entry of sources) {
      const out = await profileCloud({
        id: entry.cloud.id,
        line: profileLine.flat(),
        halfWidth,
        maxPoints: PROFILE_MAX_POINTS,
      });
      series.push({ name: entry.cloud.name, color: `rgb(${entry.solid.join(" ")})`, ...out });
    }
    profileSeries = series;
    renderProfilePlot();
    renderProfileHint();
    const total = series.reduce((sum, s) => sum + s.total, 0);
    $<HTMLButtonElement>("profile-csv").disabled = total === 0;
    setStatus(
      `Profile: ${total.toLocaleString()} points from ${series.length} ${series.length === 1 ? "cloud" : "clouds"} ` +
        `within ${fmt(halfWidth * 2)} of a ${fmt(profileLength())} line`,
    );
  } catch (err) {
    setStatus(`Profile failed: ${errorText(err)}`, true);
  }
}

/** The profile line for a session, or null without one. */
export function savedProfile(): { line: [number, number][]; halfWidth: number } | null {
  return profileLine.length >= 2 ? { line: profileLine, halfWidth: profileHalfWidth() } : null;
}

/** Restore a profile line from a session and compute it. */
export async function restoreProfile(profile: { line: [number, number][]; halfWidth: number }): Promise<void> {
  profileLine = profile.line;
  profileWidth.value = String(profile.halfWidth * 2);
  profileZ = viewer.getCamera().target.z;
  drawProfileLine();
  await computeProfile();
}

/** Round tick positions covering `[lo, hi]`, about `count` of them. */
function ticks(lo: number, hi: number, count: number): number[] {
  const span = hi - lo;
  if (!(span > 0)) return [lo];
  const raw = span / count;
  const p = 10 ** Math.floor(Math.log10(raw));
  const step = [1, 2, 5, 10].map((m) => m * p).find((s) => s >= raw) ?? 10 * p;
  const out: number[] = [];
  for (let v = Math.ceil(lo / step) * step; v <= hi + step * 1e-9; v += step) out.push(Math.abs(v) < step * 1e-9 ? 0 : v);
  return out;
}

/** Data window and pixel frame of the last plot, for the cursor readout. */
let plotFrame: { x0: number; x1: number; y0: number; y1: number; left: number; top: number; w: number; h: number } | null =
  null;

function renderProfilePlot(): void {
  profilePlot.hidden = profileSeries.length === 0;
  $<HTMLButtonElement>("profile-csv").disabled = profileSeries.every((s) => s.along.length === 0);
  $("profile-legend").replaceChildren(
    ...profileSeries.map((s) => {
      const item = document.createElement("span");
      const swatch = document.createElement("i");
      swatch.style.background = s.color;
      item.append(swatch, `${s.name} (${s.total.toLocaleString()})`);
      return item;
    }),
  );
  if (profilePlot.hidden) {
    plotFrame = null;
    return;
  }
  const dpr = window.devicePixelRatio || 1;
  const cw = profileCanvas.clientWidth;
  const ch = profileCanvas.clientHeight;
  profileCanvas.width = Math.round(cw * dpr);
  profileCanvas.height = Math.round(ch * dpr);
  const ctx = profileCanvas.getContext("2d")!;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cw, ch);

  let [x0, x1] = [0, profileLength()];
  let [y0, y1] = [Number.POSITIVE_INFINITY, Number.NEGATIVE_INFINITY];
  for (const s of profileSeries) {
    for (let i = 2; i < s.positions.length; i += 3) {
      y0 = Math.min(y0, s.positions[i]);
      y1 = Math.max(y1, s.positions[i]);
    }
  }
  if (!(y1 >= y0)) [y0, y1] = [0, 1];
  const pad = Math.max((y1 - y0) * 0.05, 1e-6);
  [y0, y1] = [y0 - pad, y1 + pad];
  const left = 64;
  const top = 8;
  const w = Math.max(10, cw - left - 12);
  const h = Math.max(10, ch - top - 22);
  if ($<HTMLInputElement>("profile-equal").checked) {
    // Same units per pixel on both axes: widen whichever range is short.
    const scale = Math.max((x1 - x0) / w, (y1 - y0) / h);
    const [cx, cy] = [(x0 + x1) / 2, (y0 + y1) / 2];
    [x0, x1] = [cx - (scale * w) / 2, cx + (scale * w) / 2];
    [y0, y1] = [cy - (scale * h) / 2, cy + (scale * h) / 2];
  }
  const px = (d: number) => left + ((d - x0) / (x1 - x0)) * w;
  const py = (z: number) => top + (1 - (z - y0) / (y1 - y0)) * h;

  ctx.font = "11px system-ui, sans-serif";
  ctx.lineWidth = 1;
  ctx.strokeStyle = "rgb(255 255 255 / 0.08)";
  ctx.fillStyle = "#8b93a5";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  for (const t of ticks(x0, x1, Math.max(2, Math.floor(w / 90)))) {
    ctx.beginPath();
    ctx.moveTo(px(t) + 0.5, top);
    ctx.lineTo(px(t) + 0.5, top + h);
    ctx.stroke();
    ctx.fillText(fmt(t), px(t), top + h + 4);
  }
  ctx.textAlign = "right";
  ctx.textBaseline = "middle";
  for (const t of ticks(y0, y1, Math.max(2, Math.floor(h / 40)))) {
    ctx.beginPath();
    ctx.moveTo(left, py(t) + 0.5);
    ctx.lineTo(left + w, py(t) + 0.5);
    ctx.stroke();
    ctx.fillText(fmt(t), left - 6, py(t));
  }
  ctx.save();
  ctx.beginPath();
  ctx.rect(left, top, w, h);
  ctx.clip();
  for (const s of profileSeries) {
    ctx.fillStyle = s.color;
    for (let i = 0; i < s.along.length; i++) {
      ctx.fillRect(px(s.along[i]) - 1, py(s.positions[i * 3 + 2]) - 1, 2, 2);
    }
  }
  ctx.restore();
  plotFrame = { x0, x1, y0, y1, left, top, w, h };
}

$<HTMLInputElement>("profile-equal").onchange = renderProfilePlot;
new ResizeObserver(() => {
  if (!profilePlot.hidden) renderProfilePlot();
}).observe(profileCanvas);

profileCanvas.onmousemove = (e) => {
  const f = plotFrame;
  if (!f) return;
  const rect = profileCanvas.getBoundingClientRect();
  const [mx, my] = [e.clientX - rect.left - f.left, e.clientY - rect.top - f.top];
  const inside = mx >= 0 && my >= 0 && mx <= f.w && my <= f.h;
  $("profile-readout").textContent = inside
    ? `distance ${fmt(f.x0 + (mx / f.w) * (f.x1 - f.x0))} · z ${fmt(f.y1 - (my / f.h) * (f.y1 - f.y0))}`
    : "";
};
profileCanvas.onmouseleave = () => {
  $("profile-readout").textContent = "";
};

$<HTMLButtonElement>("profile-csv").onclick = () => {
  const quote = (s: string) => (/[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s);
  const rows = ["cloud,distance,x,y,z"];
  for (const s of profileSeries) {
    const name = quote(s.name);
    for (let i = 0; i < s.along.length; i++) {
      const p = s.positions;
      rows.push(`${name},${s.along[i]},${p[i * 3]},${p[i * 3 + 1]},${p[i * 3 + 2]}`);
    }
  }
  download(new Blob([`${rows.join("\n")}\n`], { type: "text/csv" }), "profile.csv");
  setStatus(`Saved profile.csv (${(rows.length - 1).toLocaleString()} points)`);
};

renderProfileHint();
