/** Display settings, camera shortcuts and saved views. */

import type { Vec3 } from "../protocol";
import { $, compact, removeButton, typing } from "./dom";
import { entries, isMesh, toOriginal, toRender, viewer } from "./state";

const backgroundInput = $<HTMLInputElement>("background");
const pointSizeInput = $<HTMLInputElement>("point-size");
const pointSizeMode = $<HTMLSelectElement>("point-size-mode");
const pointBudgetSelect = $<HTMLSelectElement>("point-budget");
const edlToggle = $<HTMLInputElement>("edl");
const edlStrength = $<HTMLInputElement>("edl-strength");

backgroundInput.value = getComputedStyle(document.documentElement).getPropertyValue("--viewport").trim();
viewer.setBackground(backgroundInput.value);
backgroundInput.oninput = () => viewer.setBackground(backgroundInput.value);

$<HTMLButtonElement>("fit").onclick = () => viewer.fit();
const VIEWS: Record<string, Vec3> = {
  top: [0, -1e-3, 1],
  front: [0, -1, 0],
  side: [1, 0, 0],
  iso: [1, -1, 1],
};
for (const button of document.querySelectorAll<HTMLButtonElement>("[data-view]")) {
  button.onclick = () => {
    const [x, y, z] = VIEWS[button.dataset.view!];
    viewer.view({ x, y, z });
  };
}

export function setPointSize(size: number): void {
  pointSizeInput.value = String(size);
  viewer.setPointSize(size);
}
pointSizeInput.oninput = () => viewer.setPointSize(Number(pointSizeInput.value));

edlToggle.onchange = edlStrength.oninput = () => {
  viewer.setEdl(edlToggle.checked, Number(edlStrength.value));
  edlStrength.disabled = !edlToggle.checked;
};

function applyPointSizeMode(): void {
  const adaptive = pointSizeMode.value === "adaptive";
  viewer.setPointSizeMode(adaptive ? "adaptive" : "fixed");
  // EDL does not shade adaptive points (see Viewer.useEdl).
  edlToggle.disabled = adaptive;
  edlStrength.disabled = adaptive || !edlToggle.checked;
  edlToggle.parentElement!.title = adaptive
    ? "Eye-Dome Lighting is off with adaptive point size"
    : "Eye-Dome Lighting: shade by depth so the shape of a cloud is easier to read";
}
pointSizeMode.onchange = applyPointSizeMode;
pointBudgetSelect.onchange = () => viewer.setPointBudget(Number(pointBudgetSelect.value));

viewer.onDrawn = (points) => {
  const total = [...entries.values()].reduce((sum, e) => sum + (e.visible && !isMesh(e) ? e.cloud.count : 0), 0);
  $("drawn").textContent = total ? `Drawing ${compact(points)} of ${compact(total)} points` : "";
};
window.addEventListener("keydown", (e) => {
  if (typing(e)) return;
  if (e.key === "f" || e.key === "F") viewer.fit();
});

/** Display settings as saved in a session. */
export interface DisplaySettings {
  pointSize: number;
  edl: boolean;
  edlStrength: number;
  pointBudget: number;
  background?: string;
  pointSizeMode?: "fixed" | "adaptive";
  views?: SavedView[];
}

export function captureDisplay(): Required<DisplaySettings> {
  return {
    pointSize: Number(pointSizeInput.value),
    edl: edlToggle.checked,
    edlStrength: Number(edlStrength.value),
    pointBudget: Number(pointBudgetSelect.value),
    background: backgroundInput.value,
    pointSizeMode: pointSizeMode.value as "fixed" | "adaptive",
    views: savedViews,
  };
}

export function applyDisplay(settings: DisplaySettings): void {
  setPointSize(settings.pointSize);
  if (settings.background) {
    backgroundInput.value = settings.background;
    viewer.setBackground(settings.background);
  }
  pointSizeMode.value = settings.pointSizeMode ?? "fixed";
  applyPointSizeMode();
  savedViews.splice(0, savedViews.length, ...(settings.views ?? []));
  renderViews();
  edlToggle.checked = settings.edl;
  edlStrength.value = String(settings.edlStrength);
  viewer.setEdl(settings.edl, settings.edlStrength);
  edlStrength.disabled = !settings.edl;
  if ([...pointBudgetSelect.options].some((o) => Number(o.value) === settings.pointBudget)) {
    pointBudgetSelect.value = String(settings.pointBudget);
    viewer.setPointBudget(settings.pointBudget);
  }
}

// ---------------------------------------------------------------- saved views

interface SavedView {
  name: string;
  /** Camera position and target in original coordinates. */
  position: Vec3;
  target: Vec3;
}

const savedViews: SavedView[] = [];

export function goToView(view: { position: Vec3; target: Vec3 }): void {
  viewer.setCamera(toRender(view.position), toRender(view.target));
}

/** The camera in original coordinates. */
export function currentView(): { position: Vec3; target: Vec3 } {
  const { position, target } = viewer.getCamera();
  return { position: toOriginal(position), target: toOriginal(target) };
}

function renderViews(): void {
  $("view-list").replaceChildren(
    ...savedViews.map((view, i) => {
      const li = document.createElement("li");
      const name = document.createElement("input");
      name.type = "text";
      name.value = view.name;
      name.setAttribute("aria-label", `View ${i + 1} name`);
      name.oninput = () => {
        view.name = name.value;
      };
      const go = document.createElement("button");
      go.textContent = "Go";
      go.title = "Move the camera to this view";
      go.onclick = () => goToView(view);
      const remove = removeButton(() => {
        savedViews.splice(i, 1);
        renderViews();
      });
      li.append(name, go, remove);
      return li;
    }),
  );
}

$<HTMLButtonElement>("view-save").onclick = () => {
  savedViews.push({ name: `View ${savedViews.length + 1}`, ...currentView() });
  renderViews();
};
