/** Classification panel: show or hide ASPRS classes in every cloud. */

import { classColor, className } from "../colormap";
import { refreshColors } from "./colors";
import { $ } from "./dom";
import { entries, hiddenClasses, listChanged } from "./state";

function refreshClassified(): void {
  for (const entry of entries.values()) if (entry.cloud.classification) refreshColors(entry);
}

export function renderClasses(): void {
  const counts = new Map<number, number>();
  for (const entry of entries.values()) {
    const classes = entry.cloud.classification;
    if (!classes) continue;
    const local = new Uint32Array(256);
    for (let i = 0; i < classes.length; i++) local[classes[i]]++;
    local.forEach((n, code) => n && counts.set(code, (counts.get(code) ?? 0) + n));
  }
  $("class-panel").hidden = counts.size === 0;
  $("class-list").replaceChildren(
    ...[...counts.entries()]
      .sort(([a], [b]) => a - b)
      .map(([code, n]) => {
        const li = document.createElement("li");
        const box = document.createElement("input");
        box.type = "checkbox";
        box.checked = !hiddenClasses.has(code);
        box.onchange = () => {
          if (box.checked) hiddenClasses.delete(code);
          else hiddenClasses.add(code);
          refreshClassified();
        };
        const swatch = document.createElement("span");
        swatch.className = "swatch";
        swatch.style.background = `rgb(${classColor(code).join(" ")})`;
        const label = document.createElement("span");
        label.textContent = `${code} · ${className(code)}`;
        const count = document.createElement("span");
        count.className = "meta";
        count.textContent = n.toLocaleString();
        const row = document.createElement("label");
        row.append(box, swatch, label, count);
        li.append(row);
        return li;
      }),
  );
}
listChanged.add(renderClasses);

/** Hide exactly these classes (sessions). */
export function setHiddenClasses(codes: Iterable<number>): void {
  hiddenClasses.clear();
  for (const c of codes) hiddenClasses.add(c);
  renderClasses();
}

for (const [id, show] of [
  ["class-all", true],
  ["class-none", false],
] as const) {
  $<HTMLButtonElement>(id).onclick = () => {
    hiddenClasses.clear();
    if (!show) for (let c = 0; c < 256; c++) hiddenClasses.add(c);
    refreshClassified();
    renderClasses();
  };
}
