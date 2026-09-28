/** Panels that derive new clouds: filters, ground extraction, merge / split and normals. */

import { estimateNormals, extractGround, filterCloud, mergeClouds, splitCloud } from "../api";
import { classColor, className } from "../colormap";
import { refreshColors } from "./colors";
import { $, errorText, roundUp, setStatus } from "./dom";
import { addEntry, renderList } from "./entries";
import { record } from "./history";
import { clouds, type Entry, entries, listChanged } from "./state";

/** Refill a cloud picker, keeping its choice if still there, else the newest cloud. */
export function fillCloudSelect(select: HTMLSelectElement, options: Entry[]): void {
  const previous = select.value;
  select.replaceChildren(...options.map((e) => new Option(e.cloud.name, String(e.cloud.id))));
  select.value = options.some((e) => String(e.cloud.id) === previous)
    ? previous
    : String(options.at(-1)?.cloud.id ?? "");
}

// ---------------------------------------------------------------- filters

const filterCloudSelect = $<HTMLSelectElement>("filter-cloud");
const filterOp = $<HTMLSelectElement>("filter-op");
const filterRun = $<HTMLButtonElement>("filter-run");
const voxelInput = $<HTMLInputElement>("filter-voxel");

listChanged.add(() => {
  const options = clouds();
  fillCloudSelect(filterCloudSelect, options);
  filterRun.disabled = options.length === 0;
  suggestVoxel();
});

/** Default voxel: about 1/200 of the cloud's largest extent, rounded. */
function suggestVoxel(): void {
  const entry = entries.get(Number(filterCloudSelect.value));
  if (!entry || voxelInput.dataset.cloud === filterCloudSelect.value) return;
  const b = entry.cloud.bounds;
  voxelInput.value = String(roundUp(Math.max(b[3] - b[0], b[4] - b[1], b[5] - b[2]) / 200));
  voxelInput.dataset.cloud = filterCloudSelect.value;
}

filterCloudSelect.onchange = suggestVoxel;
filterOp.onchange = () => {
  for (const group of document.querySelectorAll<HTMLElement>("#filter-panel [data-op]")) {
    group.hidden = group.dataset.op !== filterOp.value;
  }
};

filterRun.onclick = async () => {
  const entry = entries.get(Number(filterCloudSelect.value));
  if (!entry) return;
  if (filterOp.value === "ground") {
    await runGround(entry);
    return;
  }
  const op = filterOp.value as "voxel" | "random" | "sor";
  let a = 0;
  let b = 0;
  if (op === "voxel") a = Number(voxelInput.value);
  if (op === "random") a = Math.round((entry.cloud.count * Number($<HTMLInputElement>("filter-percent").value)) / 100);
  if (op === "sor") {
    a = Number($<HTMLInputElement>("filter-k").value);
    b = Number($<HTMLInputElement>("filter-ratio").value);
  }
  if (!(a > 0)) {
    setStatus("Enter a positive value", true);
    return;
  }
  filterRun.disabled = true;
  setStatus(`Filtering ${entry.cloud.name}…`);
  try {
    const cloud = await filterCloud(entry.cloud.id, op, a, b);
    const label = { voxel: "the voxel filter", random: "the random subsampling", sor: "the outlier filter" }[op];
    record({ label, added: [addEntry(cloud)], hide: [entry] });
    renderList();
    const removed = entry.cloud.count - cloud.count;
    setStatus(
      `${cloud.name}: kept ${cloud.count.toLocaleString()} of ${entry.cloud.count.toLocaleString()} points ` +
        `(${removed.toLocaleString()} removed) in ${Math.round(cloud.timings.index)} ms`,
    );
  } catch (err) {
    setStatus(`Filter failed: ${errorText(err)}`, true);
  } finally {
    filterRun.disabled = false;
  }
};

async function runGround(entry: Entry): Promise<void> {
  const clothResolution = Number($<HTMLInputElement>("csf-resolution").value);
  const classThreshold = Number($<HTMLInputElement>("csf-threshold").value);
  if (!(clothResolution > 0) || !(classThreshold > 0)) {
    setStatus("Enter a positive cloth resolution and threshold", true);
    return;
  }
  const output = $<HTMLSelectElement>("csf-output").value as "classified" | "ground" | "objects";
  filterRun.disabled = true;
  setStatus(`Extracting ground from ${entry.cloud.name}…`);
  try {
    const cloud = await extractGround({
      id: entry.cloud.id,
      clothResolution,
      classThreshold,
      rigidness: $<HTMLSelectElement>("csf-rigidness").value as "flat" | "relief" | "steep",
      output,
    });
    const added = addEntry(cloud);
    if (output === "classified") {
      added.mode = "classification";
      refreshColors(added);
    }
    record({ label: "the ground extraction", added: [added], hide: [entry] });
    renderList();
    let detail = `${cloud.count.toLocaleString()} points`;
    if (output === "classified" && cloud.classification) {
      const ground = cloud.classification.reduce((n, c) => n + (c === 2 ? 1 : 0), 0);
      detail = `${ground.toLocaleString()} of ${cloud.count.toLocaleString()} points are ground`;
    }
    setStatus(`${cloud.name}: ${detail} (${Math.round(cloud.timings.index)} ms)`);
  } catch (err) {
    setStatus(`Ground extraction failed: ${errorText(err)}`, true);
  } finally {
    filterRun.disabled = false;
  }
}

// ---------------------------------------------------------------- merge / split

const mergeRun = $<HTMLButtonElement>("merge-run");
const splitCloudSelect = $<HTMLSelectElement>("split-cloud");
const splitBy = $<HTMLSelectElement>("split-by");
const splitRun = $<HTMLButtonElement>("split-run");

function renderMergeSplit(): void {
  mergeRun.disabled = clouds().filter((e) => e.visible).length < 2;
  fillCloudSelect(
    splitCloudSelect,
    clouds().filter((e) => e.cloud.classification || e.cloud.sources),
  );
  updateSplit();
}
listChanged.add(renderMergeSplit);

function updateSplit(): void {
  const entry = entries.get(Number(splitCloudSelect.value));
  const can = { classification: !!entry?.cloud.classification, source: !!entry?.cloud.sources };
  for (const option of splitBy.options) option.disabled = !can[option.value as keyof typeof can];
  if (splitBy.selectedOptions[0]?.disabled) {
    splitBy.value = [...splitBy.options].find((o) => !o.disabled)?.value ?? splitBy.value;
  }
  splitRun.disabled = !entry || !can[splitBy.value as keyof typeof can];
}
splitCloudSelect.onchange = splitBy.onchange = updateSplit;

mergeRun.onclick = async () => {
  const sources = clouds().filter((e) => e.visible);
  if (sources.length < 2) return;
  mergeRun.disabled = true;
  setStatus(`Merging ${sources.length} clouds…`);
  try {
    const cloud = await mergeClouds(
      sources.map((e) => e.cloud.id),
      sources.map((e) => e.solid),
    );
    record({ label: "the merge", added: [addEntry(cloud)], hide: sources });
    renderList();
    setStatus(
      `${cloud.name}: ${cloud.count.toLocaleString()} points from ${sources.map((e) => e.cloud.name).join(", ")} ` +
        `(${Math.round(cloud.timings.index)} ms)`,
    );
  } catch (err) {
    setStatus(`Merge failed: ${errorText(err)}`, true);
  } finally {
    renderMergeSplit();
  }
};

splitRun.onclick = async () => {
  const entry = entries.get(Number(splitCloudSelect.value));
  if (!entry) return;
  const by = splitBy.value as "classification" | "source";
  splitRun.disabled = true;
  setStatus(`Splitting ${entry.cloud.name} by ${by}…`);
  try {
    const parts = await splitCloud(entry.cloud.id, by);
    const added = parts.map((cloud) => {
      const part = addEntry(cloud);
      if (by === "classification") part.solid = classColor(cloud.classification?.[0] ?? 0);
      refreshColors(part);
      return part;
    });
    record({ label: "the split", added, hide: [entry] });
    renderList();
    const names = parts.map((c) =>
      by === "classification" ? `${className(c.classification?.[0] ?? 0)} (${c.count.toLocaleString()})` : c.name,
    );
    setStatus(`Split ${entry.cloud.name} into ${parts.length} clouds: ${names.join(", ")}`);
  } catch (err) {
    setStatus(`Split failed: ${errorText(err)}`, true);
  } finally {
    updateSplit();
  }
};

// ---------------------------------------------------------------- normals

const normalsCloud = $<HTMLSelectElement>("normals-cloud");
const normalsRun = $<HTMLButtonElement>("normals-run");

listChanged.add(() => {
  fillCloudSelect(normalsCloud, clouds());
  normalsRun.disabled = clouds().length === 0;
});

normalsRun.onclick = async () => {
  const entry = entries.get(Number(normalsCloud.value));
  if (!entry) return;
  normalsRun.disabled = true;
  setStatus(`Estimating normals of ${entry.cloud.name}…`);
  const start = performance.now();
  try {
    const k = Math.max(3, Number($<HTMLInputElement>("normals-k").value) || 12);
    const orientation = $<HTMLSelectElement>("normals-orient").value as "up" | "outward";
    entry.cloud.normals = await estimateNormals(entry.cloud.id, k, orientation);
    entry.mode = "shade";
    refreshColors(entry);
    renderList();
    setStatus(
      `Normals of ${entry.cloud.count.toLocaleString()} points in ${Math.round(performance.now() - start)} ms ` +
        `(${k} neighbours, facing ${orientation === "up" ? "up" : "away from the centre"})`,
    );
  } catch (err) {
    setStatus(`Normals failed: ${errorText(err)}`, true);
  } finally {
    normalsRun.disabled = false;
  }
};
