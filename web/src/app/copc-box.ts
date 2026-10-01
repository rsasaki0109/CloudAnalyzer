/** A bounded working cloud from every COPC density level. */
import { readCopcBox, removeCloud } from "../api";
import { CANCELLED, type Vec3 } from "../protocol";
import { clipBox } from "./clip";
import { $, errorText, setStatus } from "./dom";
import { addEntry, renderList } from "./entries";
import { record } from "./history";
import { clouds, entries, listChanged, pointsInvalidated, viewer } from "./state";
import { endTask, showProgress, startTask } from "./tasks";

const sourceSelect = $<HTMLSelectElement>("copc-box-source");
const run = $<HTMLButtonElement>("copc-box-run");
const fields = ["xmin", "ymin", "zmin", "xmax", "ymax", "zmax"].map((a) =>
  $<HTMLInputElement>(`copc-box-${a}`));
const limit = $<HTMLInputElement>("copc-box-limit");
const result = $("copc-box-result");
let busy = false;
let revision = 0;
let activeId: number | null = null;
const selected = () => entries.get(Number(sourceSelect.value));
const changed = () => { revision++; result.textContent = ""; };
function fillBounds(): void {
  const entry = selected();
  if (entry) fields.forEach((f, i) => { f.value = String(entry.cloud.bounds[i]); });
  changed();
}
function updateSources(): void {
  const previous = sourceSelect.value;
  sourceSelect.replaceChildren(...clouds().filter((e) => e.cloud.copcBoxAvailable).map((e) =>
    new Option(e.cloud.name, String(e.cloud.id))));
  if ([...sourceSelect.options].some((o) => o.value === previous)) sourceSelect.value = previous;
  else fillBounds();
  run.disabled = busy || !selected();
}
sourceSelect.onchange = fillBounds;
for (const f of [...fields, limit]) f.oninput = changed;
$<HTMLButtonElement>("copc-box-use-clip").onclick = () => {
  const box = clipBox();
  if (!box) { setStatus("Enable the clipping box first", true); return; }
  fields.forEach((f, i) => { f.value = String([...box.min, ...box.max][i]); });
  changed();
};
pointsInvalidated.add((id) => { if (id === activeId) changed(); });
listChanged.add(updateSources);
updateSources();
run.onclick = async () => {
  const source = selected();
  if (!source || busy) return;
  const box = fields.map((f) => f.value.trim() ? Number(f.value) : NaN);
  const maxPoints = Number(limit.value);
  if (!box.every(Number.isFinite) || box.slice(0, 3).some((v, i) => v > box[i + 3]) ||
      !Number.isSafeInteger(maxPoints) || maxPoints < 1 || maxPoints > 1_000_000) {
    setStatus("Enter ordered finite XYZ bounds and a point limit in 1..1000000", true); return;
  }
  busy = true;
  activeId = source.cloud.id;
  const before = revision;
  run.disabled = true;
  const signal = startTask();
  try {
    const cloud = await readCopcBox(source.cloud.id, box.slice(0, 3) as Vec3, box.slice(3) as Vec3,
      maxPoints, showProgress, signal);
    if (signal.aborted || before !== revision || entries.get(source.cloud.id) !== source) {
      await removeCloud(cloud.id);
      throw new Error("Selection inputs or original cloud changed; result discarded");
    }
    const added = addEntry(cloud);
    record({ label: "the full-density box", added: [added], hide: [source] });
    renderList();
    viewer.fit();
    const stats = cloud.copcBox!;
    result.textContent = `${cloud.count.toLocaleString()} points from ${stats.nodes} nodes; ${stats.sourceReadBytes.toLocaleString()} source bytes read`;
    setStatus(`Full-density box: ${result.textContent}`);
  } catch (error) {
    const message = errorText(error);
    setStatus(message === CANCELLED ? "Full-density selection cancelled" : `Full-density box: ${message}`, message !== CANCELLED);
  } finally {
    endTask(signal);
    activeId = null;
    busy = false;
    updateSources();
  }
};
