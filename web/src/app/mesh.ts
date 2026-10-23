/** Mesh panel: a cloud triangulated in the XY plane (2.5D Delaunay). */

import { exportMesh, meshCloud, removeCloud, workerBusy } from "../api";
import type { MeshFormat, MeshOutput } from "../protocol";
import { projectChanged } from "../project-change";
import { $, download, errorText, fmt, setStatus } from "./dom";
import { addEntry, renderList } from "./entries";
import { cloudHistoryReady, record, requireDerivedUndo } from "./history";
import { fillCloudSelect } from "./processing";
import { clouds, distanceChanged, entries, listChanged, pointsInvalidated, viewer, type Entry } from "./state";
import { addReportSection } from "./report";
import { endTask, startTask, taskActive } from "./tasks";

const cloudSelect = $<HTMLSelectElement>("mesh-cloud");
const maxEdgeInput = $<HTMLInputElement>("mesh-max-edge");
const runButton = $<HTMLButtonElement>("mesh-run");
const wireframe = $<HTMLInputElement>("mesh-wireframe");
type Snapshot = { entry: Entry; cloud: Entry["cloud"]; name: string };
type Result = { source: Snapshot; output: Entry; out: MeshOutput; requestedMaxEdge: number | null };
let last: Result | null = null, running = false, generation = 0, active: Snapshot | null = null;
const validSource = (s: Snapshot) => entries.get(s.cloud.id) === s.entry && s.entry.cloud === s.cloud;
const validResult = (r: Result) => validSource(r.source) && entries.get(r.out.mesh.id) === r.output && r.output.cloud === r.out.mesh;
function clear(): void {
  const had = last !== null; generation++; last = null; $("mesh-result").hidden = true;
  if (had) distanceChanged.emit();
}
function render(): void {
  runButton.disabled = running || clouds().length === 0;
  cloudSelect.disabled = maxEdgeInput.disabled = wireframe.disabled = running;
  for (const button of $("mesh-result").querySelectorAll<HTMLButtonElement>("button")) button.disabled = running;
}

listChanged.add(() => {
  fillCloudSelect(cloudSelect, clouds());
  if (last && !validResult(last)) clear();
  render();
});
cloudSelect.onchange = () => { clear(); render(); };
maxEdgeInput.oninput = clear;
pointsInvalidated.add(id => { if (id === active?.cloud.id || id === last?.source.cloud.id || id === last?.out.mesh.id) clear(); });
wireframe.onchange = () => {
  if (!last || !validResult(last)) return;
  viewer.setMeshWireframe(last.out.mesh.id, wireframe.checked); projectChanged();
};

runButton.onclick = async () => {
  if (running || taskActive() || workerBusy() || !cloudHistoryReady()) return setStatus("Finish the current operation before meshing", true);
  const source = entries.get(Number(cloudSelect.value));
  if (!source || source.cloud.kind !== "cloud") return;
  // Empty: automatic (a few times the median edge); 0 keeps every triangle.
  const text = maxEdgeInput.value.trim();
  const maxEdge = text === "" ? null : Number(text);
  if (maxEdge !== null && (!Number.isFinite(maxEdge) || maxEdge < 0)) {
    setStatus("Enter a finite max edge length of 0 or more, or leave it empty for auto", true);
    return;
  }
  clear(); const token = generation, snapshot: Snapshot = { entry: source, cloud: source.cloud, name: source.cloud.name }, signal = startTask();
  running = true; active = snapshot; render();
  const cancelled = () => setStatus("Cancel requested; the current triangulation finishes before its result is discarded.");
  signal.addEventListener("abort", cancelled);
  let out: MeshOutput | null = null, adopted = false;
  setStatus(`Meshing ${snapshot.name}…`);
  try {
    out = await meshCloud(snapshot.cloud.id, maxEdge);
    signal.throwIfAborted();
    if (token !== generation || !validSource(snapshot)) throw new Error("Meshing inputs/settings changed; mesh again");
    try { requireDerivedUndo([out.mesh], [source]); }
    catch { throw new Error("Mesh and source do not fit one Undo step; increase Memory and Undo limits or crop the source"); }
    const output = addEntry(out.mesh); adopted = true;
    record({ label: "the mesh", added: [output], hide: [source] });
    renderList();
    last = { source: snapshot, output, out, requestedMaxEdge: maxEdge };
    wireframe.checked = false; $("mesh-result").hidden = false;
    const thinned =
      out.voxel > 0
        ? ` (thinned from ${snapshot.cloud.count.toLocaleString()} points with a ${fmt(out.voxel)} voxel filter)`
        : "";
    const filtered = Number.isFinite(out.maxEdge)
      ? `, ${out.removed.toLocaleString()} longer than ${fmt(out.maxEdge)} removed`
      : "";
    const summary = `${out.mesh.name}: ${out.mesh.triangles.toLocaleString()} triangles from ${out.points.toLocaleString()} points${thinned}${filtered}`;
    $("mesh-summary").textContent = `${summary}. XY triangulation of the loaded source; source heights retained. Max edge is a length rule, not an observed-surface mask.`;
    distanceChanged.emit();
    setStatus(`${summary}, in ${Math.round(out.millis)} ms`);
  } catch (err) {
    if (out && !adopted) await removeCloud(out.mesh.id);
    setStatus(`Meshing failed: ${signal.aborted ? "Cancelled" : errorText(err)}`, true);
  } finally {
    signal.removeEventListener("abort", cancelled); active = null; running = false; endTask(signal); render();
  }
};

function summary(r: Result) {
  const c = r.source.cloud;
  return { schema: "cloudanalyzer.terrain_mesh.v1", method: "2.5D XY Delaunay; source heights retained; no ground classification",
    source: { id: c.id, name: r.source.name, loadedPoints: c.count, filePoints: c.filePoints, keepEvery: c.keepEvery,
      preview: r.source.entry.origin.displayPreview === true, copcLevels: c.copcLevels },
    output: { id: r.out.mesh.id, name: r.out.mesh.name, vertices: r.out.mesh.count, triangles: r.out.mesh.triangles },
    settings: { maxEdgeMode: r.requestedMaxEdge === null ? "auto" : r.requestedMaxEdge === 0 ? "unlimited" : "length", requestedMaxEdge: r.requestedMaxEdge,
      effectiveMaxEdge: Number.isFinite(r.out.maxEdge) ? r.out.maxEdge : null },
    processing: { triangulationPoints: r.out.points, voxel: r.out.voxel, removedTriangles: r.out.removed },
    scope: "Loaded source points, optionally voxel-thinned by the existing engine. One terrain height per XY; not a closed-solid or multi-layer reconstruction. Max edge limits horizontal lengths and does not guarantee gap preservation. Wireframe is display-only; OBJ/PLY retain faces." };
}
$("mesh-json").onclick = () => {
  if (last && validResult(last)) download(new Blob([JSON.stringify(summary(last), null, 2)], { type: "application/json" }), "terrain-mesh-summary.json");
};
async function save(format: MeshFormat): Promise<void> {
  const result = last;
  if (!result || !validResult(result) || running || taskActive() || workerBusy()) return;
  try {
    const bytes = await exportMesh(result.out.mesh.id, format);
    if (last !== result || !validResult(result)) throw new Error("Mesh inputs/result changed; export again");
    download(bytes, `${result.out.mesh.name}.${format}`); setStatus(`Saved terrain mesh ${format.toUpperCase()}.`);
  } catch (error) { setStatus(`Mesh export failed: ${errorText(error)}`, true); }
}
$("mesh-ply").onclick = () => void save("ply");
$("mesh-obj").onclick = () => void save("obj");
$("mesh-volume-after").onclick = () => {
  if (!last || !validResult(last) || running || taskActive() || workerBusy()) return;
  const select = $<HTMLSelectElement>("volume-after"); select.value = String(last.out.mesh.id); select.dispatchEvent(new Event("change"));
  if ($<HTMLSelectElement>("volume-before").value === select.value) {
    const before = $<HTMLSelectElement>("volume-before"); before.value = "constant"; before.dispatchEvent(new Event("change"));
  }
  const fill = $<HTMLInputElement>("volume-fill"); fill.checked = false; fill.dispatchEvent(new Event("change"));
  $("volume-panel").scrollIntoView({ block: "start" });
  setStatus("Terrain mesh selected as After surface; interpolation off. Review Before, cell size and coordinate unit, then Compute volume.");
};
addReportSection("terrain-mesh", () => last && validResult(last) ? { title: `Terrain mesh of ${last.source.name}`,
  metrics: { sourcePoints: { label: "Loaded source points", value: last.source.cloud.count },
    meshingPoints: { label: "Points used for triangulation", value: last.out.points },
    vertices: { label: "Mesh vertices", value: last.out.mesh.count }, triangles: { label: "Mesh triangles", value: last.out.mesh.triangles },
    removedTriangles: { label: "Triangles removed by max edge", value: last.out.removed } },
  notes: [`XY Delaunay; requested max edge ${last.requestedMaxEdge === null ? "auto" : last.requestedMaxEdge}; effective ${Number.isFinite(last.out.maxEdge) ? last.out.maxEdge : "unlimited"}; voxel ${last.out.voxel}.`,
    "Loaded surface only; max edge is not an observation footprint or a guarantee against filled holes. Wireframe does not change geometry."] } : null);
