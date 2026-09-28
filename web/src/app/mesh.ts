/** Mesh panel: a cloud triangulated in the XY plane (2.5D Delaunay). */

import { meshCloud } from "../api";
import { $, errorText, fmt, setStatus } from "./dom";
import { addEntry, renderList } from "./entries";
import { record } from "./history";
import { fillCloudSelect } from "./processing";
import { clouds, entries, listChanged } from "./state";

const cloudSelect = $<HTMLSelectElement>("mesh-cloud");
const maxEdgeInput = $<HTMLInputElement>("mesh-max-edge");
const runButton = $<HTMLButtonElement>("mesh-run");

listChanged.add(() => {
  fillCloudSelect(cloudSelect, clouds());
  runButton.disabled = clouds().length === 0;
});

runButton.onclick = async () => {
  const source = entries.get(Number(cloudSelect.value));
  if (!source) return;
  // Empty: automatic (a few times the median edge); 0 keeps every triangle.
  const text = maxEdgeInput.value.trim();
  const maxEdge = text === "" ? null : Number(text);
  if (maxEdge !== null && !(maxEdge >= 0)) {
    setStatus("Enter a max edge length of 0 or more, or leave it empty for auto", true);
    return;
  }
  runButton.disabled = true;
  setStatus(`Meshing ${source.cloud.name}…`);
  try {
    const out = await meshCloud(source.cloud.id, maxEdge);
    record({ label: "the mesh", added: [addEntry(out.mesh)], hide: [source] });
    renderList();
    const thinned =
      out.voxel > 0
        ? ` (thinned from ${source.cloud.count.toLocaleString()} points with a ${fmt(out.voxel)} voxel filter)`
        : "";
    const filtered = Number.isFinite(out.maxEdge)
      ? `, ${out.removed.toLocaleString()} longer than ${fmt(out.maxEdge)} removed`
      : "";
    setStatus(
      `${out.mesh.name}: ${out.mesh.triangles.toLocaleString()} triangles from ` +
        `${out.points.toLocaleString()} points${thinned}${filtered}, in ${Math.round(out.millis)} ms`,
    );
  } catch (err) {
    setStatus(`Meshing failed: ${errorText(err)}`, true);
  } finally {
    runButton.disabled = clouds().length === 0;
  }
};
