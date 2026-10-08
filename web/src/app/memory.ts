import * as THREE from "three";
import { memoryStats, releaseUnusedPool } from "../api";
import { configureHistory, geometryBytes, retainedBuffers } from "../memory-budget";
import { clearCloudHistory, historyMemory } from "./history";
import { clearGraphHistory, graphHistoryMemory } from "./posegraph";
import { clearMapHistory } from "./vectormap";
import { entries, viewer } from "./state";
import { $, errorText, setStatus } from "./dom";
import { taskActive } from "./tasks";

const mib = (bytes: number) => `${(bytes / 1024 / 1024).toFixed(1)} MiB`;
let refreshing = false;
export async function refreshMemory(): Promise<void> {
  if (refreshing) return;
  refreshing = true;
  try {
    const stats = await memoryStats(), cloud = historyMemory(), graph = graphHistoryMemory();
    const cached = viewer.memoryData();
    const buffers = retainedBuffers([...entries.values(), cloud.data, graph.data, cached.data]);
    const geometries = new Set<THREE.BufferGeometry>(cached.geometries);
    for (const scene of [viewer.scene, viewer.overlay]) scene.traverse(object => {
      const geometry = (object as THREE.Mesh).geometry;
      if (geometry) geometries.add(geometry);
    });
    const gpuAttributes: {owner: object; array: ArrayBufferView}[] = [];
    for (const geometry of geometries) {
      const attributes = [...Object.values(geometry.attributes), ...(geometry.index ? [geometry.index] : [])];
      const arrays = attributes.map(attribute => {
        const owner = attribute instanceof THREE.InterleavedBufferAttribute ? attribute.data : attribute;
        gpuAttributes.push({owner, array: owner.array});
        return owner.array;
      });
      for (const buffer of retainedBuffers(arrays)) buffers.add(buffer);
    }
    const gpu = geometryBytes(gpuAttributes);
    const cpu = [...buffers].reduce((sum,b) => sum + b.byteLength,0);
    $("memory-report").textContent = `Main WASM ${mib(stats.main)} · parallel WASM ${mib(stats.pool)} · retained arrays ${mib(cpu)} · geometry GPU estimate ${mib(gpu)}. Undo: clouds ${cloud.steps} (${mib(cloud.bytes)}), poses ${graph.steps} (${mib(graph.bytes)}), map ${stats.mapSteps} (${mib(stats.mapHistory)}).`;
    const threshold = Number($<HTMLInputElement>("memory-warning").value) * 1024 * 1024;
    $("memory-warning-report").textContent = stats.main + stats.pool + cpu + gpu > threshold ? "Estimated memory exceeds the warning threshold. Save a project, release Undo and idle workers, or remove clouds you no longer need." : "";
  } catch (error) { $("memory-report").textContent = errorText(error); }
  finally { refreshing = false; }
}
$("memory-apply").onclick = () => {
  try {
    configureHistory(Number($<HTMLInputElement>("memory-steps").value), Number($<HTMLInputElement>("memory-undo-budget").value) * 1024 * 1024);
    setStatus("Undo limits applied to cloud, pose graph and map histories."); void refreshMemory();
  } catch (error) { setStatus(errorText(error), true); }
};
$("memory-release").onclick = async () => {
  if (taskActive()) return setStatus("Finish the active operation before releasing memory.");
  try {
    await releaseUnusedPool();
    if (!clearCloudHistory() || !clearGraphHistory()) throw new Error("Finish the active operation before clearing Undo");
    await clearMapHistory();
    viewer.releaseUnusedDetails();
    setStatus("Undo cleared and idle workers released. Current clouds, map and poses retained.");
    await refreshMemory();
  } catch (error) { setStatus(`Could not release memory: ${errorText(error)}`, true); }
};
$("memory-panel").addEventListener("toggle", () => { if ($<HTMLDetailsElement>("memory-panel").open) void refreshMemory(); });
setInterval(() => { if ($<HTMLDetailsElement>("memory-panel").open) void refreshMemory(); }, 3000);
