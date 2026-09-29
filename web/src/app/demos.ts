/** Demos: synthetic samples (see `scripts/make-samples.mjs`) and the analysis each one runs. */

import { setPointSize } from "./display";
import { runButton } from "./distance";
import { $, choose, setStatus } from "./dom";
import { renderList } from "./entries";
import { loadUrls } from "./loading";
import { addGravity, compareWithStart, findLoops, open as openPoseGraph } from "./posegraph";
import { poseGraphDemoFiles } from "./posegraph-demo";
import { entries, hideEntry, viewer } from "./state";

const idOf = (name: string) => String([...entries.values()].find((e) => e.cloud.name === name)?.cloud.id ?? "");

function fill(values: Record<string, string>): void {
  for (const [id, value] of Object.entries(values)) $<HTMLInputElement>(id).value = value;
}

/** Hide input clouds so a demo shows its result alone. */
function hide(...names: string[]): void {
  for (const e of entries.values()) if (names.includes(e.cloud.name)) hideEntry(e);
  renderList();
}

const DEMOS: Record<string, { files: string[]; run: () => void }> = {
  c2c: {
    files: ["lidar_reference.pcd", "lidar_candidate.pcd"],
    run: () => runButton.click(),
  },
  volume: {
    files: ["stockpile_before.ply", "stockpile_after.ply"],
    run: () => {
      choose("volume-before", idOf("stockpile_before.ply"));
      choose("volume-after", idOf("stockpile_after.ply"));
      fill({ "volume-cell": "0.4" });
      $<HTMLButtonElement>("volume-run").click();
      hide("stockpile_before.ply", "stockpile_after.ply");
    },
  },
  ground: {
    files: ["town.ply"],
    run: () => {
      choose("filter-cloud", idOf("town.ply"));
      choose("filter-op", "ground");
      fill({ "csf-resolution": "1", "csf-threshold": "0.3" });
      choose("csf-rigidness", "relief");
      choose("csf-output", "classified");
      $<HTMLButtonElement>("filter-run").click();
    },
  },
  m3c2: {
    files: ["slope_before.ply", "slope_after.ply"],
    run: () => {
      choose("distance-method", "m3c2");
      choose("c2c-compared", idOf("slope_after.ply"));
      choose("c2c-reference", idOf("slope_before.ply"));
      fill({ "m3c2-normal": "1", "m3c2-projection": "0.5", "m3c2-depth": "2", "m3c2-core": "0.3" });
      runButton.click();
      hide("slope_before.ply");
    },
  },
};

/**
 * A drive round a city block, made in the browser: open its drifting pose
 * graph, close the loop, tie it to the IMU's gravity, and show how far the
 * corrections moved each point of the map.
 */
async function poseGraphDemo(): Promise<void> {
  setStatus("Making a drive round a city block…");
  const { scans, poses, gravity } = poseGraphDemoFiles();
  await openPoseGraph([poses, ...scans]);
  await findLoops();
  await addGravity([gravity]);
  $<HTMLInputElement>("pg-show-scans").click();
  await compareWithStart();
  $("pose-graph-panel").scrollIntoView({ block: "start" });
  viewer.view({ x: 0.4, y: -1, z: 1.1 });
  viewer.fit();
}

/** The same drive, only opened: to replay it or close a loop by hand. */
async function poseGraphDrive(): Promise<void> {
  setStatus("Making a drive round a city block…");
  const { scans, poses } = poseGraphDemoFiles();
  await openPoseGraph([poses, ...scans]);
  $("pose-graph-panel").scrollIntoView({ block: "start" });
}

/** Load a demo's sample files, then run its analysis. */
export async function runDemo(name: string): Promise<void> {
  if (name === "posegraph") {
    await poseGraphDemo();
    return;
  }
  if (name === "posegraph-drive") {
    await poseGraphDrive();
    return;
  }
  const demo = DEMOS[name];
  if (!demo) {
    setStatus(`Unknown demo "${name}" (try ${[...Object.keys(DEMOS), "posegraph"].join(", ")})`, true);
    return;
  }
  const missing = demo.files.filter((f) => !idOf(f));
  await loadUrls(missing.map((f) => `${import.meta.env.BASE_URL}samples/${f}`));
  if (!demo.files.every((f) => idOf(f))) return;
  // The synthetic samples are dense grids: bigger points close the gaps.
  if (name !== "c2c") setPointSize(5);
  demo.run();
}

$<HTMLButtonElement>("load-sample").onclick = () => void runDemo("c2c");
for (const button of document.querySelectorAll<HTMLButtonElement>("[data-demo]")) {
  button.onclick = () => void runDemo(button.dataset.demo!);
}
