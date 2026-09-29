// Turn the frames written by `media/readme.spec.ts` into the README GIFs:
// the 3D view only (no sidebar or toolbar). demo.gif is the analysis tour
// (C2C → M3C2 → volume → ground → pose graph); odometry.gif and loop.gif
// show a drive being replayed and a loop being closed by hand.
import { execFileSync } from "node:child_process";
import { readdirSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const frames = fileURLToPath(new URL("../media-frames/", import.meta.url));
const images = new URL("../../docs/images/web/", import.meta.url);
const all = readdirSync(frames)
  .filter((f) => f.endsWith(".png"))
  .sort();
if (!all.length) throw new Error("no frames: run the media spec first");

// Frames sort by scene prefix (a-c2c, b-m3c2, …) then number.
const GIFS = [
  { name: "demo.gif", scenes: /^[a-e]-/, seconds: 0.125 },
  { name: "odometry.gif", scenes: /^f-/, seconds: 0.18 },
  { name: "loop.gif", scenes: /^g-/, seconds: 0.16 },
];
const filter =
  "crop=1000:660:280:74,scale=720:-1:flags=lanczos,split[a][b];" +
  "[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4";

for (const { name, scenes, seconds } of GIFS) {
  const names = all.filter((f) => scenes.test(f));
  if (!names.length) {
    console.log(`skipped ${name}: no frames`);
    continue;
  }
  // A concat list works everywhere (ffmpeg's glob input does not on Windows).
  const list = `${frames}${name}.txt`;
  writeFileSync(list, names.map((n) => `file '${n}'\nduration ${seconds}\n`).join("") + `file '${names.at(-1)}'\n`);
  const out = fileURLToPath(new URL(name, images));
  execFileSync(
    "ffmpeg",
    ["-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", list, "-vf", filter, "-loop", "0", out],
    { stdio: "inherit" },
  );
  console.log(`wrote ${out} from ${names.length} frames`);
}
