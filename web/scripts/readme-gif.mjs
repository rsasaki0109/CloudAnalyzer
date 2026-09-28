// Turn the orbit frames written by `media/readme.spec.ts` into the README GIF:
// the 3D view only (no sidebar or toolbar), C2C → M3C2 → volume → ground.
import { execFileSync } from "node:child_process";
import { readdirSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const frames = fileURLToPath(new URL("../media-frames/", import.meta.url));
const out = fileURLToPath(new URL("../../docs/images/web/demo.gif", import.meta.url));
const names = readdirSync(frames)
  .filter((f) => f.endsWith(".png"))
  .sort();
const count = names.length;
if (!count) throw new Error("no frames: run the media spec first");
// A concat list works everywhere (ffmpeg's glob input does not on Windows).
const list = `${frames}frames.txt`;
writeFileSync(list, names.map((n) => `file '${n}'
duration 0.125
`).join("") + `file '${names.at(-1)}'
`);

// Frames sort by scene prefix (a-c2c, b-m3c2, …) then number.
const filter =
  "crop=1000:660:280:74,scale=720:-1:flags=lanczos,split[a][b];" +
  "[a]palettegen=max_colors=128:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4";
execFileSync(
  "ffmpeg",
  ["-loglevel", "error", "-y", "-f", "concat", "-safe", "0", "-i", list, "-vf", filter, "-loop", "0", out],
  { stdio: "inherit" },
);
console.log(`wrote ${out} from ${count} frames`);
