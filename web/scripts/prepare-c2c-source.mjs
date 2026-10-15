// Split measured source coordinates without generating or moving points.
import { existsSync, mkdirSync, readFileSync, writeFileSync } from "node:fs";
import { basename, resolve } from "node:path";
import { fileURLToPath } from "node:url";
import { createHash } from "node:crypto";
import { Cloud, announcedPoints, initSync } from "../src/wasm/ca_wasm.js";

const args = process.argv.slice(2);
const value = (key, fallback) => args.includes(key) ? args[args.indexOf(key) + 1] : fallback;
const source = value("--source"), output = value("--output");
if (!source || !output) throw new Error("Supply --source CLOUD and a new --output DIRECTORY; optional --source-url, --attribution, --license, --preparation JSON");
if (existsSync(output)) throw new Error(`Output already exists: ${output}`);
const hash = bytes => createHash("sha256").update(bytes).digest("hex");
const wasmPath = fileURLToPath(new URL("../src/wasm/ca_wasm_bg.wasm", import.meta.url));
const wasm = readFileSync(wasmPath);
initSync({ module: wasm });
const bytes = readFileSync(source);
const cloud = Cloud.parse(basename(source), bytes);
let xyz;
try { xyz = cloud.rawPositions(); } finally { cloud.free(); }
const count = xyz.length / 3;
if (count < 2 || !xyz.every(Number.isFinite)) throw new Error("Need at least two decoded finite points");
const preparationPath = value("--preparation");
const preparationBytes = preparationPath && readFileSync(preparationPath);
const manifest = {
  schemaVersion: 1,
  source: { path: source, bytes: bytes.length, sha256: hash(bytes), declaredPoints: announcedPoints(basename(source), bytes),
    decodedPoints: count, url: value("--source-url"), attribution: value("--attribution"), license: value("--license"),
    preparation: preparationBytes ? { path: preparationPath, sha256: hash(preparationBytes), metadata: JSON.parse(preparationBytes) } : undefined },
  split: { method: "Alternating decoded source indices before LOD indexing: reference 0,2,4,...; compared 1,3,5,...",
    disjointSourceIndices: true, geometricDuplicatesPossible: true,
    coordinates: "Decoded f64 XYZ unchanged; no offset, noise, registration, averaging or generated points",
    attributes: "Only XYZ is exported; original source attributes are not modified",
    interpretation: "Two subsets of one measured cloud, not independent surveys or map-accuracy ground truth" },
  decoderWasmSha256: hash(wasm),
  preparationScriptSha256: hash(readFileSync(fileURLToPath(import.meta.url))),
  inputs: {},
};
// Validate provenance before creating any output files.
mkdirSync(output);
for (const [name, first] of [["reference", 0], ["compared", 1]]) {
  const points = Math.ceil((count - first) / 2);
  const header = Buffer.from(`ply\nformat binary_little_endian 1.0\nelement vertex ${points}\nproperty double x\nproperty double y\nproperty double z\nend_header\n`);
  const prepared = Buffer.allocUnsafe(header.length + points * 24);
  header.copy(prepared);
  for (let i = 0; i < points; i++) for (let axis = 0; axis < 3; axis++)
    prepared.writeDoubleLE(xyz[(i * 2 + first) * 3 + axis], header.length + i * 24 + axis * 8);
  // The production decoder must reproduce every selected f64 bit, including -0.
  const decoded = Cloud.parse(`${name}.ply`, prepared);
  try {
    const actual = decoded.rawPositions();
    const sourceBits = new BigUint64Array(xyz.buffer, xyz.byteOffset, xyz.length);
    const actualBits = new BigUint64Array(actual.buffer, actual.byteOffset, actual.length);
    if (actual.length !== points * 3 || actualBits.some((v, i) => v !== sourceBits[(Math.floor(i / 3) * 2 + first) * 3 + i % 3]))
      throw new Error("Prepared source coordinates changed");
  } finally { decoded.free(); }
  const path = resolve(output, `${name}.ply`);
  writeFileSync(path, prepared, { flag: "wx" });
  manifest.inputs[name] = { file: `${name}.ply`, points, bytes: prepared.length, sha256: hash(prepared) };
}
writeFileSync(resolve(output, "inputs.json"), JSON.stringify(manifest, null, 2) + "\n", { flag: "wx" });
console.log(`Prepared ${count} measured points: reference ${manifest.inputs.reference.points}, compared ${manifest.inputs.compared.points}; every selected f64 coordinate bit preserved`);
