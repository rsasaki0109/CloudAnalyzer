// Generate the synthetic demo clouds in public/samples (binary PLY, float
// xyz + uchar rgb). Deterministic: `node scripts/make-samples.mjs`.

import { writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";

const out = new URL("../public/samples/", import.meta.url);

/** xorshift in [0, 1), seeded, so the files are reproducible. */
function random(seed) {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13;
    s ^= s >>> 17;
    s ^= s << 5;
    return (s >>> 0) / 4294967296;
  };
}

function writePly(name, points) {
  const header =
    "ply\nformat binary_little_endian 1.0\n" +
    "comment CloudAnalyzer synthetic demo\n" +
    `element vertex ${points.length}\n` +
    "property float x\nproperty float y\nproperty float z\n" +
    "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n";
  const body = Buffer.alloc(points.length * 15);
  points.forEach(([x, y, z, r, g, b], i) => {
    const o = i * 15;
    body.writeFloatLE(x, o);
    body.writeFloatLE(y, o + 4);
    body.writeFloatLE(z, o + 8);
    body[o + 12] = r;
    body[o + 13] = g;
    body[o + 14] = b;
  });
  const path = fileURLToPath(new URL(name, out));
  writeFileSync(path, Buffer.concat([Buffer.from(header), body]));
  console.log(`${name}: ${points.length} points`);
}

const clamp = (v) => Math.max(0, Math.min(255, Math.round(v)));
/** Soil to grass tint by a 0..1 value, with some speckle. */
const earth = (t, rnd) => {
  const n = (rnd() - 0.5) * 24;
  return [clamp(120 + 40 * t + n), clamp(105 + 50 * t + n), clamp(80 + 10 * t + n)];
};

// ---------------------------------------------------------------- stockpile

// A 40 x 40 m yard on a gentle slope; "after" adds a 4 m heap (about
// 400 m³ of fill) and a 1.5 m deep pit (45 m³ of cut).
{
  const ground = (x, y) => 0.02 * x + 0.01 * y;
  const heap = (x, y) => {
    const r = Math.hypot(x - 14, y - 22);
    return r < 8 ? 4 * (1 - (r / 8) ** 2) : 0; // a paraboloid: π r² h / 2 ≈ 402 m³
  };
  const pit = (x, y) => (Math.abs(x - 29) < 3 && Math.abs(y - 10) < 2.5 ? -1.5 : 0);
  for (const [name, after, seed] of [
    ["stockpile_before.ply", false, 11],
    ["stockpile_after.ply", true, 12],
  ]) {
    const rnd = random(seed);
    const points = [];
    for (let j = 0; j < 200; j++) {
      for (let i = 0; i < 200; i++) {
        const x = i * 0.2 + (rnd() - 0.5) * 0.1;
        const y = j * 0.2 + (rnd() - 0.5) * 0.1;
        const h = after ? heap(x, y) : 0;
        const d = after ? pit(x, y) : 0;
        const z = ground(x, y) + h + d + (rnd() - 0.5) * 0.02;
        const [r, g, b] = h > 0.05 ? [clamp(150 + rnd() * 30), clamp(120 + rnd() * 25), clamp(85 + rnd() * 20)] : earth(0.6, rnd);
        points.push([x, y, z, r, g, b]);
      }
    }
    writePly(name, points);
  }
}

// ---------------------------------------------------------------- town

// 60 x 60 m of rolling ground with two buildings and a row of trees, for
// ground extraction.
{
  const rnd = random(21);
  const terrain = (x, y) => 0.8 * Math.sin(x / 11) + 0.6 * Math.cos(y / 9);
  const buildings = [
    { x0: 10, x1: 22, y0: 30, y1: 44, h: 7, color: [175, 170, 165] },
    { x0: 34, x1: 50, y0: 8, y1: 18, h: 10, color: [150, 120, 110] },
  ];
  const trees = Array.from({ length: 7 }, (_, k) => ({ x: 30 + k * 4, y: 40 + (k % 2) * 6, crown: 2 + (k % 3) * 0.5 }));
  const inside = (b, x, y) => x >= b.x0 && x <= b.x1 && y >= b.y0 && y <= b.y1;
  const points = [];
  for (let j = 0; j < 240; j++) {
    for (let i = 0; i < 240; i++) {
      const x = i * 0.25 + (rnd() - 0.5) * 0.1;
      const y = j * 0.25 + (rnd() - 0.5) * 0.1;
      const g = terrain(x, y);
      const roof = buildings.find((b) => inside(b, x, y));
      if (roof) {
        const base = terrain((roof.x0 + roof.x1) / 2, (roof.y0 + roof.y1) / 2);
        points.push([x, y, base + roof.h + (rnd() - 0.5) * 0.03, ...roof.color.map((c) => clamp(c + (rnd() - 0.5) * 20))]);
        continue;
      }
      const tree = trees.find((t) => Math.hypot(x - t.x, y - t.y) < t.crown);
      if (tree && rnd() < 0.85) {
        const r = Math.hypot(x - tree.x, y - tree.y) / tree.crown;
        const top = g + 5 + tree.crown * Math.sqrt(1 - r * r);
        points.push([x, y, top - rnd() * 0.6, clamp(50 + rnd() * 30), clamp(110 + rnd() * 50), clamp(40 + rnd() * 25)]);
        continue;
      }
      points.push([x, y, g + (rnd() - 0.5) * 0.03, ...earth(0.8, rnd)]);
    }
  }
  // Walls, so the buildings read as boxes.
  for (const b of buildings) {
    const base = terrain((b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2);
    for (let z = 0; z < b.h; z += 0.3) {
      for (let t = 0; t <= 1; t += 0.02) {
        for (const [x, y] of [
          [b.x0 + t * (b.x1 - b.x0), b.y0],
          [b.x0 + t * (b.x1 - b.x0), b.y1],
          [b.x0, b.y0 + t * (b.y1 - b.y0)],
          [b.x1, b.y0 + t * (b.y1 - b.y0)],
        ]) {
          points.push([x, y, Math.max(terrain(x, y), base) + z, ...b.color.map((c) => clamp(c - 30 + (rnd() - 0.5) * 20))]);
        }
      }
    }
  }
  writePly("town.ply", points);
}

// ---------------------------------------------------------------- landslide

// A 30 x 30 m slope rising 12 m; "after" has a scarp that dropped up to
// 0.5 m near the top and the material piled up near the toe.
{
  const slope = (x) => 0.4 * x;
  const bump = (x, y, cx, cy, sx, sy) => Math.exp(-(((x - cx) / sx) ** 2) - ((y - cy) / sy) ** 2);
  for (const [name, after, seed] of [
    ["slope_before.ply", false, 31],
    ["slope_after.ply", true, 32],
  ]) {
    const rnd = random(seed);
    const points = [];
    for (let j = 0; j < 200; j++) {
      for (let i = 0; i < 200; i++) {
        const x = i * 0.15 + (rnd() - 0.5) * 0.08;
        const y = j * 0.15 + (rnd() - 0.5) * 0.08;
        const change = after ? -0.5 * bump(x, y, 22, 15, 3, 5) + 0.4 * bump(x, y, 8, 15, 3, 6) : 0;
        const z = slope(x) + change + (rnd() - 0.5) * 0.02;
        points.push([x, y, z, ...earth(0.3 + 0.5 * (x / 30), rnd)]);
      }
    }
    writePly(name, points);
  }
}
