/**
 * The pose graph demo's data, made in the browser: a drive once and a bit
 * around a city block over a gentle hill, with the drifting odometry a
 * LiDAR odometry leaves (a slow turn, a slow pitch and 1 % of scale), a
 * Velodyne-like scan from every pose and an IMU's up direction for each.
 * Deterministic, so the demo and the README pictures always look the same.
 */

type Vec3 = [number, number, number];
/** Row-major 3x3 rotation and a translation. */
interface Pose {
  r: number[];
  t: Vec3;
}

/** Ground height: a gentle hill across the block. */
const ground = (x: number, y: number) => 1.5 * Math.sin(x / 25) + 0.3 * Math.cos(y / 17);
/** Metres between poses along the path. */
const STEP = 2.5;
/** The path runs round this rectangle (half sizes, metres). */
const [HALF_X, HALF_Y] = [50, 30];
const SENSOR_HEIGHT = 1.8;
const SCAN_RANGE = 35;
/** Within this range a scan keeps every point; further out, fewer with the square of the distance, like a LiDAR's. */
const DENSE_RANGE = 12;

function random(seed: number): () => number {
  let s = seed;
  return () => (s = (s * 1103515245 + 12345) % 2147483648) / 2147483648;
}

/** Points of the block with an intensity each: x, y, z, i. */
function world(): Float32Array {
  const rnd = random(11);
  const out: number[] = [];
  const add = (x: number, y: number, z: number, i: number) => out.push(x, y, z, i);
  for (let x = -75; x <= 75; x += 0.6) {
    for (let y = -55; y <= 55; y += 0.6) add(x, y, ground(x, y), 0.15 + 0.05 * rnd());
  }
  // Buildings: facades 12 m from the road on both sides, one every 14 m, of varied height.
  for (const inset of [-12, 12]) {
    const [hx, hy] = [HALF_X + inset, HALF_Y + inset];
    const sides: [Vec3, Vec3][] = [
      [[-hx, -hy, 0], [hx, -hy, 0]],
      [[hx, -hy, 0], [hx, hy, 0]],
      [[hx, hy, 0], [-hx, hy, 0]],
      [[-hx, hy, 0], [-hx, -hy, 0]],
    ];
    for (const [a, b] of sides) {
      const length = Math.hypot(b[0] - a[0], b[1] - a[1]);
      for (let s = 0; s < length; s += 14) {
        const height = 6 + 10 * rnd();
        const setback = (rnd() - 0.5) * 2;
        const gap = rnd() < 0.15;
        if (gap) continue;
        for (let d = s; d < Math.min(length, s + 13); d += 0.35) {
          const f = d / length;
          const [x, y] = [a[0] + (b[0] - a[0]) * f, a[1] + (b[1] - a[1]) * f];
          const nx = Math.sign(x) * setback * (Math.abs(b[1] - a[1]) > 0 ? 1 : 0);
          const ny = Math.sign(y) * setback * (Math.abs(b[0] - a[0]) > 0 ? 1 : 0);
          const base = ground(x, y);
          for (let z = 0; z < height; z += 0.35) add(x + nx, y + ny, base + z, 0.55 + 0.1 * Math.sin(z * 2));
        }
      }
    }
  }
  // Street trees and poles along the road, on both sides.
  for (let k = 0; k < 70; k++) {
    const along = rnd() * 2 * (2 * HALF_X + 2 * HALF_Y);
    const side = rnd() < 0.5 ? -5 : 5;
    const [px, py] = pointOnPath(along, side);
    const base = ground(px, py);
    if (rnd() < 0.6) {
      for (let z = 0; z < 3; z += 0.2) add(px, py, base + z, 0.35);
      for (let i = 0; i < 260; i++) {
        const [u, v, w] = [rnd() * 2 - 1, rnd() * 2 - 1, rnd() * 2 - 1];
        const n = Math.hypot(u, v, w) || 1;
        add(px + (2 * u) / n, py + (2 * v) / n, base + 4.5 + (1.7 * w) / n, 0.4 + 0.1 * rnd());
      }
    } else {
      for (let z = 0; z < 6; z += 0.15) add(px, py, base + z, 0.8);
    }
  }
  return Float32Array.from(out);
}

/** The point `along` metres round the path from its start, `offset` metres to its left. */
function pointOnPath(along: number, offset = 0): [number, number, number] {
  const sides = [2 * HALF_X, 2 * HALF_Y, 2 * HALF_X, 2 * HALF_Y];
  const perimeter = sides.reduce((a, b) => a + b);
  let s = ((along % perimeter) + perimeter) % perimeter;
  const corners: [number, number][] = [
    [-HALF_X, -HALF_Y],
    [HALF_X, -HALF_Y],
    [HALF_X, HALF_Y],
    [-HALF_X, HALF_Y],
  ];
  for (let k = 0; k < 4; k++) {
    if (s <= sides[k] || k === 3) {
      const [a, b] = [corners[k], corners[(k + 1) % 4]];
      const yaw = Math.atan2(b[1] - a[1], b[0] - a[0]);
      const f = s / sides[k];
      return [
        a[0] + (b[0] - a[0]) * f - Math.sin(yaw) * offset,
        a[1] + (b[1] - a[1]) * f + Math.cos(yaw) * offset,
        yaw,
      ];
    }
    s -= sides[k];
  }
  return [0, 0, 0];
}

const mul = (a: number[], b: number[]) =>
  Array.from({ length: 9 }, (_, k) => [0, 1, 2].reduce((sum, j) => sum + a[(k - (k % 3)) + j] * b[j * 3 + (k % 3)], 0));
const rotZ = (a: number) => [Math.cos(a), -Math.sin(a), 0, Math.sin(a), Math.cos(a), 0, 0, 0, 1];
const rotY = (a: number) => [Math.cos(a), 0, Math.sin(a), 0, 1, 0, -Math.sin(a), 0, Math.cos(a)];
const apply = (r: number[], v: Vec3): Vec3 => [
  r[0] * v[0] + r[1] * v[1] + r[2] * v[2],
  r[3] * v[0] + r[4] * v[1] + r[5] * v[2],
  r[6] * v[0] + r[7] * v[1] + r[8] * v[2],
];
const transpose = (r: number[]) => [r[0], r[3], r[6], r[1], r[4], r[7], r[2], r[5], r[8]];

/** The true poses: along the road, pitched with the hill, the sensor above the ground. */
function truePoses(): Pose[] {
  const perimeter = 2 * (2 * HALF_X + 2 * HALF_Y);
  const count = Math.round((1.1 * perimeter) / STEP);
  return Array.from({ length: count }, (_, k) => {
    const [x, y, yaw] = pointOnPath(k * STEP);
    const [dx, dy] = [Math.cos(yaw), Math.sin(yaw)];
    const slope = (ground(x + dx, y + dy) - ground(x - dx, y - dy)) / 2;
    return { r: mul(rotZ(yaw), rotY(-Math.atan(slope))), t: [x, y, ground(x, y) + SENSOR_HEIGHT] };
  });
}

/** Dead-reckoned odometry: each true step turned 0.1° left, pitched 0.08° up and 1 % long. */
function drifted(truth: Pose[]): Pose[] {
  const bias = mul(rotZ((0.1 * Math.PI) / 180), rotY((-0.08 * Math.PI) / 180));
  const out: Pose[] = [truth[0]];
  for (let k = 1; k < truth.length; k++) {
    const [a, b, prev] = [truth[k - 1], truth[k], out[k - 1]];
    const at = transpose(a.r);
    const stepR = mul(at, b.r);
    const stepT = apply(at, [b.t[0] - a.t[0], b.t[1] - a.t[1], b.t[2] - a.t[2]]).map((v) => v * 1.01) as Vec3;
    const t = apply(prev.r, stepT);
    out.push({ r: mul(mul(prev.r, stepR), bias), t: [prev.t[0] + t[0], prev.t[1] + t[1], prev.t[2] + t[2]] });
  }
  return out;
}

const kitti = (p: Pose) =>
  [0, 1, 2].map((row) => [p.r[row * 3], p.r[row * 3 + 1], p.r[row * 3 + 2], p.t[row]].join(" ")).join(" ");

/** The demo's files: the drifted poses (KITTI), a scan per pose (KITTI .bin) and gravity.txt. */
export function poseGraphDemoFiles(): { scans: File[]; poses: File; gravity: File } {
  const points = world();
  const n = points.length / 4;
  const truth = truePoses();
  const rnd = random(5);
  const scans = truth.map((pose, k) => {
    const seen: number[] = [];
    for (let i = 0; i < n; i++) {
      const [dx, dy] = [points[i * 4] - pose.t[0], points[i * 4 + 1] - pose.t[1]];
      const d2 = dx * dx + dy * dy;
      if (d2 < SCAN_RANGE * SCAN_RANGE && (d2 < DENSE_RANGE * DENSE_RANGE || rnd() < (DENSE_RANGE * DENSE_RANGE) / d2)) {
        seen.push(i);
      }
    }
    const body = new Float32Array(seen.length * 4);
    const rt = transpose(pose.r);
    seen.forEach((i, j) => {
      const local = apply(rt, [points[i * 4] - pose.t[0], points[i * 4 + 1] - pose.t[1], points[i * 4 + 2] - pose.t[2]]);
      body.set([...local, points[i * 4 + 3]], j * 4);
    });
    return new File([body], `${String(k).padStart(6, "0")}.bin`);
  });
  const poses = new File([`${drifted(truth).map(kitti).join("\n")}\n`], "poses.txt");
  // Up in each scan's frame: the last row of its rotation.
  const gravity = new File([truth.map((p, k) => `${k} ${p.r[6]} ${p.r[7]} ${p.r[8]}`).join("\n")], "gravity.txt");
  return { scans, poses, gravity };
}

/** The demo's true poses (KITTI lines), to check a correction against. */
export function poseGraphDemoTruth(): string[] {
  return truePoses().map(kitti);
}

