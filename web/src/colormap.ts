// Scalar-field color ramps.

type Stop = [number, number, number, number]; // position, r, g, b (0-255)

export const RAMPS: Record<string, Stop[]> = {
  "Blue > Green > Yellow > Red": [
    [0, 0, 0, 255],
    [1 / 3, 0, 255, 0],
    [2 / 3, 255, 255, 0],
    [1, 255, 0, 0],
  ],
  "Blue > White > Red": [
    [0, 33, 102, 172],
    [0.5, 247, 247, 247],
    [1, 178, 24, 43],
  ],
  Viridis: [
    [0, 68, 1, 84],
    [0.25, 59, 82, 139],
    [0.5, 33, 145, 140],
    [0.75, 94, 201, 98],
    [1, 253, 231, 37],
  ],
  Grey: [
    [0, 30, 30, 30],
    [1, 245, 245, 245],
  ],
};

export type RampName = keyof typeof RAMPS;

/** Sample a ramp into a 256-entry RGB lookup table. */
export function lut(name: RampName): Uint8Array {
  const stops = RAMPS[name];
  const out = new Uint8Array(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = i / 255;
    let k = 1;
    while (k < stops.length - 1 && stops[k][0] < t) k++;
    const [p0, r0, g0, b0] = stops[k - 1];
    const [p1, r1, g1, b1] = stops[k];
    const f = p1 === p0 ? 0 : (t - p0) / (p1 - p0);
    out[i * 3] = Math.round(r0 + (r1 - r0) * f);
    out[i * 3 + 1] = Math.round(g0 + (g1 - g0) * f);
    out[i * 3 + 2] = Math.round(b0 + (b1 - b0) * f);
  }
  return out;
}

/** Map scalar values into RGBA bytes (opaque), clamping to `[lo, hi]`. */
export function colorize(values: Float32Array, lo: number, hi: number, table: Uint8Array): Uint8Array {
  const out = new Uint8Array(values.length * 4);
  const scale = hi > lo ? 255 / (hi - lo) : 0;
  for (let i = 0; i < values.length; i++) {
    if (!Number.isFinite(values[i])) {
      // No value (e.g. M3C2 without enough points): neutral grey.
      out.set([128, 128, 128, 255], i * 4);
      continue;
    }
    const idx = Math.min(255, Math.max(0, Math.round((values[i] - lo) * scale))) * 3;
    out[i * 4] = table[idx];
    out[i * 4 + 1] = table[idx + 1];
    out[i * 4 + 2] = table[idx + 2];
    out[i * 4 + 3] = 255;
  }
  return out;
}

/** Interleaved rgb to opaque rgba. */
export function toRgba(rgb: Uint8Array): Uint8Array {
  const n = rgb.length / 3;
  const out = new Uint8Array(n * 4);
  for (let i = 0; i < n; i++) {
    out[i * 4] = rgb[i * 3];
    out[i * 4 + 1] = rgb[i * 3 + 1];
    out[i * 4 + 2] = rgb[i * 3 + 2];
    out[i * 4 + 3] = 255;
  }
  return out;
}

/** The `p`-quantile of `values` (by sorting a strided sample). */
export function quantile(values: Float32Array, p: number): number {
  const step = Math.max(1, Math.floor(values.length / 100_000));
  const sample: number[] = [];
  for (let i = 0; i < values.length; i += step) sample.push(values[i]);
  sample.sort((a, b) => a - b);
  return sample[Math.min(sample.length - 1, Math.max(0, Math.round(p * (sample.length - 1))))] ?? 0;
}

/** ASPRS LAS standard classes: name and display color. */
export const CLASSES: Record<number, [string, [number, number, number]]> = {
  0: ["Never classified", [160, 160, 160]],
  1: ["Unclassified", [200, 200, 200]],
  2: ["Ground", [166, 118, 58]],
  3: ["Low vegetation", [161, 214, 108]],
  4: ["Medium vegetation", [77, 175, 74]],
  5: ["High vegetation", [27, 120, 55]],
  6: ["Building", [228, 87, 76]],
  7: ["Low point (noise)", [255, 0, 255]],
  8: ["Model key point", [255, 215, 0]],
  9: ["Water", [66, 133, 244]],
  10: ["Rail", [120, 90, 160]],
  11: ["Road surface", [110, 110, 110]],
  13: ["Wire guard", [255, 160, 60]],
  14: ["Wire conductor", [255, 200, 40]],
  15: ["Transmission tower", [200, 60, 160]],
  16: ["Wire connector", [240, 140, 200]],
  17: ["Bridge deck", [140, 110, 90]],
  18: ["High noise", [255, 80, 200]],
};

export function classColor(code: number): [number, number, number] {
  if (CLASSES[code]) return CLASSES[code][1];
  // Stable pseudo-random color for user-defined classes.
  const h = (code * 137.508) % 360;
  const c = new Array(3).fill(0).map((_, i) => {
    const k = (i * 4 + h / 30) % 12;
    return Math.round(255 * (0.55 - 0.35 * Math.max(-1, Math.min(k - 3, 9 - k, 1))));
  });
  return [c[0], c[1], c[2]];
}

export function className(code: number): string {
  return CLASSES[code]?.[0] ?? `Class ${code}`;
}

/** CSS linear-gradient for a colorbar. */
export function gradientCss(name: RampName, direction = "to top"): string {
  const stops = RAMPS[name].map(([p, r, g, b]) => `rgb(${r} ${g} ${b}) ${(p * 100).toFixed(1)}%`);
  return `linear-gradient(${direction}, ${stops.join(", ")})`;
}
