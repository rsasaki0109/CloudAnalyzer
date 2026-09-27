// Scalar-field color ramps.

type Stop = [number, number, number, number]; // position, r, g, b (0-255)

export const RAMPS: Record<string, Stop[]> = {
  "Blue > Green > Yellow > Red": [
    [0, 0, 0, 255],
    [1 / 3, 0, 255, 0],
    [2 / 3, 255, 255, 0],
    [1, 255, 0, 0],
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

/** Map scalar values into RGB bytes, clamping to `[lo, hi]`. */
export function colorize(values: Float32Array, lo: number, hi: number, table: Uint8Array): Uint8Array {
  const out = new Uint8Array(values.length * 3);
  const scale = hi > lo ? 255 / (hi - lo) : 0;
  for (let i = 0; i < values.length; i++) {
    const idx = Math.min(255, Math.max(0, Math.round((values[i] - lo) * scale))) * 3;
    out[i * 3] = table[idx];
    out[i * 3 + 1] = table[idx + 1];
    out[i * 3 + 2] = table[idx + 2];
  }
  return out;
}

/** CSS linear-gradient for a colorbar. */
export function gradientCss(name: RampName, direction = "to top"): string {
  const stops = RAMPS[name].map(([p, r, g, b]) => `rgb(${r} ${g} ${b}) ${(p * 100).toFixed(1)}%`);
  return `linear-gradient(${direction}, ${stops.join(", ")})`;
}
