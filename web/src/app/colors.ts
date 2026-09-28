/** Point colors for each color mode. */

import { classColor, colorize, lut, quantile, toRgba } from "../colormap";
import type { LoadedCloud } from "../protocol";
import { type ColorMode, display, type Entry, hiddenClasses, viewer } from "./state";

function solidColors(entry: Entry): Uint8Array {
  const out = new Uint8Array(entry.cloud.count * 4);
  const [r, g, b] = entry.solid;
  for (let i = 0; i < out.length; i += 4) {
    out[i] = r;
    out[i + 1] = g;
    out[i + 2] = b;
    out[i + 3] = 255;
  }
  return out;
}

function classificationColors(classes: Uint8Array): Uint8Array {
  const table = new Uint8Array(256 * 3);
  for (let c = 0; c < 256; c++) table.set(classColor(c), c * 3);
  const out = new Uint8Array(classes.length * 4);
  for (let i = 0; i < classes.length; i++) {
    const t = classes[i] * 3;
    out[i * 4] = table[t];
    out[i * 4 + 1] = table[t + 1];
    out[i * 4 + 2] = table[t + 2];
    out[i * 4 + 3] = 255;
  }
  return out;
}

/**
 * Colors from normals: their direction as RGB, or a grey hillshade lit
 * from the north-west at 45 degrees. Points without a normal are grey.
 */
function normalColors(normals: Float32Array, shade: boolean): Uint8Array {
  const out = new Uint8Array((normals.length / 3) * 4);
  const light = [-0.5, 0.5, Math.SQRT1_2];
  for (let i = 0; i < normals.length / 3; i++) {
    const [x, y, z] = [normals[i * 3], normals[i * 3 + 1], normals[i * 3 + 2]];
    const o = i * 4;
    out[o + 3] = 255;
    if (x === 0 && y === 0 && z === 0) {
      out.fill(128, o, o + 3);
    } else if (shade) {
      const lit = Math.max(0, x * light[0] + y * light[1] + z * light[2]);
      out.fill(Math.round(35 + 220 * lit), o, o + 3);
    } else {
      out[o] = Math.round((x + 1) * 127.5);
      out[o + 1] = Math.round((y + 1) * 127.5);
      out[o + 2] = Math.round((z + 1) * 127.5);
    }
  }
  return out;
}

/** Interleaved rgba for the entry's color mode, with hidden classes at alpha 0. */
export function colorsFor(entry: Entry): Uint8Array {
  const { cloud } = entry;
  let out: Uint8Array;
  if (entry.mode === "rgb" && cloud.colors) {
    out = toRgba(cloud.colors);
  } else if (entry.mode === "intensity" && cloud.intensity) {
    // Stretch between the 2nd and 98th percentile so a few bright returns
    // do not wash everything out.
    const lo = quantile(cloud.intensity, 0.02);
    const hi = quantile(cloud.intensity, 0.98);
    out = colorize(cloud.intensity, lo, hi > lo ? hi : lo + 1, lut("Grey"));
  } else if (entry.mode === "classification" && cloud.classification) {
    out = classificationColors(cloud.classification);
  } else if ((entry.mode === "normal" || entry.mode === "shade") && cloud.normals) {
    out = normalColors(cloud.normals, entry.mode === "shade");
  } else if (entry.mode === "c2c" && entry.c2c) {
    const { lo, hi } = display.range ?? { lo: entry.c2c.stats.min, hi: entry.c2c.stats.max };
    out = colorize(entry.c2c.distances, lo, hi, lut(display.ramp));
  } else {
    out = solidColors(entry);
  }
  if (cloud.classification && hiddenClasses.size > 0) {
    const classes = cloud.classification;
    for (let i = 0; i < classes.length; i++) if (hiddenClasses.has(classes[i])) out[i * 4 + 3] = 0;
  }
  return out;
}

/** The color mode a newly loaded cloud starts with. */
export function defaultMode(cloud: LoadedCloud): ColorMode {
  if (cloud.colors) return "rgb";
  if (cloud.intensity) return "intensity";
  return "solid";
}

/** The color modes a cloud has the data for. */
export function availableModes(entry: Entry): Record<ColorMode, boolean> {
  const { cloud } = entry;
  return {
    rgb: cloud.colors !== null,
    intensity: cloud.intensity !== null,
    classification: cloud.classification !== null,
    solid: true,
    c2c: !!entry.c2c,
    normal: cloud.normals !== null,
    shade: cloud.normals !== null,
  };
}

export function refreshColors(entry: Entry): void {
  if (entry.cloud.kind === "mesh") viewer.setMeshColor(entry.cloud.id, entry.solid);
  else viewer.setColors(entry.cloud.id, colorsFor(entry));
}

/** Label for a distance result, e.g. "C2M distance → part.stl". */
export function distanceLabel(c2c: Entry["c2c"]): string {
  if (!c2c) return "Distance";
  if (c2c.kind === "volume") return `Height difference vs ${c2c.referenceName}`;
  if (c2c.kind === "m3c2") return `M3C2 distance from ${c2c.referenceName}`;
  if (c2c.kind === "raster") return `Height (raster of ${c2c.referenceName})`;
  return `${c2c.kind === "c2m" ? "C2M" : "C2C"} distance → ${c2c.referenceName}`;
}
