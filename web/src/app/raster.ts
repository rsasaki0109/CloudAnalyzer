/** Rasterize panel: a cloud as a height grid (DEM / DSM), saved as GeoTIFF or PNG. */

import { rasterGeotiff, rasterizeCloud } from "../api";
import { className, colorize, lut } from "../colormap";
import type { RasterGrid, RasterOutput } from "../protocol";
import { refreshColors } from "./colors";
import { finiteStats } from "./distance";
import { $, download, errorText, roundUp, setStatus } from "./dom";
import { addEntry, renderList } from "./entries";
import { record } from "./history";
import { addReportSection } from "./report";
import { clouds, display, distanceChanged, type Entry, entries, listChanged } from "./state";

const cloudSelect = $<HTMLSelectElement>("raster-cloud");
const classSelect = $<HTMLSelectElement>("raster-class");
const heightSelect = $<HTMLSelectElement>("raster-height");
const cellInput = $<HTMLInputElement>("raster-cell");
const runButton = $<HTMLButtonElement>("raster-run");
const tiffButton = $<HTMLButtonElement>("raster-tiff");
const pngButton = $<HTMLButtonElement>("raster-png");
const ALL = "all";

/** The latest raster, kept for saving; `id` is its cell cloud. */
let last: { grid: RasterGrid; name: string; id: number } | null = null;
let lastStats: {
  source: string;
  nx: number;
  ny: number;
  cell: number;
  populated: number;
  filled: number;
  heights: { min: number; max: number; mean: number };
} | null = null;

addReportSection("raster", () =>
  lastStats
    ? {
        title: `Raster of ${lastStats.source}`,
        metrics: {
          cells: { label: "Cells", value: lastStats.nx * lastStats.ny },
          coverage: { label: "Cells with points", value: (100 * lastStats.populated) / (lastStats.nx * lastStats.ny), unit: "%" },
          filled: { label: "Cells filled", value: lastStats.filled },
          min: { label: "Lowest cell", value: lastStats.heights.min },
          max: { label: "Highest cell", value: lastStats.heights.max },
          mean: { label: "Mean height", value: lastStats.heights.mean },
        },
        notes: [`${lastStats.nx} × ${lastStats.ny} cells of ${lastStats.cell}`],
      }
    : null,
);

function renderSelects(): void {
  const ids = clouds().map((e) => String(e.cloud.id));
  const current = ids.includes(cloudSelect.value) ? cloudSelect.value : (ids.at(-1) ?? "");
  cloudSelect.replaceChildren(...ids.map((id) => new Option(entries.get(Number(id))!.cloud.name, id)));
  cloudSelect.value = current;
  updateForm();
}
listChanged.add(renderSelects);

function updateForm(): void {
  const entry = entries.get(Number(cloudSelect.value));
  runButton.disabled = !entry;
  $("raster-percentile-row").hidden = heightSelect.value !== "percentile";
  if (!entry || cellInput.dataset.for === String(entry.cloud.id)) return;
  cellInput.dataset.for = String(entry.cloud.id);
  // Default cell: about 1/200 of the larger horizontal extent, as for volumes.
  const b = entry.cloud.bounds;
  cellInput.value = String(roundUp(Math.max(b[3] - b[0], b[4] - b[1]) / 200 || 1));
  // Offer the class codes the cloud has, e.g. ground only for a DTM.
  const classes = entry.cloud.classification;
  const codes = classes ? [...new Set(classes)].sort((a, b) => a - b) : [];
  classSelect.replaceChildren(
    new Option("All points", ALL),
    ...codes.map((c) => new Option(`${className(c)} (${c})`, String(c))),
  );
  $("raster-class-row").hidden = codes.length === 0;
}
cloudSelect.onchange = heightSelect.onchange = updateForm;

/** Add a raster's cells to the list, colored by height in the ramp, and make it the one to save. */
export function showRaster(source: Entry, out: RasterOutput): Entry {
  const entry = addEntry(out.cells);
  // Color the cells by height in the current ramp.
  entry.c2c = {
    kind: "raster",
    signed: false,
    distances: out.cellHeights,
    stats: finiteStats(out.cellHeights),
    millis: out.millis,
    workers: 1,
    referenceName: source.cloud.name,
  };
  entry.mode = "c2c";
  display.activeC2c = entry.cloud.id;
  display.range = null;
  refreshColors(entry);
  record({ label: "the raster", added: [entry], hide: [source] });
  renderList();
  distanceChanged.emit();
  lastStats = { source: source.cloud.name, nx: out.nx, ny: out.ny, cell: out.cell, populated: out.populatedCells, filled: out.cells.count - out.populatedCells, heights: entry.c2c.stats };
  const { nx, ny, minX, minY, heights } = out;
  last = { grid: { nx, ny, minX, minY, cell: out.cell, heights }, name: out.cells.name, id: entry.cloud.id };
  tiffButton.disabled = pngButton.disabled = false;
  const total = out.nx * out.ny;
  const filled = out.cells.count - out.populatedCells;
  setStatus(
    `Raster: ${out.nx.toLocaleString()} × ${out.ny.toLocaleString()} cells, ` +
      `${out.populatedCells.toLocaleString()} with points (${((100 * out.populatedCells) / total).toFixed(1)} %)` +
      (filled > 0 ? `, ${filled.toLocaleString()} filled` : "") +
      ` in ${Math.round(out.millis)} ms`,
  );
  return entry;
}

runButton.onclick = async () => {
  const source = entries.get(Number(cloudSelect.value));
  const cell = Number(cellInput.value);
  if (!source) return;
  if (!(cell > 0)) {
    setStatus("Enter a positive cell size", true);
    return;
  }
  runButton.disabled = true;
  setStatus(`Rasterizing ${source.cloud.name}…`);
  try {
    const out = await rasterizeCloud({
      id: source.cloud.id,
      cell,
      height: heightSelect.value as "mean" | "min" | "max" | "percentile",
      percentile: Number($<HTMLInputElement>("raster-percentile").value),
      fillEmpty: $<HTMLInputElement>("raster-fill").checked,
      class: classSelect.value === ALL || $("raster-class-row").hidden ? null : Number(classSelect.value),
    });
    showRaster(source, out);
  } catch (err) {
    setStatus(`Rasterize failed: ${errorText(err)}`, true);
  } finally {
    updateForm();
  }
};

tiffButton.onclick = async () => {
  if (!last) return;
  const filename = `${last.name}.tif`;
  try {
    const bytes = await rasterGeotiff(last.grid);
    download(bytes, filename);
    setStatus(`Saved ${filename} (${(bytes.byteLength / 1e6).toFixed(1)} MB)`);
  } catch (err) {
    setStatus(`GeoTIFF failed: ${errorText(err)}`, true);
  }
};

/**
 * Hillshade factor per cell (0-1), lit from the north-west at 45 degrees as
 * the "shade" color mode; slopes from central differences, falling back to
 * one-sided ones at edges and empty neighbours.
 */
function hillshade({ nx, ny, cell, heights }: RasterGrid): Float32Array {
  const out = new Float32Array(nx * ny);
  const light = [-0.5, 0.5, Math.SQRT1_2];
  const at = (i: number, j: number, fallback: number) => {
    const v = i >= 0 && j >= 0 && i < nx && j < ny ? heights[j * nx + i] : Number.NaN;
    return Number.isNaN(v) ? fallback : v;
  };
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const z = heights[j * nx + i];
      if (Number.isNaN(z)) continue;
      const dx = (at(i + 1, j, z) - at(i - 1, j, z)) / (2 * cell);
      const dy = (at(i, j + 1, z) - at(i, j - 1, z)) / (2 * cell);
      const lit = (-dx * light[0] - dy * light[1] + light[2]) / Math.hypot(dx, dy, 1);
      out[j * nx + i] = Math.max(0, lit);
    }
  }
  return out;
}

/** The raster as a PNG in the current ramp (north up), empty cells transparent. */
function rasterImage(grid: RasterGrid, shade: boolean): Promise<Blob> {
  const { nx, ny, heights } = grid;
  const stats = finiteStats(heights);
  // The colorbar's range if it is showing this raster, else the full range.
  const { lo, hi } =
    display.activeC2c === last?.id && display.range ? display.range : { lo: stats.min, hi: stats.max };
  const colors = colorize(heights, lo, hi, lut(display.ramp));
  const light = shade ? hillshade(grid) : null;
  const canvas = document.createElement("canvas");
  [canvas.width, canvas.height] = [nx, ny];
  const ctx = canvas.getContext("2d");
  if (!ctx) return Promise.reject(new Error(`the raster is too large for an image (${nx} × ${ny})`));
  const image = ctx.createImageData(nx, ny);
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const k = j * nx + i;
      const o = ((ny - 1 - j) * nx + i) * 4; // image rows run from the top (highest y)
      if (Number.isNaN(heights[k])) continue; // stays transparent
      // Keep some color in the shadows.
      const f = light ? 0.35 + 0.65 * light[k] : 1;
      for (let c = 0; c < 3; c++) image.data[o + c] = colors[k * 4 + c] * f;
      image.data[o + 3] = 255;
    }
  }
  ctx.putImageData(image, 0, 0);
  return new Promise((resolve, reject) =>
    canvas.toBlob((blob) => (blob ? resolve(blob) : reject(new Error("could not encode the image"))), "image/png"),
  );
}

pngButton.onclick = async () => {
  if (!last) return;
  const filename = `${last.name}.png`;
  try {
    const blob = await rasterImage(last.grid, $<HTMLInputElement>("raster-hillshade").checked);
    download(blob, filename);
    setStatus(`Saved ${filename} (${last.grid.nx} × ${last.grid.ny} px)`);
  } catch (err) {
    setStatus(`PNG failed: ${errorText(err)}`, true);
  }
};
