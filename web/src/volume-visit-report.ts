/** Standalone, read-only report of a completed live comparison; no point data. */
import type { VisitVolumeRow, IntervalVolumeRow, VisitSupportMode } from "./volume-visits.ts";
import { heightReviewScope } from "./surface-volume.ts";

export type VolumeReportInput = { constantHeight: number | null } | {
  id: number; name: string; kind: string; loadedPoints: number; filePoints: number;
  keepEvery: number; preview: boolean; copcLevels: number | null;
};
type Summary<T> = T & { net: number; uncomparedCells: number; comparisonAvailable: boolean };
export interface VolumeVisitSnapshot {
  schema: "cloudanalyzer.area_volume_visits.v1";
  baseline: VolumeReportInput;
  visits: (VolumeReportInput & { visitIndex: number })[];
  ordering: { policy: string; cloudIds: number[] };
  settings: { cell: number; height: string; fillEmpty: boolean; coordinateUnit: string; heightThreshold?: number };
  screeningScope?: string;
  sharedGrid: { min: number[]; nx: number; ny: number; cell: number; origin: string };
  support: { mode: VisitSupportMode; commonGridCells?: number; commonGridArea?: number;
    mask?: { file: string; sha256?: string; byteLength: number; encoding: string } };
  scope: string;
  rows: Summary<VisitVolumeRow>[];
  intervals?: { scope: string; rows: Summary<IntervalVolumeRow>[] };
}
const escape = (value: unknown): string => String(value).replaceAll("&", "&amp;").replaceAll("<", "&lt;")
  .replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#39;");
const cell = (value: unknown) => `<td>${escape(value)}</td>`;
function table(headers: string[], rows: string[][], id: string): string {
  return `<div class="scroll"><table id="${id}"><thead><tr>${headers.map(h => `<th scope="col">${escape(h)}</th>`).join("")}</tr></thead><tbody>${rows.map(row => `<tr>${row.map(cell).join("")}</tr>`).join("")}</tbody></table></div>`;
}

export function volumeVisitReport(snapshot: VolumeVisitSnapshot, exportedAt = new Date().toISOString()): string {
  const { settings: s, support, sharedGrid: g } = snapshot;
  const screening = (rows: (VisitVolumeRow | IntervalVolumeRow)[], id: string): string => s.heightThreshold === undefined ? "" :
    `<h2>Height screening: strict ±${escape(s.heightThreshold)} ${escape(s.coordinateUnit)}</h2><p>${escape(snapshot.screeningScope ?? heightReviewScope)}</p>` +
    table(["Comparison", "Area", "Rise cells", "Fall cells", "Within cells", `Rise area (${s.coordinateUnit}²)`, `Fall area (${s.coordinateUnit}²)`, `Rise volume (${s.coordinateUnit}³)`, `Fall volume (${s.coordinateUnit}³)`, `Screened net (${s.coordinateUnit}³)`], rows.map(r => {
      const h = r.heightReview;
      return ["visitIndex" in r ? `${r.visitIndex}. ${r.visit}` : `${r.fromVisitIndex}. ${r.fromVisit} → ${r.toVisitIndex}. ${r.toVisit}`, r.name,
        ...(h ? [h.riseCells, h.fallCells, h.withinCells].map(String) : ["Unavailable", "Unavailable", "Unavailable"]),
        ...(h && r.matchedCells ? [h.riseArea, h.fallArea, h.riseVolume, h.fallVolume, h.riseVolume - h.fallVolume].map(String) : Array<string>(5).fill("Unavailable"))];
    }), id);
  const inputRows = [snapshot.baseline, ...snapshot.visits].map((input, i) => [
    i ? `Visit ${i}` : "Baseline", ...("constantHeight" in input
      ? [`Constant height ${input.constantHeight}`, "—", "Plane", "—", "—", "—", "—", "—"]
      : [input.name, String(input.id), input.kind, String(input.loadedPoints), String(input.filePoints), String(input.keepEvery), String(input.preview), input.copcLevels === null ? "n/a" : String(input.copcLevels)]),
  ]);
  const quantities = (r: Summary<VisitVolumeRow | IntervalVolumeRow>): string[] => [r.name, `[${r.bounds.join(", ")}]`,
    ...(r.comparisonAvailable ? [r.added, r.removed, r.net, r.addedArea, r.removedArea].map(String) : Array<string>(5).fill("Unavailable")),
    String(r.matchedCells), String(r.uncomparedCells), String(r.outsideRegionCells)];
  const headers = ["Area", "World XY bounds", `Fill (${s.coordinateUnit}³)`, `Cut (${s.coordinateUnit}³)`, `Net (${s.coordinateUnit}³)`,
    `Fill area (${s.coordinateUnit}²)`, `Cut area (${s.coordinateUnit}²)`, "Compared cells", "Uncompared cells", "Outside cells"];
  return `<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>2.5D survey volume report</title><style>
body{font:16px/1.5 system-ui,sans-serif;color:#17202a;background:white;max-width:1400px;margin:2rem auto;padding:0 1rem}
h1,h2{line-height:1.25}p,li,td,pre{overflow-wrap:anywhere}table{border-collapse:collapse;width:100%;font-size:14px;margin:1rem 0}th,td{border:1px solid #aab3bb;padding:.5rem;text-align:left;vertical-align:top}th{background:#eef2f5}.scroll{overflow-x:auto}pre{white-space:pre-wrap;font-size:12px}.scope{border-left:4px solid #637a8b;padding:.5rem 1rem;background:#f3f6f8}
@media print{body{max-width:none;margin:0;font-size:11px}table{font-size:9px}th,td{padding:.25rem}.scroll{overflow:visible}thead{display:table-header-group}tr{break-inside:avoid}details{display:none}}
</style></head><body><h1>2.5D survey volume report</h1>
<p>CloudAnalyzer · Exported <time>${escape(exportedAt)}</time> (export time, not acquisition time).</p>
<p class="scope">Loaded-data surface-height estimates. No automatic registration, unit conversion, closed-solid reconstruction, uncertainty calibration or independent field-accuracy guarantee. This report does not certify quantities for payment. Overlapping areas and gross interval volumes must not be summed as unique earthwork. No combined total is calculated.</p>
<h2>Inputs and explicit visit order</h2><p>${escape(snapshot.ordering.policy)}</p>
${table(["Role", "Name / height", "Runtime ID", "Kind", "Loaded points", "File points", "Keep every", "Display preview", "COPC levels"], inputRows, "inputs")}
<p>Counts and preview/thinning flags describe loaded inputs. Display LOD does not establish full-density processing. Runtime IDs and names are not source-content fingerprints. A constant baseline is a plane, not an observed survey footprint.</p>
<h2>Calculation settings and support</h2>
<p>Cell size: ${escape(s.cell)} ${escape(s.coordinateUnit)}; height: ${escape(s.height)}; interpolation: ${s.fillEmpty ? "enabled (estimated heights may be included)" : "disabled"}; coordinate-unit label: ${escape(s.coordinateUnit)} (no conversion).</p>
<p>Shared union grid: min [${escape(g.min.join(", "))}], ${escape(g.nx)} × ${escape(g.ny)} cells. ${escape(g.origin)}. Whole square cells selected by closed center bounds; no fractional clipping. Counts cover only the loaded union grid; they do not establish the surveyed percentage of the full rectangle. Missing heights are excluded, never zero change.</p>
<p>Support mode: <strong>${escape(support.mode)}</strong>. ${support.mode === "common_observed"
    ? `The same raw observed cells are used across the baseline and every selected visit. Full-grid common cells: ${escape(support.commonGridCells)}; area: ${escape(support.commonGridArea)} ${escape(s.coordinateUnit)}². Uncompared includes cells missing in any selected input.`
    : "Each baseline/visit pair may have different support, even if counts match. A shared grid does not imply a common observed footprint. Do not subtract totals to infer between-visit earthwork."}</p>
${support.mask ? `<p>Companion mask: ${escape(support.mask.file)}; ${escape(support.mask.byteLength)} bytes; SHA-256: ${escape(support.mask.sha256)}. ${escape(support.mask.encoding)}. The mask is not embedded: retain the separately saved binary with the JSON for exact cell provenance.</p>` : ""}
<h2>Relative to the fixed baseline</h2><p>${escape(snapshot.scope)}</p>
${table(["Visit", ...headers], snapshot.rows.map(r => [`${r.visitIndex}. ${r.visit}`, ...quantities(r)]), "baseline-volumes")}
${screening(snapshot.rows, "baseline-screening")}
<h2>Direct adjacent-visit intervals</h2>${snapshot.intervals
    ? `<p>${escape(snapshot.intervals.scope)}</p>${table(["From → to", ...headers], snapshot.intervals.rows.map(r => [`${r.fromVisitIndex}. ${r.fromVisit} → ${r.toVisitIndex}. ${r.toVisit}`, ...quantities(r)]), "interval-volumes")}${screening(snapshot.intervals.rows, "interval-screening")}`
    : "<p>Not computed. Baseline-relative gross fill/cut differences are not direct interval fill/cut.</p>"}
<p>Unavailable means no compared cells, not zero change. Fill and cut are nonnegative height-difference magnitudes; net = fill − cut. Visit indices express the user's chosen order and do not infer dates or elapsed time.</p>
<details><summary>Complete comparison JSON snapshot</summary><p>Same comparison fields as Save visit summaries JSON; no point coordinates or binary mask included.</p><pre id="comparison-json">${escape(JSON.stringify(snapshot, null, 2))}</pre></details>
</body></html>`;
}
