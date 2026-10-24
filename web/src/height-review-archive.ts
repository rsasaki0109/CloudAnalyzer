/** Bounded, read-only historical cell reviews; consistency is not authenticity. */
import { editDecision, emptyDecision, reviewCounts } from "./candidate-review.ts";
import { heightCellReviewScope, heightCellReviewSnapshot, heightCellReviewsCsv, type HeightCellReviewSession } from "./height-cell-review.ts";
import { validateExpectedOverlap } from "./distance-regions.ts";
import { VOLUME_GRID_LIMIT } from "./surface-volume.ts";
export const HEIGHT_ARCHIVE_BYTES = 64 * 1024 * 1024;
// 32 areas × (8 baseline visits + 7 direct intervals).
export const HEIGHT_ARCHIVE_COMPARISONS = 480;
export const heightArchiveScope = "Historical file only; no decisions applied to current clouds, analysis, projects or QA. Internal consistency checked; original point files, mask contents, reviewer identity and field accuracy are unverified. Opened time is first inspection, not acquisition or analysis time. Only opened comparison lists and their largest 50 exceedances are recorded. " + heightCellReviewScope;
function fail(): never { throw new Error("Invalid or inconsistent height-cell review archive"); }
function object(value: unknown): Record<string, unknown> {
  if (!value || typeof value !== "object" || Array.isArray(value)) fail();
  return value as Record<string, unknown>;
}
function finite(value: unknown): number { if (typeof value !== "number" || !Number.isFinite(value)) fail(); return value; }
function natural(value: unknown, max = Number.MAX_SAFE_INTEGER): number {
  const n = finite(value); if (!Number.isSafeInteger(n) || n < 0 || n > max) fail(); return n;
}
function positive(value: unknown): number { const n = finite(value); if (n <= 0) fail(); return n; }
function nonnegative(value: unknown): number { const n = finite(value); if (n < 0) fail(); return n; }
function text(value: unknown, limit = 2048): string { if (typeof value !== "string" || value.length > limit) fail(); return value; }
function bool(value: unknown): boolean { if (typeof value !== "boolean") fail(); return value; }
function region(value: unknown): readonly [number, number, number, number] {
  if (!Array.isArray(value) || value.length !== 4) fail();
  return validateExpectedOverlap([finite(value[0]), finite(value[1]), finite(value[2]), finite(value[3])])!;
}
function input(value: unknown): Record<string, unknown> {
  const v = object(value);
  if ("constantHeight" in v) return { constantHeight: finite(v.constantHeight) };
  const id = natural(v.id), name = text(v.name), loadedPoints = natural(v.loadedPoints), filePoints = natural(v.filePoints), keepEvery = positive(v.keepEvery);
  if (!["cloud", "mesh"].includes(v.kind as string) || !Number.isSafeInteger(keepEvery) || filePoints < loadedPoints) fail();
  return { id, name, kind: v.kind, loadedPoints, filePoints, keepEvery, preview: bool(v.preview),
    ...(v.copcLevels === undefined ? {} : { copcLevels: v.copcLevels === null ? null : natural(v.copcLevels) }) };
}
const close = (a: number, b: number) => a === b || Math.abs(a - b) <= 1e-9 * Math.max(Math.abs(a), Math.abs(b));
export function parseHeightReviewArchive(value: unknown): HeightCellReviewSession[] {
  try {
    const root = object(value);
    if (root.schema !== "cloudanalyzer.height_cell_reviews.v1" || !Array.isArray(root.comparisons) || root.comparisons.length > HEIGHT_ARCHIVE_COMPARISONS) fail();
    return root.comparisons.map(value => {
      const c = object(value), s = object(c.settings), h = object(c.screening), support = object(c.support);
      const openedAt = text(c.openedAt, 40); editDecision("unreviewed", "", openedAt);
      const before = input(c.before), after = input(c.after), cell = positive(s.cell), area = cell * cell;
      const threshold = nonnegative(h.threshold), riseCells = natural(h.riseCells, VOLUME_GRID_LIMIT), fallCells = natural(h.fallCells, VOLUME_GRID_LIMIT), withinCells = natural(h.withinCells, VOLUME_GRID_LIMIT);
      const exceedances = riseCells + fallCells, riseArea = nonnegative(h.riseArea), fallArea = nonnegative(h.fallArea), riseVolume = nonnegative(h.riseVolume), fallVolume = nonnegative(h.fallVolume);
      if (!Number.isFinite(area) || exceedances + withinCells > VOLUME_GRID_LIMIT || s.heightThreshold !== threshold ||
        riseArea !== riseCells * area || fallArea !== fallCells * area || h.cellLimit !== 50 ||
        !["mean", "min", "max"].includes(s.height as string) || !["m", "ft", "unit"].includes(s.coordinateUnit as string) ||
        !Array.isArray(c.cells) || c.cells.length !== Math.min(exceedances, 50) || h.unlistedCells !== exceedances - c.cells.length) fail();
      const settings = { cell, height: s.height as string, coordinateUnit: s.coordinateUnit as string, fillEmpty: bool(s.fillEmpty), heightThreshold: threshold,
        ...(s.region === undefined ? {} : { region: region(s.region) }),
        ...(s.areaName === undefined ? {} : { areaName: text(s.areaName, 1024) }) };
      if (support.mode !== "independent" && support.mode !== "common_observed") fail();
      let normalizedSupport: HeightCellReviewSession["context"]["support"] = { mode: support.mode };
      if (support.mode === "common_observed") {
        const sha256 = text(support.sha256, 64), commonGridCells = natural(support.commonGridCells, VOLUME_GRID_LIMIT);
        if (!/^[0-9a-f]{64}$/.test(sha256) || commonGridCells < exceedances + withinCells || settings.fillEmpty || before.kind !== "cloud" || after.kind !== "cloud") fail();
        normalizedSupport = { mode: support.mode, sha256, commonGridCells };
      } else if (support.sha256 !== undefined || support.commonGridCells !== undefined) fail();
      const seen = new Set<number>();
      const cells = c.cells.map((value, i) => {
        const r = object(value), index = natural(r.index, VOLUME_GRID_LIMIT - 1), x = finite(r.x), y = finite(r.y), b = finite(r.before), a = finite(r.after), difference = finite(r.difference);
        if (r.listedRank !== i + 1 || seen.has(index) || difference !== a - b || Math.abs(difference) <= threshold ||
          settings.region && (x < settings.region[0] || y < settings.region[1] || x > settings.region[2] || y > settings.region[3])) fail();
        seen.add(index); return { index, x, y, before: b, after: a, difference };
      });
      for (let i = 1; i < cells.length; i++) if (Math.abs(cells[i - 1].difference) < Math.abs(cells[i].difference) ||
        Math.abs(cells[i - 1].difference) === Math.abs(cells[i].difference) && cells[i - 1].index >= cells[i].index) fail();
      const rise = cells.filter(c => c.difference > 0), fall = cells.filter(c => c.difference < 0);
      const listedRise = rise.reduce((v, c) => v + c.difference * area, 0), listedFall = fall.reduce((v, c) => v - c.difference * area, 0);
      if (rise.length > riseCells || fall.length > fallCells || !Number.isFinite(listedRise) || !Number.isFinite(listedFall) ||
        !close(riseVolume, listedRise) && (rise.length === riseCells || riseVolume < listedRise) ||
        !close(fallVolume, listedFall) && (fall.length === fallCells || fallVolume < listedFall)) fail();
      const decisions = c.cells.map(value => {
        const r = object(value);
        if (r.updatedAt === null) { if (r.status !== "unreviewed" || r.note !== "") fail(); return emptyDecision(); }
        return editDecision(text(r.status, 20), text(r.note, 2000), text(r.updatedAt, 40));
      });
      const declared = object(c.counts);
      if (Object.entries(reviewCounts(decisions)).some(([key, value]) => declared[key] !== value)) fail();
      return { openedAt, context: { label: text(c.label, 4096), before, after, settings, support: normalizedSupport },
        screening: { threshold, riseCells, fallCells, withinCells, riseArea, fallArea, riseVolume, fallVolume, cellLimit: 50, unlistedCells: exceedances - cells.length, cells }, decisions };
    });
  } catch { fail(); }
}
export function heightArchiveCsv(sessions: readonly HeightCellReviewSession[]): string { return heightCellReviewsCsv(sessions); }
export function heightArchiveSummary(s: HeightCellReviewSession): string {
  const c = s.context, h = s.screening, counts = reviewCounts(s.decisions);
  return `${c.label}; opened ${s.openedAt}. ${counts.reviewed} reviewed, ${counts.deferred} deferred, ${counts.unreviewed} unreviewed; ${h.unlistedCells} exceedance cells unlisted. Strict threshold ${h.threshold} ${c.settings.coordinateUnit}; cell ${c.settings.cell}, ${c.settings.height}, interpolation ${c.settings.fillEmpty}. Parent bounds ${JSON.stringify(c.settings.region ?? null)}; support ${JSON.stringify(c.support)}. Before ${JSON.stringify(c.before)}; After ${JSON.stringify(c.after)}. Full screening summary ${JSON.stringify({ ...h, cells: undefined })}.`;
}
export function heightArchiveHtml(sessions: readonly HeightCellReviewSession[]): string {
  const escape = (s: string) => s.replaceAll("&", "&amp;").replaceAll("<", "&lt;").replaceAll(">", "&gt;").replaceAll('"', "&quot;").replaceAll("'", "&#39;");
  return `<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Archived height-cell reviews</title><style>body{font:16px/1.5 system-ui;max-width:1100px;margin:2rem auto;padding:0 1rem}p,pre{white-space:pre-wrap;overflow-wrap:anywhere}section,li{border-top:1px solid #aaa;padding:1rem 0}h2{font-size:1.2rem}</style><h1>Archived height-cell reviews</h1><p>${escape(heightArchiveScope)}</p><p>${sessions.length} opened comparison lists; no combined unique-change total.</p>` + sessions.map(s => `<section><h2>${escape(s.context.label)}</h2><p>${escape(heightArchiveSummary(s))}</p><ol>${(s.screening.cells ?? []).map((c, i) => `<li><p>${escape(`Grid index ${c.index}; world XY [${c.x}, ${c.y}]; Before ${c.before}; After ${c.after}; difference ${c.difference}; ${s.decisions[i].status}; updated ${s.decisions[i].updatedAt ?? "never"}`)}</p><p>${escape(s.decisions[i].note)}</p></li>`).join("")}</ol></section>`).join("") + `<details><summary>Complete normalized JSON snapshot</summary><pre>${escape(JSON.stringify(heightCellReviewSnapshot(sessions), null, 2))}</pre></details></html>`;
}
