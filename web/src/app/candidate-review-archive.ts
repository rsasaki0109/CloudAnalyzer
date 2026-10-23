/** Independent historical viewer: never modifies live diagnoses or their QA. */
import { parseReviewArchive, archiveCsv, archiveHtml, archiveSummary, REVIEW_ARCHIVE_BYTES, type ReviewArchive } from "../candidate-review-archive";
import { compareReviewArchives, reviewComparisonSummary, reviewComparisonCsv, reviewComparisonHtml, type ReviewComparison } from "../candidate-review-comparison";
import { $, download, errorText, setStatus } from "./dom";
const fileInput = $<HTMLInputElement>("review-archive-file");
let current: { file: File; archive: ReviewArchive } | null = null, generation = 0;
const comparisonFile = $<HTMLInputElement>("review-compare-file");
let comparison: { file: File; result: ReviewComparison } | null = null, comparisonGeneration = 0;
function clearComparison(): void {
  comparisonGeneration++; comparison = null;
  $("review-compare-result").hidden = true; $("review-compare-list").replaceChildren();
}
$("review-archive-open").onclick = () => fileInput.click();
$("review-archive-close").onclick = () => { generation++; clearComparison(); current = null; $("review-archive-result").hidden = true; $("review-archive-list").replaceChildren(); setStatus("Archived review closed; current analysis unchanged."); };
fileInput.onchange = async () => {
  const file = fileInput.files?.[0]; fileInput.value = ""; if (!file) return;
  const token = ++generation;
  comparisonGeneration++; // A new A read supersedes pending B reads, even if invalid.
  try {
    if (file.size > REVIEW_ARCHIVE_BYTES) throw new Error("Review archive exceeds the 2 MiB limit");
    const value = await file.text();
    if (token !== generation) return;
    const archive = parseReviewArchive(JSON.parse(value));
    clearComparison();
    current = { file, archive };
    $("review-archive-name").textContent = file.name;
    $("review-archive-scope").textContent = "Historical file only; no decisions applied to current analysis or QA. Internal consistency checked; source contents and reviewer identity unverified. Reviewed means inspection completed, not confirmed map change.";
    $("review-archive-list").replaceChildren(...archive.comparisons.map(c => {
      const section = document.createElement("details"); section.open = true;
      const summary = document.createElement("summary"); summary.textContent = `${c.direction}: ${c.source.name} → ${c.reference.name}`;
      const counts = document.createElement("p");
      counts.textContent = `${c.review.counts.reviewed} reviewed, ${c.review.counts.deferred} deferred, ${c.review.counts.unreviewed} unreviewed; ${c.analysis.eligibleRegions - c.analysis.regions.length} eligible regions unlisted.`;
      const contextDetails = document.createElement("details"), contextTitle = document.createElement("summary");
      contextTitle.textContent = "Inputs and analysis conditions";
      const context = document.createElement("p"); context.className = "hint small"; context.textContent = archiveSummary(c);
      contextDetails.append(contextTitle, context);
      const list = document.createElement("ol");
      list.append(...c.review.candidates.map(r => {
        const li = document.createElement("li"), metrics = document.createElement("p"), note = document.createElement("p");
        const detail = document.createElement("details"), title = document.createElement("summary");
        title.textContent = `Region ${r.rank}: ${r.status}; ${r.points} points`;
        metrics.textContent = `Max ${r.maxDistance}; mean ${r.meanDistance}; world XYZ bounds [${r.bounds.join(", ")}]; updated ${r.updatedAt ?? "never"}`;
        detail.append(title, metrics);
        note.className = "archive-comment"; note.textContent = r.note || "(No comment)"; li.append(detail, note); return li;
      }));
      section.append(summary, counts, contextDetails, list); return section;
    }));
    $("review-archive-result").hidden = false;
    setStatus("Archived candidate reviews opened read-only; current analysis unchanged.");
  } catch (error) { if (token === generation) setStatus("Could not open review archive: " + errorText(error), true); }
};
$("review-archive-json").onclick = () => { if (current) download(current.file, "archived-candidate-reviews.json"); };
$("review-archive-csv").onclick = () => { if (current) download(new Blob([archiveCsv(current.archive)], { type: "text/csv" }), "archived-candidate-reviews.csv"); };
$("review-archive-html").onclick = () => { if (current) download(new Blob([archiveHtml(current.archive)], { type: "text/html;charset=utf-8" }), "archived-candidate-reviews.html"); };

$("review-compare-open").onclick = () => { if (current) comparisonFile.click(); };
$("review-compare-clear").onclick = () => { clearComparison(); setStatus("Archive comparison cleared; archive A and live analysis unchanged."); };
comparisonFile.onchange = async () => {
  const file = comparisonFile.files?.[0]; comparisonFile.value = "";
  const base = current, baseGeneration = generation, token = ++comparisonGeneration;
  if (!file || !base) return;
  const valid = () => token === comparisonGeneration && base === current && baseGeneration === generation;
  try {
    if (file.size > REVIEW_ARCHIVE_BYTES) throw new Error("Review archive exceeds the 2 MiB limit");
    const value = await file.text();
    if (!valid()) return;
    const archive = parseReviewArchive(JSON.parse(value));
    const result = compareReviewArchives(base.archive, archive, base.file.name, file.name);
    comparison = { file, result };
    $("review-compare-names").textContent = `A: ${base.file.name}; B: ${file.name}. A → B is file selection order, not an inferred time order.`;
    $("review-compare-scope").textContent = result.scope;
    $("review-compare-list").replaceChildren(...result.directions.map(d => {
      const section = document.createElement("details"); section.open = true;
      const heading = document.createElement("summary"); heading.textContent = reviewComparisonSummary(d);
      section.append(heading);
      for (const [label, c] of [["A", d.a], ["B", d.b]] as const) {
        const detail = document.createElement("details"), title = document.createElement("summary"), p = document.createElement("p");
        title.textContent = `${label}: inputs and analysis conditions`;
        p.className = "hint small"; p.textContent = c ? archiveSummary(c) : "Direction absent";
        detail.append(title, p); section.append(detail);
      }
      const list = document.createElement("ol");
      for (const r of d.rows) {
        const li = document.createElement("li"), detail = document.createElement("details"), title = document.createElement("summary"), metrics = document.createElement("p");
        title.textContent = `Region ${r.rank}: ${r.a.status} → ${r.b.status}; changed: ${r.changed.join(", ") || "none"}`;
        metrics.textContent = `${r.geometry.points} points; max ${r.geometry.maxDistance}; mean ${r.geometry.meanDistance}; world XYZ bounds [${r.geometry.bounds.join(", ")}].`;
        detail.append(title, metrics);
        li.append(detail);
        for (const [label, value] of [["A", r.a], ["B", r.b]] as const) {
          const p = document.createElement("p"); p.className = "archive-comment";
          p.textContent = `${label}: ${value.note || "(No comment)"}\nUpdated ${label}: ${value.updatedAt ?? "never"}`; li.append(p);
        }
        list.append(li);
      }
      section.append(list); return section;
    }));
    $("review-compare-result").hidden = false;
    setStatus("Archive review comparison ready, read-only; no decisions applied to current analysis.");
  } catch (error) { if (valid()) setStatus("Could not compare review archive: " + errorText(error), true); }
};
$("review-compare-original").onclick = () => { if (comparison) download(comparison.file, "archive-b-candidate-reviews.json"); };
$("review-compare-json").onclick = () => { if (comparison) download(new Blob([JSON.stringify(comparison.result, null, 2)], { type: "application/json" }), "candidate-review-comparison.json"); };
$("review-compare-csv").onclick = () => { if (comparison) download(new Blob([reviewComparisonCsv(comparison.result)], { type: "text/csv" }), "candidate-review-comparison.csv"); };
$("review-compare-html").onclick = () => { if (comparison) download(new Blob([reviewComparisonHtml(comparison.result)], { type: "text/html;charset=utf-8" }), "candidate-review-comparison.html"); };
