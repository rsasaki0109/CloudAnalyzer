/** Independent historical viewer: never modifies live diagnoses or their QA. */
import { parseReviewArchive, archiveCsv, archiveHtml, archiveSummary, REVIEW_ARCHIVE_BYTES, type ReviewArchive } from "../candidate-review-archive";
import { $, download, errorText, setStatus } from "./dom";
const fileInput = $<HTMLInputElement>("review-archive-file");
let current: { file: File; archive: ReviewArchive } | null = null, generation = 0;
$("review-archive-open").onclick = () => fileInput.click();
$("review-archive-close").onclick = () => { generation++; current = null; $("review-archive-result").hidden = true; $("review-archive-list").replaceChildren(); setStatus("Archived review closed; current analysis unchanged."); };
fileInput.onchange = async () => {
  const file = fileInput.files?.[0]; fileInput.value = ""; if (!file) return;
  const token = ++generation;
  try {
    if (file.size > REVIEW_ARCHIVE_BYTES) throw new Error("Review archive exceeds the 2 MiB limit");
    const value = await file.text();
    if (token !== generation) return;
    const archive = parseReviewArchive(JSON.parse(value));
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
