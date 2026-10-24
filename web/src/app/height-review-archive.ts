/** Historical viewer deliberately independent of live cells, clouds and QA. */
import { HEIGHT_ARCHIVE_BYTES, parseHeightReviewArchive, heightArchiveScope, heightArchiveSummary, heightArchiveCsv, heightArchiveHtml } from "../height-review-archive";
import type { HeightCellReviewSession } from "../height-cell-review";
import { $, download, errorText, setStatus } from "./dom";
const input = $<HTMLInputElement>("height-archive-file");
let current: { file: File; sessions: HeightCellReviewSession[] } | null = null, generation = 0;
$("height-archive-open").onclick = () => input.click();
$("height-archive-close").onclick = () => {
  generation++; current = null; $("height-archive-list").replaceChildren(); $("height-archive-result").hidden = true;
  setStatus("Archived height-cell reviews closed; current analysis unchanged.");
};
input.onchange = async () => {
  const file = input.files?.[0]; input.value = ""; if (!file) return;
  const token = ++generation;
  try {
    if (file.size > HEIGHT_ARCHIVE_BYTES) throw new Error("Height-cell review archive exceeds the 64 MiB limit");
    const bytes = await file.text(); if (token !== generation) return;
    const sessions = parseHeightReviewArchive(JSON.parse(bytes));
    const sections = sessions.map(s => {
      const detail = document.createElement("details"), title = document.createElement("summary"), context = document.createElement("p"), list = document.createElement("ol");
      title.textContent = s.context.label; context.className = "hint small"; context.style.overflowWrap = "anywhere"; context.textContent = heightArchiveSummary(s);
      let rendered = false;
      detail.ontoggle = () => {
        if (!detail.open || rendered) return; rendered = true;
        list.append(...(s.screening.cells ?? []).map((c, i) => {
          const li = document.createElement("li"), metrics = document.createElement("p"), note = document.createElement("p");
          metrics.textContent = `Grid index ${c.index}; world XY [${c.x}, ${c.y}]; Before ${c.before}; After ${c.after}; difference ${c.difference}; ${s.decisions[i].status}; updated ${s.decisions[i].updatedAt ?? "never"}`;
          note.style.whiteSpace = "pre-wrap"; note.style.overflowWrap = "anywhere"; note.className = "height-archive-note";
          note.textContent = s.decisions[i].note || "(No comment)"; li.append(metrics, note); return li;
        }));
      };
      detail.append(title, context, list); return detail;
    });
    current = { file, sessions };
    $("height-archive-name").textContent = file.name; $("height-archive-scope").textContent = heightArchiveScope;
    $("height-archive-count").textContent = `${sessions.length} opened comparison lists in this file. Expand a list to read its cell evidence and notes; no combined unique-change total.`;
    $("height-archive-list").replaceChildren(...sections); $("height-archive-result").hidden = false;
    setStatus("Archived height-cell reviews opened read-only; current analysis unchanged.");
  } catch (error) { if (token === generation) setStatus("Could not open height-cell review archive: " + errorText(error), true); }
};
$("height-archive-json").onclick = () => { if (current) download(current.file, "archived-height-cell-reviews.json"); };
$("height-archive-csv").onclick = () => { if (current) download(new Blob([heightArchiveCsv(current.sessions)], { type: "text/csv" }), "archived-height-cell-reviews.csv"); };
$("height-archive-html").onclick = () => { if (current) download(new Blob([heightArchiveHtml(current.sessions)], { type: "text/html;charset=utf-8" }), "archived-height-cell-reviews.html"); };
