/** Open the verified point/HD pair and inspect its frozen source audits together. */
import { readMappingReview, parseSavedAudits, type SavedAudit } from "../mapping-review";
import { loadCloud, removeCloud, vectorMap, workerBusy } from "../api";
import { addEntry, renderList } from "./entries";
import { refreshColors } from "./colors";
import { entries, listChanged, pointsInvalidated, viewer } from "./state";
import { captureMapProject, openVectorMap, showSavedSourceQuality, vectorMapChanged, mapProjectReady } from "./vectormap";
import { $, errorText, setStatus } from "./dom";
import { startTask, endTask, showProgress, taskActive } from "./tasks";

const button = $<HTMLButtonElement>("mapping-review-open"), input = $<HTMLInputElement>("mapping-review-file");
const select = $<HTMLSelectElement>("mapping-review-audit");
let version = 0, selection = 0;
let current: { cloud: number; audits: SavedAudit[]; map: string; stale: boolean; sourceCount?: number } | null = null;
function invalidate(message: string): void {
  if (!current) return;
  current.stale = true;
  select.disabled = true;
  $("mapping-review-state").textContent = message + " Saved audits describe the exported pair. Reopen the package to review that pair again, or run a new source check on your edits.";
}
vectorMapChanged.add(() => { version++; selection++; invalidate("The HD map changed."); });
pointsInvalidated.add(id => { if (id === current?.cloud) { selection++; invalidate("The imported point map changed or was removed."); } });
listChanged.add(() => { if (current && !entries.has(current.cloud)) invalidate("The imported point map was removed."); });

async function showAudit(): Promise<void> {
  if (!current || current.stale || !mapProjectReady() || taskActive()) return;
  const session = current, ticket = ++selection;
  try {
    const map = await captureMapProject();
    if (session !== current || session.stale || ticket !== selection) return;
    if (map !== session.map || !entries.has(session.cloud)) return invalidate("The map pair changed.");
    const audit = session.audits[Number(select.value)];
    if (!audit) return;
    showSavedSourceQuality(audit.report, audit.label, session.cloud);
    $("mapping-review-state").textContent = (session.sourceCount ? `Display preview only. Saved audits describe the original ${session.sourceCount.toLocaleString()} points; load the full original point map for new source checks. ` : "") + "Saved audit locations are shown under Source coverage. They describe the full exported point map; displayed points follow your loading limit. Traffic rules and independent accuracy remain unverified.";
  } catch (error) { setStatus(`Could not display saved audit: ${errorText(error)}`, true); }
}
select.onchange = () => { void showAudit(); };
button.onclick = () => {
  if (!mapProjectReady() || taskActive()) return setStatus("Finish the current operation before opening generated maps.", true);
  input.click();
};
input.onchange = async () => {
  const file = input.files?.[0]; input.value = "";
  if (!file) return;
  if (!mapProjectReady() || taskActive()) return setStatus("Finish the current operation before opening generated maps.", true);
  const signal = startTask();
  button.disabled = true;
  let loaded: number | null = null, committed = false;
  try {
    const waitingSince = performance.now();
    while (workerBusy()) {
      signal.throwIfAborted();
      if (performance.now() - waitingSince > 5000) throw new Error("Finish the current worker operation before opening generated maps");
      await new Promise(resolve => setTimeout(resolve, 25));
    }
    const initialVersion = version;
    setStatus(`Verifying all files in ${file.name}…`);
    const review = await readMappingReview(file, signal, (done, total) => showProgress({ note: "Verifying generated maps", fraction: done / total }));
    const source = review.members.get(review.preview ? review.roles.preview_map : review.roles.map)!, hd = review.members.get(review.roles.hd_editable_map)!;
    const text = await hd.text();
    const audits = parseSavedAudits(JSON.parse(await review.members.get(review.roles.hd_source_audits)!.text()));
    await vectorMap("check-project", { name: hd.name, text });
    signal.throwIfAborted();
    const limit = Number($<HTMLSelectElement>("max-points").value);
    setStatus("Opening the verified point map…");
    const cloud = await loadCloud(source, limit || Number.POSITIVE_INFINITY, showProgress, signal);
    loaded = cloud.id;
    signal.throwIfAborted();
    if (initialVersion !== version) throw new Error("The HD map changed while opening; choose the package again");
    await openVectorMap(hd.name, text);
    const entry = addEntry(cloud, { kind: "file", file: source, loadMaxPoints: limit, displayPreview: !!review.preview });
    entry.mode = "solid";
    refreshColors(entry);
    committed = true;
    renderList(); viewer.fit();
    current = { cloud: cloud.id, audits, map: await captureMapProject(), stale: false, sourceCount: review.preview?.sourceCount };
    select.replaceChildren(...audits.map((audit, i) => new Option(audit.label, String(i))));
    select.disabled = false;
    $("mapping-review-result").hidden = false;
    const diagnosis = review.review.diagnosis as { extent?: { generated_length_m?: number; trajectory_length_m?: number; passes_requested_extent?: boolean } } | undefined;
    const extent = diagnosis?.extent;
    const extentText = typeof extent?.generated_length_m === "number" && typeof extent.trajectory_length_m === "number"
      ? ` Source extent ${extent.generated_length_m.toFixed(1)} / ${extent.trajectory_length_m.toFixed(1)} m; ${extent.passes_requested_extent ? "requested extent met" : "requested extent unmet"}.` : "";
    $("mapping-review-summary").textContent = `${review.verifiedFiles} files verified. ${cloud.count.toLocaleString()} loaded point records${review.preview ? ` (display preview from ${review.preview.sourceCount.toLocaleString()})` : ""}.${extentText} HD draft needs review.`;
    $("mapping-review-credit").textContent = review.attribution;
    endTask(signal);
    await showAudit();
    setStatus(`Opened generated maps: ${cloud.count.toLocaleString()} loaded points and the delivered HD draft. Saved review holds are retained.`);
  } catch (error) {
    if (loaded !== null && !committed) await removeCloud(loaded);
    setStatus(`Could not open generated maps: ${errorText(error)}`, true);
  } finally { button.disabled = false; endTask(signal); }
};
