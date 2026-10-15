import { parseProject, type Project } from "./project";
import type { ReviewStatus } from "./lane-review";

export const MAX_RECOVERY_BYTES = 64 * 1024 * 1024;
export interface ReviewDraft { lane: number; status: ReviewStatus; notes: string }
export interface Recovery {
  version: 1; token: string; savedAt: string; project: Project; draft?: ReviewDraft; snapshot?: Blob;
}
const DB = "cloudanalyzer-recovery", STORE = "projects", KEY = "latest";
let database: Promise<IDBDatabase> | undefined;
function open(): Promise<IDBDatabase> {
  if (database) return database;
  const attempt = new Promise<IDBDatabase>((resolve, reject) => {
    let blocked = false;
    const request = indexedDB.open(DB, 1);
    request.onupgradeneeded = () => request.result.createObjectStore(STORE);
    request.onerror = () => { database = undefined; reject(request.error); };
    request.onblocked = () => { blocked = true; database = undefined; reject(new Error("Close the other CloudAnalyzer tabs and retry")); };
    request.onsuccess = () => {
      const db = request.result;
      if (blocked) { db.close(); return; }
      db.onversionchange = () => { db.close(); database = undefined; };
      resolve(db);
    };
  });
  database = attempt;
  void attempt.catch(() => { if (database === attempt) database = undefined; });
  return attempt;
}
export function parseRecovery(value: unknown): Recovery {
  if (!value || typeof value !== "object") throw new Error("Invalid browser recovery copy");
  const r = value as Recovery;
  if (r.version !== 1 || typeof r.token !== "string" || !r.token || typeof r.savedAt !== "string" || !Number.isFinite(Date.parse(r.savedAt))) throw new Error("Invalid browser recovery copy");
  checkSize(r);
  if (r.draft && (!Number.isSafeInteger(r.draft.lane) || r.draft.lane < 0 || !["unreviewed","reviewed","needs-fix","deferred"].includes(r.draft.status) || typeof r.draft.notes !== "string" || r.draft.notes.length > 10000)) throw new Error("Invalid recovered review draft");
  return {...r,project:parseProject(r.project)};
}
export async function readRecovery(): Promise<Recovery | null> {
  const db = await open();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE, "readonly");
    const request = transaction.objectStore(STORE).get(KEY);
    let value: unknown;
    request.onsuccess = () => { value = request.result; };
    transaction.onabort = () => reject(transaction.error);
    transaction.onerror = () => reject(transaction.error);
    transaction.oncomplete = () => {
      try { resolve(value === undefined ? null : parseRecovery(value)); } catch (error) { reject(error); }
    };
  });
}
/** Compare-and-swap in one transaction prevents silent cross-tab overwrites. */
export async function writeRecovery(value: Recovery | null, expected: string | null): Promise<void> {
  if (value) checkSize(value);
  const db = await open();
  return new Promise((resolve, reject) => {
    const transaction = db.transaction(STORE, "readwrite"), store = transaction.objectStore(STORE);
    let conflict: Error | undefined;
    const request = store.get(KEY);
    request.onsuccess = () => {
      if ((request.result?.token ?? null) !== expected) {
        conflict = new Error("Another tab changed the browser copy. Reload to choose which work to resume; use Save project to keep this tab's work.");
        transaction.abort(); return;
      }
      try { if (value) store.put(value, KEY); else store.delete(KEY); }
      catch (error) { conflict = error instanceof Error ? error : new Error(String(error)); transaction.abort(); }
    };
    transaction.oncomplete = () => resolve();
    transaction.onabort = () => reject(conflict ?? transaction.error ?? new Error("Could not save the browser copy"));
    transaction.onerror = () => reject(transaction.error);
  });
}

function checkSize(value: Recovery): void {
  if (value.snapshot !== undefined && !(value.snapshot instanceof Blob)) throw new Error("Invalid browser point-record snapshot");
  if (new Blob([JSON.stringify(value)]).size + (value.snapshot?.size ?? 0) > MAX_RECOVERY_BYTES) throw new Error("Browser recovery exceeds the 64 MiB content limit; download a workspace snapshot or export large results separately");
}
