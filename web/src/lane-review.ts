/** Human decisions stay valid only for the geometry and rules reviewed. */
export type ReviewStatus = "unreviewed" | "reviewed" | "needs-fix" | "deferred";
export interface LaneReview {
  lane: number; status: ReviewStatus; notes: string; signature: string;
  updated: string; staleReason?: string; previousStatus?: ReviewStatus;
}
export type LaneReviewRow = Omit<LaneReview, "signature">;

/** UTF-8 CSV for sharing saved decisions, including untouched lanes. */
export function reviewCsv(rows: readonly LaneReviewRow[]): string {
  const cell = (value: string | number | undefined): string => {
    if (typeof value === "number") return String(value);
    let text = value ?? "";
    // Keep imported notes and timestamps literal when opened in a spreadsheet.
    if (/^[\s\uFEFF]*[=+\-@]/.test(text) || /^[\t\r\n]/.test(text)) text = "'" + text;
    return `"${text.replaceAll('"', '""')}"`;
  };
  return "\uFEFFlane_id,status,notes,updated_at,previous_status,stale_reason\r\n" + rows.map(row =>
    [row.lane, row.status, row.notes, row.updated, row.previousStatus, row.staleReason].map(cell).join(",") + "\r\n",
  ).join("");
}
const statuses: ReviewStatus[] = ["unreviewed", "reviewed", "needs-fix", "deferred"];
export function parseReviews(input: unknown): LaneReview[] {
  if (!Array.isArray(input)) throw new Error("Invalid lane reviews");
  const ids = new Set<number>();
  return input.map(value => {
    if (!value || typeof value !== "object") throw new Error("Invalid lane review");
    const v = value as LaneReview;
    if (!Number.isSafeInteger(v.lane) || v.lane < 0 || ids.has(v.lane) || !statuses.includes(v.status) || typeof v.notes !== "string" || v.notes.length > 10000 || typeof v.signature !== "string" || typeof v.updated !== "string" || (v.staleReason !== undefined && typeof v.staleReason !== "string") || (v.previousStatus !== undefined && !statuses.includes(v.previousStatus))) throw new Error("Invalid lane review");
    ids.add(v.lane);
    return { lane: v.lane, status: v.status, notes: v.notes, signature: v.signature, updated: v.updated, staleReason: v.staleReason, previousStatus: v.previousStatus };
  });
}
export class LaneReviews {
  private records = new Map<number, LaneReview>();
  get(lane: number): LaneReview | undefined { return this.records.get(lane); }
  rows(lanes: Iterable<number>): LaneReviewRow[] {
    return [...new Set(lanes)].sort((a, b) => a - b).map(lane => {
      const record = this.records.get(lane);
      return { lane, status: record?.status ?? "unreviewed", notes: record?.notes ?? "", updated: record?.updated ?? "", previousStatus: record?.previousStatus, staleReason: record?.staleReason };
    });
  }
  save(lane: number, signature: string, status: ReviewStatus, notes: string): void {
    this.records.set(lane, { lane, signature, status, notes: notes.slice(0,10000), updated: new Date().toISOString() });
  }
  private invalidate(record: LaneReview, reason: string): void {
    if (record.status !== "unreviewed") record.previousStatus = record.status;
    record.status = "unreviewed"; record.staleReason = reason;
  }
  reconcile(signatures: Map<number, string>): void {
    for (const [id, record] of this.records) {
      const current = signatures.get(id);
      if (current === undefined) this.records.delete(id);
      else if (current !== record.signature) this.invalidate(record, "Lane geometry or associated rules changed; review again.");
    }
  }
  invalidateSource(): void {
    for (const record of this.records.values()) this.invalidate(record, "Review source changed; check against the current points.");
  }
  snapshot(): LaneReview[] { return structuredClone([...this.records.values()]); }
  restore(records: LaneReview[], signatures: Map<number,string>): void {
    this.records = new Map(parseReviews(records).map(r => [r.lane,r])); this.reconcile(signatures);
  }
  clear(): void { this.records.clear(); }
}
