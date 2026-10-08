/** Human decisions stay valid only for the geometry and rules reviewed. */
export type ReviewStatus = "unreviewed" | "reviewed" | "needs-fix" | "deferred";
export interface LaneReview {
  lane: number; status: ReviewStatus; notes: string; signature: string;
  updated: string; staleReason?: string; previousStatus?: ReviewStatus;
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
