/** Bounded summaries of immutable recorded mapping actions, never executable commands. */
export const HISTORY_BYTES = 8 * 1024 * 1024;
export interface MappingHistoryStep {
  sequence: number;
  proposalDigest: string | null;
  action: string;
  status: string;
  reason: string;
  candidates: number[];
  decisions: { candidate: number; action: string; from: number; to: number; reason: string }[];
  decisionsTotal: number;
}
export interface MappingHistory { steps: MappingHistoryStep[]; unavailable: string | null }
const object = (value: unknown): Record<string, unknown> => {
  if (!value || typeof value !== 'object' || Array.isArray(value)) throw new Error('Expected an object');
  return value as Record<string, unknown>;
};
const text = (value: unknown, limit = 4000): string => {
  if (typeof value !== 'string' || !value.trim() || value.length > limit) throw new Error('Invalid action text');
  return value;
};
const id = (value: unknown): number => {
  if (!Number.isSafeInteger(value) || Number(value) < 1) throw new Error('Invalid action identifier');
  return Number(value);
};
const distance = (value: unknown): number => {
  if (typeof value !== 'number' || !Number.isFinite(value) || value < 0) throw new Error('Invalid source interval');
  return value;
};
export function parseMappingHistory(value: unknown): MappingHistory {
  try {
    const run = object(value);
    if (run.schema !== 'cloudanalyzer.mapping_run.v1') return { steps: [], unavailable: 'This package has no supported mapping-run history.' };
    if (!Array.isArray(run.history) || run.history.length > 128) throw new Error('History must contain at most 128 actions');
    let previous = 0;
    const steps = run.history.map(value => {
      const row = object(value), action = object(row.action), sequence = id(row.sequence);
      if (sequence <= previous) throw new Error('Action sequence is repeated or out of order');
      previous = sequence;
      const ids = action.candidate_ids ?? (action.candidate_id == null ? [] : [action.candidate_id]);
      if (!Array.isArray(ids) || ids.length > 256) throw new Error('Invalid candidate list');
      const raw = action.decisions ?? [];
      if (!Array.isArray(raw) || raw.length > 4096) throw new Error('Invalid interval decisions');
      // The whole archive remains available; limit detailed display per action.
      const decisions = raw.slice(0, 256).map(value => {
        const d = object(value), from = distance(d.from_m), to = distance(d.to_m);
        if (to < from || !['include', 'defer'].includes(String(d.action))) throw new Error('Invalid interval decision');
        return { candidate: id(d.candidate_id), action: String(d.action), from, to, reason: text(d.reason) };
      });
      const proposalDigest = typeof row.proposal_sha256 === "string" && /^[a-f0-9]{64}$/.test(row.proposal_sha256) ? row.proposal_sha256 : null;
      return { proposalDigest, sequence, action: text(action.type, 80), status: text(row.status, 80), reason: text(row.reason),
        candidates: [...new Set(ids.map(id))], decisions, decisionsTotal: raw.length };
    });
    return { steps, unavailable: null };
  } catch (error) {
    return { steps: [], unavailable: `Saved history cannot be displayed: ${error instanceof Error ? error.message : 'invalid data'}.` };
  }
}
