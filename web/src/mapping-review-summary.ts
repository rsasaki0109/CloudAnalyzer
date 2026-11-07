/** Bounded display of saved diagnoses; missing measurements stay unavailable. */
export function mappingReviewSummary(review: Record<string, unknown>): string[] {
  const obj = (v: unknown): Record<string, unknown> => v && typeof v === 'object' && !Array.isArray(v) ? v as Record<string, unknown> : {};
  const diagnosis = obj(review.diagnosis), holds = obj(diagnosis.review_holds);
  const count = (v: unknown) => Number.isSafeInteger(v) && Number(v) >= 0 ? String(v) : 'unavailable';
  const flag = (v: unknown) => v === true ? 'yes' : v === false ? 'no' : 'unavailable';
  const strings = (v: unknown) => Array.isArray(v) ? v.filter((x): x is string => typeof x === 'string').slice(0, 32).map(x => x.slice(0, 256)).join(', ') || 'none recorded' : 'unavailable';
  return [
    `Requested source extent met: ${flag(holds.passes_requested_extent ?? obj(diagnosis.extent).passes_requested_extent)}.`,
    `Held traces: ${count(holds.traces_total)}; unexplained: ${count(holds.unexplained_traces_total)}.`,
    `Ground estimators disagree: ${flag(holds.ground_estimators_disagree)}.`,
    `Editable map and reopened Lanelet2 agree: ${flag(holds.editable_and_reopened_osm_agree)}.`,
    `Export warnings: ${strings(holds.export_warnings)}.`,
    `Unverified assumptions: ${strings(holds.unverified)}.`,
    'Select a saved source audit to inspect problem locations. These checks do not establish independent accuracy or road-use readiness.',
  ];
}
