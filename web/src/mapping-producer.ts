/** Summaries of hash-verified archive evidence; never runtime attestations. */
export const PRODUCER_BYTES = 4 * 1024 * 1024;
type Identity = {sha256: string; bytes: number};
const object = (v: unknown): Record<string, unknown> => {
  if (!v || typeof v !== 'object' || Array.isArray(v)) throw new Error('Invalid evidence object');
  return v as Record<string, unknown>;
};
const text = (v: unknown): string => {
  if (typeof v !== 'string' || !v.trim() || v.length > 2048) throw new Error('Invalid producer text');
  return v;
};
const matches = (v: unknown, expected: Identity | undefined): boolean => {
  const actual = object(v);
  return !!expected && actual.sha256 === expected.sha256 && actual.bytes === expected.bytes;
};
function configurationTopics(run: Record<string, unknown>, points: string, imu: string): string[] {
  if (run.config_topics === undefined || run.config_topics === null)
    return ['Configuration topic match: not recorded.'];
  try {
    const check = object(run.config_topics), inventory = object(run.inputs).configuration;
    if (check.schema !== 'cloudanalyzer.glim_config_topics.v1' || check.status !== 'matched' ||
        check.points_topic !== points || check.imu_topic !== imu ||
        !Array.isArray(check.files) || check.files.length !== 2 ||
        !Array.isArray(inventory) || !inventory.length || inventory.length > 10000)
      throw new Error('Invalid configuration check');
    // Match paths as opaque recorded strings; never access original local files.
    for (const value of check.files) {
      const file = object(value), path = text(file.path);
      if (typeof file.sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(file.sha256) ||
          !Number.isSafeInteger(file.bytes) || Number(file.bytes) < 1 || Number(file.bytes) > 1024 * 1024)
        throw new Error('Invalid configuration identity');
      const entries = inventory.map(object).filter(entry => entry.path === path);
      if (entries.length !== 1 || !matches(entries[0], file as Identity))
        throw new Error('Configuration identity mismatch');
    }
    return ['Configuration topics: recorded match with inspected point/IMU topics; both configuration file identities match the run inventory. Calibration, scaling, offsets and module compatibility remain unverified.'];
  } catch {
    return ['Configuration topic evidence is unavailable or inconsistent; no configuration match is shown.'];
  }
}
function inputPreflight(run: Record<string, unknown>): string[] {
  const status = run.input_preflight_status, value = run.input_preflight;
  if (status === undefined && value === undefined) return ['Input preflight: not recorded in this execution report.'];
  if (status === 'not_requested' && value === null) return ['Input preflight: not performed (not requested).'];
  try {
    const checked = object(value), inventory = object(run.inputs).bag;
    if (status !== 'basic_checks_passed' || checked.schema !== 'cloudanalyzer.glim_preflight.v1' ||
        checked.status !== status || !Array.isArray(inventory) || inventory.length !== 1) throw new Error('Unsupported preflight');
    const bag = object(inventory[0]);
    if (typeof bag.sha256 !== 'string' || !/^[0-9a-f]{64}$/.test(bag.sha256) ||
        !Number.isSafeInteger(bag.bytes) || Number(bag.bytes) < 1 ||
        !matches(checked.input, bag as Identity)) throw new Error('Different inspected recording');
    const points = text(checked.points_topic), imu = text(checked.imu_topic), gaps = object(checked.imu_gaps);
    if (points === imu || !Array.isArray(checked.issues) || checked.issues.length ||
        !Number.isSafeInteger(gaps.limit_ns) || Number(gaps.limit_ns) < 1 || Number(gaps.limit_ns) > 3_600_000_000_000 ||
        gaps.count !== 0 || !Number.isSafeInteger(gaps.max_observed_ns) || Number(gaps.max_observed_ns) <= 0 ||
        Number(gaps.max_observed_ns) > Number(gaps.limit_ns)) throw new Error('Inconsistent passed checks');
    const warnings = checked.warnings ?? [];
    if (!Array.isArray(warnings) || warnings.length > 32 || warnings.some(v => typeof v !== 'string' || v.length > 2048)) throw new Error('Invalid warnings');
    return [
      `Input preflight: recorded basic checks passed; inspected recording hash matches the run inventory.`,
      `Inspected topics: ${points} / ${imu}. IMU gap limit: ${Number(gaps.limit_ns) / 1e6} ms; maximum observed: ${Number(gaps.max_observed_ns) / 1e6} ms.`,
      `${warnings.length} recorded preflight warnings. Diagnostic topics are not proof of GLIM configuration, calibration or accuracy.`,
      ...configurationTopics(run, points, imu),
    ];
  } catch {
    return ['Input preflight is unavailable or inconsistent; no diagnostic pass is shown.'];
  }
}
export function mappingProducer(imported: unknown, execution: unknown,
  trajectory: Identity | undefined, report: Identity | undefined): string[] {
  if (imported === undefined) return ['External SLAM producer information is unavailable in this package.'];
  const lines: string[] = [];
  try {
    const source = object(imported);
    if (source.schema !== 'cloudanalyzer.mapping_slam_import.v1') throw new Error('Unsupported import schema');
    const producer = object(source.producer_declaration);
    lines.push(`Declared producer: ${text(producer.backend)} · ${text(producer.version)}`);
    lines.push('Original generation evidence. Frame alignment, calibration and accuracy remain unverified.');
    if (!report) { lines.push('No linked GLIM execution report is included.'); return lines; }
    const inputs = object(source.inputs), run = object(execution);
    if (!matches(inputs.source_trajectory, trajectory) || !matches(inputs.glim_execution_report, report) ||
        !matches(run.trajectory, trajectory) || !matches(run.trajectory_check, trajectory)) throw new Error('Evidence identity mismatch');
    if (text(producer.backend).trim().toLowerCase() !== 'glim' || run.schema !== 'cloudanalyzer.glim_run.v1' ||
        run.status !== 'process_succeeded' || run.exit_code !== 0 || run.inputs_unchanged !== true || run.configuration_copy_matches !== true ||
        typeof run.image_id !== 'string' || !/^sha256:[0-9a-f]{64}$/.test(run.image_id)) throw new Error('Unsupported or unsuccessful execution');
    lines.push(`Recorded GLIM image: ${run.image_id}`);
    lines.push('Recorded successful execution; included source trajectory hashes match. This is a local record, not proof of execution or same-state PLY export.');
    lines.push(...inputPreflight(run));
  } catch {
    lines.push('Producer/execution evidence is unavailable, unsupported or inconsistent. No verified execution link is shown.');
  }
  return lines;
}
