/**
 * QA report: panels publish the metrics of their latest results, the user
 * sets pass / fail thresholds on any of them, and the report is saved as a
 * self-contained HTML page and as JSON whose gate summary matches `ca check`
 * (schema cloudanalyzer.gate_summary.v0.1).
 */

import { $, download, fmt, removeButton, setStatus } from "./dom";
import { viewImage } from "./image";
import { distanceChanged, entries, isMesh, listChanged } from "./state";

export interface Metric {
  label: string;
  value: number;
  unit?: string;
}

export interface ReportSection {
  title: string;
  /** Metrics keyed by a short id, e.g. "mean"; gates refer to `<section id>/<metric id>`. */
  metrics: Record<string, Metric>;
  /** Extra lines shown with the section (not gated). */
  notes?: string[];
}

type Severity = "fail" | "warn";

export interface Gate {
  metric: string;
  op: "<=" | ">=";
  threshold: number;
  severity: Severity;
}

const providers = new Map<string, () => Map<string, ReportSection> | ReportSection | null>();

/** Publish a section (or several, keyed by id) for the report; called when the report is made. */
export function addReportSection(
  id: string,
  provider: () => Map<string, ReportSection> | ReportSection | null,
): void {
  providers.set(id, provider);
}

/** Every section available now, keyed by id. */
function sections(): Map<string, ReportSection> {
  const out = new Map<string, ReportSection>();
  for (const [id, provider] of providers) {
    const got = provider();
    if (got instanceof Map) for (const [sub, section] of got) out.set(`${id}:${sub}`, section);
    else if (got) out.set(id, got);
  }
  return out;
}

function metricsOf(all: Map<string, ReportSection>): Map<string, { section: ReportSection; metric: Metric }> {
  const out = new Map<string, { section: ReportSection; metric: Metric }>();
  for (const [id, section] of all) {
    for (const [key, metric] of Object.entries(section.metrics)) out.set(`${id}/${key}`, { section, metric });
  }
  return out;
}

// ---------------------------------------------------------------- gates

const gates: Gate[] = [];

export function savedGates(): Gate[] {
  return gates.map((g) => ({ ...g }));
}

export function setGates(saved: Gate[]): void {
  gates.splice(0, gates.length, ...saved);
  renderGates();
}

type Status = "pass" | "fail" | "warn" | "info";

interface Check {
  id: string;
  metric: string;
  label: string;
  value: number | null;
  op: Gate["op"];
  threshold: number;
  severity: Severity;
  passed: boolean | null;
  status: Status;
}

function evaluate(all: Map<string, ReportSection>): Check[] {
  const metrics = metricsOf(all);
  return gates.map((gate, i) => {
    const found = metrics.get(gate.metric);
    const value = found?.metric.value ?? null;
    const passed =
      value === null || !Number.isFinite(value) ? null : gate.op === "<=" ? value <= gate.threshold : value >= gate.threshold;
    const status: Status = passed === null ? "info" : passed ? "pass" : gate.severity;
    return {
      id: `gate-${i + 1}`,
      metric: gate.metric,
      label: found ? `${found.section.title} · ${found.metric.label}` : gate.metric,
      value,
      op: gate.op,
      threshold: gate.threshold,
      severity: gate.severity,
      passed,
      status,
    };
  });
}

/** Summary in the shape of `ca check`'s gate policy block (default mode). */
function summarize(checks: Check[]) {
  const ids = (status: Status) => checks.filter((c) => c.status === status).map((c) => c.id);
  const failed = ids("fail");
  return {
    schema_version: "cloudanalyzer.gate_summary.v0.1",
    mode: "default",
    passed: failed.length === 0,
    exit_code: failed.length === 0 ? 0 : 1,
    blocking_failed_ids: failed,
    pass_count: ids("pass").length,
    fail_count: failed.length,
    warn_count: ids("warn").length,
    soft_fail_count: 0,
    skip_count: 0,
    not_applicable_count: 0,
    info_count: ids("info").length,
    passed_ids: ids("pass"),
    failed_ids: failed,
    warning_ids: ids("warn"),
    soft_failed_ids: [],
    skipped_ids: [],
    not_applicable_ids: [],
    ungated_ids: ids("info"),
  };
}

function renderGates(): void {
  const metrics = metricsOf(sections());
  const checks = evaluate(sections());
  $("gate-list").replaceChildren(
    ...gates.map((gate, i) => {
      const li = document.createElement("li");
      const select = document.createElement("select");
      select.setAttribute("aria-label", `Gate ${i + 1} metric`);
      const known = [...metrics.entries()];
      if (!metrics.has(gate.metric)) select.add(new Option(`${gate.metric} (no result yet)`, gate.metric));
      for (const [id, { section, metric }] of known) select.add(new Option(`${section.title} · ${metric.label}`, id));
      select.value = gate.metric;
      select.onchange = () => {
        gate.metric = select.value;
        renderGates();
      };
      const op = document.createElement("select");
      op.setAttribute("aria-label", `Gate ${i + 1} comparison`);
      op.add(new Option("≤", "<="));
      op.add(new Option("≥", ">="));
      op.value = gate.op;
      op.onchange = () => {
        gate.op = op.value as Gate["op"];
        renderGates();
      };
      const threshold = document.createElement("input");
      threshold.type = "number";
      threshold.step = "any";
      threshold.value = String(gate.threshold);
      threshold.setAttribute("aria-label", `Gate ${i + 1} threshold`);
      threshold.onchange = () => {
        gate.threshold = Number(threshold.value);
        renderGates();
      };
      const severity = document.createElement("select");
      severity.setAttribute("aria-label", `Gate ${i + 1} severity`);
      severity.add(new Option("fail", "fail"));
      severity.add(new Option("warn", "warn"));
      severity.value = gate.severity;
      severity.onchange = () => {
        gate.severity = severity.value as Severity;
        renderGates();
      };
      const badge = document.createElement("span");
      const check = checks[i];
      badge.className = `badge ${check.status}`;
      badge.textContent = check.value === null ? "no value" : `${check.status} · ${value(check.value)}`;
      li.append(select, op, threshold, severity, badge, removeButton(() => {
        gates.splice(i, 1);
        renderGates();
      }));
      return li;
    }),
  );
  $<HTMLButtonElement>("gate-add").disabled = metrics.size === 0;
  $("report-hint").textContent =
    metrics.size === 0
      ? "Run an analysis (distances, volume, ICP, trajectories…) to get metrics to gate on."
      : `${metrics.size} metrics available.`;
}

$<HTMLButtonElement>("gate-add").onclick = () => {
  const first = metricsOf(sections()).entries().next().value;
  if (!first) return;
  const [metric, { metric: m }] = first;
  gates.push({ metric, op: "<=", threshold: Number(m.value.toPrecision(3)), severity: "fail" });
  renderGates();
};
// New results change the metrics gates can use and their values.
listChanged.add(renderGates);
distanceChanged.add(renderGates);
$<HTMLButtonElement>("gate-refresh").onclick = renderGates;

// ---------------------------------------------------------------- output

function reportJson(all: Map<string, ReportSection>, checks: Check[]) {
  return {
    schema_version: "cloudanalyzer.web_report.v0.1",
    generated: new Date().toISOString(),
    clouds: [...entries.values()].map((e) => ({
      name: e.cloud.name,
      kind: e.cloud.kind,
      points: isMesh(e) ? null : e.cloud.count,
      triangles: isMesh(e) ? e.cloud.triangles : null,
      bounds: e.cloud.bounds,
      visible: e.visible,
    })),
    sections: Object.fromEntries(
      [...all].map(([id, s]) => [
        id,
        {
          title: s.title,
          metrics: Object.fromEntries(Object.entries(s.metrics).map(([k, m]) => [k, { ...m }])),
          notes: s.notes ?? [],
        },
      ]),
    ),
    checks: checks.map(({ label: _label, ...c }) => c),
    gate_summary: summarize(checks),
  };
}

const escape = (s: string) =>
  s.replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]!);

/** Counts as whole numbers, everything else as `fmt` does. */
function value(v: number): string {
  return Number.isInteger(v) && Math.abs(v) < 1e15 ? v.toLocaleString("en-US") : fmt(v);
}

function number(m: Metric): string {
  return `${value(m.value)}${m.unit ? ` ${m.unit}` : ""}`;
}

async function reportHtml(all: Map<string, ReportSection>, checks: Check[]): Promise<string> {
  const summary = summarize(checks);
  let image = "";
  try {
    const blob = await viewImage();
    const bytes = new Uint8Array(await blob.arrayBuffer());
    let binary = "";
    for (let i = 0; i < bytes.length; i += 0x8000) binary += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
    image = `<img alt="View" src="data:image/png;base64,${btoa(binary)}">`;
  } catch {
    // No view to show (e.g. WebGL unavailable): the tables still make a report.
  }
  const verdict =
    checks.length === 0
      ? `<p class="verdict info">No gates set</p>`
      : `<p class="verdict ${summary.passed ? "pass" : "fail"}">${summary.passed ? "PASS" : "FAIL"} · ` +
        `${summary.pass_count} passed, ${summary.fail_count} failed, ${summary.warn_count} warnings` +
        `${summary.info_count ? `, ${summary.info_count} without a value` : ""}</p>`;
  const gateRows = checks
    .map(
      (c) =>
        `<tr class="${c.status}"><td>${escape(c.label)}</td><td>${c.value === null ? "–" : value(c.value)}</td>` +
        `<td>${c.op === "<=" ? "≤" : "≥"} ${value(c.threshold)}</td><td>${c.severity}</td><td><b>${c.status}</b></td></tr>`,
    )
    .join("");
  const cloudRows = [...entries.values()]
    .map(
      (e) =>
        `<tr><td>${escape(e.cloud.name)}</td><td>${
          isMesh(e) ? `${e.cloud.triangles.toLocaleString()} triangles` : `${e.cloud.count.toLocaleString()} points`
        }</td><td>${e.visible ? "shown" : "hidden"}</td></tr>`,
    )
    .join("");
  const sectionHtml = [...all.values()]
    .map(
      (s) =>
        `<section><h2>${escape(s.title)}</h2><table>${Object.values(s.metrics)
          .map((m) => `<tr><th>${escape(m.label)}</th><td>${escape(number(m))}</td></tr>`)
          .join("")}</table>${(s.notes ?? []).map((n) => `<p class="note">${escape(n)}</p>`).join("")}</section>`,
    )
    .join("");
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>CloudAnalyzer QA report</title>
<style>
body{font:14px/1.5 system-ui,sans-serif;margin:0 auto;max-width:960px;padding:24px 16px;color:#1d2230;background:#fff}
h1{font-size:22px;margin:0 0 4px}h2{font-size:16px;margin:24px 0 8px}
table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:4px 8px;border-bottom:1px solid #e3e6ee;vertical-align:top}
th{font-weight:500;color:#5b6272;width:40%}img{max-width:100%;border-radius:6px;margin-top:12px}
.verdict{display:inline-block;padding:6px 12px;border-radius:6px;font-weight:600}
.pass{color:#11632f;background:#e3f6e9}.fail{color:#8c1d1d;background:#fde7e7}.warn{color:#7a4b00;background:#fff3d6}.info{color:#475069;background:#eef0f5}
tr.pass td,tr.fail td,tr.warn td,tr.info td{background:none}tr.fail b{color:#b42323}tr.pass b{color:#1a7f3c}tr.warn b{color:#a86400}
.note,.meta{color:#5b6272}
</style></head><body>
<h1>CloudAnalyzer QA report</h1>
<p class="meta">${escape(new Date().toLocaleString())} · CloudAnalyzer Web</p>
${verdict}
${checks.length ? `<h2>Gates</h2><table><tr><th>Metric</th><th>Value</th><th>Threshold</th><th>Severity</th><th>Status</th></tr>${gateRows}</table>` : ""}
<h2>Data</h2><table>${cloudRows || "<tr><td>No clouds</td></tr>"}</table>
${sectionHtml}
${image}
</body></html>
`;
}

$<HTMLButtonElement>("report-html").onclick = async () => {
  const all = sections();
  const checks = evaluate(all);
  download(new Blob([await reportHtml(all, checks)], { type: "text/html" }), "cloudanalyzer-report.html");
  const summary = summarize(checks);
  setStatus(
    `Saved cloudanalyzer-report.html` +
      (checks.length ? `: ${summary.passed ? "PASS" : "FAIL"} (${summary.pass_count} of ${checks.length} gates passed)` : ""),
    checks.length > 0 && !summary.passed,
  );
};

$<HTMLButtonElement>("report-json").onclick = () => {
  const all = sections();
  const json = JSON.stringify(reportJson(all, evaluate(all)), null, 2);
  download(new Blob([json], { type: "application/json" }), "cloudanalyzer-report.json");
  setStatus("Saved cloudanalyzer-report.json");
};

renderGates();
