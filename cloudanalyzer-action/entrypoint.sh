#!/usr/bin/env bash
set -euo pipefail

CONFIG="${INPUT_CONFIG:?config input is required}"
BASELINE="${INPUT_BASELINE:-}"
COMMENT="${INPUT_COMMENT:-true}"
FAIL_ON_GATE="${INPUT_FAIL_ON_GATE:-true}"
PROJECT="${INPUT_PROJECT:-}"
MARKER="${INPUT_MARKER:-cloudanalyzer-qa}"

WORKDIR="${RUNNER_TEMP:-/tmp}/cloudanalyzer-action"
mkdir -p "$WORKDIR"

# Dogfood: editable install when the caller repo is CloudAnalyzer itself.
if [[ -f "${GITHUB_WORKSPACE}/cloudanalyzer/pyproject.toml" ]]; then
  pip install -q -e "${GITHUB_WORKSPACE}/cloudanalyzer"
fi

CONFIG_ABS="${GITHUB_WORKSPACE}/${CONFIG}"
if [[ ! -f "$CONFIG_ABS" ]]; then
  echo "::error::Config not found: ${CONFIG_ABS}"
  exit 1
fi

SUMMARY_JSON="${WORKDIR}/summary.json"
COMMENT_MD="${WORKDIR}/body.md"

set +e
xvfb-run ca check "$CONFIG_ABS" --output-json "$SUMMARY_JSON"
CHECK_EXIT=$?
set -e

if [[ ! -f "$SUMMARY_JSON" ]]; then
  echo "::error::CloudAnalyzer summary JSON was not produced"
  exit 1
fi

EXTRA=()
if [[ -n "$BASELINE" ]]; then
  BASELINE_ABS="${GITHUB_WORKSPACE}/${BASELINE}"
  if [[ ! -f "$BASELINE_ABS" ]]; then
    echo "::error::Baseline summary not found: ${BASELINE_ABS}"
    exit 1
  fi
  EXTRA+=(--baseline "$BASELINE_ABS")
fi
if [[ -n "$PROJECT" ]]; then
  EXTRA+=(--project "$PROJECT")
fi

ca report-pr-comment "$SUMMARY_JSON" "${EXTRA[@]}" --output "$COMMENT_MD"

read -r PASSED WORST_CHECK < <(python3 - <<'PY'
import json
import sys
from pathlib import Path

summary_path = Path(sys.argv[1])
data = json.loads(summary_path.read_text(encoding="utf-8"))
passed = "false"
worst = ""
if isinstance(data.get("summary"), dict):
    passed = "true" if data["summary"].get("passed") else "false"
    checks = data.get("checks") or []
    for check in checks:
        if isinstance(check, dict) and check.get("passed") is False:
            worst = str(check.get("id") or "")
            break
elif "overall_quality_gate" in data:
    gate = data.get("overall_quality_gate") or {}
    passed = "true" if gate.get("passed") else "false"
print(passed, worst)
PY
"$SUMMARY_JSON")

{
  printf '<!-- %s -->\n' "$MARKER"
  cat "$COMMENT_MD"
} > "${COMMENT_MD}.with-marker"
mv "${COMMENT_MD}.with-marker" "$COMMENT_MD"

if [[ -n "${GITHUB_OUTPUT:-}" ]]; then
  {
    echo "summary_json=${SUMMARY_JSON}"
    echo "passed=${PASSED}"
    echo "worst_check=${WORST_CHECK}"
    echo "comment_path=${COMMENT_MD}"
  } >> "$GITHUB_OUTPUT"
fi

if [[ "$COMMENT" == "true" ]]; then
  PR=""
  if [[ -n "${GITHUB_EVENT_PATH:-}" && -f "$GITHUB_EVENT_PATH" ]]; then
    PR="$(python3 - <<'PY'
import json
import os
from pathlib import Path

event = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text(encoding="utf-8"))
pull_request = event.get("pull_request") or {}
number = pull_request.get("number")
print(number if number is not None else "")
PY
)"
  fi
  if [[ -z "$PR" ]]; then
    echo "::warning::No pull request context; skipping comment post."
  elif [[ -z "${GITHUB_TOKEN:-}" ]]; then
    echo "::warning::GITHUB_TOKEN not available; skipping comment post."
  else
    export GH_TOKEN="$GITHUB_TOKEN"
    marker_pattern="<!-- ${MARKER} -->"
    existing_id="$(gh api \
      "repos/${GITHUB_REPOSITORY}/issues/${PR}/comments" \
      --paginate \
      --jq "[.[] | select(.body | contains(\"${marker_pattern}\"))] | .[0].id // empty")"
    if [[ -n "$existing_id" ]]; then
      echo "Updating existing comment ${existing_id}"
      gh api \
        -X PATCH \
        "repos/${GITHUB_REPOSITORY}/issues/comments/${existing_id}" \
        -f body="$(cat "$COMMENT_MD")"
    else
      echo "Posting new comment on PR #${PR}"
      gh pr comment "$PR" \
        --repo "$GITHUB_REPOSITORY" \
        --body-file "$COMMENT_MD"
    fi
  fi
fi

if [[ "$FAIL_ON_GATE" == "true" && "$CHECK_EXIT" -ne 0 ]]; then
  echo "CloudAnalyzer quality gate failed"
  exit "$CHECK_EXIT"
fi

exit 0
