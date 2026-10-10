# Apply an agent's explicit supported HD plan through MCP

The calling agent reads fresh gap and reference-support reports, chooses observed
interval IDs and explicitly permits coincident geometric endpoint links.
`apply_supported_hd_plan` then runs child preparation, source inspection, geometry
and lane drafting, patch inspection, all four final audits, retained-map checks,
child finish and comparison. The root remains unfinished until the agent reads
the result and explicitly adopts it with `finish_retry`.

| Bundled drive | Retained seed | Delivered source extent | New HD attempts |
|---|---:|---:|---:|
| 2012-06-15 | 120 m | 124 m (+4 m; no loss) | 3 of 4 |
| 2012-04-29 | 110 m | 124 m (+14 m; no loss) | 3 of 3 |

June selects global interval IDs 3 and 11 from two pages of the same frozen
index: 222–224 and 224–226 m. This is a controlled replay on a known session,
not a new autonomy or generalization benchmark. Its local route grows from 2 to
6 m, while the global longest span remains 66 m. All 646,309 point records,
motion, proposal, retained lane geometry, metadata and directed edges are exact.
The nine held reference intervals and legacy 363 quantile height failures remain.
Repeating the same application leaves root/child job and run bytes unchanged;
no further native generation runs.

April inspects 15 width/occupancy-eligible intervals across two frozen indices.
The caller chooses the seven contiguous intervals spanning 184–198 m, supported
by both estimators. An unsupported 22–24 m interval is rejected before any child,
policy or HD attempt is created; the recorded job/run hashes remain identical.
The final patch retains every old lane and adds seven new lanes with explicit
endpoint pairs. No point fusion or source extraction is repeated.

The four complete final audits (quantile and consensus, editable IR and reopened
OSM) were repeated exactly on each full point map after adoption. Earlier finished
run/job states and the pinned native extension remain unchanged. Both outputs
still miss full-drive extent; traffic semantics, independent map accuracy and
deployment readiness remain unresolved.

The packet includes source pages/indexes, full bounded reference observations,
explicit policies, action histories, maps, audit reports, comparison and retention
checks. June's unchanged source proposal is referenced from the earlier HD-only
packet; April's is included. `files-sha256.json` hashes portable copies. Original
artifact descriptors identify execution files, with prefixes replaced by
`run/`, `april-run/`, `repo/`, `native/` and `env/`. Full point files remain outside
git. The verifier independently reconstructs interval indexing and audit gates.

```sh
PYTHONPATH=cloudanalyzer python benchmarks/vector-map/nclt-hd-plan-application/verify_packet.py
# With the original execution files and matching native extension:
PYTHONPATH=/workspace/.cloudanalyzer-env/path-refinement-final-native:cloudanalyzer \
python benchmarks/vector-map/nclt-hd-plan-application/verify_packet.py \
  --generated-root /workspace/.cloudanalyzer-env/nclt-hd-plan-application-june \
  --april-generated-root /workspace/.cloudanalyzer-env/nclt-hd-plan-application-april
```

These are recorded assisted agent workflows with inherited maps and explicit
layout hypotheses. Synthetic tests additionally exercise interruption/resume,
failed native attempts, stale revisions, copied/unseen pages, unsupported heights,
overlapping choices and isolated endpoint policies. The helper embeds no model
and makes no quality ranking or parent adoption decision.

Derived NCLT maps and observations retain ODbL-1.0 and DBCL-1.0 terms. See
[NCLT attribution](../../../web/public/samples/ATTRIBUTION.md).
