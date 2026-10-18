# Exact target-count lookup comparison

This packet audits the single-lookup counter used by target-balanced odometry.
It reuses all 16 original closed-loop real-data conditions and all 64 predetermined
states from the target-balance and stage studies. The four original empty-source
controls remain separate from the 60 alignment-eligible fixed states.

The count stage took about 24% less time. Whole balanced H/b took 9.7% less time
with four workers on the fixed states. Complete native registration did **not**
establish an overall speedup: the sum of case medians increased about 0.8%, while
the unchanged default control increased about 3.9%. Defaults, numerical results
and stopping policies are unchanged. No whole-application or GPU speedup is claimed.

See [method, complete cohort results and limitations](../../../docs/slam-flatwall-count-lookup.md),
[all measured summaries](summary.json), [numerical checks](verification.json),
[frozen protocol](protocol.json) and [output catalog](catalog.json).

```text
python benchmarks/slam/flatwall-count-lookup/verify.py
python -m pytest cloudanalyzer/tests/test_flatwall_count_lookup_packet.py -q -o addopts=
```

`fixed/`, `time/` and `trace/` retain the original output bytes inside deterministic
gzip files, original logs, invocations and closing receipts. `source-footprints.json`
resolves the two 84-file native build inventories. Unchanged sources reuse the
preceding study's archive; changed executed source content is included here.
`controllers/` contains the original executed controllers with their original
study paths; these are archival records, not a portable installation command.

`prior-validation/` preserves the validator that rejected a sub-clock empty-input
component. The zero-duration correction keeps all native values and samples
unchanged. `history/` preserves the prior public metadata and affected test source.
The independent verifier permits only tiny floating-point roundoff in derived
summary arithmetic; raw elapsed integers, correspondence identities, native
pose/matrix bits and archived-byte hashes remain strict.

Real recordings are the same Flatwall data attributed in the
[original target-balance packet](../flatwall-target-balance/README.md) and
[recurrence study](../flatwall-balance-dynamics/README.md). Raw recordings are not
redistributed. Local full Windows release tests reproduce three existing NaN bit
failures in both builds; the odometry tests and strict clippy pass. Details and
the full validation scope are recorded in the linked study.
