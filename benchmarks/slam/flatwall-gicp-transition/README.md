# Real-recording GICP outer-transition diagnostics

All **686 accepted steps** reduce their frozen trial cost. After matching and
source-shape transport refresh, **278** have a larger cost. The matched source
set changes in **zero** transitions: these increases are not caused by gaining
or dropping source rows. This is a diagnostic of **36 fixed real problems with
two shape variants (72 histories)**, not 72 independent recording scans.

The two variants retain their original 27 step-converged, five iteration-limited
and four empty-map outcomes. The optimizer is not rerun and no new pose is
accepted. Both converged and limited histories can have refreshed increases;
an increase alone does not classify failure or define a new stopping rule.

## Fixed-support accounting

For the common source IDs, evaluate an eight-cell cube with independent choices
of baseline/accepted target assignment, source-shape rotation and residual pose.
An assignment selects both a world target point and its world shape. One order
accounts for motion, source-metric refresh, then assignment; the alternate order
refreshes assignment first. Their interaction is reported explicitly. These
are declared counterfactual cost evaluations at captured poses, not estimator
updates. Newly matched and removed rows have separate costs, even though this
recording's transitions have no support changes.

Every cell agrees with the unchanged native factor/frozen-cost APIs:
**5,488 cost checks** and **2,744 combined-shape checks**. Maximum observed
cost disagreement is `2.388e-12`, shape disagreement `4.441e-16`, and accounting
residual `1.422e-14`. Full baseline, frozen accepted and refreshed costs agree
with the captured history. This uses the complete PR300 private native package
verbatim; its 111 committed build inputs agree under LF normalization. No new
native compilation or installed-extension replacement occurs.

The five limited problems in each variant contribute 100 final-ten transitions.
All 100 lower the frozen cost; 60 increase after refresh. Accounting effects
depend on order and can cancel. Tail sums need not cover a complete recurrence
period, so they are not a causal attribution or a new objective.

## Recurrence and controls

All lags 1..5 for the final ten baseline states are published separately for
ordered correspondence IDs, pose distances and combined shapes. Raw-fine row60
and rotation-legacy row63 show three-step correspondence repeats and close pose
returns; rotation-fine row63 shows five-step repeats and close returns. Raw-fine
row63 approaches a two-step pattern during the tail. Rotation-fine row49 has
seven distinct final-ten correspondence sets and none repeat at lags1..5.
No approximate-cycle threshold, convergence relabeling or iteration increase
is introduced. Repeating IDs alone do not imply a repeating pose or metric.

Twenty-three closed-form controls cover anisotropic rotations, the complete
cube, target point/shape switches, added/removed/empty support, ownership and
invalid inputs. Four controls separate correspondence, pose and metric
recurrence, including zero/small/half-turn rotation distances. Two packet
tests check reported identities and reject changed evidence bytes.

Input provenance: [immutable-target audit](../flatwall-gicp-target/README.md)
and its original/native shape inputs. Kenji Koide/AIST,
[flatwall recording](https://doi.org/10.5281/zenodo.7641866), CC BY 4.0.
`flatwall_01.bag`, 184,597,802 bytes,
SHA256 `4721e667166e11b948aea83b221f99ca0fec21fabdce5b292662a4b442efbab6`.
Thirteen JSON files preserve identities, scope, frame summaries, limited tails,
recurrence and reported accounting. All 686 complete cost arrays/transitions
remain outside Git. Public scripts and packet use LF checkout rules.

```powershell
$env:RAYON_NUM_THREADS = '4'
$env:OPENBLAS_NUM_THREADS = '1'
python scripts/audit_flatwall_gicp_transition.py `
  --input /path/to/frozen/target/audit --out /path/to/fresh/transitions `
  --slice-protocol benchmarks/slam/flatwall-gicp-transition/slice-protocol.json `
  --reused-native benchmarks/slam/flatwall-gicp-transition/reused-native.json
python scripts/analyze_flatwall_gicp_transitions.py `
  --audit /path/to/fresh/transitions --input /path/to/frozen/target/audit `
  --out /path/to/fresh/analysis
python benchmarks/slam/flatwall-gicp-transition/verify.py
```

Reproduction requires the original hashed arrays and the compatible complete
native package. The verifier checks public bytes and reported identities/
accounting; it does not rerun the recording or independently certify numerical
or physical accuracy. No solver, acceptance, gate, frontend, default, GPU or
performance change follows from these diagnostics. See [cost semantics](../../../docs/slam-gicp-transitions.md).
