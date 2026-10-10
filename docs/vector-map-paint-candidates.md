# Why bright returns do not become lane paint

Optional paint fits now report the outcomes of their bright-return checks in
`candidate_diagnostics`. The corridor and interior-divider reports expose the
same scan outcomes, and Web displays them in the build report. RGB remains the
default channel; selecting retained intensity still requires explicit opt-in.

Each bright candidate receives one outcome, in the existing gate order:

| Outcome | Existing check |
| --- | --- |
| `local_ground_missing` | Fewer than three source returns within 0.3 m |
| `local_height_mismatch` | Height differs from local P20 by more than 0.12 m |
| `trace_ground_missing` | Fewer than three source returns within 0.75 m of the trace at this station |
| `trace_height_mismatch` | Height differs from trace-neighbourhood P15 by more than 0.3 m |
| `flank_support_missing` | Either lateral flank lacks three same-level channel values |
| `flank_contrast_insufficient` | Either flank median is less than 40 brightness units darker |
| `accepted` | Candidate reaches the later connected-component checks |

These outcomes are mutually exclusive; the first failed check owns the candidate.
Intensity brightness uses temporary ROI normalization; RGB uses its retained
8-bit values. Support checks and height quantiles do not certify ground or classify buildings,
vehicles or road paint. **Accepted candidates are not detected lane boundaries.**
Narrow longitudinal components, a unique supported bundle, heading, residual and
curb checks still decide whether a correction can be applied.

`complete` means that the bright-candidate loop finished, even if a later
component/bundle check holds the fit. Complete outcomes sum to `bright_candidates`.
An interrupted scan reports only the examined prefix, with `complete: false`.
Its pending candidate has no outcome; Web displays it separately and says that
later candidates were not examined. A hold before scanning starts omits the
field rather than inventing zero measurements. Existing partial-scan
`contrasted_points` behavior is retained; use diagnostic `accepted` to inspect
the prefix. No correction is adopted from a partial scan.

## Cached source result

The six existing Tokyo development courses were regenerated without reference
geometry, with their original operator inputs. All corridor/divider corrections
remain held; every generated IR/OSM and source-profile position remains identical
to the previous bounded-search stage. No map-accuracy improvement is claimed.

The central course completes its intensity candidate loop:

| Check outcome | Points |
| --- | ---: |
| Bright candidates | 7,147 |
| Local support missing | 0 |
| Local height mismatch | 6,881 |
| Trace support missing | 0 |
| Trace height mismatch | 240 |
| Flank support missing | 2 |
| Flank contrast insufficient | 0 |
| Accepted for component checks | 24 |

None of the 24 accepted points forms a narrow longitudinal component. Most
bright returns fail the existing height checks, so their brightness alone is
insufficient evidence for lane paint. This does not establish an object class
for the rejected points. The unchanged ROI P10/P99.9 intensity calibration still
uses all ROI returns regardless of height; no ground mask or new normalization
domain is added.

West-south and east-south still encounter the existing neighbourhood cap and
show **incomplete** counts. Their examined prefixes are:

| Course | Bright prefix | Local height mismatch | Trace support missing | Trace height mismatch | Accepted | Pending |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| west-south | 120 | 112 | 0 | 7 | 0 | 1 |
| east-south | 441 | 367 | 6 | 66 | 1 | 1 |

All other rejection outcomes in these prefixes are zero. These are not totals
for the full courses; later bright returns were not examined. Curved traces and
the northern ROI limit still hold before candidate scanning. Reused development
courses are not held-out evidence.
No reference map was opened or threshold tuned in this iteration.

[Source-only generation hashes, exclusive outcomes and native/browser checks](../benchmarks/vector-map/paint-candidates/README.md)
retain the runtime and baseline identities. Previous verification packages,
source screenshots and README GIF remain historical and unchanged. This package
adds report/hash records rather than duplicate source clouds or generated maps.

[Bounded neighbourhood search](vector-map-paint-search.md) describes the unchanged
4,096-point examination cap and conservative cell overhead. [Explicit intensity
paint](vector-map-intensity-paint.md) retains source preparation, attribution and
the original held source-fit evidence.
