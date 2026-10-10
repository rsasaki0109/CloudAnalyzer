# Exact processed point records in a portable workspace

Chromium loaded the full generated-map review package, retained its 646,309-point
map and HD draft, created a 1% random subset (6,463 points), and saved both clouds
with current project metadata in `project.cloudanalyzer.zip`. Opening that one
file restored both clouds, their display names, hidden/visible state and all 21
HD lanes. Native PLY exports after reopening were byte-identical to the two PLY
members in the snapshot. `restored.png` is the actual browser screenshot.

The full cloud's **multiset of 32-byte records** matches the original canonical
PLY exactly: double XYZ and float intensity/correction. The filtered cloud is
an exact subset of those same records, including duplicate multiplicities. Point
order can change during native loading, so this comparison intentionally does not
assume source order. Python's independent ZIP reader also verified every CRC and
manifest member hash. The ZIP is 20,917,783 bytes; its SHA-256 and external path are
recorded in `receipt.json`. Large point data is kept outside git.

This validation exposed a separate PLY loader defect: an additional float
`correction` property was previously discarded. The native reader now retains
additional float32 and unsigned-byte vertex attributes in ASCII, little-endian,
big-endian and streaming paths. Synthetic browser checks also cover unsigned-byte
attributes, large double coordinates, a baked manual transform without applying
it twice, processed-cloud names, mesh faces and signed mesh distances. Arbitrary
float64/integer attribute types and raw Gaussian rotation/SH records are outside
this additional-attribute guarantee.

## Reproduce

```sh
cd web
PW_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium npm run test:e2e -- project-snapshot.spec.ts
CA_REVIEW_NCLT_BUNDLE=/path/to/nclt-hd-preflight-portable-review.zip \
  PW_CHROMIUM_EXECUTABLE_PATH=/usr/bin/chromium \
  npm run test:e2e -- project-snapshot-nclt.spec.ts --workers=1
```

The optional real-data test writes its ZIP and screenshot under `/tmp`. It skips
when the external NCLT review bundle is not provided. Run the stdlib-only verifier
against the retained ZIP and original canonical map to repeat the record checks:

```sh
python3 -I -S benchmarks/vector-map/nclt-workspace-snapshot/verify.py \
  --bundle /path/to/nclt-point-workspace-snapshot.zip \
  --source /path/to/local_map.ply
```

Without arguments, the verifier checks the committed evidence hashes and receipt
consistency. With `--bundle` it additionally checks ZIP CRCs, member hashes,
metadata/visibility, lane count and source fingerprints. With `--source` it checks
the original full-record multiset and subset. It does not rerun browser actions,
generate geometry or establish independent accuracy.

## Scope and provenance

A workspace snapshot stores **currently loaded** point/mesh geometry and the
current project metadata. It is bounded to 64 MiB of ZIP content and 127 clouds/
meshes. Unloaded original density, Undo history, pose-graph input files, mapping
job continuation state and the four frozen source-audit reports remain external.
Keep the original generated-map review bundle for those reports and provenance.
Review notes in project metadata are distinct from those source-audit reports.
Normal JSON project files and browser autosaves still contain metadata only;
processed results require a workspace snapshot or their own point-file exports.

No root/child run or job record changed, no mapping attempts were consumed, and
no HD geometry or saved source audit was regenerated. The prior NCLT result still
has extent 124 / 252.586 m and 363 legacy height disagreements. Saving it does
not resolve those limitations or establish deployment readiness.

Data derives from the University of Michigan North Campus Long-Term (NCLT)
dataset, session 2012-06-15; N. Carlevaris-Bianco, A. K. Ushani and R. M. Eustice,
*University of Michigan North Campus long-term vision and lidar dataset*, IJRR
2016. Dataset: <http://robots.engin.umich.edu/nclt/>. Database terms:
[ODbL 1.0](https://opendatacommons.org/licenses/odbl/1-0/); contents:
[DBCL 1.0](https://opendatacommons.org/licenses/dbcl/1-0/). This derived database is
offered under the same terms. See the [input attribution](../../../web/public/samples/ATTRIBUTION.md).
The generic workspace ZIP does not copy dataset attribution automatically; keep
this credit and the original attributed review bundle when sharing these data.
