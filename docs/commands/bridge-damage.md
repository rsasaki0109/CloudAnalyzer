# Damage candidates on a bridge soffit

`ca bridge-damage` searches the underside of a bridge deck for damage candidates
and places each on the numbered plan from [`bridge-plan`](bridge-plan.md). It also
writes an orthophoto of the soffit, which makes a base for drawing damage maps by
hand. Agents can call the same search as the `find_bridge_damage_candidates`
[MCP](mcp.md) tool.

```bash
ca bridge-damage bridge.las -o out --classes 0,1,2,3 --axis-classes 2 --member-classes 1,2 --surface-classes 1,2
```

The point cloud needs colour. `--classes`, `--axis-classes`, `--member-classes` and
`--supports` draw the plan as in `bridge-plan`; `--surface-classes` selects the
points the soffit is taken from.

## How it searches

The deck top is fitted over the inner 80% of the width, with a crown term, through
its upper returns; returns within 8 cm of a slab thickness below it are the soffit.
The soffit is rasterised at 2 cm into colour (CIE L\*a\*b\*) and height, and every
cell is compared with the median of its 0.5 m neighbourhood:

| Kind | Rule | Label |
|---|---|---|
| `light` | L\* at least 15 above the neighbourhood | 遊離石灰・白色変色の候補 (efflorescence, whitening) |
| `dark` | L\* at least 15 below | 漏水・暗色変色の候補 (water or dark staining) |
| `rust` | b\* at least 12 and a\* at least 4 above | 錆汁の候補 (rust staining) |
| `recess` | at least 12 mm receded into the slab | 剥離・欠損の候補 (spalling, section loss) |

Regions of at least 0.005 m² become candidates, with the deck slab element (`Ds`) they
lie in. Left out, because shading, haunches and edges mimic all four kinds:

- bands 0.15 m beyond the visible faces of girders and cross beams and around supports;
- 0.3 m inside the deck edges and ends, and 0.1 m around scan holes;
- straight runs at least 0.4 m long along or across the deck, even broken into pieces:
  joints between slab units and formwork seams. Their length is reported apart, and a
  0.2 m band beside them is left out for every kind (a dark joint has a receded groove);
- cells whose 0.5 m neighbourhood has returns in fewer than half of its cells: far from
  the scanner or between scan lines, single returns' colour and height noise reads as
  damage. The area left out is reported as too sparse;
- for `recess` only, cells whose neighbourhood heights have an interquartile range over
  2.4 cm: a step in the soffit (a haunch, a beam edge at a skew) puts both levels in the
  neighbourhood and one side reads as receded. A spall covers much less than a quarter
  of the neighbourhood and hardly moves that range, but a broadly rough surface does,
  so wide rough areas are not searched for recess.

Cracks are narrower than the raster and are not searched. Girder faces and the
deck top are not searched either.

## Outputs

- `damage_plan.svg`: the plan over the soffit orthophoto with each candidate boxed
  and numbered `D001`, `D002`, … in the colour of its kind.
- `soffit_ortho.png`: the orthophoto (2 cm, looking down through the deck like the plan).
- `damage.csv`: id, kind, label, element, area, centre and contrast per candidate.
- `damage.json`: the candidates with their bounds, the area searched, the seams left
  out and the thresholds.

When the underside was not scanned, nothing is searched and no damage is implied.

## Limits

Candidates are colour or height departures from the surrounding soffit, not
diagnoses. Formwork marks, shadows, wet patches, repairs and paint can look the
same, and no inspection records were available to measure how many candidates are
real or how much damage is missed. Candidates on the SemanticBridge scans were checked
by eye only, which led to leaving out joint grooves, sparse stretches and soffit steps
(see [`benchmarks/bridge/semanticbridge-tls`](../../benchmarks/bridge/semanticbridge-tls/README.md)). Use the candidates as pointers for an inspector,
and the orthophoto to draw the damage map.
