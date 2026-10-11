# Bridge cross sections and dimension table

`ca bridge-sections` cuts cross sections perpendicular to a bridge's axis and
aggregates a member dimension table, a section drawing and the per-section
evidence. Agents can call the same measurement as the `measure_bridge_sections`
[MCP](mcp.md) tool.

```bash
ca bridge-sections bridge.las -o out --classes 0,1,2,3 --axis-classes 2
ca bridge-sections bridge.laz -o out --axis 12.0,40.5,15.2,10.1 --spacing 0.5
```

- `--classes` keeps only these classification codes, for example structure
  classes without vegetation. Without it every point is measured.
- `--axis-classes` gives the points that define the axis and the deck ends,
  usually the deck. Otherwise the measured points are used.
- `--axis x1,y1,x2,y2` fixes the axis explicitly instead.
- `--spacing` (default 1 m) and `--thickness` (default 0.1 m) set the stations
  and the slab of points each section takes.

## How it measures

The axis starts as the principal XY direction and is then turned parallel to
the deck's side edges: a skewed deck is a parallelogram whose principal
direction leans toward its diagonal. The deck end lines give the skew and the
centreline deck length. Sections are cut only where the full width lies between
the ends, so a skewed end never cuts a section partway across an abutment.

In each section the points are binned laterally into height layers, and layers
continuing smoothly between bins form surfaces. The deck top is the highest of
the long surfaces; a scanned soffit or the ground can be as long as the deck.
A crowned deck is reported with its crown height and separate left and right
cross slopes. Beyond each deck end, the curb (地覆) top is the most populated
level 5–60 cm above the deck; returns above it give the parapet height, and
returns continuing down from it give the visible outer-face depth. The slab
thickness is the first surface of at least three returns beneath the deck top,
reported only when such returns exist in at least 30% of the deck's width.

## Outputs

`dimensions.csv` and the `table` in `bridge_sections.json` give, per item, the
median, p10, p90 and the number of sections that observed it. Each item has a status:

| Status | Meaning |
|---|---|
| `observed` | Measured from returns in the sections that list it. |
| `lower_bound` | A visible extent, such as the outer-face depth; the member can be larger. |
| `unobserved` | No supporting returns, for example no soffit when scanned from the deck. No value is assumed. |
| `not_applicable` | The feature is absent, such as side slopes on a deck without a crown. |

`section.svg` draws the median-width section viewed from the start of the axis
toward its end, with the measured outline, dimension lines in millimetres and
the unobserved items. The JSON also keeps every section's values and the axis.

Results on three public bridges are in
[`benchmarks/bridge/figshare-rc-bridges`](../../benchmarks/bridge/figshare-rc-bridges/README.md).

## Limits

These are measurements from the point cloud, not a certified survey. Interior
girders, bearings and abutment dimensions are not measured yet, and no drawing
beyond the cross section is produced. A one-bin (5 cm) feature or a curb lower
than 5 cm is not separated from the deck. Without labels, vegetation over the
deck is not guaranteed to be excluded; use `--classes` or crop first.
