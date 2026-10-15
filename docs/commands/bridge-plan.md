# Bridge plan with numbered elements

`ca bridge-plan` draws a bridge's plan as a base for damage maps: the deck
outline with its skew, the girders and cross beams that the scan shows beneath
the slab, and numbered deck, girder and cross-beam elements. Agents can call the
same drawing as the `draw_bridge_plan` [MCP](mcp.md) tool.

```bash
ca bridge-plan bridge.las -o out --classes 0,1,2,3 --axis-classes 2 --member-classes 1,2
```

- `--classes` and `--axis-classes` work as in [`bridge-sections`](bridge-sections.md).
- `--member-classes` selects the points searched for girders and cross beams.
  Leaving out the abutments keeps their walls under the deck ends from reading as
  end cross beams.
- `--spacing` (default 0.25 m) sets the cross sections searched for girders.
- `--supports 18.5,37.0` gives the piers as distances from the start along the
  centreline, for example from the bridge register. Without it, piers are detected.

## How it finds members

The deck outline comes from the deck's side edges and its skewed end lines, as in
`bridge-sections`. In each cross section the slab thickness is measured from the
returns beneath the deck. A girder is a run of returns at least 0.2 m below the
soffit, at most 1.2 m wide, that appears at the same lateral position in at least
half of the sections with a measured soffit. A cross beam is the same kind of run
between two adjacent girders, parallel to the deck ends, in at least half of the
girder bays. A member line seen in fewer sections is drawn dotted in red, is not
numbered, and adds a warning: a member hidden from every scan position is missing
and would shift the numbers after it.

What is drawn depends on the underside:

| `underside` | Meaning | Elements |
|---|---|---|
| `girders_observed` | Girders found beneath the slab. | Deck slab divided by girders and cross beams; girder and cross-beam elements. |
| `no_girders_observed` | The soffit was scanned and nothing protrudes 0.2 m below it (a slab deck, or members outside the scan). | Deck slab divided lengthwise only. |
| `unobserved` | No returns beneath most of the deck. | One deck element; nothing below the deck is drawn or assumed. |

## Spans

Without `--supports`, a pier is structure found 0.5–3.5 m below the lowest girder
(or the slab) that meets all of these:

- From end to end it spans at least 30% of the deck width, searched to 1 m beyond the
  deck edges. A row of columns counts.
- It is at most 3.5 m long along the axis. Two scanned faces with an unscanned
  interior up to 2 m count as one.
- Its upper and lower halves stand within 0.5 m of each other.
- There is open space beneath the deck for 1.5 m on both sides.
- It is at least 2 m from both deck ends.

An abutment's sloped face leans and has its body on the side of its deck end, so it
is not taken as a pier.

On the 20 labelled bridges of the public SemanticBridge dataset, 32 of 33 piers are
found, with a mean position error of 0.15 m (at most 0.8 m). There are two errors:

- A vertical abutment wall 5 m inside a deck end is reported as a pier, because the
  ground behind it is not among the classes.
- A pier merged with a deep diaphragm above it is missed.

No pier is reported on the single-span bridges of either dataset (see
[`benchmarks/bridge/semanticbridge-tls`](../../benchmarks/bridge/semanticbridge-tls/README.md)).
Check the detected spans, and give `--supports` when the span layout is known.

Piers are labelled P1, P2, … from the start and the abutments A1 (start) and A2
(end). Each span boundary also divides the deck rows; a cross beam over a pier is
the same line. Every element records its span, and the drawing adds the span lengths.

## Numbering

Element numbers have four digits: the row along the axis from the start, then the
column across from the left facing the end, each from 01; rows continue across
spans. `Mg` is a main girder, `Cr` a cross beam and `Ds` a deck slab element. Rows run between the deck ends and
the cross beams; deck columns run between the deck edges and the girders, so the
overhangs are their own columns. Girders are numbered G1, G2, … from the left;
a girder element's column is its girder number, and a cross-beam element's column
is its girder bay. This follows the row-and-column rule of the 2004 draft road
bridge inspection guidelines; check it against the guidelines your inspection uses.

`plan.svg` looks down through the deck with the start on the left and the left side
facing the end on top. Members beneath the deck are dashed. Dimensions are in
millimetres: the centreline deck length, the total width, the girder spacings and
the cross-beam spacings along the centreline.

## Outputs

- `plan.svg`: the drawing.
- `members.csv`: one row per element with its span, along-axis and lateral bounds
  (metres in the axis frame) and status.
- `bridge_plan.json`: the deck, girders (lateral offset, visible width, depth of the
  lowest returns below the deck top, sections observed), cross beams, supports and
  spans, elements, underside status and warnings.

Results on the nine bridges of a public dataset are in
[`benchmarks/bridge/figshare-rc-bridges`](../../benchmarks/bridge/figshare-rc-bridges/README.md).

## Limits

These are drawings of what the point cloud shows, not a certified survey. A girder's
visible width is the extent of its returns, which is narrower than the web when only
one face was scanned. Bearings and lateral bracing are not detected. Detected piers
can include a vertical abutment wall or miss a pier hidden in the superstructure
(see Spans).
