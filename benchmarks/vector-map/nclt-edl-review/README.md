# Readable NCLT point context with EDL

The old screen-space EDL pass assigned an occlusion response of 1 to every
empty neighbouring pixel. At strength 1 this made isolated two-pixel returns
nearly black even when their solid color was light grey. The regression test
observed a visible-point ratio of zero with EDL on versus EDL off.

The pass now distinguishes background using the raw depth-buffer value and
only compares measured neighbouring depths. Sparse point colors remain visible;
actual closer surfaces still produce depth-step shading. Raw depth also avoids
confusing a measured distance of exactly one metre (`log2(distance) == 0`) with
background. No source data, cloud visibility, audit or map geometry is changed.

Actual Chromium review of the retained NCLT display package loaded 161,578
records, switched all four full-source saved audits, retained nine legacy review
lanes and 19 problem intervals, and showed readable context points under EDL.
`preview-after.png` is the actual screenshot. Compare the prior screenshot at
`../nclt-preview-bundle/review.png`. Root/child run and job hashes remained unchanged.
The extent shortfall and original audit disagreements remain unresolved.

Two synthetic pixel checks compare EDL with unshaded rendering: isolated returns
stay visible against empty background, while a dense measured depth step still
has shaded pixels. Dark background comparisons allow five sRGB levels for the
existing 8-bit linear render-target quantization. TypeScript/Vite build, the full
and preview NCLT browser cases, and both source-coverage editing cases pass.

```sh
cd web
npm run build
npm run test:e2e -- edl.spec.ts source-coverage-locations.spec.ts
```

`receipt.json` identifies the exact unchanged review package and original run
records. This display fix establishes no new source coverage or map accuracy.
