// Strict, measurement-only Rust transformation; the production source is untouched.
export const collectionSource = 'rust/crates/ca-core/src/kdtree.rs';
export const collectionBefore = `        Some(Self {
            nodes,
            points: items.iter().map(|&(p, _)| p).collect(),
            indices: items.iter().map(|&(_, i)| i).collect(),
        })`;
export const collectionAfter = `        let (points, indices): (Vec<_>, Vec<_>) = items.into_iter().unzip();
        Some(Self { nodes, points, indices })`;

export function transformCollectionSource(original, mode) {
  if (!['baseline', 'unzip'].includes(mode)) throw new Error('Unknown collection mode');
  const source = original.replaceAll('\r\n', '\n');
  if (source.split(collectionBefore).length !== 2) throw new Error('KD tree collection source changed');
  return mode === 'baseline' ? source : source.replace(collectionBefore, collectionAfter);
}
