//! Nested octree for level-of-detail rendering (Potree-style).
//!
//! Every point is stored in exactly one node. An inner node keeps a
//! spatially uniform subsample of its cube (at most one point per cell of a
//! `grid³` lattice) and passes the remaining points on to its eight children,
//! so rendering a node together with all of its ancestors shows the cloud at
//! that node's density. Points are reordered so each node's points are one
//! contiguous range, laid out in depth-first pre-order.

/// Marker for a missing child.
pub const NO_CHILD: u32 = u32::MAX;

#[derive(Debug, Clone, PartialEq)]
pub struct OctreeNode {
    /// Range of this node's own points in [`Octree::order`].
    pub start: u32,
    pub count: u32,
    /// Minimum corner and edge length of the node's cube.
    pub min: [f64; 3],
    pub size: f64,
    pub level: u8,
    /// Child node indices by octant (`x | y << 1 | z << 2`), or [`NO_CHILD`].
    pub children: [u32; 8],
}

impl OctreeNode {
    /// Approximate distance between neighbouring points kept in this node.
    pub fn spacing(&self, grid: u32) -> f64 {
        self.size / grid as f64
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct Octree {
    /// `order[i]` is the index of the original point stored at position `i`.
    pub order: Vec<u32>,
    /// Nodes in depth-first pre-order; the root is `nodes[0]`.
    pub nodes: Vec<OctreeNode>,
    /// Subsampling lattice resolution per node edge.
    pub grid: u32,
}

/// Parameters for [`Octree::build_in_place`].
#[derive(Debug, Clone, Copy)]
pub struct OctreeParams {
    /// Nodes with at most this many points become leaves.
    pub max_leaf: usize,
    /// Subsampling lattice resolution per node edge; a power of two.
    pub grid: u32,
    /// Hard depth limit, which also bounds recursion on duplicate points.
    /// Capped at [`MORTON_BITS`] levels.
    pub max_depth: u8,
}

impl Default for OctreeParams {
    fn default() -> Self {
        Self {
            max_leaf: 20_000,
            grid: 128,
            max_depth: 20,
        }
    }
}

/// Quantisation bits per axis of the Morton codes used during the build.
pub const MORTON_BITS: u32 = 21;

impl Octree {
    /// Build an octree over `points`, reordering them in place into octree
    /// order (`order[i]` is the original index of the point now at `i`).
    ///
    /// The build works on 64-bit Morton codes of the points (quantised to
    /// [`MORTON_BITS`] bits per axis within the root cube): octants and
    /// lattice cells are just bit fields of the code, so no floating point is
    /// needed per level. Points move together with their codes; the octant
    /// partition writes eight sequential streams, which stays cache-friendly
    /// even for randomly ordered input (a single final gather would not).
    ///
    /// Returns `None` for an empty input, more than `u32::MAX` points, or a
    /// `grid` that is not a power of two.
    pub fn build_in_place(points: &mut [[f64; 3]], params: OctreeParams) -> Option<Self> {
        Self::build(points, None, params)
    }

    /// Like [`Octree::build_in_place`], reordering the cloud's colors along
    /// with its positions.
    pub fn build_for_cloud(cloud: &mut crate::PointCloud, params: OctreeParams) -> Option<Self> {
        let (lo, size) = cube(&cloud.positions);
        let attributes = cloud.attributes.iter_mut().map(|a| &mut a.values).collect();
        Self::build_cube(
            &mut cloud.positions,
            cloud.colors.as_deref_mut(),
            attributes,
            lo,
            size,
            0,
            params,
        )
    }

    fn build(
        points: &mut [[f64; 3]],
        colors: Option<&mut [[u8; 3]]>,
        params: OctreeParams,
    ) -> Option<Self> {
        let (lo, size) = cube(points);
        Self::build_cube(points, colors, Vec::new(), lo, size, 0, params)
    }

    /// Build over a given cube starting at `level`. Colors and attributes are
    /// reordered together with the points.
    #[allow(clippy::too_many_arguments)]
    fn build_cube(
        points: &mut [[f64; 3]],
        colors: Option<&mut [[u8; 3]]>,
        attributes: Vec<&mut crate::AttributeValues>,
        lo: [f64; 3],
        size: f64,
        level: u8,
        params: OctreeParams,
    ) -> Option<Self> {
        if colors.as_ref().is_some_and(|c| c.len() != points.len())
            || attributes.iter().any(|a| a.len() != points.len())
        {
            return None;
        }
        if points.is_empty() || points.len() > u32::MAX as usize || !params.grid.is_power_of_two() {
            return None;
        }
        let cells = (1u64 << MORTON_BITS) as f64;
        let scale = cells / size;
        let quantize = |v: f64, a: usize| -> u64 {
            (((v - lo[a]) * scale) as u64).min((1 << MORTON_BITS) - 1)
        };
        let codes: Vec<u64> = points
            .iter()
            .map(|p| {
                spread_bits(quantize(p[0], 0))
                    | spread_bits(quantize(p[1], 1)) << 1
                    | spread_bits(quantize(p[2], 2)) << 2
            })
            .collect();
        let grid_bits = params.grid.trailing_zeros();
        let order = (0..points.len() as u32).collect();
        let mut builder = Builder {
            points,
            colors,
            attributes,
            codes,
            order,
            params: OctreeParams {
                // The codes resolve MORTON_BITS levels below the build cube.
                max_depth: params
                    .max_depth
                    .min(level.saturating_add(MORTON_BITS as u8)),
                ..params
            },
            base_level: level,
            grid_bits,
            stamps: vec![0; 1usize << (3 * grid_bits)],
            stamp: 0,
            nodes: Vec::new(),
        };
        let len = builder.codes.len();
        builder.node(0, len, lo, size, level);
        Some(Self {
            order: builder.order,
            nodes: builder.nodes,
            grid: params.grid,
        })
    }
}

/// Insert two zero bits between each of the low 21 bits of `v`.
fn spread_bits(v: u64) -> u64 {
    let mut x = v & 0x1f_ffff;
    x = (x | x << 32) & 0x1f_0000_0000_ffff;
    x = (x | x << 16) & 0x1f_0000_ff00_00ff;
    x = (x | x << 8) & 0x100f_00f0_0f00_f00f;
    x = (x | x << 4) & 0x10c3_0c30_c30c_30c3;
    x = (x | x << 2) & 0x1249_2492_4924_9249;
    x
}

struct Builder<'a> {
    points: &'a mut [[f64; 3]],
    colors: Option<&'a mut [[u8; 3]]>,
    attributes: Vec<&'a mut crate::AttributeValues>,
    /// Morton code of each point, permuted together with `points` and `order`.
    codes: Vec<u64>,
    order: Vec<u32>,
    params: OctreeParams,
    /// Level of the build cube; code bits are relative to it.
    base_level: u8,
    grid_bits: u32,
    /// Per-cell "last seen" marker, reused across nodes to avoid clearing.
    stamps: Vec<u32>,
    stamp: u32,
    nodes: Vec<OctreeNode>,
}

impl Builder<'_> {
    fn swap(&mut self, a: usize, b: usize) {
        self.points.swap(a, b);
        if let Some(colors) = self.colors.as_deref_mut() {
            colors.swap(a, b);
        }
        for values in &mut self.attributes {
            values.swap(a, b);
        }
        self.codes.swap(a, b);
        self.order.swap(a, b);
    }

    /// Build the node for `codes[lo..hi]` and return its index.
    fn node(&mut self, lo: usize, hi: usize, min: [f64; 3], size: f64, level: u8) -> u32 {
        let id = self.nodes.len();
        self.nodes.push(OctreeNode {
            start: lo as u32,
            count: (hi - lo) as u32,
            min,
            size,
            level,
            children: [NO_CHILD; 8],
        });
        if hi - lo <= self.params.max_leaf || level >= self.params.max_depth {
            return id as u32;
        }

        // Keep the first point of every occupied lattice cell here, moving
        // those points to the front of the range. The cell is the next
        // `grid_bits` octree levels below this node, i.e. a bit field of the
        // code (coarser near the bottom where fewer bits remain).
        self.stamp = self.stamp.wrapping_add(1);
        if self.stamp == 0 {
            self.stamps.fill(0);
            self.stamp = 1;
        }
        let below = MORTON_BITS - u32::from(level - self.base_level);
        let cell_levels = self.grid_bits.min(below);
        let shift = 3 * (below - cell_levels);
        let mask = (1u64 << (3 * cell_levels)) - 1;
        let mut kept = lo;
        for i in lo..hi {
            let key = ((self.codes[i] >> shift) & mask) as usize;
            if self.stamps[key] != self.stamp {
                self.stamps[key] = self.stamp;
                self.swap(kept, i);
                kept += 1;
            }
        }
        self.nodes[id].count = (kept - lo) as u32;

        // In-place 8-way partition of the remaining points by octant
        // (American flag sort) on the next three code bits.
        let octant_shift = 3 * (below - 1);
        let octant = |code: u64| ((code >> octant_shift) & 7) as usize;
        let mut counts = [0usize; 8];
        for &code in &self.codes[kept..hi] {
            counts[octant(code)] += 1;
        }
        let mut starts = [0usize; 8];
        let mut acc = kept;
        for o in 0..8 {
            starts[o] = acc;
            acc += counts[o];
        }
        let mut next = starts;
        for o in 0..8 {
            let end = starts[o] + counts[o];
            while next[o] < end {
                let target = octant(self.codes[next[o]]);
                if target == o {
                    next[o] += 1;
                } else {
                    self.swap(next[o], next[target]);
                    next[target] += 1;
                }
            }
        }

        let half = 0.5 * size;
        for o in 0..8 {
            if counts[o] == 0 {
                continue;
            }
            let child_min: [f64; 3] =
                std::array::from_fn(|a| min[a] + if o >> a & 1 == 1 { half } else { 0.0 });
            let child = self.node(starts[o], starts[o] + counts[o], child_min, half, level + 1);
            self.nodes[id].children[o] = child;
        }
        id as u32
    }
}

/// Minimum corner and edge length of the cube enclosing `points`.
fn cube(points: &[[f64; 3]]) -> ([f64; 3], f64) {
    let (lo, hi) = bounds(points);
    let size = (0..3)
        .map(|a| hi[a] - lo[a])
        .fold(0.0f64, f64::max)
        .max(f64::MIN_POSITIVE);
    (lo, size)
}

fn bounds(points: &[[f64; 3]]) -> ([f64; 3], [f64; 3]) {
    let mut lo = [f64::INFINITY; 3];
    let mut hi = [f64::NEG_INFINITY; 3];
    for p in points {
        for a in 0..3 {
            lo[a] = lo[a].min(p[a]);
            hi[a] = hi[a].max(p[a]);
        }
    }
    (lo, hi)
}

// ---------------------------------------------------------------- bucketed build
//
// A build whose every heavy step runs on independent slices, for workers
// that share no memory:
//
// 1. [`bucket_order`] on contiguous chunks of the input: a stable sort by the
//    point's level-2 node (64 "buckets") under the root cube.
// 2. Concatenating each bucket's pieces in chunk order keeps the input order
//    within a bucket, so [`build_bucket`] on one bucket can pick the root's
//    and its level-1 node's subsample (their lattice cells never straddle
//    buckets) and build the level-2 subtree over the rest.
// 3. [`BucketLayout`] places every bucket's three sections and stitches the
//    nodes together.
//
// The root keeps exactly the points a plain build keeps; lower levels may pick
// other (equally valid) representatives.

/// Number of level-2 nodes, i.e. buckets.
pub const BUCKETS: usize = 64;

/// Minimum corner and edge of the root cube of `points`, as used by a build.
pub fn root_cube(points: &[[f64; 3]]) -> ([f64; 3], f64) {
    cube(points)
}

/// Morton codes of `points` relative to the root cube.
fn codes_in(points: &[[f64; 3]], lo: [f64; 3], size: f64) -> Vec<u64> {
    let scale = (1u64 << MORTON_BITS) as f64 / size;
    let quantize =
        |v: f64, a: usize| -> u64 { (((v - lo[a]) * scale) as u64).min((1 << MORTON_BITS) - 1) };
    points
        .iter()
        .map(|p| {
            spread_bits(quantize(p[0], 0))
                | spread_bits(quantize(p[1], 1)) << 1
                | spread_bits(quantize(p[2], 2)) << 2
        })
        .collect()
}

/// Bucket (level-0 octant << 3 | level-1 octant) of a Morton code.
fn bucket_of(code: u64) -> usize {
    (code >> (3 * (MORTON_BITS - 2))) as usize & (BUCKETS - 1)
}

/// Step 1: the stable order of `points` by bucket, and the size of each bucket.
pub fn bucket_order(points: &[[f64; 3]], lo: [f64; 3], size: f64) -> (Vec<u32>, [u32; BUCKETS]) {
    let buckets: Vec<u8> = codes_in(points, lo, size)
        .into_iter()
        .map(|c| bucket_of(c) as u8)
        .collect();
    let mut counts = [0u32; BUCKETS];
    for &b in &buckets {
        counts[b as usize] += 1;
    }
    let mut next = [0u32; BUCKETS];
    let mut acc = 0;
    for b in 0..BUCKETS {
        next[b] = acc;
        acc += counts[b];
    }
    let mut order = vec![0u32; points.len()];
    for (i, &b) in buckets.iter().enumerate() {
        order[next[b as usize] as usize] = i as u32;
        next[b as usize] += 1;
    }
    (order, counts)
}

/// Cube of bucket `key` under the root cube.
pub fn bucket_cube(lo: [f64; 3], size: f64, key: usize) -> ([f64; 3], f64) {
    let (outer, inner) = (key >> 3, key & 7);
    let min = std::array::from_fn(|a| {
        lo[a]
            + if outer >> a & 1 == 1 { 0.5 * size } else { 0.0 }
            + if inner >> a & 1 == 1 {
                0.25 * size
            } else {
                0.0
            }
    });
    (min, 0.25 * size)
}

/// Result of [`build_bucket`]: the bucket's points were reordered to
/// `[root's | level-1 node's | subtree]`.
#[derive(Debug, Clone, PartialEq)]
pub struct Bucket {
    pub root_kept: usize,
    pub level1_kept: usize,
    /// Nodes of the level-2 subtree over the remaining points (ranges relative
    /// to them); empty when no points remain.
    pub nodes: Vec<OctreeNode>,
    /// `order[i]` is the bucket-local index of the point now at `i`.
    pub order: Vec<u32>,
}

/// Step 2: one bucket's points (in input order), reordered in place.
pub fn build_bucket(
    points: &mut [[f64; 3]],
    mut colors: Option<&mut [[u8; 3]]>,
    lo: [f64; 3],
    size: f64,
    key: usize,
    params: OctreeParams,
) -> Option<Bucket> {
    if key >= BUCKETS
        || !params.grid.is_power_of_two()
        || params.max_depth < 2
        || colors.as_ref().is_some_and(|c| c.len() != points.len())
    {
        return None;
    }
    let codes = codes_in(points, lo, size);
    let grid_bits = params.grid.trailing_zeros();
    // The lattice cell of a node at `level`, as in `Builder::node`.
    let cell = |code: u64, level: u32| {
        let below = MORTON_BITS - level;
        let levels = grid_bits.min(below);
        ((code >> (3 * (below - levels))) & ((1u64 << (3 * levels)) - 1)) as usize
    };
    let mut seen = vec![0u64; (1usize << (3 * grid_bits)).div_ceil(64)];
    let first = |c: usize, seen: &mut [u64]| {
        let (w, b) = (c / 64, 1u64 << (c % 64));
        let new = seen[w] & b == 0;
        seen[w] |= b;
        new
    };
    let mut section = vec![2u8; points.len()];
    for (i, &code) in codes.iter().enumerate() {
        if first(cell(code, 0), &mut seen) {
            section[i] = 0;
        }
    }
    seen.fill(0);
    for (i, &code) in codes.iter().enumerate() {
        if section[i] == 2 && first(cell(code, 1), &mut seen) {
            section[i] = 1;
        }
    }
    // Stable three-way split.
    let mut order: Vec<u32> = Vec::with_capacity(points.len());
    for s in 0..3 {
        order.extend((0..points.len() as u32).filter(|&i| section[i as usize] == s));
    }
    let root_kept = section.iter().filter(|&&s| s == 0).count();
    let level1_kept = section.iter().filter(|&&s| s == 1).count();
    let gather = |v: &mut [[f64; 3]]| {
        let old = v.to_vec();
        for (dst, &o) in v.iter_mut().zip(&order) {
            *dst = old[o as usize];
        }
    };
    gather(points);
    if let Some(c) = colors.as_deref_mut() {
        let old = c.to_vec();
        for (dst, &o) in c.iter_mut().zip(&order) {
            *dst = old[o as usize];
        }
    }

    let rest = root_kept + level1_kept..points.len();
    let mut nodes = Vec::new();
    if !rest.is_empty() {
        let (min, sub_size) = bucket_cube(lo, size, key);
        let tree = Octree::build_cube(
            &mut points[rest.clone()],
            colors.map(|c| &mut c[rest.clone()]),
            Vec::new(),
            min,
            sub_size,
            2,
            params,
        )?;
        let head = order[rest.clone()].to_vec();
        for (dst, &o) in order[rest].iter_mut().zip(&tree.order) {
            *dst = head[o as usize];
        }
        nodes = tree.nodes;
    }
    Some(Bucket {
        root_kept,
        level1_kept,
        nodes,
        order,
    })
}

/// Step 3: where each bucket's sections go, and the final octree.
///
/// Layout: the root's points of every bucket (in bucket order), then for each
/// level-1 node its points from each of its buckets, then those buckets'
/// subtrees.
#[derive(Debug, Clone)]
pub struct BucketLayout {
    lo: [f64; 3],
    size: f64,
    params: OctreeParams,
    sizes: [u32; BUCKETS],
    root_kept: [u32; BUCKETS],
    level1_kept: [u32; BUCKETS],
    /// Destination of each bucket's three sections.
    offsets: [[usize; 3]; BUCKETS],
    order: Vec<u32>,
    subtrees: Vec<Option<Vec<OctreeNode>>>,
}

impl BucketLayout {
    /// `sizes`, `root_kept` and `level1_kept` per bucket, from
    /// [`build_bucket`]. Returns `None` if they are inconsistent.
    pub fn new(
        lo: [f64; 3],
        size: f64,
        params: OctreeParams,
        sizes: [u32; BUCKETS],
        root_kept: [u32; BUCKETS],
        level1_kept: [u32; BUCKETS],
    ) -> Option<Self> {
        if (0..BUCKETS).any(|b| root_kept[b] as u64 + level1_kept[b] as u64 > sizes[b] as u64) {
            return None;
        }
        let mut offsets = [[0usize; 3]; BUCKETS];
        let mut at = 0usize;
        for b in 0..BUCKETS {
            offsets[b][0] = at;
            at += root_kept[b] as usize;
        }
        for outer in 0..8 {
            let keys = outer * 8..outer * 8 + 8;
            for b in keys.clone() {
                offsets[b][1] = at;
                at += level1_kept[b] as usize;
            }
            for b in keys {
                offsets[b][2] = at;
                at += (sizes[b] - root_kept[b] - level1_kept[b]) as usize;
            }
        }
        if at > u32::MAX as usize {
            return None;
        }
        Some(Self {
            lo,
            size,
            params,
            sizes,
            root_kept,
            level1_kept,
            offsets,
            order: vec![0; at],
            subtrees: vec![None; BUCKETS],
        })
    }

    /// Total number of points.
    pub fn len(&self) -> usize {
        self.order.len()
    }

    pub fn is_empty(&self) -> bool {
        self.order.is_empty()
    }

    /// Place bucket `key` (as reordered by [`build_bucket`]) into `positions`
    /// and `colors` of the whole cloud. `order` gives the original index of
    /// each of its points.
    #[allow(clippy::too_many_arguments)]
    pub fn put(
        &mut self,
        key: usize,
        positions: &mut [[f64; 3]],
        colors: Option<&mut [[u8; 3]]>,
        bucket_positions: &[[f64; 3]],
        bucket_colors: Option<&[[u8; 3]]>,
        order: &[u32],
        nodes: Vec<OctreeNode>,
    ) -> Option<()> {
        let n = *self.sizes.get(key)? as usize;
        if bucket_positions.len() != n
            || order.len() != n
            || positions.len() != self.len()
            || bucket_colors.is_some_and(|c| c.len() != n)
        {
            return None;
        }
        let cuts = [
            0,
            self.root_kept[key] as usize,
            (self.root_kept[key] + self.level1_kept[key]) as usize,
            n,
        ];
        let mut colors = colors;
        for s in 0..3 {
            let (from, to) = (cuts[s]..cuts[s + 1], self.offsets[key][s]);
            let dst = to..to + from.len();
            positions[dst.clone()].copy_from_slice(&bucket_positions[from.clone()]);
            if let (Some(c), Some(src)) = (colors.as_deref_mut(), bucket_colors) {
                c[dst.clone()].copy_from_slice(&src[from.clone()]);
            }
            self.order[dst].copy_from_slice(&order[from]);
        }
        self.subtrees[key] = Some(nodes);
        Some(())
    }

    /// The octree, once every non-empty bucket was [`put`](Self::put).
    /// Reorders the cloud's attributes to match.
    pub fn finish(self, attributes: &mut [crate::Attribute]) -> Option<Octree> {
        let missing = (0..BUCKETS).any(|b| self.sizes[b] > 0 && self.subtrees[b].is_none());
        if missing || self.is_empty() {
            return None;
        }
        for a in attributes.iter_mut() {
            if a.values.len() != self.len() {
                return None;
            }
            a.values.permute_range(0..self.len(), &self.order);
        }
        let root_count: u32 = self.root_kept.iter().sum();
        let mut nodes = vec![OctreeNode {
            start: 0,
            count: root_count,
            min: self.lo,
            size: self.size,
            level: 0,
            children: [NO_CHILD; 8],
        }];
        let half = 0.5 * self.size;
        let mut subtrees = self.subtrees;
        for outer in 0..8 {
            let keys = outer * 8..outer * 8 + 8;
            let total: u32 = keys
                .clone()
                .map(|b| self.sizes[b] - self.root_kept[b])
                .sum();
            if total == 0 {
                continue;
            }
            let id = nodes.len() as u32;
            nodes[0].children[outer] = id;
            let leaf = total as usize <= self.params.max_leaf || self.params.max_depth <= 1;
            nodes.push(OctreeNode {
                start: self.offsets[outer * 8][1] as u32,
                count: if leaf {
                    total
                } else {
                    keys.clone().map(|b| self.level1_kept[b]).sum()
                },
                min: std::array::from_fn(|a| {
                    self.lo[a] + if outer >> a & 1 == 1 { half } else { 0.0 }
                }),
                size: half,
                level: 1,
                children: [NO_CHILD; 8],
            });
            if leaf {
                continue;
            }
            for b in keys {
                let sub = subtrees[b].take().unwrap_or_default();
                if sub.is_empty() {
                    continue;
                }
                let base = nodes.len() as u32;
                let start = self.offsets[b][2] as u32;
                nodes[id as usize].children[b & 7] = base;
                nodes.extend(sub.into_iter().map(|mut n| {
                    n.start += start;
                    n.children = n.children.map(|c| if c == NO_CHILD { c } else { base + c });
                    n
                }));
            }
        }
        Some(Octree {
            order: self.order,
            nodes,
            grid: self.params.grid,
        })
    }
}

impl Octree {
    /// The bucketed build run here (chunks and buckets in turn; with the
    /// `parallel` feature, on the rayon pool). Reorders the cloud like
    /// [`Octree::build_for_cloud`].
    pub fn build_bucketed(cloud: &mut crate::PointCloud, params: OctreeParams) -> Option<Self> {
        if cloud.is_empty()
            || cloud.len() > u32::MAX as usize
            || cloud
                .colors
                .as_ref()
                .is_some_and(|c| c.len() != cloud.len())
        {
            return None;
        }
        if cloud.len() <= params.max_leaf || params.max_depth < 2 {
            return Self::build_for_cloud(cloud, params);
        }
        let (lo, size) = root_cube(&cloud.positions);
        let (order, sizes) = bucket_order(&cloud.positions, lo, size);
        // Gather each bucket's points (input order within a bucket).
        let mut starts = [0usize; BUCKETS + 1];
        for b in 0..BUCKETS {
            starts[b + 1] = starts[b] + sizes[b] as usize;
        }
        type Built = (Vec<[f64; 3]>, Option<Vec<[u8; 3]>>, Vec<u32>, Bucket);
        let run = |b: usize| -> Option<Built> {
            let idx = &order[starts[b]..starts[b + 1]];
            let mut positions: Vec<[f64; 3]> =
                idx.iter().map(|&i| cloud.positions[i as usize]).collect();
            let mut colors: Option<Vec<[u8; 3]>> = cloud
                .colors
                .as_ref()
                .map(|c| idx.iter().map(|&i| c[i as usize]).collect());
            let bucket = build_bucket(&mut positions, colors.as_deref_mut(), lo, size, b, params)?;
            let global = bucket.order.iter().map(|&o| idx[o as usize]).collect();
            Some((positions, colors, global, bucket))
        };
        let keys: Vec<usize> = (0..BUCKETS).filter(|&b| sizes[b] > 0).collect();
        #[cfg(feature = "parallel")]
        let built: Vec<_> = {
            use rayon::prelude::*;
            keys.par_iter().map(|&b| run(b)).collect()
        };
        #[cfg(not(feature = "parallel"))]
        let built: Vec<_> = keys.iter().map(|&b| run(b)).collect();

        let mut root_kept = [0u32; BUCKETS];
        let mut level1_kept = [0u32; BUCKETS];
        let mut done = Vec::with_capacity(built.len());
        for (&b, result) in keys.iter().zip(built) {
            let (positions, colors, global, bucket) = result?;
            root_kept[b] = bucket.root_kept as u32;
            level1_kept[b] = bucket.level1_kept as u32;
            done.push((b, positions, colors, global, bucket.nodes));
        }
        let mut layout = BucketLayout::new(lo, size, params, sizes, root_kept, level1_kept)?;
        for (b, positions, colors, global, nodes) in done {
            layout.put(
                b,
                &mut cloud.positions,
                cloud.colors.as_deref_mut(),
                &positions,
                colors.as_deref(),
                &global,
                nodes,
            )?;
        }
        layout.finish(&mut cloud.attributes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn grid_points(n: usize) -> Vec<[f64; 3]> {
        (0..n * n)
            .map(|i| {
                [
                    (i % n) as f64 * 0.1,
                    (i / n) as f64 * 0.1,
                    ((i % 7) as f64).sin(),
                ]
            })
            .collect()
    }

    fn params(max_leaf: usize, grid: u32) -> OctreeParams {
        OctreeParams {
            max_leaf,
            grid,
            max_depth: 20,
        }
    }

    #[test]
    fn every_point_is_stored_once_inside_its_node() {
        let original = grid_points(200);
        let mut points = original.clone();
        let tree = Octree::build_in_place(&mut points, params(500, 16)).unwrap();
        for (i, &o) in tree.order.iter().enumerate() {
            assert_eq!(points[i], original[o as usize]);
        }
        let mut seen = tree.order.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..points.len() as u32).collect::<Vec<_>>());
        let total: u32 = tree.nodes.iter().map(|n| n.count).sum();
        assert_eq!(total as usize, points.len());
        for node in &tree.nodes {
            for p in &points[node.start as usize..(node.start + node.count) as usize] {
                for (a, &v) in p.iter().enumerate() {
                    assert!(v >= node.min[a] - 1e-9 && v <= node.min[a] + node.size + 1e-9);
                }
            }
        }
    }

    #[test]
    fn subtrees_are_contiguous_in_preorder() {
        let mut points = grid_points(150);
        let tree = Octree::build_in_place(&mut points, params(300, 8)).unwrap();
        // A node's subtree spans from its own start to the end of its last descendant.
        fn end(tree: &Octree, id: usize) -> u32 {
            let node = &tree.nodes[id];
            node.children
                .iter()
                .filter(|&&c| c != NO_CHILD)
                .map(|&c| end(tree, c as usize))
                .max()
                .unwrap_or(node.start + node.count)
        }
        assert_eq!(end(&tree, 0) as usize, points.len());
        for (id, node) in tree.nodes.iter().enumerate() {
            let mut next = node.start + node.count;
            for &c in node.children.iter().filter(|&&c| c != NO_CHILD) {
                assert_eq!(tree.nodes[c as usize].start, next, "node {id}");
                assert!(c as usize > id);
                next = end(&tree, c as usize);
            }
        }
    }

    #[test]
    fn inner_nodes_keep_at_most_one_point_per_cell() {
        let mut points = grid_points(300);
        let tree = Octree::build_in_place(&mut points, params(1000, 32)).unwrap();
        let root = &tree.nodes[0];
        assert!(root.children.iter().any(|&c| c != NO_CHILD));
        assert!(root.count as usize <= 32 * 32 * 32);
        assert!(root.count > 0);
    }

    #[test]
    fn attributes_follow_their_points() {
        let original = grid_points(80);
        let n = original.len();
        let mut cloud = crate::PointCloud {
            positions: original.clone(),
            colors: None,
            attributes: vec![
                crate::Attribute {
                    name: crate::INTENSITY.into(),
                    values: crate::AttributeValues::F32((0..n).map(|i| i as f32).collect()),
                },
                crate::Attribute {
                    name: crate::CLASSIFICATION.into(),
                    values: crate::AttributeValues::U8((0..n).map(|i| (i % 251) as u8).collect()),
                },
            ],
        };
        let tree = Octree::build_for_cloud(&mut cloud, params(200, 8)).unwrap();
        let crate::AttributeValues::F32(intensity) = &cloud.attributes[0].values else {
            panic!()
        };
        let crate::AttributeValues::U8(class) = &cloud.attributes[1].values else {
            panic!()
        };
        for (i, &o) in tree.order.iter().enumerate() {
            assert_eq!(cloud.positions[i], original[o as usize]);
            assert_eq!(intensity[i], o as f32);
            assert_eq!(class[i], (o % 251) as u8);
        }
    }

    #[test]
    fn colors_follow_their_points() {
        let original = grid_points(120);
        let mut cloud = crate::PointCloud {
            positions: original.clone(),
            colors: Some(
                (0..original.len())
                    .map(|i| [i as u8, (i >> 8) as u8, (i >> 16) as u8])
                    .collect(),
            ),
            attributes: Vec::new(),
        };
        let tree = Octree::build_for_cloud(&mut cloud, params(400, 16)).unwrap();
        let colors = cloud.colors.unwrap();
        for (i, &o) in tree.order.iter().enumerate() {
            assert_eq!(cloud.positions[i], original[o as usize]);
            assert_eq!(colors[i], [o as u8, (o >> 8) as u8, (o >> 16) as u8]);
        }
    }

    #[test]
    fn rejects_non_power_of_two_grid() {
        let mut points = grid_points(10);
        assert!(Octree::build_in_place(&mut points, params(10, 100)).is_none());
    }

    #[test]
    fn duplicates_stop_at_max_depth() {
        let mut points = vec![[1.0, 1.0, 1.0]; 5000];
        let tree = Octree::build_in_place(&mut points, params(10, 4)).unwrap();
        assert!(tree.nodes.iter().all(|n| n.level <= 20));
        let total: u32 = tree.nodes.iter().map(|n| n.count).sum();
        assert_eq!(total, 5000);
    }

    /// Structural checks shared by the build variants.
    fn assert_valid(tree: &Octree, original: &[[f64; 3]], positions: &[[f64; 3]]) {
        let mut seen = tree.order.clone();
        seen.sort_unstable();
        assert_eq!(seen, (0..original.len() as u32).collect::<Vec<_>>());
        for (i, &o) in tree.order.iter().enumerate() {
            assert_eq!(positions[i], original[o as usize]);
        }
        let total: u32 = tree.nodes.iter().map(|n| n.count).sum();
        assert_eq!(total as usize, original.len());
        fn end(tree: &Octree, id: usize) -> u32 {
            let node = &tree.nodes[id];
            node.children
                .iter()
                .filter(|&&c| c != NO_CHILD)
                .map(|&c| end(tree, c as usize))
                .max()
                .unwrap_or(node.start + node.count)
        }
        assert_eq!(end(tree, 0) as usize, original.len());
        for (id, node) in tree.nodes.iter().enumerate() {
            for p in &positions[node.start as usize..(node.start + node.count) as usize] {
                for (a, &v) in p.iter().enumerate() {
                    assert!(v >= node.min[a] - 1e-9 && v <= node.min[a] + node.size + 1e-9);
                }
            }
            let mut next = node.start + node.count;
            for &c in node.children.iter().filter(|&&c| c != NO_CHILD) {
                let child = &tree.nodes[c as usize];
                assert!(c as usize > id);
                assert_eq!(child.start, next, "node {id}");
                assert_eq!(child.level, node.level + 1);
                assert!((child.size - node.size / 2.0).abs() < 1e-12);
                next = end(tree, c as usize);
            }
        }
    }

    fn scattered(n: usize, seed: u64) -> Vec<[f64; 3]> {
        let mut s = seed | 1;
        let mut next = move || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s >> 11) as f64 / (1u64 << 53) as f64
        };
        // Dense near one corner, so buckets are uneven and some are empty.
        (0..n)
            .map(|_| {
                let r = next().powi(3);
                [r * 40.0, next() * 10.0 * r, next() * 3.0]
            })
            .collect()
    }

    #[test]
    fn bucketed_build_is_a_valid_octree_with_the_plain_root() {
        for (original, p) in [
            (grid_points(250), params(300, 16)),
            (scattered(60_000, 3), params(500, 32)),
            (scattered(20_000, 5), params(2_000, 8)),
        ] {
            let n = original.len();
            let mut cloud = crate::PointCloud {
                positions: original.clone(),
                colors: Some(
                    (0..n)
                        .map(|i| [i as u8, (i >> 8) as u8, (i >> 16) as u8])
                        .collect(),
                ),
                attributes: vec![crate::Attribute {
                    name: crate::INTENSITY.into(),
                    values: crate::AttributeValues::F32((0..n).map(|i| i as f32).collect()),
                }],
            };
            let tree = Octree::build_bucketed(&mut cloud, p).unwrap();
            assert_valid(&tree, &original, &cloud.positions);
            let colors = cloud.colors.as_ref().unwrap();
            let crate::AttributeValues::F32(intensity) = &cloud.attributes[0].values else {
                panic!()
            };
            for (i, &o) in tree.order.iter().enumerate() {
                assert_eq!(colors[i], [o as u8, (o >> 8) as u8, (o >> 16) as u8]);
                assert_eq!(intensity[i], o as f32);
            }
            let mut again = original.clone();
            let plain = Octree::build_in_place(&mut again, p).unwrap();
            assert_valid(&plain, &original, &again);
            // The root keeps the same points as a plain build.
            let root = |t: &Octree| {
                let mut r = t.order[..t.nodes[0].count as usize].to_vec();
                r.sort_unstable();
                r
            };
            assert_eq!(root(&tree), root(&plain));
        }
    }

    #[test]
    fn bucket_order_of_chunks_concatenates_to_the_whole() {
        let points = scattered(10_000, 9);
        let (lo, size) = root_cube(&points);
        let (whole, counts) = bucket_order(&points, lo, size);
        let chunks: Vec<_> = points
            .chunks(3_001)
            .map(|c| bucket_order(c, lo, size))
            .collect();
        let mut joined = Vec::new();
        for b in 0..BUCKETS {
            let mut offset = 0u32;
            for (k, (order, c)) in chunks.iter().enumerate() {
                let start: u32 = c[..b].iter().sum();
                joined.extend(
                    order[start as usize..(start + c[b]) as usize]
                        .iter()
                        .map(|&i| i + offset),
                );
                offset = ((k + 1) * 3_001).min(points.len()) as u32;
            }
        }
        assert_eq!(joined, whole);
        assert_eq!(counts.iter().sum::<u32>() as usize, points.len());
    }

    #[test]
    fn small_clouds_fall_back_to_the_plain_build() {
        let original = grid_points(20);
        let mut cloud = crate::PointCloud {
            positions: original.clone(),
            colors: None,
            attributes: Vec::new(),
        };
        let tree = Octree::build_bucketed(&mut cloud, params(1_000, 8)).unwrap();
        assert_eq!(tree.nodes.len(), 1);
        assert_valid(&tree, &original, &cloud.positions);
    }
}
