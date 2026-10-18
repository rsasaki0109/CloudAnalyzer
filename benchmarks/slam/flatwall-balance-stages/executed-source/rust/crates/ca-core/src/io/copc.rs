//! COPC (Cloud Optimized Point Cloud): a LAZ 1.4 file whose points form an
//! octree of independently compressed chunks, listed in a hierarchy of pages,
//! so a reader can fetch just the nodes it needs (e.g. with HTTP range
//! requests). Coarse levels hold a spatially uniform subsample of the whole
//! cloud, so reading every node down to some level gives an even density.

use std::collections::{HashMap, HashSet};

use laz::record::{LayeredPointRecordDecompressor, RecordDecompressor};

use super::IoError;
use super::las::{LasDecoder, LasHeader};

const FORMAT: &str = "COPC";
/// Size of a LAS 1.4 header, where the COPC info VLR must start.
const HEADER_SIZE: usize = 375;
const VLR_HEADER: usize = 54;

/// What a reader needs from the file's first bytes.
pub struct CopcHeader {
    las: LasHeader,
    laz: laz::LazVlr,
    /// Octree root cube: centre and half its edge.
    pub center: [f64; 3],
    pub halfsize: f64,
    /// Point spacing at the root.
    pub spacing: f64,
    /// Byte range of the root hierarchy page.
    pub root_page: (u64, u64),
    /// Points in the file.
    pub total_points: u64,
}

impl CopcHeader {
    /// Bytes in each uncompressed LAS point record.
    pub fn record_len(&self) -> usize {
        self.las.record_len
    }

    pub fn point_format(&self) -> u8 {
        self.las.format
    }

    pub fn scale(&self) -> [f64; 3] {
        self.las.scale
    }

    pub fn offset(&self) -> [f64; 3] {
        self.las.offset
    }

    /// Bytes from the start of the file needed by [`CopcHeader::parse`], or
    /// `None` if `head` is too short to tell (read at least 100 bytes).
    pub fn needed(head: &[u8]) -> Option<usize> {
        let at = head.get(96..100)?;
        Some(u32::from_le_bytes(at.try_into().ok()?) as usize)
    }

    /// Whether `head` starts like a COPC file (LAS 1.4 with the COPC info
    /// VLR first). Needs about 400 bytes.
    pub fn is_copc(head: &[u8]) -> bool {
        head.starts_with(b"LASF")
            && head.get(24..26) == Some(&[1, 4])
            && head
                .get(HEADER_SIZE + 2..HEADER_SIZE + 18)
                .is_some_and(|id| id.starts_with(b"copc\0"))
            && head.get(HEADER_SIZE + 18..HEADER_SIZE + 20) == Some(&[1, 0])
    }

    /// Parse the header and VLRs (the file's first [`CopcHeader::needed`] bytes).
    pub fn parse(head: &[u8]) -> Result<Self, IoError> {
        if !Self::is_copc(head) {
            return Err(IoError::header(
                FORMAT,
                "not a COPC file (no COPC info VLR)",
            ));
        }
        let las = LasHeader::parse(head)?;
        if !las.compressed {
            return Err(IoError::header(FORMAT, "points are not compressed"));
        }
        let info = head
            .get(HEADER_SIZE + VLR_HEADER..HEADER_SIZE + VLR_HEADER + 160)
            .ok_or(IoError::Truncated(FORMAT))?;
        let f = |i: usize| f64::from_le_bytes(info[8 * i..8 * i + 8].try_into().unwrap());
        let u = |i: usize| u64::from_le_bytes(info[8 * i..8 * i + 8].try_into().unwrap());
        let laz = laz::LazVlr::from_buffer(super::las::laszip_vlr(head)?)
            .map_err(|e| IoError::header(FORMAT, format!("bad LASzip VLR: {e}")))?;
        Ok(Self {
            center: [f(0), f(1), f(2)],
            halfsize: f(3),
            spacing: f(4),
            root_page: (u(5), u(6)),
            total_points: las.count,
            las,
            laz,
        })
    }

    /// Decompress one node's chunk of `count` points.
    pub fn decode_node(&self, chunk: &[u8], count: usize) -> Result<CopcPoints, IoError> {
        let record_len = self.las.record_len;
        let records = self.decode_records(chunk, count, isize::MAX as u64)?;
        let mut decoder = LasDecoder::new(self.las.clone(), count)?;
        for record in records.chunks_exact(record_len) {
            decoder.decode(record);
        }
        Ok(decoder.into_raw())
    }

    /// Original LAS records, including GPS time, return flags, 16-bit RGB,
    /// NIR and extra bytes. Validate layer sizes before laz can allocate from
    /// them. `max_raw_bytes` bounds the output; compressed input and decoder
    /// models remain separate costs. COPC point formats 6/7/8 are supported.
    pub fn decode_records(
        &self,
        chunk: &[u8],
        count: usize,
        max_raw_bytes: u64,
    ) -> Result<Vec<u8>, IoError> {
        use laz::LazItemType;
        let record_len = self.las.record_len;
        let size = count
            .checked_mul(record_len)
            .ok_or_else(|| IoError::header(FORMAT, "node record span overflow"))?;
        if count == 0 || size as u64 > max_raw_bytes {
            return Err(IoError::header(
                FORMAT,
                "empty node or raw node exceeds byte limit",
            ));
        }
        let items = self.laz.items();
        let mut item_bytes = 0usize;
        let mut layers = 0usize;
        let base_items = if self.las.format == 6 { 1 } else { 2 };
        if !(6..=8).contains(&self.las.format)
            || !(base_items..=base_items + 1).contains(&items.len())
        {
            return Err(IoError::header(FORMAT, "unsupported COPC LASzip items"));
        }
        for (index, item) in items.iter().enumerate() {
            if !matches!(item.version(), 3 | 4) {
                return Err(IoError::header(
                    FORMAT,
                    "unsupported layered LASzip version",
                ));
            }
            let n = match (index, self.las.format, item.item_type()) {
                (0, _, LazItemType::Point14) if item.size() == 30 => 9,
                (1, 7, LazItemType::RGB14) if item.size() == 6 => 1,
                (1, 8, LazItemType::RGBNIR14) if item.size() == 8 => 2,
                (i, _, LazItemType::Byte14(n)) if i == base_items && n > 0 && item.size() == n => {
                    n as usize
                }
                _ => {
                    return Err(IoError::header(
                        FORMAT,
                        "LASzip item does not match the point format",
                    ));
                }
            };
            layers += n;
            item_bytes += item.size() as usize;
        }
        if item_bytes != record_len {
            return Err(IoError::header(
                FORMAT,
                "LASzip record size does not match LAS header",
            ));
        }
        let prefix = record_len + 4 + layers * 4;
        let header = chunk.get(..prefix).ok_or(IoError::Truncated(FORMAT))?;
        let encoded_count =
            u32::from_le_bytes(header[record_len..record_len + 4].try_into().unwrap());
        if encoded_count as u64 != count as u64 {
            return Err(IoError::header(
                FORMAT,
                "layered chunk count does not match hierarchy",
            ));
        }
        let end = header[record_len + 4..]
            .as_chunks::<4>()
            .0
            .iter()
            .try_fold(prefix as u64, |sum, word| {
                sum.checked_add(u32::from_le_bytes(*word) as u64)
            })
            .ok_or_else(|| IoError::header(FORMAT, "layer byte span overflow"))?;
        if end > chunk.len() as u64 {
            return Err(IoError::header(
                FORMAT,
                "layer sizes exceed the compressed node",
            ));
        }
        let mut decompressor = LayeredPointRecordDecompressor::new(std::io::Cursor::new(chunk));
        decompressor
            .set_fields_from(items)
            .map_err(|e| IoError::Unsupported(format!("COPC: {e}")))?;
        let mut records = Vec::new();
        records
            .try_reserve_exact(size)
            .map_err(|e| IoError::Unsupported(format!("COPC allocation: {e}")))?;
        records.resize(size, 0);
        decompressor
            .decompress_many(&mut records)
            .map_err(|e| IoError::Unsupported(format!("COPC: {e}")))?;
        Ok(records)
    }
}

#[cfg(feature = "parallel")]
impl CopcHeader {
    /// [`CopcHeader::decode_node`] for many nodes (`(chunk, count)`) on the
    /// rayon pool, merged in the given order.
    pub fn decode_nodes_par(&self, nodes: &[(&[u8], usize)]) -> Result<CopcPoints, IoError> {
        use rayon::prelude::*;
        let parts: Vec<CopcPoints> = nodes
            .par_iter()
            .map(|&(chunk, count)| self.decode_node(chunk, count))
            .collect::<Result<_, _>>()?;
        let mut out = CopcPoints::default();
        for p in parts {
            out.extend(p);
        }
        Ok(out)
    }
}

/// Decoded points of one or more nodes, colors still 16-bit (the 8-bit
/// scaling is decided once all are in).
pub use super::las::RawLasPoints as CopcPoints;

/// A node of the COPC octree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct VoxelKey {
    pub level: i32,
    pub x: i32,
    pub y: i32,
    pub z: i32,
}

impl VoxelKey {
    pub fn parent(self) -> Option<Self> {
        (self.level > 0).then(|| Self {
            level: self.level - 1,
            x: self.x >> 1,
            y: self.y >> 1,
            z: self.z >> 1,
        })
    }
}

/// A hierarchy entry: a node's chunk, or (`point_count == -1`) a child page.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Entry {
    pub key: VoxelKey,
    pub offset: u64,
    pub byte_size: i32,
    pub point_count: i32,
}

/// Entries of one hierarchy page (32 bytes each).
pub fn parse_page(bytes: &[u8]) -> Vec<Entry> {
    bytes
        .as_chunks::<32>()
        .0
        .iter()
        .map(|e| {
            let i = |k: usize| i32::from_le_bytes(e[4 * k..4 * k + 4].try_into().unwrap());
            Entry {
                key: VoxelKey {
                    level: i(0),
                    x: i(1),
                    y: i(2),
                    z: i(3),
                },
                offset: u64::from_le_bytes(e[16..24].try_into().unwrap()),
                byte_size: i(6),
                point_count: i(7),
            }
        })
        .collect()
}

/// Walks the hierarchy to choose nodes: every node down to the deepest
/// level whose total still fits a point budget. Feed it pages as it asks.
#[derive(Debug, Default)]
pub struct NodeSelector {
    nodes: HashMap<VoxelKey, Entry>,
    /// Pages referenced but not yet read, by the level of their root node.
    pages: Vec<Entry>,
    read: HashSet<u64>,
}

impl NodeSelector {
    pub fn new(root_page: (u64, u64)) -> Self {
        let root = Entry {
            key: VoxelKey {
                level: 0,
                x: 0,
                y: 0,
                z: 0,
            },
            offset: root_page.0,
            byte_size: root_page.1 as i32,
            point_count: -1,
        };
        Self {
            pages: vec![root],
            ..Self::default()
        }
    }

    /// Byte ranges of the pages still needed to know every node down to
    /// `level` (pages whose root is at or above it).
    pub fn pages_for(&self, level: i32) -> Vec<(u64, u64)> {
        self.pages
            .iter()
            .filter(|p| p.key.level <= level && !self.read.contains(&p.offset))
            .map(|p| (p.offset, p.byte_size as u64))
            .collect()
    }

    /// Add a page read from `offset`.
    pub fn add_page(&mut self, offset: u64, bytes: &[u8]) {
        self.read.insert(offset);
        self.pages.retain(|p| p.offset != offset);
        for e in parse_page(bytes) {
            if e.point_count < 0 {
                self.pages.push(e);
            } else if e.point_count > 0 {
                self.nodes.insert(e.key, e);
            }
        }
    }

    /// Points in all nodes at `level` (call once its pages are read).
    pub fn level_points(&self, level: i32) -> u64 {
        self.nodes
            .values()
            .filter(|e| e.key.level == level)
            .map(|e| e.point_count as u64)
            .sum()
    }

    /// Whether any node or page lies below `level`.
    pub fn deeper_than(&self, level: i32) -> bool {
        self.nodes.keys().any(|k| k.level > level) || self.pages.iter().any(|p| p.key.level > level)
    }

    /// Nodes down to `level`, largest first.
    pub fn nodes_to(&self, level: i32) -> Vec<Entry> {
        let mut out: Vec<Entry> = self
            .nodes
            .values()
            .filter(|e| e.key.level <= level)
            .copied()
            .collect();
        out.sort_by_key(|e| (-e.point_count, e.offset));
        out
    }
}

/// Write a minimal COPC file (point format 7, one hierarchy page) with the
/// given nodes' points, for tests and test fixtures. Intensity, classes
/// (2 and 3) and colours are made up.
#[doc(hidden)]
pub fn write_minimal_copc(
    nodes: &[(VoxelKey, Vec<[f64; 3]>)],
    center: [f64; 3],
    halfsize: f64,
) -> Vec<u8> {
    use laz::record::{LayeredPointRecordCompressor, RecordCompressor};
    let items = laz::LazItemRecordBuilder::default_for_point_format_id(7, 0).unwrap();
    let vlr = laz::LazVlrBuilder::new(items)
        .with_variable_chunk_size()
        .build();
    let mut laz_payload = Vec::new();
    vlr.write_to(&mut laz_payload).unwrap();
    let record_len = 36usize;
    let scale = 0.001;
    let total: usize = nodes.iter().map(|n| n.1.len()).sum();

    let vlr_bytes = |user: &[u8], record: u16, payload: &[u8]| {
        let mut v = vec![0u8; VLR_HEADER];
        v[2..2 + user.len()].copy_from_slice(user);
        v[18..20].copy_from_slice(&record.to_le_bytes());
        v[20..22].copy_from_slice(&(payload.len() as u16).to_le_bytes());
        v.extend_from_slice(payload);
        v
    };
    let data_offset = HEADER_SIZE + (VLR_HEADER + 160) + (VLR_HEADER + laz_payload.len());
    // Chunks after the header, then the hierarchy page.
    let mut body = Vec::new();
    let mut entries = Vec::new();
    for (key, points) in nodes {
        let mut raw = Vec::with_capacity(points.len() * record_len);
        for (k, p) in points.iter().enumerate() {
            let mut r = vec![0u8; record_len];
            for a in 0..3 {
                let v = ((p[a] - center[a]) / scale).round() as i32;
                r[4 * a..4 * a + 4].copy_from_slice(&v.to_le_bytes());
            }
            r[12..14].copy_from_slice(&((k % 1000) as u16).to_le_bytes()); // intensity
            r[14] = 0x11; // return 1 of 1
            r[16] = 2 + (k % 2) as u8; // class
            for c in 0..3 {
                r[30 + 2 * c..32 + 2 * c].copy_from_slice(&((k as u16 % 200) << 8).to_le_bytes());
            }
            raw.extend_from_slice(&r);
        }
        let mut chunk = std::io::Cursor::new(Vec::new());
        let mut compressor = LayeredPointRecordCompressor::new(&mut chunk);
        compressor.set_fields_from(vlr.items()).unwrap();
        compressor.compress_many(&raw).unwrap();
        compressor.done().unwrap();
        drop(compressor);
        let chunk = chunk.into_inner();
        entries.push(Entry {
            key: *key,
            offset: (data_offset + body.len()) as u64,
            byte_size: chunk.len() as i32,
            point_count: points.len() as i32,
        });
        body.extend_from_slice(&chunk);
    }
    let page_offset = (data_offset + body.len()) as u64;
    let mut page = Vec::new();
    for e in &entries {
        for v in [e.key.level, e.key.x, e.key.y, e.key.z] {
            page.extend_from_slice(&v.to_le_bytes());
        }
        page.extend_from_slice(&e.offset.to_le_bytes());
        page.extend_from_slice(&e.byte_size.to_le_bytes());
        page.extend_from_slice(&e.point_count.to_le_bytes());
    }

    let mut info = Vec::new();
    for v in [center[0], center[1], center[2], halfsize, halfsize / 64.0] {
        info.extend_from_slice(&v.to_le_bytes());
    }
    info.extend_from_slice(&page_offset.to_le_bytes());
    info.extend_from_slice(&(page.len() as u64).to_le_bytes());
    info.resize(160, 0);

    let mut h = vec![0u8; HEADER_SIZE];
    h[0..4].copy_from_slice(b"LASF");
    h[24] = 1;
    h[25] = 4;
    h[94..96].copy_from_slice(&(HEADER_SIZE as u16).to_le_bytes());
    h[96..100].copy_from_slice(&(data_offset as u32).to_le_bytes());
    h[100..104].copy_from_slice(&2u32.to_le_bytes());
    h[104] = 7 | 0x80;
    h[105..107].copy_from_slice(&(record_len as u16).to_le_bytes());
    for a in 0..3 {
        h[131 + 8 * a..139 + 8 * a].copy_from_slice(&scale.to_le_bytes());
        h[155 + 8 * a..163 + 8 * a].copy_from_slice(&center[a].to_le_bytes());
    }
    h[247..255].copy_from_slice(&(total as u64).to_le_bytes());
    let mut file = h;
    file.extend(vlr_bytes(b"copc", 1, &info));
    file.extend(vlr_bytes(b"laszip encoded", 22204, &laz_payload));
    assert_eq!(file.len(), data_offset);
    file.extend(body);
    file.extend(page);
    file
}

#[cfg(test)]
mod tests {
    use super::*;

    fn key(level: i32, x: i32, y: i32, z: i32) -> VoxelKey {
        VoxelKey { level, x, y, z }
    }

    fn grid(n: usize, z: f64) -> Vec<[f64; 3]> {
        (0..n * n)
            .map(|k| [(k % n) as f64 * 0.5, (k / n) as f64 * 0.5, z])
            .collect()
    }

    #[test]
    fn reads_header_hierarchy_and_nodes() {
        let nodes = vec![
            (key(0, 0, 0, 0), grid(10, 1.0)),
            (key(1, 0, 0, 0), grid(20, 2.0)),
            (key(1, 1, 0, 0), grid(15, 3.0)),
            (key(2, 0, 0, 0), grid(30, 4.0)),
        ];
        let file = write_minimal_copc(&nodes, [5.0, 5.0, 0.0], 8.0);
        assert!(CopcHeader::is_copc(&file));
        let header = CopcHeader::parse(&file[..CopcHeader::needed(&file).unwrap()]).unwrap();
        assert_eq!(header.total_points, 100 + 400 + 225 + 900);
        assert_eq!(header.halfsize, 8.0);

        let mut selector = NodeSelector::new(header.root_page);
        let pages = selector.pages_for(0);
        assert_eq!(pages, vec![header.root_page]);
        let (o, s) = pages[0];
        selector.add_page(o, &file[o as usize..(o + s) as usize]);
        assert!(selector.pages_for(5).is_empty());
        assert_eq!(selector.level_points(1), 625);
        assert!(selector.deeper_than(1) && !selector.deeper_than(2));
        let chosen = selector.nodes_to(1);
        assert_eq!(chosen.len(), 3);

        // Decode the level-1 node with 400 points.
        let e = chosen.iter().find(|e| e.point_count == 400).unwrap();
        let chunk = &file[e.offset as usize..e.offset as usize + e.byte_size as usize];
        let points = header.decode_node(chunk, 400).unwrap();
        let cloud = points.into_cloud();
        assert_eq!(cloud.len(), 400);
        for (p, q) in cloud.positions.iter().zip(&nodes[1].1) {
            for a in 0..3 {
                assert!((p[a] - q[a]).abs() < 1e-3, "{p:?} vs {q:?}");
            }
        }
        let class = cloud.attribute(crate::CLASSIFICATION).unwrap();
        assert_eq!(class.values.len(), 400);
        assert!(cloud.colors.is_some());
    }

    #[test]
    fn rejects_plain_las() {
        let mut las = vec![0u8; 400];
        las[..4].copy_from_slice(b"LASF");
        assert!(!CopcHeader::is_copc(&las));
        assert!(CopcHeader::parse(&las).is_err());
    }

    #[test]
    fn raw_decode_preserves_fields_and_rejects_untrusted_layer_sizes() {
        let file = write_minimal_copc(&[(key(0, 0, 0, 0), grid(2, 0.0))], [0.0; 3], 8.0);
        let header = CopcHeader::parse(&file).unwrap();
        let (offset, size) = header.root_page;
        let entry = parse_page(&file[offset as usize..(offset + size) as usize])[0];
        let chunk = &file[entry.offset as usize..entry.offset as usize + entry.byte_size as usize];
        let raw = header.decode_records(chunk, 4, 144).unwrap();
        assert_eq!(raw.len(), 4 * 36);
        assert_eq!(
            u16::from_le_bytes(raw[36 + 12..36 + 14].try_into().unwrap()),
            1
        );
        assert_eq!(
            u16::from_le_bytes(raw[36 + 30..36 + 32].try_into().unwrap()),
            256
        );
        assert_eq!(raw[36 + 16], 3);
        assert!(header.decode_records(chunk, 4, 143).is_err());
        assert!(header.decode_records(chunk, 3, 144).is_err());
        let mut corrupt = chunk.to_vec();
        corrupt[40..44].copy_from_slice(&u32::MAX.to_le_bytes());
        assert!(header.decode_records(&corrupt, 4, 144).is_err());
    }

    #[test]
    fn raw_formats_6_7_8_keep_extra_bytes_gps_flags_and_nir() {
        use laz::record::{LayeredPointRecordCompressor, RecordCompressor};
        for format in 6..=8 {
            let extra = 5;
            let vlr = laz::LazVlrBuilder::new(
                laz::LazItemRecordBuilder::default_for_point_format_id(format, extra).unwrap(),
            )
            .with_variable_chunk_size()
            .build();
            let record_len = match format {
                6 => 30,
                7 => 36,
                _ => 38,
            } + extra as usize;
            let mut raw = vec![0u8; 5 * record_len];
            for (k, record) in raw.chunks_exact_mut(record_len).enumerate() {
                record[..4].copy_from_slice(&(k as i32 * 123).to_le_bytes());
                record[12..14].copy_from_slice(&(k as u16 * 117).to_le_bytes());
                record[14] = 0x11;
                record[15] = k as u8;
                record[16] = 2;
                record[22..30].copy_from_slice(&(1_000_000.123 + k as f64).to_le_bytes());
                for (j, byte) in record[30..].iter_mut().enumerate() {
                    *byte = (k * 31 + j) as u8;
                }
            }
            let mut output = std::io::Cursor::new(Vec::new());
            let mut compressor = LayeredPointRecordCompressor::new(&mut output);
            compressor.set_fields_from(vlr.items()).unwrap();
            compressor.compress_many(&raw).unwrap();
            compressor.done().unwrap();
            drop(compressor);
            let header = CopcHeader {
                las: LasHeader {
                    data_offset: 0,
                    record_len,
                    count: 5,
                    compressed: true,
                    format,
                    scale: [0.01; 3],
                    offset: [0.0; 3],
                },
                laz: vlr,
                center: [0.0; 3],
                halfsize: 8.,
                spacing: 1.,
                root_page: (0, 0),
                total_points: 5,
            };
            let decoded = header
                .decode_records(&output.into_inner(), 5, raw.len() as u64)
                .unwrap();
            assert_eq!(decoded, raw, "point format {format}");
        }
    }

    #[test]
    fn count_is_not_narrowed_to_the_target_address_width() {
        let mut file = write_minimal_copc(&[(key(0, 0, 0, 0), grid(2, 0.0))], [0.0; 3], 8.0);
        file[247..255].copy_from_slice(&10_000_000_000u64.to_le_bytes());
        assert_eq!(
            CopcHeader::parse(&file).unwrap().total_points,
            10_000_000_000
        );
    }
}
