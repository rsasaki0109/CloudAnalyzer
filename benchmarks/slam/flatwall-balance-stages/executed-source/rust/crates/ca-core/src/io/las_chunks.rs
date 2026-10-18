//! Random access to a plain LAS/LAZ file, one chunk of points at a time, so
//! a file larger than memory can be read in parallel pieces and parts of it
//! decoded again later at full density.
//!
//! LAZ files are already split into independently compressed chunks (usually
//! 50,000 points), listed in a chunk table after the points; uncompressed LAS
//! is split into runs of [`LAS_CHUNK_POINTS`] records. A [`LasLayout`] knows
//! every chunk's byte range and first point; a [`ChunkDecoder`] turns the
//! bytes of consecutive chunks into points and each chunk's bounding box.

use std::io::Cursor;

use laz::LazVlr;
use laz::laszip::ChunkTable;
use laz::record::{
    LayeredPointRecordDecompressor, RecordDecompressor, SequentialPointRecordDecompressor,
};

use super::IoError;
use super::las::{LasDecoder, LasHeader, RawLasPoints, laszip_vlr};

const FORMAT: &str = "LAZ";

/// Records per chunk of an uncompressed LAS file, the usual LAZ chunk size.
pub const LAS_CHUNK_POINTS: u64 = 50_000;

/// Consecutive points of the file that decode on their own.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct LasChunk {
    /// Index of the chunk's first point in the file.
    pub first: u64,
    pub count: u64,
    /// Byte range in the file.
    pub offset: u64,
    pub size: u64,
}

/// What [`LasLayout`] still has to read before the chunks are known.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Need {
    Nothing,
    /// The writer could not seek back, so the table offset is the file's last 8 bytes.
    OffsetAtEnd,
    /// Version and chunk count at the start of the table.
    TableHead(u64),
    Table {
        offset: u64,
        chunks: u32,
    },
}

/// The chunks of a LAS/LAZ file. Open it from the file's first
/// [`LasLayout::header_len`] bytes, then give it the byte ranges it asks for
/// with [`LasLayout::needs`] / [`LasLayout::supply`] (for LAZ: the chunk table).
#[derive(Debug)]
pub struct LasLayout {
    header: LasHeader,
    laz: Option<LazVlr>,
    file_size: u64,
    need: Need,
    chunks: Vec<LasChunk>,
}

impl LasLayout {
    /// Bytes from the file start that [`LasLayout::open`] needs: header and
    /// VLRs, plus for LAZ the chunk table offset that starts the point data.
    /// `None` if `head` is too short to tell (read at least 227 bytes) or not LAS.
    pub fn header_len(head: &[u8]) -> Option<usize> {
        let header = LasHeader::parse(head).ok()?;
        header
            .data_offset
            .checked_add(if header.compressed { 8 } else { 0 })
    }

    /// Open a file of `file_size` bytes. `Ok(None)` when it is not LAS or is
    /// LAZ without chunks (the oldest point-wise compression).
    pub fn open(head: &[u8], file_size: u64) -> Result<Option<Self>, IoError> {
        if !head.starts_with(b"LASF") {
            return Ok(None);
        }
        let header = LasHeader::parse(head)?;
        let data = header.data_offset as u64;
        let (laz, need, chunks) = if header.compressed {
            let payload = laszip_vlr(head)?;
            let vlr = LazVlr::from_buffer(payload)
                .map_err(|e| IoError::header(FORMAT, format!("bad LASzip VLR: {e}")))?;
            // The compressor type leads the VLR: 2 and 3 are chunked.
            if !matches!(payload.get(..2), Some([2 | 3, 0])) {
                return Ok(None);
            }
            let at = header.data_offset;
            let table = head
                .get(at..at + 8)
                .ok_or(IoError::Truncated(FORMAT))
                .map(|b| i64::from_le_bytes(b.try_into().unwrap()))?;
            let need =
                Self::table_at(table, data, file_size).map_or(Need::OffsetAtEnd, Need::TableHead);
            (Some(vlr), need, Vec::new())
        } else {
            let record_len = header.record_len as u64;
            let total = header.count;
            let end = total
                .checked_mul(record_len)
                .and_then(|n| data.checked_add(n));
            if end.is_none_or(|end| end > file_size) {
                return Err(IoError::Truncated("LAS"));
            }
            if total.div_ceil(LAS_CHUNK_POINTS) > 2_000_000 {
                return Err(IoError::Unsupported(
                    "LAS layout exceeds 2 million chunks".into(),
                ));
            }
            let chunks = (0..total.div_ceil(LAS_CHUNK_POINTS))
                .map(|k| {
                    let first = k * LAS_CHUNK_POINTS;
                    let count = LAS_CHUNK_POINTS.min(total - first);
                    LasChunk {
                        first,
                        count,
                        offset: data + first * record_len,
                        size: count * record_len,
                    }
                })
                .collect();
            (None, Need::Nothing, chunks)
        };
        Ok(Some(Self {
            header,
            laz,
            file_size,
            need,
            chunks,
        }))
    }

    /// A chunk table offset, if it points past the data start and into the file.
    fn table_at(offset: i64, data: u64, file_size: u64) -> Option<u64> {
        (offset >= 0
            && offset as u64 > data
            && (offset as u64)
                .checked_add(8)
                .is_some_and(|end| end <= file_size))
        .then_some(offset as u64)
    }

    /// The byte range (`offset, length`) to read next, or `None` once the
    /// chunks are known.
    pub fn needs(&self) -> Option<(u64, u64)> {
        match self.need {
            Need::Nothing => None,
            Need::OffsetAtEnd => Some((self.file_size.saturating_sub(8), 8)),
            Need::TableHead(offset) => Some((offset, 8)),
            Need::Table { offset, chunks } => {
                // Arithmetic-coded counts and sizes: well under 24 bytes a chunk.
                let len = (8 + 24 * chunks as u64 + 64).min(self.file_size - offset);
                Some((offset, len))
            }
        }
    }

    /// The bytes of the range [`LasLayout::needs`] asked for.
    pub fn supply(&mut self, bytes: &[u8]) -> Result<(), IoError> {
        let data = self.header.data_offset as u64;
        match self.need {
            Need::Nothing => {}
            Need::OffsetAtEnd => {
                let offset = bytes
                    .get(..8)
                    .map(|b| i64::from_le_bytes(b.try_into().unwrap()))
                    .ok_or(IoError::Truncated(FORMAT))?;
                let at = Self::table_at(offset, data, self.file_size)
                    .ok_or_else(|| IoError::Unsupported("LAZ file without a chunk table".into()))?;
                self.need = Need::TableHead(at);
            }
            Need::TableHead(offset) => {
                let chunks = bytes
                    .get(4..8)
                    .map(|b| u32::from_le_bytes(b.try_into().unwrap()))
                    .ok_or(IoError::Truncated(FORMAT))?;
                if chunks > 2_000_000 {
                    return Err(IoError::Unsupported(
                        "LAZ layout exceeds 2 million chunks".into(),
                    ));
                }
                self.need = Need::Table { offset, chunks };
            }
            Need::Table { chunks, .. } => {
                if bytes.get(4..8) != Some(chunks.to_le_bytes().as_slice()) {
                    return Err(IoError::header(FORMAT, "chunk table count changed"));
                }
                let vlr = self.laz.as_ref().expect("only LAZ reads a table");
                let variable = vlr.uses_variable_size_chunks();
                let table = ChunkTable::read(&mut Cursor::new(bytes), variable)
                    .map_err(|e| IoError::header(FORMAT, format!("bad chunk table: {e}")))?;
                self.chunks = Self::layout(&table, vlr, data + 8, self.header.count)?;
                if self.chunks.iter().any(|c| {
                    c.offset
                        .checked_add(c.size)
                        .is_none_or(|end| end > self.file_size)
                }) {
                    return Err(IoError::Truncated(FORMAT));
                }
                self.need = Need::Nothing;
            }
        }
        Ok(())
    }

    /// Chunks from the table: they follow each other from `start`. Fixed-size
    /// tables do not store counts, and the last chunk holds what is left.
    fn layout(
        table: &ChunkTable,
        vlr: &LazVlr,
        start: u64,
        total: u64,
    ) -> Result<Vec<LasChunk>, IoError> {
        let mut chunks = Vec::with_capacity(table.len());
        let (mut first, mut offset) = (0, start);
        for entry in table {
            if first >= total {
                break;
            }
            let count = if vlr.uses_variable_size_chunks() {
                entry.point_count
            } else {
                vlr.chunk_size() as u64
            }
            .min(total - first);
            chunks.push(LasChunk {
                first,
                count,
                offset,
                size: entry.byte_count,
            });
            first += count;
            offset = offset
                .checked_add(entry.byte_count)
                .ok_or_else(|| IoError::header(FORMAT, "chunk byte span overflow"))?;
        }
        if first < total {
            return Err(IoError::header(
                FORMAT,
                format!("the chunk table covers {first} of {total} points"),
            ));
        }
        Ok(chunks)
    }

    pub fn chunks(&self) -> &[LasChunk] {
        &self.chunks
    }

    /// Points in the file.
    pub fn total_points(&self) -> u64 {
        self.header.count
    }

    /// Bytes of the file start a [`ChunkDecoder`] needs (header and VLRs).
    pub fn data_offset(&self) -> usize {
        self.header.data_offset
    }
}

/// Decodes chunks, e.g. on another thread: needs only the file's first
/// [`LasLayout::data_offset`] bytes.
pub struct ChunkDecoder {
    header: LasHeader,
    laz: Option<LazVlr>,
}

/// Decoded points of consecutive chunks, and each chunk's bounds.
#[derive(Debug, Default)]
pub struct DecodedChunks {
    pub points: RawLasPoints,
    /// `min, max` corners per chunk, over all its points (kept or not).
    pub bounds: Vec<[f64; 6]>,
}

impl ChunkDecoder {
    pub fn new(head: &[u8]) -> Result<Self, IoError> {
        let header = LasHeader::parse(head)?;
        let laz = if header.compressed {
            Some(
                LazVlr::from_buffer(laszip_vlr(head)?)
                    .map_err(|e| IoError::header(FORMAT, format!("bad LASzip VLR: {e}")))?,
            )
        } else {
            None
        };
        Ok(Self { header, laz })
    }

    /// Decode chunks whose bytes lie back to back in `bytes`, keeping every
    /// `keep_every`-th point counted from the start of the file (as a whole
    /// file read does).
    pub fn decode(
        &self,
        bytes: &[u8],
        chunks: &[LasChunk],
        keep_every: u64,
    ) -> Result<DecodedChunks, IoError> {
        let keep_every = keep_every.max(1);
        let len = self.header.record_len;
        let kept: u64 = chunks.iter().try_fold(0u64, |sum, c| {
            let end = c
                .first
                .checked_add(c.count)
                .filter(|&end| end <= self.header.count)
                .ok_or_else(|| {
                    IoError::header(FORMAT, "chunk point span overflow or past total")
                })?;
            sum.checked_add(end.div_ceil(keep_every) - c.first.div_ceil(keep_every))
                .ok_or_else(|| IoError::header(FORMAT, "output point count overflow"))
        })?;
        let capacity = usize::try_from(kept)
            .map_err(|_| IoError::Unsupported("LAZ output exceeds address space".into()))?;
        let mut decoder = LasDecoder::new(self.header.clone(), capacity)?;
        let mut bounds = Vec::with_capacity(chunks.len());
        let mut records = Vec::new();
        let mut at = 0usize;
        for chunk in chunks {
            let end = usize::try_from(chunk.size)
                .ok()
                .and_then(|size| at.checked_add(size))
                .ok_or_else(|| IoError::header(FORMAT, "chunk byte span exceeds address space"))?;
            let data = bytes.get(at..end).ok_or(IoError::Truncated(FORMAT))?;
            at = end;
            let record_bytes = chunk
                .count
                .checked_mul(len as u64)
                .and_then(|n| usize::try_from(n).ok())
                .ok_or_else(|| IoError::header(FORMAT, "chunk records exceed address space"))?;
            let records: &[u8] = match &self.laz {
                None => data.get(..record_bytes).ok_or(IoError::Truncated("LAS"))?,
                Some(vlr) => {
                    records
                        .try_reserve(record_bytes.saturating_sub(records.len()))
                        .map_err(|e| IoError::Unsupported(format!("LAZ allocation: {e}")))?;
                    records.resize(record_bytes, 0);
                    decompress(vlr, data, &mut records)?;
                    &records
                }
            };
            let mut lo = [i32::MAX; 3];
            let mut hi = [i32::MIN; 3];
            for (k, record) in records.chunks_exact(len).enumerate() {
                for i in 0..3 {
                    let v = i32::from_le_bytes(record[4 * i..4 * i + 4].try_into().unwrap());
                    lo[i] = lo[i].min(v);
                    hi[i] = hi[i].max(v);
                }
                if (chunk.first + k as u64).is_multiple_of(keep_every) {
                    decoder.decode(record);
                }
            }
            bounds.push(self.to_bounds(lo, hi));
        }
        Ok(DecodedChunks {
            points: decoder.into_raw(),
            bounds,
        })
    }

    /// Integer box to coordinates (a negative scale swaps the corners).
    fn to_bounds(&self, lo: [i32; 3], hi: [i32; 3]) -> [f64; 6] {
        let (s, o) = (self.header.scale, self.header.offset);
        let mut out = [0.0; 6];
        for i in 0..3 {
            let (a, b) = (lo[i] as f64 * s[i] + o[i], hi[i] as f64 * s[i] + o[i]);
            out[i] = a.min(b);
            out[3 + i] = a.max(b);
        }
        out
    }
}

/// Decompress one LAZ chunk into `out` (a whole number of records).
fn decompress(vlr: &LazVlr, chunk: &[u8], out: &mut [u8]) -> Result<(), IoError> {
    let err = |e: &dyn std::fmt::Display| IoError::Unsupported(format!("LAZ: {e}"));
    let items = vlr.items();
    // Point formats 6-10 compress in layers (item version 3+), older ones point by point.
    let mut decompressor: Box<dyn RecordDecompressor<Cursor<&[u8]>>> =
        if items.first().is_some_and(|i| i.version() >= 3) {
            Box::new(LayeredPointRecordDecompressor::new(Cursor::new(chunk)))
        } else {
            Box::new(SequentialPointRecordDecompressor::new(Cursor::new(chunk)))
        };
    decompressor.set_fields_from(items).map_err(|e| err(&e))?;
    decompressor.decompress_many(out).map_err(|e| err(&e))
}

#[cfg(test)]
mod tests {
    use super::*;
    use laz::{LasZipCompressor, LazItemRecordBuilder, LazVlrBuilder};

    const DATA: usize = 227;

    #[test]
    fn large_layout_count_and_overflow_are_checked_without_point_data() {
        let mut h = header(0, 20, 0);
        h.resize(375, 0);
        h[25] = 4;
        h[96..100].copy_from_slice(&375u32.to_le_bytes());
        h[247..255].copy_from_slice(&10_000_000_000u64.to_le_bytes());
        let layout = LasLayout::open(&h, 375 + 200_000_000_000).unwrap().unwrap();
        assert_eq!(layout.total_points(), 10_000_000_000);
        assert_eq!(layout.chunks().len(), 200_000);
        assert_eq!(layout.chunks().last().unwrap().first, 9_999_950_000);
        h[247..255].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(LasLayout::open(&h, u64::MAX).is_err());
    }

    /// A LAS 1.2 header for `n` records of `format` (`record_len` bytes)
    /// with 1 cm scale and offset (1000, 2000, 0).
    fn header(format: u8, record_len: u16, n: usize) -> Vec<u8> {
        let mut h = vec![0u8; DATA];
        h[..4].copy_from_slice(b"LASF");
        h[24] = 1;
        h[25] = 2;
        h[94..96].copy_from_slice(&(DATA as u16).to_le_bytes());
        h[96..100].copy_from_slice(&(DATA as u32).to_le_bytes());
        h[104] = format;
        h[105..107].copy_from_slice(&record_len.to_le_bytes());
        h[107..111].copy_from_slice(&(n as u32).to_le_bytes());
        for i in 0..3 {
            h[131 + 8 * i..139 + 8 * i].copy_from_slice(&0.01f64.to_le_bytes());
        }
        for (i, o) in [1000.0f64, 2000.0, 0.0].iter().enumerate() {
            h[155 + 8 * i..163 + 8 * i].copy_from_slice(&o.to_le_bytes());
        }
        h
    }

    /// Format 1 records along a line: point `i` at (i, -2i, i % 50) cm.
    fn records(n: usize) -> Vec<u8> {
        let mut out = Vec::with_capacity(n * 28);
        for i in 0..n as i32 {
            let mut r = [0u8; 28];
            for (a, v) in [i, -2 * i, i % 50].iter().enumerate() {
                r[4 * a..4 * a + 4].copy_from_slice(&v.to_le_bytes());
            }
            r[12..14].copy_from_slice(&(i as u16).to_le_bytes());
            r[15] = (i % 3) as u8;
            out.extend_from_slice(&r);
        }
        out
    }

    fn las(n: usize) -> Vec<u8> {
        let mut file = header(1, 28, n);
        file.extend_from_slice(&records(n));
        file
    }

    /// LAZ of `n` format-1 records, in fixed chunks of `chunk` points or, with
    /// `variable`, in chunks of those sizes. `seek_back` false leaves the
    /// table offset for the end of the file, as streaming writers do.
    fn laz(n: usize, chunk: u32, variable: Option<&[usize]>, seek_back: bool) -> Vec<u8> {
        let items = LazItemRecordBuilder::default_for_point_format_id(1, 0).unwrap();
        let vlr = match variable {
            Some(_) => LazVlrBuilder::new(items).with_variable_chunk_size().build(),
            None => LazVlrBuilder::new(items)
                .with_fixed_chunk_size(chunk)
                .build(),
        };
        let mut payload = Vec::new();
        vlr.write_to(&mut payload).unwrap();
        let mut file = header(0x80 | 1, 28, n);
        let mut vlr_header = vec![0u8; 54];
        vlr_header[2..16].copy_from_slice(b"laszip encoded");
        vlr_header[18..20].copy_from_slice(&22204u16.to_le_bytes());
        vlr_header[20..22].copy_from_slice(&(payload.len() as u16).to_le_bytes());
        file.extend_from_slice(&vlr_header);
        file.extend_from_slice(&payload);
        let data = file.len() as u32;
        file[96..100].copy_from_slice(&data.to_le_bytes());
        file[100..104].copy_from_slice(&1u32.to_le_bytes());
        let mut cursor = Cursor::new(file);
        cursor.set_position(data as u64);
        let points = records(n);
        {
            let mut compressor = LasZipCompressor::new(&mut cursor, vlr).unwrap();
            match variable {
                Some(sizes) => {
                    let mut at = 0;
                    for s in sizes {
                        compressor
                            .compress_many(&points[at * 28..(at + s) * 28])
                            .unwrap();
                        compressor.finish_current_chunk().unwrap();
                        at += s;
                    }
                }
                None => compressor.compress_many(&points).unwrap(),
            }
            compressor.done().unwrap();
        }
        let mut file = cursor.into_inner();
        if !seek_back {
            let table =
                i64::from_le_bytes(file[data as usize..data as usize + 8].try_into().unwrap());
            file[data as usize..data as usize + 8].copy_from_slice(&(-1i64).to_le_bytes());
            file.extend_from_slice(&table.to_le_bytes());
        }
        file
    }

    /// Open `file` and read what the layout asks for.
    fn layout(file: &[u8]) -> LasLayout {
        let head = &file[..LasLayout::header_len(file).unwrap()];
        let mut layout = LasLayout::open(head, file.len() as u64).unwrap().unwrap();
        while let Some((offset, len)) = layout.needs() {
            layout
                .supply(&file[offset as usize..(offset + len) as usize])
                .unwrap();
        }
        layout
    }

    /// Decode every chunk, one call per `per_call` chunks.
    fn decode_all(file: &[u8], keep_every: u64, per_call: usize) -> DecodedChunks {
        let layout = layout(file);
        let decoder = ChunkDecoder::new(&file[..layout.data_offset()]).unwrap();
        let mut out = DecodedChunks::default();
        for group in layout.chunks().chunks(per_call) {
            let start = group[0].offset as usize;
            let end = group.last().map(|c| (c.offset + c.size) as usize).unwrap();
            let part = decoder
                .decode(&file[start..end], group, keep_every)
                .unwrap();
            out.points.extend(part.points);
            out.bounds.extend(part.bounds);
        }
        out
    }

    #[test]
    fn las_splits_into_fixed_runs_of_records() {
        let n = 2 * LAS_CHUNK_POINTS as usize + 123;
        let file = las(n);
        let layout = layout(&file);
        let chunks = layout.chunks();
        assert_eq!(chunks.len(), 3);
        assert_eq!(chunks[1].first, LAS_CHUNK_POINTS);
        assert_eq!(
            chunks[1].offset,
            (DATA + 28 * LAS_CHUNK_POINTS as usize) as u64
        );
        assert_eq!(chunks[2].count, 123);
        assert_eq!(chunks[2].size, 123 * 28);
        // A file cut short is refused up front.
        let head = &file[..DATA];
        assert!(LasLayout::open(head, file.len() as u64 - 1).is_err());
    }

    #[test]
    fn laz_chunk_table_gives_offsets_counts_and_bounds() {
        let n = 25_000;
        let file = laz(n, 10_000, None, true);
        let layout = layout(&file);
        let chunks = layout.chunks();
        assert_eq!(
            chunks
                .iter()
                .map(|c| (c.first, c.count))
                .collect::<Vec<_>>(),
            [(0, 10_000), (10_000, 10_000), (20_000, 5_000)]
        );
        assert_eq!(
            chunks[0].offset,
            file[96..100]
                .iter()
                .rev()
                .fold(0, |a, &b| a << 8 | b as u64)
                + 8
        );
        assert_eq!(chunks[1].offset, chunks[0].offset + chunks[0].size);
        let decoded = decode_all(&file, 1, 1);
        assert_eq!(decoded.points.len(), n);
        // Chunk 1 holds points 10,000..19,999: x = 1000 + i cm, y = 2000 - 2i cm.
        let b = decoded.bounds[1];
        assert!((b[0] - 1100.0).abs() < 1e-9 && (b[3] - 1199.99).abs() < 1e-9);
        assert!((b[1] - (2000.0 - 399.98)).abs() < 1e-9 && (b[4] - 1800.0).abs() < 1e-9);
        assert!((b[2] - 0.0).abs() < 1e-9 && (b[5] - 0.49).abs() < 1e-9);
    }

    #[test]
    fn laz_table_offset_at_the_end_and_variable_chunks() {
        let sizes = [7_000, 13_000, 1];
        let file = laz(20_001, 0, Some(&sizes), false);
        let layout = layout(&file);
        assert_eq!(
            layout.chunks().iter().map(|c| c.count).collect::<Vec<_>>(),
            sizes.map(|s| s as u64)
        );
        let decoded = decode_all(&file, 1, 2);
        assert_eq!(decoded.points.len(), 20_001);
        assert_eq!(decoded.bounds.len(), 3);
        assert!((decoded.bounds[2][0] - 1200.0).abs() < 1e-9);
    }

    #[test]
    fn chunks_decode_like_a_whole_file_read() {
        let n = 2 * LAS_CHUNK_POINTS as usize + 999;
        for file in [las(n), laz(n, 50_000, None, true)] {
            for keep_every in [1, 3, 7] {
                let whole = super::super::las::read(&file, keep_every).unwrap();
                for per_call in [1, 2] {
                    let decoded = decode_all(&file, keep_every as u64, per_call);
                    assert_eq!(decoded.points.into_cloud(), whole, "1 in {keep_every}");
                }
            }
        }
    }

    #[test]
    fn layered_laz_14_with_colors_and_extra_bytes() {
        // Formats 6-10 compress in layers; the core's writer makes format 7.
        let n = 120_000;
        let cloud = crate::PointCloud {
            positions: (0..n)
                .map(|i| [i as f64 * 0.01, 5.0, (i % 7) as f64])
                .collect(),
            colors: Some((0..n).map(|i| [(i % 256) as u8, 0, 200]).collect()),
            attributes: Vec::new(),
        };
        let values = vec![0.5f32; n];
        let scalar = super::super::ScalarField {
            name: "d",
            values: &values,
        };
        let file = super::super::write_las(&cloud, &[scalar], true).unwrap();
        let whole = super::super::las::read(&file, 4).unwrap();
        let decoded = decode_all(&file, 4, 3);
        assert!(decoded.bounds.len() >= 2);
        assert_eq!(decoded.points.into_cloud(), whole);
    }

    #[test]
    fn laz_without_chunks_or_table_is_declined() {
        // Point-wise compression has no chunks to seek to.
        let mut file = laz(10, 50_000, None, true);
        file[DATA + 54..DATA + 56].copy_from_slice(&1u16.to_le_bytes());
        assert!(LasLayout::open(&file, file.len() as u64).unwrap().is_none());

        // No table offset anywhere.
        let mut file = laz(10, 50_000, None, false);
        let n = file.len();
        file[n - 8..].copy_from_slice(&(-1i64).to_le_bytes());
        let head = &file[..LasLayout::header_len(&file).unwrap()];
        let mut layout = LasLayout::open(head, n as u64).unwrap().unwrap();
        let (offset, len) = layout.needs().unwrap();
        assert!(
            layout
                .supply(&file[offset as usize..(offset + len) as usize])
                .is_err()
        );
        assert!(LasLayout::open(b"ply\n", 4).unwrap().is_none());
    }
}
