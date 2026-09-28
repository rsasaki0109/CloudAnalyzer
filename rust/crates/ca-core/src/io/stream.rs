//! Incremental reading of fixed-size point records, so a file never has to
//! be held in memory as a whole. Supports uncompressed LAS, binary PLY whose
//! first element is a fixed-size vertex element, and row-major binary PCD.
//! Points can be thinned while reading (keep every n-th record).

use super::{Format, IoError};
use crate::PointCloud;

/// Turns fixed-size records into points.
pub(crate) trait RecordDecoder {
    fn record_len(&self) -> usize;
    /// Decode a block of whole records.
    fn decode_block(&mut self, records: &[u8]);
    fn finish(self: Box<Self>) -> PointCloud;
}

/// Decode the records in `records`, keeping every `keep_every`-th record
/// counted from the start of the file (`index` carries the count across
/// calls).
pub(crate) fn decode_records(
    decoder: &mut dyn RecordDecoder,
    records: &[u8],
    keep_every: u64,
    index: &mut u64,
) {
    let len = decoder.record_len();
    let n = (records.len() / len) as u64;
    if keep_every <= 1 {
        decoder.decode_block(records);
    } else {
        let mut kept = Vec::with_capacity(records.len() / keep_every as usize + len);
        for (k, record) in records.chunks_exact(len).enumerate() {
            if (*index + k as u64).is_multiple_of(keep_every) {
                kept.extend_from_slice(record);
            }
        }
        decoder.decode_block(&kept);
    }
    *index += n;
}

/// A point file being read piece by piece. Feed it the bytes from
/// [`PointStream::data_offset`] onward, in order, with [`PointStream::push`].
pub struct PointStream {
    decoder: Box<dyn RecordDecoder>,
    data_offset: u64,
    total_points: u64,
    remaining: u64,
    pending: Vec<u8>,
    keep_every: u64,
    index: u64,
}

impl PointStream {
    /// How many leading bytes [`PointStream::open`] needs: the whole header.
    /// `None` when `head` is not the start of a streamable file or the header
    /// is not complete within `head`.
    pub fn header_len(name: &str, head: &[u8]) -> Option<usize> {
        match Format::detect(name, head) {
            Format::Las => Some(375.min(head.len())),
            Format::Ply => find_line_end(head, b"end_header"),
            Format::Pcd => find_data_line_end(head),
            Format::Xyz | Format::E57 | Format::Splat => None,
        }
    }

    /// Start streaming. Returns `Ok(None)` when the file cannot be streamed
    /// (compressed LAS, ASCII or mesh PLY, ASCII or compressed PCD, text).
    pub fn open(name: &str, head: &[u8]) -> Result<Option<Self>, IoError> {
        let opened = match Format::detect(name, head) {
            Format::Las => super::las::stream(head)?,
            Format::Ply => super::ply::stream(head)?,
            Format::Pcd => super::pcd::stream(head)?,
            Format::Xyz | Format::E57 | Format::Splat => None,
        };
        Ok(opened.map(|(decoder, data_offset, total_points)| {
            let remaining = total_points * decoder.record_len() as u64;
            Self {
                decoder,
                data_offset: data_offset as u64,
                total_points,
                remaining,
                pending: Vec::new(),
                keep_every: 1,
                index: 0,
            }
        }))
    }

    /// Byte offset in the file where the point records start.
    pub fn data_offset(&self) -> u64 {
        self.data_offset
    }

    /// Number of point records in the file.
    pub fn total_points(&self) -> u64 {
        self.total_points
    }

    /// Keep only every `n`-th point (set before the first push).
    pub fn set_keep_every(&mut self, n: u64) {
        self.keep_every = n.max(1);
    }

    /// Record bytes still expected.
    pub fn remaining(&self) -> u64 {
        self.remaining
    }

    /// Feed the next bytes of the record section. Bytes past the last record
    /// are ignored.
    pub fn push(&mut self, mut bytes: &[u8]) {
        let take = (bytes.len() as u64).min(self.remaining) as usize;
        bytes = &bytes[..take];
        self.remaining -= take as u64;
        let len = self.decoder.record_len();
        if !self.pending.is_empty() {
            let need = len - self.pending.len();
            let n = need.min(bytes.len());
            self.pending.extend_from_slice(&bytes[..n]);
            bytes = &bytes[n..];
            if self.pending.len() < len {
                return;
            }
            let record = std::mem::take(&mut self.pending);
            decode_records(
                self.decoder.as_mut(),
                &record,
                self.keep_every,
                &mut self.index,
            );
        }
        let whole = bytes.len() / len * len;
        decode_records(
            self.decoder.as_mut(),
            &bytes[..whole],
            self.keep_every,
            &mut self.index,
        );
        self.pending.extend_from_slice(&bytes[whole..]);
    }

    /// The points read so far. Errors if the record section was cut short.
    pub fn finish(self) -> Result<PointCloud, IoError> {
        if self.remaining > 0 {
            return Err(IoError::Truncated("point"));
        }
        let cloud = self.decoder.finish();
        if cloud.is_empty() {
            return Err(IoError::Empty);
        }
        Ok(cloud)
    }
}

/// Byte length up to and including the line that starts with `marker`.
fn find_line_end(head: &[u8], marker: &[u8]) -> Option<usize> {
    let mut start = 0;
    while start < head.len() {
        let end = head[start..].iter().position(|&b| b == b'\n')? + start;
        if head[start..end].trim_ascii().starts_with(marker) {
            return Some(end + 1);
        }
        start = end + 1;
    }
    None
}

fn find_data_line_end(head: &[u8]) -> Option<usize> {
    let mut start = 0;
    while start < head.len() {
        let end = head[start..].iter().position(|&b| b == b'\n')? + start;
        if head[start..end]
            .trim_ascii()
            .to_ascii_uppercase()
            .starts_with(b"DATA")
        {
            return Some(end + 1);
        }
        start = end + 1;
    }
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{AttributeValues, CLASSIFICATION};

    fn las_file(n: usize) -> Vec<u8> {
        let mut h = vec![0u8; 227];
        h[..4].copy_from_slice(b"LASF");
        h[24] = 1;
        h[25] = 2;
        h[94..96].copy_from_slice(&227u16.to_le_bytes());
        h[96..100].copy_from_slice(&227u32.to_le_bytes());
        h[104] = 1;
        h[105..107].copy_from_slice(&28u16.to_le_bytes());
        h[107..111].copy_from_slice(&(n as u32).to_le_bytes());
        for i in 0..3 {
            h[131 + 8 * i..139 + 8 * i].copy_from_slice(&0.5f64.to_le_bytes());
        }
        for i in 0..n {
            let mut r = [0u8; 28];
            r[0..4].copy_from_slice(&(i as i32).to_le_bytes());
            r[15] = (i % 7) as u8;
            h.extend_from_slice(&r);
        }
        h
    }

    fn ply_file(n: usize) -> Vec<u8> {
        let mut f = format!(
            "ply\nformat binary_little_endian 1.0\nelement vertex {n}\nproperty float x\n\
property float y\nproperty float z\nproperty uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
        )
        .into_bytes();
        for i in 0..n {
            for v in [i as f32, 0.0, 1.0] {
                f.extend_from_slice(&v.to_le_bytes());
            }
            f.extend_from_slice(&[i as u8, 0, 0]);
        }
        f
    }

    /// Stream `file` in chunks of `chunk` bytes.
    fn stream_file(name: &str, file: &[u8], chunk: usize, keep_every: u64) -> PointCloud {
        let head_len = PointStream::header_len(name, file).unwrap();
        let mut s = PointStream::open(name, &file[..head_len]).unwrap().unwrap();
        s.set_keep_every(keep_every);
        let body = &file[s.data_offset() as usize..];
        for piece in body.chunks(chunk) {
            s.push(piece);
        }
        s.finish().unwrap()
    }

    #[test]
    fn streamed_las_matches_whole_file_read_for_any_chunking() {
        let file = las_file(1000);
        let whole = crate::read("a.las", &file).unwrap();
        for chunk in [1, 7, 28, 1000, 1 << 20] {
            assert_eq!(
                stream_file("a.las", &file, chunk, 1),
                whole,
                "chunk {chunk}"
            );
        }
    }

    #[test]
    fn streamed_ply_matches_whole_file_read() {
        let file = ply_file(777);
        let whole = crate::read("a.ply", &file).unwrap();
        for chunk in [5, 15, 4096] {
            assert_eq!(
                stream_file("a.ply", &file, chunk, 1),
                whole,
                "chunk {chunk}"
            );
        }
    }

    #[test]
    fn keep_every_thins_consistently() {
        let file = las_file(1000);
        let thinned = stream_file("a.las", &file, 100, 3);
        assert_eq!(thinned.len(), 334);
        assert_eq!(thinned.positions[1][0], 3.0 * 0.5);
        let AttributeValues::U8(c) = &thinned.attribute(CLASSIFICATION).unwrap().values else {
            panic!()
        };
        assert_eq!(c[1], 3);
        // Same result as thinning a whole-file read.
        let whole = crate::io::read_thinned("a.las", &file, 3).unwrap();
        assert_eq!(thinned, whole);
    }

    #[test]
    fn truncated_stream_is_an_error() {
        let file = las_file(10);
        let mut s = PointStream::open("a.las", &file[..375.min(file.len())])
            .unwrap()
            .unwrap();
        s.push(&file[227..227 + 28 * 5]);
        assert!(matches!(s.finish(), Err(IoError::Truncated(_))));
    }

    #[test]
    fn unstreamable_files_are_declined() {
        assert!(PointStream::open("a.xyz", b"1 2 3\n").unwrap().is_none());
        let ascii =
            b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\n\
property float z\nend_header\n1 2 3\n";
        let head = PointStream::header_len("a.ply", ascii).unwrap();
        assert!(
            PointStream::open("a.ply", &ascii[..head])
                .unwrap()
                .is_none()
        );
    }
}
