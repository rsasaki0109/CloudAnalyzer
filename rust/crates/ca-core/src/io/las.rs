//! LAS 1.0–1.4 and LAZ reader (point formats 0–10).

use super::IoError;
use super::stream::{RecordDecoder, decode_records};
use crate::{Attribute, AttributeValues, CLASSIFICATION, INTENSITY, PointCloud};

const FORMAT: &str = "LAS";

fn u8_at(b: &[u8], at: usize) -> Result<u8, IoError> {
    b.get(at).copied().ok_or(IoError::Truncated(FORMAT))
}

fn bytes_at<const N: usize>(b: &[u8], at: usize) -> Result<[u8; N], IoError> {
    b.get(at..at + N)
        .map(|s| s.try_into().unwrap())
        .ok_or(IoError::Truncated(FORMAT))
}

fn u16_at(b: &[u8], at: usize) -> Result<u16, IoError> {
    bytes_at(b, at).map(u16::from_le_bytes)
}

fn u32_at(b: &[u8], at: usize) -> Result<u32, IoError> {
    bytes_at(b, at).map(u32::from_le_bytes)
}

fn u64_at(b: &[u8], at: usize) -> Result<u64, IoError> {
    bytes_at(b, at).map(u64::from_le_bytes)
}

fn f64_at(b: &[u8], at: usize) -> Result<f64, IoError> {
    bytes_at(b, at).map(f64::from_le_bytes)
}

/// Byte offset of the RGB triple within a point record, if the format has one.
fn rgb_offset(format: u8) -> Option<usize> {
    match format {
        2 => Some(20),
        3 | 5 => Some(28),
        7 | 8 | 10 => Some(30),
        _ => None,
    }
}

/// What the LAS header says about the point records.
#[derive(Debug, Clone)]
pub(crate) struct LasHeader {
    pub data_offset: usize,
    pub record_len: usize,
    pub count: u64,
    pub compressed: bool,
    format: u8,
    pub scale: [f64; 3],
    pub offset: [f64; 3],
}

impl LasHeader {
    pub fn parse(b: &[u8]) -> Result<Self, IoError> {
        if !b.starts_with(b"LASF") {
            return Err(IoError::header(FORMAT, "missing 'LASF' signature"));
        }
        let minor = u8_at(b, 25)?;
        let raw_format = u8_at(b, 104)?;
        let format = raw_format & 0x3f;
        if format > 10 {
            return Err(IoError::header(
                FORMAT,
                format!("unknown point format {format}"),
            ));
        }
        let record_len = u16_at(b, 105)? as usize;
        let legacy_count = u32_at(b, 107)? as u64;
        // LAS 1.4's extended count is authoritative, including legacy mode.
        // Keep it independent of the address width (notably wasm32).
        let count = if minor >= 4 {
            u64_at(b, 247)?
        } else {
            legacy_count
        };
        if record_len < rgb_offset(format).map_or(20, |o| o + 6) {
            return Err(IoError::header(FORMAT, "point record too short"));
        }
        Ok(Self {
            data_offset: u32_at(b, 96)? as usize,
            record_len,
            count,
            // LASzip marks compressed files by setting the high bits of the format id.
            compressed: raw_format & 0xc0 != 0,
            format,
            scale: [f64_at(b, 131)?, f64_at(b, 139)?, f64_at(b, 147)?],
            offset: [f64_at(b, 155)?, f64_at(b, 163)?, f64_at(b, 171)?],
        })
    }
}

/// Turns LAS point records into points, colors, intensity and classes.
pub(crate) struct LasDecoder {
    header: LasHeader,
    rgb: Option<usize>,
    class_at: usize,
    class_mask: u8,
    positions: Vec<[f64; 3]>,
    /// LAS stores 16-bit color, but many writers only fill the low byte, so
    /// the scaling is decided once all colors are known.
    wide_colors: Vec<[u16; 3]>,
    intensity: Vec<f32>,
    classification: Vec<u8>,
}

impl LasDecoder {
    pub fn new(header: LasHeader, capacity: usize) -> Result<Self, IoError> {
        // Legacy formats pack the class into the low 5 bits of byte 15; the
        // 1.4 formats (6-10) give it the whole of byte 16.
        let (class_at, class_mask) = if header.format >= 6 {
            (16, 0xff)
        } else {
            (15, 0x1f)
        };
        let rgb = rgb_offset(header.format);
        let mut out = Self {
            rgb,
            class_at,
            class_mask,
            positions: Vec::new(),
            wide_colors: Vec::new(),
            intensity: Vec::new(),
            classification: Vec::new(),
            header,
        };
        let allocation = |e| IoError::Unsupported(format!("LAS allocation: {e}"));
        out.positions
            .try_reserve_exact(capacity)
            .map_err(allocation)?;
        out.wide_colors
            .try_reserve_exact(if rgb.is_some() { capacity } else { 0 })
            .map_err(allocation)?;
        out.intensity
            .try_reserve_exact(capacity)
            .map_err(allocation)?;
        out.classification
            .try_reserve_exact(capacity)
            .map_err(allocation)?;
        Ok(out)
    }
}

impl RecordDecoder for LasDecoder {
    fn record_len(&self) -> usize {
        self.header.record_len
    }

    fn decode_block(&mut self, records: &[u8]) {
        for record in records.chunks_exact(self.header.record_len) {
            self.decode(record);
        }
    }

    fn finish(self: Box<Self>) -> PointCloud {
        self.finish_cloud()
    }
}

impl LasDecoder {
    pub(crate) fn decode(&mut self, record: &[u8]) {
        let h = &self.header;
        self.positions.push(std::array::from_fn(|i| {
            let v = i32::from_le_bytes(record[4 * i..4 * i + 4].try_into().unwrap());
            v as f64 * h.scale[i] + h.offset[i]
        }));
        self.intensity
            .push(u16::from_le_bytes([record[12], record[13]]) as f32);
        self.classification
            .push(record[self.class_at] & self.class_mask);
        if let Some(o) = self.rgb {
            let c: [u16; 3] = std::array::from_fn(|k| {
                u16::from_le_bytes([record[o + 2 * k], record[o + 2 * k + 1]])
            });
            self.wide_colors.push(c);
        }
    }

    /// The decoded points with colors still 16-bit.
    pub(crate) fn into_raw(self) -> RawLasPoints {
        RawLasPoints {
            positions: self.positions,
            colors: self.rgb.map(|_| self.wide_colors),
            intensity: self.intensity,
            classification: self.classification,
        }
    }

    fn finish_cloud(self) -> PointCloud {
        self.into_raw().into_cloud()
    }
}

/// Decoded LAS points before colors are narrowed to 8 bits, which depends
/// on all of them (see [`RawLasPoints::into_cloud`]).
#[derive(Debug, Clone, Default)]
pub struct RawLasPoints {
    pub positions: Vec<[f64; 3]>,
    pub colors: Option<Vec<[u16; 3]>>,
    pub intensity: Vec<f32>,
    pub classification: Vec<u8>,
}

impl RawLasPoints {
    pub fn len(&self) -> usize {
        self.positions.len()
    }

    pub fn is_empty(&self) -> bool {
        self.positions.is_empty()
    }

    /// Append `other` (from the same file).
    pub fn extend(&mut self, other: RawLasPoints) {
        self.positions.extend(other.positions);
        match (&mut self.colors, other.colors) {
            (Some(a), Some(b)) => a.extend(b),
            (None, Some(b)) if self.intensity.is_empty() => self.colors = Some(b),
            _ => {}
        }
        self.intensity.extend(other.intensity);
        self.classification.extend(other.classification);
    }

    /// How far to shift 16-bit colors right to get 8 bits: 8, or 0 when
    /// every channel fits 8 bits already (many writers only fill the low byte).
    pub fn color_shift(&self) -> u32 {
        let max_channel = self
            .colors
            .iter()
            .flatten()
            .flat_map(|c| c.iter().copied())
            .max()
            .unwrap_or(0);
        if max_channel > 255 { 8 } else { 0 }
    }

    /// The cloud, with colors narrowed by [`RawLasPoints::color_shift`].
    pub fn into_cloud(self) -> PointCloud {
        let shift = self.color_shift();
        let colors = self
            .colors
            .map(|c| c.iter().map(|c| c.map(|v| (v >> shift) as u8)).collect());
        PointCloud {
            positions: self.positions,
            colors,
            attributes: vec![
                Attribute {
                    name: INTENSITY.into(),
                    values: AttributeValues::F32(self.intensity),
                },
                Attribute {
                    name: CLASSIFICATION.into(),
                    values: AttributeValues::U8(self.classification),
                },
            ],
        }
    }
}

/// Streaming decoder for an uncompressed LAS file: decoder, record offset
/// and point count. `None` for LAZ.
#[allow(clippy::type_complexity)]
pub(crate) fn stream(head: &[u8]) -> Result<Option<(Box<dyn RecordDecoder>, usize, u64)>, IoError> {
    let header = LasHeader::parse(head)?;
    if header.compressed {
        return Ok(None);
    }
    let (offset, count) = (header.data_offset, header.count);
    Ok(Some((Box::new(LasDecoder::new(header, 0)?), offset, count)))
}

/// Read a whole LAS/LAZ file, keeping every `keep_every`-th point. LAZ is
/// decompressed in blocks so that only the kept points are held in memory.
pub(crate) fn read(b: &[u8], keep_every: usize) -> Result<PointCloud, IoError> {
    let header = LasHeader::parse(b)?;
    let (data_offset, record_len, count, compressed) = (
        header.data_offset,
        header.record_len,
        header.count,
        header.compressed,
    );
    let keep_every = keep_every.max(1);
    // Validate the record span before allocating from an untrusted count.
    let records = if compressed {
        None
    } else {
        let size = count
            .checked_mul(record_len as u64)
            .and_then(|n| n.checked_add(data_offset as u64))
            .and_then(|n| usize::try_from(n).ok())
            .ok_or_else(|| IoError::header(FORMAT, "record span exceeds address space"))?;
        Some(b.get(data_offset..size).ok_or(IoError::Truncated(FORMAT))?)
    };
    let capacity = usize::try_from(count.div_ceil(keep_every as u64))
        .ok()
        .filter(|&n| n <= isize::MAX as usize / std::mem::size_of::<[f64; 3]>())
        .ok_or_else(|| {
            IoError::Unsupported("LAS output exceeds address space; use chunked reading".into())
        })?;
    let mut decoder: Box<dyn RecordDecoder> =
        Box::new(LasDecoder::new(header, capacity.min(50_000))?);
    let mut index = 0u64;
    if compressed {
        let vlr = laszip_vlr(b)?;
        let vlr = laz::LazVlr::from_buffer(vlr)
            .map_err(|e| IoError::header(FORMAT, format!("bad LASzip VLR: {e}")))?;
        let mut source = std::io::Cursor::new(b);
        source.set_position(data_offset as u64);
        let mut decompressor = laz::LasZipDecompressor::new(source, vlr)
            .map_err(|e| IoError::Unsupported(format!("LAZ: {e}")))?;
        let block_points = 50_000.min((4 << 20) / record_len).max(1);
        let mut buffer = vec![0u8; block_points * record_len];
        let mut left = count;
        while left > 0 {
            let n = left.min(block_points as u64) as usize;
            let block = &mut buffer[..n * record_len];
            decompressor
                .decompress_many(block)
                .map_err(|e| IoError::Unsupported(format!("LAZ: {e}")))?;
            decode_records(decoder.as_mut(), block, keep_every as u64, &mut index);
            left -= n as u64;
        }
    } else {
        decode_records(
            decoder.as_mut(),
            records.unwrap(),
            keep_every as u64,
            &mut index,
        );
    }
    Ok(decoder.finish())
}

/// Payload of the LASzip VLR (user id `laszip encoded`, record id 22204).
pub(crate) fn laszip_vlr(b: &[u8]) -> Result<&[u8], IoError> {
    const HEADER: usize = 54;
    let mut at = u16_at(b, 94)? as usize;
    for _ in 0..u32_at(b, 100)? {
        let user_id = b.get(at + 2..at + 18).ok_or(IoError::Truncated(FORMAT))?;
        let record_id = u16_at(b, at + 18)?;
        let len = u16_at(b, at + 20)? as usize;
        if user_id.starts_with(b"laszip encoded") && record_id == 22204 {
            return b
                .get(at + HEADER..at + HEADER + len)
                .ok_or(IoError::Truncated(FORMAT));
        }
        at += HEADER + len;
    }
    Err(IoError::header(
        FORMAT,
        "compressed file without a LASzip VLR",
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extended_count_stays_64_bit_and_is_authoritative() {
        let mut h = las(0, 20, &[]);
        h.resize(375, 0);
        h[25] = 4;
        h[107..111].copy_from_slice(&7u32.to_le_bytes());
        h[247..255].copy_from_slice(&10_000_000_000u64.to_le_bytes());
        assert_eq!(LasHeader::parse(&h).unwrap().count, 10_000_000_000);
        let (_, _, count) = stream(&h).unwrap().unwrap();
        assert_eq!(count, 10_000_000_000);
        // A header alone must fail before an allocation based on its count.
        assert!(matches!(
            read(&h, 1),
            Err(IoError::Truncated(_)) | Err(IoError::Header { .. })
        ));
        h[247..255].copy_from_slice(&u64::MAX.to_le_bytes());
        assert!(read(&h, 1).is_err());
        assert!(crate::io::PointStream::open("huge.las", &h).is_err());
    }

    fn las(format: u8, record_len: u16, records: &[Vec<u8>]) -> Vec<u8> {
        let mut h = vec![0u8; 227];
        h[..4].copy_from_slice(b"LASF");
        h[24] = 1;
        h[25] = 2;
        h[94..96].copy_from_slice(&227u16.to_le_bytes());
        h[96..100].copy_from_slice(&227u32.to_le_bytes());
        h[104] = format;
        h[105..107].copy_from_slice(&record_len.to_le_bytes());
        h[107..111].copy_from_slice(&(records.len() as u32).to_le_bytes());
        for (i, s) in [0.01f64, 0.01, 0.001].iter().enumerate() {
            h[131 + 8 * i..139 + 8 * i].copy_from_slice(&s.to_le_bytes());
        }
        for (i, o) in [500000.0f64, 4000000.0, 0.0].iter().enumerate() {
            h[155 + 8 * i..163 + 8 * i].copy_from_slice(&o.to_le_bytes());
        }
        for r in records {
            h.extend_from_slice(r);
        }
        h
    }

    fn record(xyz: [i32; 3], len: usize, rgb: Option<(usize, [u16; 3])>) -> Vec<u8> {
        let mut r = vec![0u8; len];
        for (i, v) in xyz.iter().enumerate() {
            r[4 * i..4 * i + 4].copy_from_slice(&v.to_le_bytes());
        }
        if let Some((o, c)) = rgb {
            for (i, v) in c.iter().enumerate() {
                r[o + 2 * i..o + 2 * i + 2].copy_from_slice(&v.to_le_bytes());
            }
        }
        r
    }

    #[test]
    fn reads_format_0_with_scale_and_offset() {
        let src = las(0, 20, &[record([100, -250, 1500], 20, None)]);
        let cloud = read(&src, 1).unwrap();
        let p = cloud.positions[0];
        assert!((p[0] - 500001.0).abs() < 1e-9);
        assert!((p[1] - 3999997.5).abs() < 1e-9);
        assert!((p[2] - 1.5).abs() < 1e-9);
        assert!(cloud.colors.is_none());
    }

    #[test]
    fn reads_intensity_and_classification() {
        // Format 1 (legacy class byte 15, flags in the high bits) and format 6 (byte 16).
        let mut r0 = record([0, 0, 0], 28, None);
        r0[12..14].copy_from_slice(&1234u16.to_le_bytes());
        r0[15] = 0b1110_0010; // synthetic/key-point/withheld flags + class 2
        let cloud = read(&las(1, 28, &[r0]), 1).unwrap();
        assert_eq!(
            cloud.attribute(INTENSITY).unwrap().values,
            AttributeValues::F32(vec![1234.0])
        );
        assert_eq!(
            cloud.attribute(CLASSIFICATION).unwrap().values,
            AttributeValues::U8(vec![2])
        );

        let mut r6 = record([0, 0, 0], 30, None);
        r6[12..14].copy_from_slice(&7u16.to_le_bytes());
        r6[16] = 45;
        let cloud = read(&las(6, 30, &[r6]), 1).unwrap();
        assert_eq!(
            cloud.attribute(CLASSIFICATION).unwrap().values,
            AttributeValues::U8(vec![45])
        );
    }

    #[test]
    fn reads_16bit_rgb_from_format_2() {
        let src = las(
            2,
            26,
            &[
                record([0, 0, 0], 26, Some((20, [65535, 32768, 0]))),
                record([1, 1, 1], 26, Some((20, [256, 0, 512]))),
            ],
        );
        let cloud = read(&src, 1).unwrap();
        assert_eq!(cloud.colors, Some(vec![[255, 128, 0], [1, 0, 2]]));
    }

    #[test]
    fn reads_laz_roundtrip() {
        use laz::{LasZipCompressor, LazItemRecordBuilder, LazVlrBuilder};

        let records: Vec<Vec<u8>> = (0..5000)
            .map(|i| {
                record(
                    [i * 7, -i, i % 100],
                    26,
                    Some((20, [i as u16 * 13, 0, 65535])),
                )
            })
            .collect();
        let vlr =
            LazVlrBuilder::new(LazItemRecordBuilder::default_for_point_format_id(2, 0).unwrap())
                .build();
        let mut vlr_payload = Vec::new();
        vlr.write_to(&mut vlr_payload).unwrap();

        // Header + one VLR, then the compressed points.
        let mut file = las(0x80 | 2, 26, &[]);
        let header_len = file.len();
        let mut vlr_header = vec![0u8; 54];
        vlr_header[2..16].copy_from_slice(b"laszip encoded");
        vlr_header[18..20].copy_from_slice(&22204u16.to_le_bytes());
        vlr_header[20..22].copy_from_slice(&(vlr_payload.len() as u16).to_le_bytes());
        file.extend_from_slice(&vlr_header);
        file.extend_from_slice(&vlr_payload);
        let data_offset = file.len() as u32;
        file[96..100].copy_from_slice(&data_offset.to_le_bytes());
        file[100..104].copy_from_slice(&1u32.to_le_bytes());
        file[107..111].copy_from_slice(&(records.len() as u32).to_le_bytes());
        assert_eq!(header_len, 227);

        let mut cursor = std::io::Cursor::new(file);
        cursor.set_position(data_offset as u64);
        {
            let mut compressor = LasZipCompressor::new(&mut cursor, vlr).unwrap();
            compressor.compress_many(&records.concat()).unwrap();
            compressor.done().unwrap();
        }
        let file = cursor.into_inner();

        let cloud = read(&file, 1).unwrap();
        assert_eq!(cloud.len(), 5000);
        let p = cloud.positions[4999];
        assert!((p[0] - (500000.0 + 4999.0 * 7.0 * 0.01)).abs() < 1e-6);
        assert!((p[1] - (4000000.0 - 49.99)).abs() < 1e-6);
        assert!((p[2] - 0.099).abs() < 1e-9);
        assert_eq!(cloud.colors.unwrap()[1], [0, 0, 255]);
    }
}
