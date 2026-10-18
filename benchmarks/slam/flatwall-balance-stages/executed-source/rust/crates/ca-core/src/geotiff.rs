//! A minimal GeoTIFF writer for height rasters: one Float32 band in
//! uncompressed strips, placed by ModelPixelScale + ModelTiepoint. No CRS is
//! written: the clouds carry none we could trust, and GIS tools let the user
//! assign one.

use crate::volume::Grid;

/// Value of cells without data (declared in the GDAL_NODATA tag).
pub const NODATA: f32 = -9999.0;

/// Aim for strips of about this many bytes, as TIFF writers commonly do.
const STRIP_BYTES: usize = 64 * 1024;

enum Value {
    Short(Vec<u16>),
    Long(Vec<u32>),
    Double(Vec<f64>),
    Ascii(&'static str),
}

impl Value {
    /// TIFF field type and value count.
    fn kind(&self) -> (u16, usize) {
        match self {
            Value::Short(v) => (3, v.len()),
            Value::Long(v) => (4, v.len()),
            Value::Double(v) => (12, v.len()),
            Value::Ascii(s) => (2, s.len()),
        }
    }

    fn bytes(&self) -> Vec<u8> {
        match self {
            Value::Short(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Value::Long(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Value::Double(v) => v.iter().flat_map(|x| x.to_le_bytes()).collect(),
            Value::Ascii(s) => s.as_bytes().to_vec(),
        }
    }
}

/// Encode `heights` (row-major from the grid's lowest y, as
/// [`crate::raster::Raster`]; NaN = no data) as a little-endian GeoTIFF.
/// The image's first row is the grid's highest y, as GIS tools expect.
///
/// # Panics
/// If `heights` does not have one value per grid cell.
pub fn write_geotiff(grid: &Grid, heights: &[f32]) -> Vec<u8> {
    let (nx, ny) = (grid.nx, grid.ny);
    assert_eq!(heights.len(), nx * ny, "one height per cell");
    let row_bytes = nx * 4;
    let rows_per_strip = (STRIP_BYTES / row_bytes).clamp(1, ny.max(1));
    let strips = ny.div_ceil(rows_per_strip);
    let strip_bytes: Vec<u32> = (0..strips)
        .map(|s| ((ny - s * rows_per_strip).min(rows_per_strip) * row_bytes) as u32)
        .collect();
    let top = grid.min[1] + ny as f64 * grid.cell;
    // Sorted by tag, as TIFF requires. Strip offsets are filled in below.
    let mut entries: Vec<(u16, Value)> = vec![
        (256, Value::Long(vec![nx as u32])),             // ImageWidth
        (257, Value::Long(vec![ny as u32])),             // ImageLength
        (258, Value::Short(vec![32])),                   // BitsPerSample
        (259, Value::Short(vec![1])),                    // Compression: none
        (262, Value::Short(vec![1])),                    // Photometric: BlackIsZero
        (273, Value::Long(vec![0; strips])),             // StripOffsets
        (277, Value::Short(vec![1])),                    // SamplesPerPixel
        (278, Value::Long(vec![rows_per_strip as u32])), // RowsPerStrip
        (279, Value::Long(strip_bytes)),                 // StripByteCounts
        (284, Value::Short(vec![1])),                    // PlanarConfiguration: chunky
        (339, Value::Short(vec![3])),                    // SampleFormat: IEEE float
        // ModelPixelScale: cell size in x, y (z unused).
        (33550, Value::Double(vec![grid.cell, grid.cell, 0.0])),
        // ModelTiepoint: raster (0, 0) is the grid's top-left corner.
        (
            33922,
            Value::Double(vec![0.0, 0.0, 0.0, grid.min[0], top, 0.0]),
        ),
        // GeoKeyDirectory v1.1.0 with one key: GTRasterTypeGeoKey =
        // RasterPixelIsArea, i.e. the tie point is a pixel corner.
        (34735, Value::Short(vec![1, 1, 0, 1, 1025, 0, 1, 1])),
        (42113, Value::Ascii("-9999\0")), // GDAL_NODATA
    ];

    // Layout: header, IFD, values that do not fit in 4 bytes, image data.
    let ifd_len = 2 + 12 * entries.len() + 4;
    let mut offset = 8 + ifd_len;
    let mut placed = Vec::with_capacity(entries.len());
    for (_, value) in &entries {
        let len = value.bytes().len();
        if len > 4 {
            offset = offset.next_multiple_of(8);
            placed.push(Some(offset));
            offset += len;
        } else {
            placed.push(None);
        }
    }
    let image = offset.next_multiple_of(8);
    if let Some((_, Value::Long(offsets))) = entries.iter_mut().find(|(tag, _)| *tag == 273) {
        for (s, o) in offsets.iter_mut().enumerate() {
            *o = (image + s * rows_per_strip * row_bytes) as u32;
        }
    }

    let mut out = Vec::with_capacity(image + nx * ny * 4);
    out.extend_from_slice(b"II");
    out.extend_from_slice(&42u16.to_le_bytes());
    out.extend_from_slice(&8u32.to_le_bytes());
    out.extend_from_slice(&(entries.len() as u16).to_le_bytes());
    let mut outside = Vec::new();
    for ((tag, value), at) in entries.iter().zip(&placed) {
        let (kind, count) = value.kind();
        out.extend_from_slice(&tag.to_le_bytes());
        out.extend_from_slice(&kind.to_le_bytes());
        out.extend_from_slice(&(count as u32).to_le_bytes());
        let mut bytes = value.bytes();
        match at {
            Some(at) => {
                out.extend_from_slice(&(*at as u32).to_le_bytes());
                outside.push((*at, bytes));
            }
            None => {
                bytes.resize(4, 0);
                out.extend_from_slice(&bytes);
            }
        }
    }
    out.extend_from_slice(&0u32.to_le_bytes()); // no further IFD
    for (at, bytes) in outside {
        out.resize(at, 0);
        out.extend_from_slice(&bytes);
    }
    out.resize(image, 0);
    for j in (0..ny).rev() {
        for &h in &heights[j * nx..(j + 1) * nx] {
            let v = if h.is_nan() { NODATA } else { h };
            out.extend_from_slice(&v.to_le_bytes());
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn u16_at(b: &[u8], o: usize) -> u16 {
        u16::from_le_bytes([b[o], b[o + 1]])
    }

    fn u32_at(b: &[u8], o: usize) -> u32 {
        u32::from_le_bytes(b[o..o + 4].try_into().unwrap())
    }

    /// Tag -> (type, count, raw value bytes), read back from the first IFD.
    fn parse(b: &[u8]) -> HashMap<u16, (u16, usize, Vec<u8>)> {
        assert_eq!(&b[..4], b"II\x2a\x00");
        let ifd = u32_at(b, 4) as usize;
        let n = u16_at(b, ifd) as usize;
        let mut tags = HashMap::new();
        let mut last = 0;
        for e in 0..n {
            let o = ifd + 2 + 12 * e;
            let (tag, kind, count) = (u16_at(b, o), u16_at(b, o + 2), u32_at(b, o + 4) as usize);
            assert!(tag > last, "tags must be sorted");
            last = tag;
            let size = count * [0, 1, 1, 2, 4, 8, 1, 1, 2, 4, 8, 4, 8][kind as usize];
            let at = if size <= 4 {
                o + 8
            } else {
                u32_at(b, o + 8) as usize
            };
            tags.insert(tag, (kind, count, b[at..at + size].to_vec()));
        }
        assert_eq!(u32_at(b, ifd + 2 + 12 * n), 0);
        tags
    }

    fn longs(v: &[u8]) -> Vec<u32> {
        v.chunks(4).map(|c| u32_at(c, 0)).collect()
    }

    fn doubles(v: &[u8]) -> Vec<f64> {
        v.chunks(8)
            .map(|c| f64::from_le_bytes(c.try_into().unwrap()))
            .collect()
    }

    #[test]
    fn header_tags_and_values_read_back() {
        let grid = Grid {
            min: [500_000.0, 4_000_000.0],
            cell: 0.5,
            nx: 3,
            ny: 2,
        };
        // Row j = 0 is the lowest y, so it comes out as the image's last row.
        let heights = [1.0, 2.0, f32::NAN, 4.0, 5.0, 6.5];
        let tiff = write_geotiff(&grid, &heights);
        let tags = parse(&tiff);
        let short = |tag| u16_at(&tags[&tag].2, 0);
        assert_eq!(longs(&tags[&256].2), [3]);
        assert_eq!(longs(&tags[&257].2), [2]);
        assert_eq!(short(258), 32);
        assert_eq!(short(259), 1);
        assert_eq!(short(277), 1);
        assert_eq!(short(339), 3);
        assert_eq!(doubles(&tags[&33550].2), [0.5, 0.5, 0.0]);
        assert_eq!(
            doubles(&tags[&33922].2),
            [0.0, 0.0, 0.0, 500_000.0, 4_000_001.0, 0.0]
        );
        assert_eq!(tags[&34735].1, 8);
        assert_eq!(tags[&42113].2, b"-9999\0");

        let offsets = longs(&tags[&273].2);
        let counts = longs(&tags[&279].2);
        let mut pixels = Vec::new();
        for (&o, &n) in offsets.iter().zip(&counts) {
            let strip = &tiff[o as usize..(o + n) as usize];
            pixels.extend(
                strip
                    .chunks(4)
                    .map(|c| f32::from_le_bytes(c.try_into().unwrap())),
            );
        }
        assert_eq!(pixels, [4.0, 5.0, 6.5, 1.0, 2.0, NODATA]);
        assert_eq!(tiff.len(), offsets[0] as usize + 6 * 4);
    }

    #[test]
    fn large_rasters_split_into_strips() {
        let grid = Grid {
            min: [0.0, 0.0],
            cell: 1.0,
            nx: 1000,
            ny: 50,
        };
        let heights: Vec<f32> = (0..grid.nx * grid.ny).map(|k| k as f32).collect();
        let tiff = write_geotiff(&grid, &heights);
        let tags = parse(&tiff);
        let rows = longs(&tags[&278].2)[0] as usize;
        let offsets = longs(&tags[&273].2);
        let counts = longs(&tags[&279].2);
        assert_eq!(rows, 16);
        assert_eq!(offsets.len(), 4);
        assert_eq!(counts.iter().sum::<u32>() as usize, 1000 * 50 * 4);
        // The last strip ends the file and holds the last row of the image,
        // i.e. the grid's first row.
        let last = *offsets.last().unwrap() as usize + *counts.last().unwrap() as usize;
        assert_eq!(last, tiff.len());
        let first_value = |row: usize| {
            let o = offsets[0] as usize + row * 1000 * 4;
            f32::from_le_bytes(tiff[o..o + 4].try_into().unwrap())
        };
        assert_eq!(first_value(0), (49 * 1000) as f32);
        assert_eq!(first_value(49), 0.0);
    }
}
