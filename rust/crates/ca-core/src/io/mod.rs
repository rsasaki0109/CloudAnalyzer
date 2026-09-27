//! Point cloud file readers.

mod las;
mod obj;
mod pcd;
mod ply;
mod scalar;
mod stl;
mod stream;
mod write;
mod xyz;

pub use stream::PointStream;
pub use write::{ScalarField, write_csv, write_ply};

use crate::PointCloud;
use crate::mesh::TriangleMesh;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Format {
    Ply,
    Pcd,
    Las,
    Xyz,
}

impl Format {
    /// Guess the format from the file name, then from the leading bytes.
    pub fn detect(name: &str, bytes: &[u8]) -> Self {
        let ext = name.rsplit_once('.').map(|(_, e)| e.to_ascii_lowercase());
        match ext.as_deref() {
            Some("ply") => return Self::Ply,
            Some("pcd") => return Self::Pcd,
            Some("las") | Some("laz") => return Self::Las,
            Some("xyz" | "txt" | "csv" | "pts" | "asc") => return Self::Xyz,
            _ => {}
        }
        if bytes.starts_with(b"ply") {
            Self::Ply
        } else if bytes.starts_with(b"LASF") {
            Self::Las
        } else if bytes.starts_with(b"# .PCD")
            || bytes.starts_with(b"VERSION")
            || bytes.starts_with(b"FIELDS")
        {
            Self::Pcd
        } else {
            Self::Xyz
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum IoError {
    #[error("invalid {format} header: {message}")]
    Header {
        format: &'static str,
        message: String,
    },
    #[error("unexpected end of data in {0} body")]
    Truncated(&'static str),
    #[error("could not parse {format} value {value:?}")]
    Parse { format: &'static str, value: String },
    #[error("{0}")]
    Unsupported(String),
    #[error("no points found")]
    Empty,
}

impl IoError {
    pub(crate) fn header(format: &'static str, message: impl Into<String>) -> Self {
        Self::Header {
            format,
            message: message.into(),
        }
    }

    pub(crate) fn parse(format: &'static str, value: impl Into<String>) -> Self {
        Self::Parse {
            format,
            value: value.into(),
        }
    }
}

/// Read a point cloud from an in-memory file.
pub fn read(name: &str, bytes: &[u8]) -> Result<PointCloud, IoError> {
    read_thinned(name, bytes, 1)
}

/// Read a point cloud keeping every `keep_every`-th point. LAS/LAZ thin
/// while decoding (so a large LAZ never holds all its points); other formats
/// are thinned after reading.
pub fn read_thinned(name: &str, bytes: &[u8], keep_every: usize) -> Result<PointCloud, IoError> {
    let keep_every = keep_every.max(1);
    let mut cloud = match Format::detect(name, bytes) {
        Format::Ply => ply::read(bytes)?,
        Format::Pcd => pcd::read(bytes)?,
        Format::Las => las::read(bytes, keep_every)?,
        Format::Xyz => xyz::read(bytes)?,
    };
    if keep_every > 1 && Format::detect(name, bytes) != Format::Las {
        let keep: Vec<usize> = (0..cloud.len()).step_by(keep_every).collect();
        cloud = cloud.select(&keep);
    }
    if cloud.is_empty() {
        return Err(IoError::Empty);
    }
    Ok(cloud)
}

/// Number of points a file header announces (LAS, PLY, PCD), to decide on
/// thinning before reading. `None` when the header does not say.
pub fn announced_points(name: &str, head: &[u8]) -> Option<u64> {
    match Format::detect(name, head) {
        Format::Las => las::LasHeader::parse(head).ok().map(|h| h.count as u64),
        Format::Ply | Format::Pcd => {
            let len = PointStream::header_len(name, head)?;
            let text = std::str::from_utf8(&head[..len]).ok()?;
            text.lines().find_map(|line| {
                let mut t = line.split_whitespace();
                match (t.next()?, t.next()?, t.next()) {
                    ("element", "vertex", Some(n)) => n.parse().ok(),
                    ("POINTS", n, None) => n.parse().ok(),
                    _ => None,
                }
            })
        }
        Format::Xyz => None,
    }
}

/// Read a triangle mesh (PLY with faces, OBJ, STL). Returns `Ok(None)` when
/// the file is not a mesh (a PLY without faces, or another point format).
pub fn read_mesh(name: &str, bytes: &[u8]) -> Result<Option<TriangleMesh>, IoError> {
    let ext = name.rsplit_once('.').map(|(_, e)| e.to_ascii_lowercase());
    let mesh = match ext.as_deref() {
        Some("stl") => stl::read(bytes)?,
        Some("obj") => obj::read(bytes)?,
        _ if Format::detect(name, bytes) == Format::Ply => match ply::read_mesh(bytes)? {
            Some(mesh) => mesh,
            None => return Ok(None),
        },
        _ => return Ok(None),
    };
    if mesh.triangles.is_empty() {
        return Err(IoError::Unsupported(format!("{name}: mesh has no faces")));
    }
    Ok(Some(mesh))
}

/// Split `bytes` at the end of the line that satisfies `is_last`, returning
/// the header lines (without line terminators) and the remaining body.
pub(crate) fn split_header<'a>(
    bytes: &'a [u8],
    format: &'static str,
    is_last: impl Fn(&str) -> bool,
) -> Result<(Vec<&'a str>, &'a [u8]), IoError> {
    let mut lines = Vec::new();
    let mut offset = 0;
    while offset < bytes.len() {
        let end = bytes[offset..]
            .iter()
            .position(|&b| b == b'\n')
            .map_or(bytes.len(), |i| offset + i);
        let line = std::str::from_utf8(&bytes[offset..end])
            .map_err(|_| IoError::header(format, "header is not valid UTF-8"))?
            .trim_end_matches('\r');
        offset = (end + 1).min(bytes.len());
        lines.push(line);
        if is_last(line) {
            return Ok((lines, &bytes[offset..]));
        }
    }
    Err(IoError::header(format, "header terminator not found"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detects_by_extension_then_magic() {
        assert_eq!(Format::detect("a.PLY", b""), Format::Ply);
        assert_eq!(Format::detect("scan.laz", b""), Format::Las);
        assert_eq!(Format::detect("blob", b"ply\n"), Format::Ply);
        assert_eq!(Format::detect("blob", b"LASF"), Format::Las);
        assert_eq!(Format::detect("blob", b"# .PCD v0.7"), Format::Pcd);
        assert_eq!(Format::detect("blob", b"1 2 3"), Format::Xyz);
    }
}
