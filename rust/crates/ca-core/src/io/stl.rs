//! STL reader (binary and ASCII). Vertices are not merged.

use super::IoError;
use crate::mesh::TriangleMesh;

const FORMAT: &str = "STL";

pub(crate) fn read(bytes: &[u8]) -> Result<TriangleMesh, IoError> {
    // Binary files may also start with "solid", so trust the size check first.
    if bytes.len() >= 84 {
        let count = u32::from_le_bytes(bytes[80..84].try_into().unwrap()) as usize;
        if bytes.len() == 84 + 50 * count {
            return Ok(read_binary(&bytes[84..], count));
        }
    }
    if bytes.trim_ascii_start().starts_with(b"solid") {
        return read_ascii(bytes);
    }
    Err(IoError::header(FORMAT, "neither binary nor ASCII STL"))
}

fn read_binary(records: &[u8], count: usize) -> TriangleMesh {
    let mut mesh = TriangleMesh {
        vertices: Vec::with_capacity(3 * count),
        triangles: Vec::with_capacity(count),
    };
    for record in records.chunks_exact(50) {
        let base = mesh.vertices.len() as u32;
        for corner in 0..3 {
            let at = 12 + 12 * corner;
            mesh.vertices.push(std::array::from_fn(|a| {
                let o = at + 4 * a;
                f32::from_le_bytes(record[o..o + 4].try_into().unwrap()) as f64
            }));
        }
        mesh.triangles.push([base, base + 1, base + 2]);
    }
    mesh
}

fn read_ascii(bytes: &[u8]) -> Result<TriangleMesh, IoError> {
    let text = std::str::from_utf8(bytes).map_err(|_| IoError::parse(FORMAT, "non-UTF-8 text"))?;
    let mut mesh = TriangleMesh::default();
    let mut corners = 0;
    for line in text.lines() {
        let mut tokens = line.split_whitespace();
        match tokens.next() {
            Some("vertex") => {
                let mut xyz = [0.0; 3];
                for v in &mut xyz {
                    let token = tokens.next().ok_or_else(|| IoError::parse(FORMAT, line))?;
                    *v = token.parse().map_err(|_| IoError::parse(FORMAT, token))?;
                }
                mesh.vertices.push(xyz);
                corners += 1;
            }
            Some("endloop") => {
                if corners != 3 {
                    return Err(IoError::parse(FORMAT, "facet without 3 vertices"));
                }
                let base = mesh.vertices.len() as u32 - 3;
                mesh.triangles.push([base, base + 1, base + 2]);
                corners = 0;
            }
            _ => {}
        }
    }
    Ok(mesh)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_binary_even_with_solid_header() {
        let mut src = b"solid but actually binary".to_vec();
        src.resize(80, 0);
        src.extend_from_slice(&2u32.to_le_bytes());
        for t in 0..2 {
            src.extend_from_slice(&[0u8; 12]); // normal
            for v in [[0.0f32, 0.0, t as f32], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]] {
                for c in v {
                    src.extend_from_slice(&c.to_le_bytes());
                }
            }
            src.extend_from_slice(&[0, 0]);
        }
        let mesh = read(&src).unwrap();
        assert_eq!(mesh.triangles, vec![[0, 1, 2], [3, 4, 5]]);
        assert_eq!(mesh.vertices[3], [0.0, 0.0, 1.0]);
    }

    #[test]
    fn reads_ascii() {
        let src = b"solid t\n facet normal 0 0 1\n  outer loop\n   vertex 0 0 0\n   vertex 1 0 0\n\
   vertex 0 1 2.5\n  endloop\n endfacet\nendsolid t\n";
        let mesh = read(src).unwrap();
        assert_eq!(
            mesh.vertices,
            vec![[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 2.5]]
        );
        assert_eq!(mesh.triangles, vec![[0, 1, 2]]);
    }
}
