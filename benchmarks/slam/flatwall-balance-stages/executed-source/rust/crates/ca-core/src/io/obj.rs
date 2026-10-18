//! Wavefront OBJ reader (vertices and faces; everything else is ignored).

use super::IoError;
use crate::mesh::TriangleMesh;

const FORMAT: &str = "OBJ";

/// Resolve a 1-based (or negative, relative) OBJ index.
fn resolve(token: &str, vertices: usize) -> Result<u32, IoError> {
    let index_text = token.split('/').next().unwrap_or("");
    let i: i64 = index_text
        .parse()
        .map_err(|_| IoError::parse(FORMAT, token))?;
    let resolved = if i > 0 { i - 1 } else { vertices as i64 + i };
    if resolved < 0 || resolved >= vertices as i64 {
        return Err(IoError::parse(FORMAT, token));
    }
    Ok(resolved as u32)
}

pub(crate) fn read(bytes: &[u8]) -> Result<TriangleMesh, IoError> {
    let text = std::str::from_utf8(bytes).map_err(|_| IoError::parse(FORMAT, "non-UTF-8 text"))?;
    let mut mesh = TriangleMesh::default();
    let mut polygon = Vec::new();
    for line in text.lines() {
        let mut tokens = line.split_whitespace();
        match tokens.next() {
            Some("v") => {
                let mut xyz = [0.0; 3];
                for v in &mut xyz {
                    let token = tokens.next().ok_or_else(|| IoError::parse(FORMAT, line))?;
                    *v = token.parse().map_err(|_| IoError::parse(FORMAT, token))?;
                }
                mesh.vertices.push(xyz);
            }
            Some("f") => {
                polygon.clear();
                for token in tokens {
                    polygon.push(resolve(token, mesh.vertices.len())?);
                }
                for k in 1..polygon.len().saturating_sub(1) {
                    mesh.triangles
                        .push([polygon[0], polygon[k], polygon[k + 1]]);
                }
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
    fn reads_vertices_faces_and_index_forms() {
        let src = b"# cube corner\nv 0 0 0\nv 1 0 0\nv 1 1 0\nv 0 1 0\nvn 0 0 1\n\
f 1/1/1 2/2/1 3/3/1 4/4/1\nf -4//1 -2//1 -1//1\n";
        let mesh = read(src).unwrap();
        assert_eq!(mesh.vertices.len(), 4);
        assert_eq!(mesh.triangles, vec![[0, 1, 2], [0, 2, 3], [0, 2, 3]]);
    }

    #[test]
    fn rejects_out_of_range_indices() {
        assert!(read(b"v 0 0 0\nf 1 2 3\n").is_err());
    }
}
