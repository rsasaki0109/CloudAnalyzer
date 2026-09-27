//! PLY reader (ascii, binary_little_endian, binary_big_endian).

use super::scalar::{Scalar, color_channel};
use super::{IoError, split_header};
use crate::PointCloud;

const FORMAT: &str = "PLY";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Encoding {
    Ascii,
    BinaryLe,
    BinaryBe,
}

#[derive(Debug)]
enum Property {
    Scalar { name: String, kind: Scalar },
    List { count: Scalar, item: Scalar },
}

#[derive(Debug)]
struct Element {
    name: String,
    count: usize,
    properties: Vec<Property>,
}

fn parse_scalar(name: &str) -> Result<Scalar, IoError> {
    Ok(match name {
        "char" | "int8" => Scalar::I8,
        "uchar" | "uint8" => Scalar::U8,
        "short" | "int16" => Scalar::I16,
        "ushort" | "uint16" => Scalar::U16,
        "int" | "int32" => Scalar::I32,
        "uint" | "uint32" => Scalar::U32,
        "float" | "float32" => Scalar::F32,
        "double" | "float64" => Scalar::F64,
        other => return Err(IoError::header(FORMAT, format!("unknown type {other:?}"))),
    })
}

fn parse_header(lines: &[&str]) -> Result<(Encoding, Vec<Element>), IoError> {
    if lines.first().map(|l| l.trim()) != Some("ply") {
        return Err(IoError::header(FORMAT, "missing 'ply' magic"));
    }
    let mut encoding = None;
    let mut elements: Vec<Element> = Vec::new();
    for line in &lines[1..] {
        let tokens: Vec<&str> = line.split_whitespace().collect();
        match tokens.as_slice() {
            ["format", fmt, ..] => {
                encoding = Some(match *fmt {
                    "ascii" => Encoding::Ascii,
                    "binary_little_endian" => Encoding::BinaryLe,
                    "binary_big_endian" => Encoding::BinaryBe,
                    other => {
                        return Err(IoError::header(FORMAT, format!("unknown format {other:?}")));
                    }
                });
            }
            ["element", name, count] => elements.push(Element {
                name: (*name).to_owned(),
                count: count
                    .parse()
                    .map_err(|_| IoError::header(FORMAT, format!("bad element count {count:?}")))?,
                properties: Vec::new(),
            }),
            ["property", "list", count, item, _name] => {
                let element = elements
                    .last_mut()
                    .ok_or_else(|| IoError::header(FORMAT, "property before element"))?;
                element.properties.push(Property::List {
                    count: parse_scalar(count)?,
                    item: parse_scalar(item)?,
                });
            }
            ["property", kind, name] => {
                let element = elements
                    .last_mut()
                    .ok_or_else(|| IoError::header(FORMAT, "property before element"))?;
                element.properties.push(Property::Scalar {
                    name: (*name).to_owned(),
                    kind: parse_scalar(kind)?,
                });
            }
            _ => {} // comment, obj_info, end_header
        }
    }
    let encoding = encoding.ok_or_else(|| IoError::header(FORMAT, "missing format line"))?;
    Ok((encoding, elements))
}

/// Indices (into the element's properties) of the vertex fields we read.
struct VertexLayout {
    xyz: [usize; 3],
    rgb: Option<([usize; 3], [Scalar; 3])>,
}

fn vertex_layout(element: &Element) -> Result<VertexLayout, IoError> {
    let find = |names: &[&str]| {
        element.properties.iter().position(
            |p| matches!(p, Property::Scalar { name, .. } if names.contains(&name.as_str())),
        )
    };
    let axis =
        |n: &str| find(&[n]).ok_or_else(|| IoError::header(FORMAT, format!("vertex has no {n}")));
    let xyz = [axis("x")?, axis("y")?, axis("z")?];
    let kind = |i: usize| match element.properties[i] {
        Property::Scalar { kind, .. } => kind,
        Property::List { item, .. } => item,
    };
    let rgb = match (
        find(&["red", "r", "diffuse_red"]),
        find(&["green", "g", "diffuse_green"]),
        find(&["blue", "b", "diffuse_blue"]),
    ) {
        (Some(r), Some(g), Some(b)) => Some(([r, g, b], [kind(r), kind(g), kind(b)])),
        _ => None,
    };
    Ok(VertexLayout { xyz, rgb })
}

impl VertexLayout {
    fn push(&self, cloud: &mut PointCloud, values: &[f64]) {
        cloud.positions.push(self.xyz.map(|i| values[i]));
        if let (Some(colors), Some((idx, kinds))) = (cloud.colors.as_mut(), self.rgb) {
            colors.push(std::array::from_fn(|c| {
                color_channel(values[idx[c]], kinds[c])
            }));
        }
    }

    fn empty_cloud(&self, count: usize) -> PointCloud {
        PointCloud {
            positions: Vec::with_capacity(count),
            colors: self.rgb.map(|_| Vec::with_capacity(count)),
        }
    }
}

pub(crate) fn read(bytes: &[u8]) -> Result<PointCloud, IoError> {
    let (lines, body) = split_header(bytes, FORMAT, |l| l.trim() == "end_header")?;
    let (encoding, elements) = parse_header(&lines)?;
    if !elements.iter().any(|e| e.name == "vertex") {
        return Err(IoError::header(FORMAT, "no vertex element"));
    }
    match encoding {
        Encoding::Ascii => read_ascii(body, &elements),
        Encoding::BinaryLe => read_binary(body, &elements, true),
        Encoding::BinaryBe => read_binary(body, &elements, false),
    }
}

fn read_binary(body: &[u8], elements: &[Element], le: bool) -> Result<PointCloud, IoError> {
    let mut cursor = 0usize;
    let mut take = |n: usize| -> Result<&[u8], IoError> {
        let slice = body
            .get(cursor..cursor + n)
            .ok_or(IoError::Truncated(FORMAT))?;
        cursor += n;
        Ok(slice)
    };
    for element in elements {
        let layout = if element.name == "vertex" {
            Some(vertex_layout(element)?)
        } else {
            None
        };
        let mut cloud = layout
            .as_ref()
            .map(|l| l.empty_cloud(element.count))
            .unwrap_or_default();
        let mut values = vec![0.0f64; element.properties.len()];
        for _ in 0..element.count {
            for (i, property) in element.properties.iter().enumerate() {
                match *property {
                    Property::Scalar { kind, .. } => {
                        values[i] = kind.decode(take(kind.size())?, le);
                    }
                    Property::List { count, item } => {
                        let n = count.decode(take(count.size())?, le) as usize;
                        take(n * item.size())?;
                    }
                }
            }
            if let Some(layout) = &layout {
                layout.push(&mut cloud, &values);
            }
        }
        if layout.is_some() {
            return Ok(cloud);
        }
    }
    unreachable!("vertex element presence checked by caller")
}

fn read_ascii(body: &[u8], elements: &[Element]) -> Result<PointCloud, IoError> {
    let text = std::str::from_utf8(body).map_err(|_| IoError::parse(FORMAT, "non-UTF-8 body"))?;
    let mut rows = text.lines().filter(|l| !l.trim().is_empty());
    for element in elements {
        if element.name != "vertex" {
            for _ in 0..element.count {
                rows.next().ok_or(IoError::Truncated(FORMAT))?;
            }
            continue;
        }
        if element
            .properties
            .iter()
            .any(|p| matches!(p, Property::List { .. }))
        {
            return Err(IoError::Unsupported(
                "ASCII PLY vertices with list properties are not supported".into(),
            ));
        }
        let layout = vertex_layout(element)?;
        let mut cloud = layout.empty_cloud(element.count);
        let mut values = Vec::with_capacity(element.properties.len());
        for _ in 0..element.count {
            let row = rows.next().ok_or(IoError::Truncated(FORMAT))?;
            values.clear();
            for token in row.split_whitespace() {
                values.push(
                    token
                        .parse::<f64>()
                        .map_err(|_| IoError::parse(FORMAT, token))?,
                );
            }
            if values.len() < element.properties.len() {
                return Err(IoError::parse(FORMAT, row));
            }
            layout.push(&mut cloud, &values);
        }
        return Ok(cloud);
    }
    unreachable!("vertex element presence checked by caller")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_ascii_with_colors_and_leading_element() {
        let src =
            b"ply\r\nformat ascii 1.0\r\ncomment hi\r\nelement camera 1\r\nproperty float fov\r\n\
element vertex 2\r\nproperty float x\r\nproperty float y\r\nproperty float z\r\n\
property uchar red\r\nproperty uchar green\r\nproperty uchar blue\r\nend_header\r\n\
60\r\n1 2 3 255 0 10\r\n-1.5 0 4e2 1 2 3\r\n";
        let cloud = read(src).unwrap();
        assert_eq!(cloud.positions, vec![[1.0, 2.0, 3.0], [-1.5, 0.0, 400.0]]);
        assert_eq!(cloud.colors, Some(vec![[255, 0, 10], [1, 2, 3]]));
    }

    #[test]
    fn reads_binary_le_with_faces_after_vertices() {
        let mut src = b"ply\nformat binary_little_endian 1.0\nelement vertex 2\n\
property double x\nproperty double y\nproperty double z\nproperty float nx\n\
element face 1\nproperty list uchar int vertex_indices\nend_header\n"
            .to_vec();
        for p in [[1.0f64, 2.0, 3.0], [4.0, 5.0, 6.0]] {
            for v in p {
                src.extend_from_slice(&v.to_le_bytes());
            }
            src.extend_from_slice(&0.5f32.to_le_bytes());
        }
        src.push(3);
        for i in [0i32, 1, 0] {
            src.extend_from_slice(&i.to_le_bytes());
        }
        let cloud = read(&src).unwrap();
        assert_eq!(cloud.positions, vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        assert_eq!(cloud.colors, None);
    }

    #[test]
    fn reads_binary_be_and_skips_leading_list_element() {
        let mut src = b"ply\nformat binary_big_endian 1.0\nelement meta 1\n\
property list uchar short ids\nelement vertex 1\nproperty float x\nproperty float y\n\
property float z\nproperty float red\nproperty float green\nproperty float blue\nend_header\n"
            .to_vec();
        src.push(2);
        src.extend_from_slice(&7i16.to_be_bytes());
        src.extend_from_slice(&8i16.to_be_bytes());
        for v in [1.0f32, -2.0, 3.5, 1.0, 0.5, 0.0] {
            src.extend_from_slice(&v.to_be_bytes());
        }
        let cloud = read(&src).unwrap();
        assert_eq!(cloud.positions, vec![[1.0, -2.0, 3.5]]);
        assert_eq!(cloud.colors, Some(vec![[255, 128, 0]]));
    }

    #[test]
    fn truncated_binary_is_an_error() {
        let src = b"ply\nformat binary_little_endian 1.0\nelement vertex 2\n\
property float x\nproperty float y\nproperty float z\nend_header\n\0\0\0\0";
        assert!(matches!(read(src), Err(IoError::Truncated(_))));
    }
}
