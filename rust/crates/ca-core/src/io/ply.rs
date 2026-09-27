//! PLY reader (ascii, binary_little_endian, binary_big_endian).

use super::scalar::{Scalar, color_channel};
use super::{IoError, split_header};
use crate::mesh::TriangleMesh;
use crate::{Attribute, AttributeValues, CLASSIFICATION, INTENSITY, PointCloud};

const FORMAT: &str = "PLY";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Encoding {
    Ascii,
    BinaryLe,
    BinaryBe,
}

#[derive(Debug)]
enum Property {
    Scalar {
        name: String,
        kind: Scalar,
    },
    List {
        name: String,
        count: Scalar,
        item: Scalar,
    },
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
            ["property", "list", count, item, name] => {
                let element = elements
                    .last_mut()
                    .ok_or_else(|| IoError::header(FORMAT, "property before element"))?;
                element.properties.push(Property::List {
                    name: (*name).to_owned(),
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
    intensity: Option<usize>,
    classification: Option<usize>,
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
    Ok(VertexLayout {
        xyz,
        rgb,
        // CloudCompare writes scalar fields as `scalar_<name>`.
        intensity: find(&["intensity", "scalar_intensity", "scalar_Intensity"]),
        classification: find(&[
            "classification",
            "scalar_classification",
            "scalar_Classification",
        ]),
    })
}

impl VertexLayout {
    fn push(&self, cloud: &mut PointCloud, values: &[f64]) {
        cloud.positions.push(self.xyz.map(|i| values[i]));
        let mut slot = 0;
        if let Some(i) = self.intensity {
            if let AttributeValues::F32(v) = &mut cloud.attributes[slot].values {
                v.push(values[i] as f32);
            }
            slot += 1;
        }
        if let Some(i) = self.classification
            && let AttributeValues::U8(v) = &mut cloud.attributes[slot].values
        {
            v.push(values[i].clamp(0.0, 255.0) as u8);
        }
        if let (Some(colors), Some((idx, kinds))) = (cloud.colors.as_mut(), self.rgb) {
            colors.push(std::array::from_fn(|c| {
                color_channel(values[idx[c]], kinds[c])
            }));
        }
    }

    fn empty_cloud(&self, count: usize) -> PointCloud {
        let mut attributes = Vec::new();
        if self.intensity.is_some() {
            attributes.push(Attribute {
                name: INTENSITY.into(),
                values: AttributeValues::F32(Vec::with_capacity(count)),
            });
        }
        if self.classification.is_some() {
            attributes.push(Attribute {
                name: CLASSIFICATION.into(),
                values: AttributeValues::U8(Vec::with_capacity(count)),
            });
        }
        PointCloud {
            positions: Vec::with_capacity(count),
            colors: self.rgb.map(|_| Vec::with_capacity(count)),
            attributes,
        }
    }
}

/// Index of the face element's vertex index list, if the file has faces.
fn face_list(elements: &[Element]) -> Option<(usize, usize)> {
    elements.iter().enumerate().find_map(|(e, element)| {
        if element.name != "face" || element.count == 0 {
            return None;
        }
        element
            .properties
            .iter()
            .position(|p| {
                matches!(p, Property::List { name, .. } if name == "vertex_indices" || name == "vertex_index")
            })
            .map(|p| (e, p))
    })
}

/// Read a PLY with faces as a triangle mesh (polygons are fan-triangulated).
/// Returns `Ok(None)` for a PLY without faces.
pub(crate) fn read_mesh(bytes: &[u8]) -> Result<Option<TriangleMesh>, IoError> {
    let (lines, body) = split_header(bytes, FORMAT, |l| l.trim() == "end_header")?;
    let (encoding, elements) = parse_header(&lines)?;
    let Some((face_element, face_property)) = face_list(&elements) else {
        return Ok(None);
    };
    let mut mesh = TriangleMesh::default();
    let mut vertex_xyz = None;
    let mut values = Vec::new();
    let mut polygon: Vec<u32> = Vec::new();
    let mut rows = RowReader::new(body, encoding)?;
    for (e, element) in elements.iter().enumerate() {
        if element.name == "vertex" {
            vertex_xyz = Some(vertex_layout(element)?.xyz);
        }
        for _ in 0..element.count {
            values.clear();
            polygon.clear();
            rows.start_row()?;
            for (i, property) in element.properties.iter().enumerate() {
                match property {
                    Property::Scalar { kind, .. } => values.push(rows.scalar(*kind)?),
                    Property::List { count, item, .. } => {
                        values.push(0.0);
                        let n = rows.scalar(*count)? as usize;
                        for _ in 0..n {
                            let v = rows.scalar(*item)?;
                            if e == face_element && i == face_property {
                                polygon.push(v as u32);
                            }
                        }
                    }
                }
            }
            if element.name == "vertex" {
                let xyz = vertex_xyz.expect("set above");
                mesh.vertices.push(xyz.map(|i| values[i]));
            } else if e == face_element {
                for k in 1..polygon.len().saturating_sub(1) {
                    mesh.triangles
                        .push([polygon[0], polygon[k], polygon[k + 1]]);
                }
            }
        }
    }
    mesh.validate();
    Ok(Some(mesh))
}

/// Sequential scalar reader over a PLY body in any encoding.
enum RowReader<'a> {
    Binary {
        body: &'a [u8],
        cursor: usize,
        le: bool,
    },
    Ascii {
        lines: std::str::Lines<'a>,
        tokens: std::str::SplitWhitespace<'a>,
    },
}

impl<'a> RowReader<'a> {
    fn new(body: &'a [u8], encoding: Encoding) -> Result<Self, IoError> {
        Ok(match encoding {
            Encoding::BinaryLe | Encoding::BinaryBe => Self::Binary {
                body,
                cursor: 0,
                le: encoding == Encoding::BinaryLe,
            },
            Encoding::Ascii => {
                let text = std::str::from_utf8(body)
                    .map_err(|_| IoError::parse(FORMAT, "non-UTF-8 body"))?;
                Self::Ascii {
                    lines: text.lines(),
                    tokens: "".split_whitespace(),
                }
            }
        })
    }

    /// ASCII rows are one line each; binary rows have no delimiter.
    fn start_row(&mut self) -> Result<(), IoError> {
        if let Self::Ascii { lines, tokens } = self {
            let line = loop {
                let line = lines.next().ok_or(IoError::Truncated(FORMAT))?;
                if !line.trim().is_empty() {
                    break line;
                }
            };
            *tokens = line.split_whitespace();
        }
        Ok(())
    }

    fn scalar(&mut self, kind: Scalar) -> Result<f64, IoError> {
        match self {
            Self::Binary { body, cursor, le } => {
                let bytes = body
                    .get(*cursor..*cursor + kind.size())
                    .ok_or(IoError::Truncated(FORMAT))?;
                *cursor += kind.size();
                Ok(kind.decode(bytes, *le))
            }
            Self::Ascii { tokens, .. } => {
                let token = tokens.next().ok_or(IoError::Truncated(FORMAT))?;
                token.parse().map_err(|_| IoError::parse(FORMAT, token))
            }
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

/// Record size of an element without list properties.
fn fixed_stride(element: &Element) -> Option<usize> {
    element
        .properties
        .iter()
        .map(|p| match p {
            Property::Scalar { kind, .. } => Some(kind.size()),
            Property::List { .. } => None,
        })
        .sum()
}

fn read_fixed_vertices(
    records: &[u8],
    stride: usize,
    element: &Element,
    le: bool,
) -> Result<PointCloud, IoError> {
    let layout = vertex_layout(element)?;
    let mut offsets = Vec::with_capacity(element.properties.len());
    let mut offset = 0;
    for property in &element.properties {
        offsets.push(offset);
        if let Property::Scalar { kind, .. } = property {
            offset += kind.size();
        }
    }
    let field = |i: usize| match element.properties[i] {
        Property::Scalar { kind, .. } => (offsets[i], kind),
        Property::List { .. } => unreachable!("fixed-size element"),
    };
    let xyz = layout.xyz.map(field);
    let rgb = layout.rgb.map(|(idx, _)| idx.map(field));
    let mut cloud = layout.empty_cloud(element.count);
    // The common little-endian float/double + uchar layouts get a
    // monomorphic loop; everything else goes through `Scalar::decode`.
    let plain = layout.intensity.is_none() && layout.classification.is_none();
    if le && plain && read_common_layout(records, stride, xyz, rgb, &mut cloud) {
        return Ok(cloud);
    }
    // Stride 0 would mean a vertex with no properties, which vertex_layout rejects.
    let intensity = layout.intensity.map(field);
    let classification = layout.classification.map(field);
    for record in records.chunks_exact(stride) {
        cloud
            .positions
            .push(xyz.map(|(at, kind)| kind.decode(&record[at..], le)));
        if let (Some(colors), Some(rgb)) = (cloud.colors.as_mut(), rgb) {
            colors.push(rgb.map(|(at, kind)| color_channel(kind.decode(&record[at..], le), kind)));
        }
        let mut slot = 0;
        if let Some((at, kind)) = intensity {
            if let AttributeValues::F32(v) = &mut cloud.attributes[slot].values {
                v.push(kind.decode(&record[at..], le) as f32);
            }
            slot += 1;
        }
        if let Some((at, kind)) = classification
            && let AttributeValues::U8(v) = &mut cloud.attributes[slot].values
        {
            v.push(kind.decode(&record[at..], le).clamp(0.0, 255.0) as u8);
        }
    }
    Ok(cloud)
}

/// Fast path for little-endian x/y/z that are all `float` or all `double`
/// and colors (if any) that are all `uchar`. Returns false if not applicable.
fn read_common_layout(
    records: &[u8],
    stride: usize,
    xyz: [(usize, Scalar); 3],
    rgb: Option<[(usize, Scalar); 3]>,
    cloud: &mut PointCloud,
) -> bool {
    let kind = xyz[0].1;
    if !xyz.iter().all(|&(_, k)| k == kind) || !matches!(kind, Scalar::F32 | Scalar::F64) {
        return false;
    }
    if rgb.is_some_and(|c| c.iter().any(|&(_, k)| k != Scalar::U8)) {
        return false;
    }
    let at = xyz.map(|(offset, _)| offset);
    let size = kind.size();
    if at.iter().any(|&a| a + size > stride) {
        return false;
    }
    for record in records.chunks_exact(stride) {
        let p = if kind == Scalar::F32 {
            at.map(|a| f32::from_le_bytes(record[a..a + 4].try_into().unwrap()) as f64)
        } else {
            at.map(|a| f64::from_le_bytes(record[a..a + 8].try_into().unwrap()))
        };
        cloud.positions.push(p);
    }
    if let (Some(colors), Some(rgb)) = (cloud.colors.as_mut(), rgb) {
        let at = rgb.map(|(offset, _)| offset);
        colors.extend(records.chunks_exact(stride).map(|r| at.map(|a| r[a])));
    }
    true
}

fn read_binary(body: &[u8], elements: &[Element], le: bool) -> Result<PointCloud, IoError> {
    let mut cursor = 0usize;
    for element in elements {
        // Fixed-size records (no list properties) are read field by field at
        // known offsets, skipping everything but x/y/z and colors.
        if let Some(stride) = fixed_stride(element) {
            let len = stride * element.count;
            let records = body
                .get(cursor..cursor + len)
                .ok_or(IoError::Truncated(FORMAT))?;
            cursor += len;
            if element.name == "vertex" {
                return read_fixed_vertices(records, stride, element, le);
            }
            continue;
        }
        let mut take = |n: usize| -> Result<&[u8], IoError> {
            let slice = body
                .get(cursor..cursor + n)
                .ok_or(IoError::Truncated(FORMAT))?;
            cursor += n;
            Ok(slice)
        };
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
                    Property::List { count, item, .. } => {
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
    fn reads_interleaved_float_and_uchar_fields() {
        let mut src = b"ply
format binary_little_endian 1.0
element vertex 2
property uchar red
property float x
property ushort label
property float y
property float z
property uchar green
property uchar blue
end_header
"
        .to_vec();
        for (r, p, l, g, b) in [
            (9u8, [1.5f32, 2.0, -3.0], 7u16, 8u8, 7u8),
            (1, [4.0, 5.0, 6.25], 0, 2, 3),
        ] {
            src.push(r);
            src.extend_from_slice(&p[0].to_le_bytes());
            src.extend_from_slice(&l.to_le_bytes());
            src.extend_from_slice(&p[1].to_le_bytes());
            src.extend_from_slice(&p[2].to_le_bytes());
            src.extend_from_slice(&[g, b]);
        }
        let cloud = read(&src).unwrap();
        assert_eq!(cloud.positions, vec![[1.5, 2.0, -3.0], [4.0, 5.0, 6.25]]);
        assert_eq!(cloud.colors, Some(vec![[9, 8, 7], [1, 2, 3]]));
    }

    #[test]
    fn mixed_coordinate_types_use_the_general_path() {
        let mut src = b"ply
format binary_little_endian 1.0
element vertex 1
property float x
property double y
property int z
property ushort red
property ushort green
property ushort blue
end_header
"
        .to_vec();
        src.extend_from_slice(&0.5f32.to_le_bytes());
        src.extend_from_slice(&(-2.25f64).to_le_bytes());
        src.extend_from_slice(&7i32.to_le_bytes());
        for c in [65535u16, 0, 32896] {
            src.extend_from_slice(&c.to_le_bytes());
        }
        let cloud = read(&src).unwrap();
        assert_eq!(cloud.positions, vec![[0.5, -2.25, 7.0]]);
        assert_eq!(cloud.colors, Some(vec![[255, 0, 128]]));
    }

    #[test]
    fn reads_ascii_mesh_with_quads() {
        let src = b"ply\nformat ascii 1.0\nelement vertex 4\nproperty float x\nproperty float y\n\
property float z\nelement face 2\nproperty list uchar int vertex_indices\nend_header\n\
0 0 0\n1 0 0\n1 1 0\n0 1 0\n4 0 1 2 3\n3 0 2 9\n";
        let mesh = read_mesh(src).unwrap().unwrap();
        assert_eq!(mesh.vertices.len(), 4);
        // The quad becomes two triangles; the face with a bad index is dropped.
        assert_eq!(mesh.triangles, vec![[0, 1, 2], [0, 2, 3]]);
    }

    #[test]
    fn reads_binary_mesh_and_ignores_face_extras() {
        let mut src = b"ply\nformat binary_little_endian 1.0\nelement vertex 3\n\
property double x\nproperty double y\nproperty double z\nelement face 1\n\
property uchar flags\nproperty list uchar uint vertex_index\nend_header\n"
            .to_vec();
        for v in [[0.0f64, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 1.0]] {
            for c in v {
                src.extend_from_slice(&c.to_le_bytes());
            }
        }
        src.extend_from_slice(&[7, 3]);
        for i in [0u32, 1, 2] {
            src.extend_from_slice(&i.to_le_bytes());
        }
        let mesh = read_mesh(&src).unwrap().unwrap();
        assert_eq!(mesh.vertices[2], [0.0, 2.0, 1.0]);
        assert_eq!(mesh.triangles, vec![[0, 1, 2]]);
    }

    #[test]
    fn point_cloud_ply_is_not_a_mesh() {
        let src = b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\n\
property float z\nend_header\n1 2 3\n";
        assert!(read_mesh(src).unwrap().is_none());
    }

    #[test]
    fn reads_intensity_and_classification_properties() {
        let mut src = b"ply
format binary_little_endian 1.0
element vertex 2
property float x
property float y
property float z
property ushort intensity
property uchar scalar_Classification
end_header
"
        .to_vec();
        for (p, i, c) in [([1.0f32, 2.0, 3.0], 900u16, 2u8), ([4.0, 5.0, 6.0], 15, 6)] {
            for v in p {
                src.extend_from_slice(&v.to_le_bytes());
            }
            src.extend_from_slice(&i.to_le_bytes());
            src.push(c);
        }
        let cloud = read(&src).unwrap();
        assert_eq!(cloud.positions[1], [4.0, 5.0, 6.0]);
        assert_eq!(
            cloud.attribute(INTENSITY).unwrap().values,
            AttributeValues::F32(vec![900.0, 15.0])
        );
        assert_eq!(
            cloud.attribute(CLASSIFICATION).unwrap().values,
            AttributeValues::U8(vec![2, 6])
        );
        // ASCII goes through the general path.
        let ascii = b"ply
format ascii 1.0
element vertex 1
property float x
property float y
property float z
property float intensity
end_header
1 2 3 0.25
";
        let cloud = read(ascii).unwrap();
        assert_eq!(
            cloud.attribute(INTENSITY).unwrap().values,
            AttributeValues::F32(vec![0.25])
        );
    }

    #[test]
    fn truncated_binary_is_an_error() {
        let src = b"ply\nformat binary_little_endian 1.0\nelement vertex 2\n\
property float x\nproperty float y\nproperty float z\nend_header\n\0\0\0\0";
        assert!(matches!(read(src), Err(IoError::Truncated(_))));
    }
}
