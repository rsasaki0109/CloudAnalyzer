//! PCD reader (ascii, binary, binary_compressed).

use super::scalar::{Scalar, unpack_rgb};
use super::{IoError, split_header};
use crate::{Attribute, AttributeValues, CLASSIFICATION, INTENSITY, PointCloud};

const FORMAT: &str = "PCD";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum Data {
    Ascii,
    Binary,
    BinaryCompressed,
}

#[derive(Debug)]
struct Field {
    name: String,
    kind: Scalar,
    count: usize,
}

#[derive(Debug)]
struct Header {
    fields: Vec<Field>,
    points: usize,
    data: Data,
}

impl Header {
    fn index(&self, name: &str) -> Option<usize> {
        self.fields.iter().position(|f| f.name == name)
    }

    fn xyz(&self) -> Result<[usize; 3], IoError> {
        let axis = |n: &str| {
            self.index(n)
                .ok_or_else(|| IoError::header(FORMAT, format!("no {n} field")))
        };
        Ok([axis("x")?, axis("y")?, axis("z")?])
    }

    fn rgb(&self) -> Option<usize> {
        self.index("rgb").or_else(|| self.index("rgba"))
    }

    fn intensity(&self) -> Option<usize> {
        self.index("intensity")
    }

    /// PCL writes semantic classes as `label`.
    fn classification(&self) -> Option<usize> {
        self.index("classification").or_else(|| self.index("label"))
    }

    fn empty_attributes(&self) -> Vec<Attribute> {
        let mut out = Vec::new();
        if self.intensity().is_some() {
            out.push(Attribute {
                name: INTENSITY.into(),
                values: AttributeValues::F32(Vec::with_capacity(self.points)),
            });
        }
        if self.classification().is_some() {
            out.push(Attribute {
                name: CLASSIFICATION.into(),
                values: AttributeValues::U8(Vec::with_capacity(self.points)),
            });
        }
        out
    }

    /// Byte size of one point in row-major layout.
    fn stride(&self) -> usize {
        self.fields.iter().map(|f| f.kind.size() * f.count).sum()
    }
}

fn parse_kind(kind: &str, size: &str) -> Result<Scalar, IoError> {
    Ok(match (kind, size) {
        ("I", "1") => Scalar::I8,
        ("U", "1") => Scalar::U8,
        ("I", "2") => Scalar::I16,
        ("U", "2") => Scalar::U16,
        ("I", "4") => Scalar::I32,
        ("U", "4") => Scalar::U32,
        ("I", "8") => Scalar::I64,
        ("U", "8") => Scalar::U64,
        ("F", "4") => Scalar::F32,
        ("F", "8") => Scalar::F64,
        _ => {
            return Err(IoError::header(
                FORMAT,
                format!("unsupported TYPE/SIZE {kind}/{size}"),
            ));
        }
    })
}

fn parse_header(lines: &[&str]) -> Result<Header, IoError> {
    let mut names = Vec::new();
    let mut sizes = Vec::new();
    let mut types = Vec::new();
    let mut counts = Vec::new();
    let mut width = None;
    let mut height = 1usize;
    let mut points = None;
    let mut data = None;
    let number = |v: &str| {
        v.parse::<usize>()
            .map_err(|_| IoError::header(FORMAT, format!("bad number {v:?}")))
    };
    for line in lines {
        let mut tokens = line.split_whitespace();
        let Some(key) = tokens.next() else { continue };
        let rest: Vec<&str> = tokens.collect();
        match key.to_ascii_uppercase().as_str() {
            "FIELDS" => names = rest,
            "SIZE" => sizes = rest,
            "TYPE" => types = rest,
            "COUNT" => counts = rest.iter().map(|c| number(c)).collect::<Result<_, _>>()?,
            "WIDTH" => width = Some(number(rest.first().copied().unwrap_or(""))?),
            "HEIGHT" => height = number(rest.first().copied().unwrap_or(""))?,
            "POINTS" => points = Some(number(rest.first().copied().unwrap_or(""))?),
            "DATA" => {
                data = Some(match rest.first().copied() {
                    Some("ascii") => Data::Ascii,
                    Some("binary") => Data::Binary,
                    Some("binary_compressed") => Data::BinaryCompressed,
                    other => {
                        return Err(IoError::header(FORMAT, format!("unknown DATA {other:?}")));
                    }
                });
            }
            _ => {} // comments, VERSION, VIEWPOINT
        }
    }
    if names.is_empty() || names.len() != sizes.len() || names.len() != types.len() {
        return Err(IoError::header(FORMAT, "FIELDS/SIZE/TYPE mismatch"));
    }
    if counts.is_empty() {
        counts = vec![1; names.len()];
    }
    if counts.len() != names.len() {
        return Err(IoError::header(FORMAT, "COUNT length mismatch"));
    }
    let fields = names
        .iter()
        .zip(&sizes)
        .zip(&types)
        .zip(&counts)
        .map(|(((name, size), kind), &count)| {
            Ok(Field {
                name: (*name).to_owned(),
                kind: parse_kind(kind, size)?,
                count,
            })
        })
        .collect::<Result<_, IoError>>()?;
    let points = points
        .or(width.map(|w| w * height))
        .ok_or_else(|| IoError::header(FORMAT, "missing POINTS/WIDTH"))?;
    let data = data.ok_or_else(|| IoError::header(FORMAT, "missing DATA line"))?;
    Ok(Header {
        fields,
        points,
        data,
    })
}

pub(crate) fn read(bytes: &[u8]) -> Result<PointCloud, IoError> {
    let (lines, body) = split_header(bytes, FORMAT, |l| {
        l.trim_start().to_ascii_uppercase().starts_with("DATA")
    })?;
    let header = parse_header(&lines)?;
    match header.data {
        Data::Ascii => read_ascii(body, &header),
        Data::Binary => {
            let stride = header.stride();
            let body = body
                .get(..stride * header.points)
                .ok_or(IoError::Truncated(FORMAT))?;
            let mut offsets = Vec::with_capacity(header.fields.len());
            let mut offset = 0;
            for f in &header.fields {
                offsets.push(offset);
                offset += f.kind.size() * f.count;
            }
            read_binary(&header, |field, point| {
                &body[point * stride + offsets[field]..]
            })
        }
        Data::BinaryCompressed => {
            let size = |i: usize| {
                body.get(i..i + 4)
                    .map(|b| u32::from_le_bytes(b.try_into().unwrap()) as usize)
                    .ok_or(IoError::Truncated(FORMAT))
            };
            let (compressed, raw) = (size(0)?, size(4)?);
            let payload = body
                .get(8..8 + compressed)
                .ok_or(IoError::Truncated(FORMAT))?;
            let data = lzf_decompress(payload, raw)?;
            if data.len() < header.stride() * header.points {
                return Err(IoError::Truncated(FORMAT));
            }
            // Column-major: every field's values for all points are contiguous.
            let mut starts = Vec::with_capacity(header.fields.len());
            let mut offset = 0;
            for f in &header.fields {
                starts.push(offset);
                offset += f.kind.size() * f.count * header.points;
            }
            read_binary(&header, |field, point| {
                let f = &header.fields[field];
                &data[starts[field] + point * f.kind.size() * f.count..]
            })
        }
    }
}

fn read_binary<'a>(
    header: &Header,
    at: impl Fn(usize, usize) -> &'a [u8],
) -> Result<PointCloud, IoError> {
    let xyz = header.xyz()?;
    let kinds = xyz.map(|i| header.fields[i].kind);
    let rgb = header.rgb();
    let mut cloud = PointCloud {
        positions: Vec::with_capacity(header.points),
        colors: rgb.map(|_| Vec::with_capacity(header.points)),
        attributes: header.empty_attributes(),
    };
    let intensity = header.intensity().map(|i| (i, header.fields[i].kind));
    let classification = header.classification().map(|i| (i, header.fields[i].kind));
    for point in 0..header.points {
        let p: [f64; 3] = std::array::from_fn(|a| kinds[a].decode(at(xyz[a], point), true));
        if !p.iter().all(|v| v.is_finite()) {
            continue; // organized clouds mark invalid points with NaN
        }
        cloud.positions.push(p);
        if let (Some(colors), Some(field)) = (cloud.colors.as_mut(), rgb) {
            colors.push(unpack_rgb(Scalar::bits_u32(at(field, point), true)));
        }
        push_attributes(
            &mut cloud,
            intensity.map(|(f, k)| k.decode(at(f, point), true)),
            classification.map(|(f, k)| k.decode(at(f, point), true)),
        );
    }
    Ok(cloud)
}

fn read_ascii(body: &[u8], header: &Header) -> Result<PointCloud, IoError> {
    let text = std::str::from_utf8(body).map_err(|_| IoError::parse(FORMAT, "non-UTF-8 body"))?;
    // Token index of the first value of each field.
    let mut columns = Vec::with_capacity(header.fields.len());
    let mut column = 0;
    for f in &header.fields {
        columns.push(column);
        column += f.count;
    }
    let xyz = header.xyz()?.map(|i| columns[i]);
    let rgb = header.rgb().map(|i| (columns[i], header.fields[i].kind));
    let mut cloud = PointCloud {
        positions: Vec::with_capacity(header.points),
        colors: rgb.map(|_| Vec::with_capacity(header.points)),
        attributes: header.empty_attributes(),
    };
    let intensity = header.intensity().map(|i| columns[i]);
    let classification = header.classification().map(|i| columns[i]);
    for row in text
        .lines()
        .filter(|l| !l.trim().is_empty())
        .take(header.points)
    {
        let tokens: Vec<&str> = row.split_whitespace().collect();
        if tokens.len() < column {
            return Err(IoError::parse(FORMAT, row));
        }
        let parse = |t: &str| t.parse::<f64>().map_err(|_| IoError::parse(FORMAT, t));
        let p = [
            parse(tokens[xyz[0]])?,
            parse(tokens[xyz[1]])?,
            parse(tokens[xyz[2]])?,
        ];
        if !p.iter().all(|v| v.is_finite()) {
            continue;
        }
        cloud.positions.push(p);
        if let (Some(colors), Some((col, kind))) = (cloud.colors.as_mut(), rgb) {
            let token = tokens[col];
            let bits = if kind.is_float() {
                parse(token)? as f32
            } else {
                f32::from_bits(parse(token)? as u32)
            }
            .to_bits();
            colors.push(unpack_rgb(bits));
        }
        push_attributes(
            &mut cloud,
            intensity.map(|c| parse(tokens[c])).transpose()?,
            classification.map(|c| parse(tokens[c])).transpose()?,
        );
    }
    Ok(cloud)
}

/// Append one point's attribute values (in the order of `empty_attributes`).
fn push_attributes(cloud: &mut PointCloud, intensity: Option<f64>, classification: Option<f64>) {
    let mut slot = 0;
    if let Some(v) = intensity {
        if let AttributeValues::F32(values) = &mut cloud.attributes[slot].values {
            values.push(v as f32);
        }
        slot += 1;
    }
    if let Some(v) = classification
        && let AttributeValues::U8(values) = &mut cloud.attributes[slot].values
    {
        values.push(v.clamp(0.0, 255.0) as u8);
    }
}

/// Decompress an LZF stream (as written by PCL) into exactly `expected` bytes.
fn lzf_decompress(input: &[u8], expected: usize) -> Result<Vec<u8>, IoError> {
    let corrupt = || IoError::Unsupported("corrupt LZF payload in PCD".into());
    let mut out = Vec::with_capacity(expected);
    let mut i = 0;
    while i < input.len() {
        let ctrl = input[i] as usize;
        i += 1;
        if ctrl < 32 {
            let len = ctrl + 1;
            out.extend_from_slice(input.get(i..i + len).ok_or_else(corrupt)?);
            i += len;
        } else {
            let mut len = ctrl >> 5;
            if len == 7 {
                len += *input.get(i).ok_or_else(corrupt)? as usize;
                i += 1;
            }
            let back = ((ctrl & 0x1f) << 8) + *input.get(i).ok_or_else(corrupt)? as usize + 1;
            i += 1;
            let start = out.len().checked_sub(back).ok_or_else(corrupt)?;
            for k in 0..len + 2 {
                let byte = out[start + k];
                out.push(byte);
            }
        }
    }
    if out.len() != expected {
        return Err(corrupt());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    const RGB_RED: f32 = f32::from_bits(0x00ff_0000);

    #[test]
    fn reads_ascii_with_packed_rgb_and_nan() {
        let src = format!(
            "# .PCD v0.7\nVERSION 0.7\nFIELDS x y z rgb\nSIZE 4 4 4 4\nTYPE F F F F\n\
COUNT 1 1 1 1\nWIDTH 2\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS 2\nDATA ascii\n\
1 2 3 {:e}\nnan nan nan 0\n",
            RGB_RED
        );
        let cloud = read(src.as_bytes()).unwrap();
        assert_eq!(cloud.positions, vec![[1.0, 2.0, 3.0]]);
        assert_eq!(cloud.colors, Some(vec![[255, 0, 0]]));
    }

    #[test]
    fn reads_intensity_and_label() {
        let src =
            b"FIELDS x y z intensity label\nSIZE 4 4 4 4 4\nTYPE F F F F U\nWIDTH 2\nHEIGHT 1\n\
POINTS 2\nDATA ascii\n1 2 3 0.5 7\nnan nan nan 0 0\n";
        let cloud = read(src).unwrap();
        assert_eq!(cloud.len(), 1);
        assert_eq!(
            cloud.attribute(INTENSITY).unwrap().values,
            AttributeValues::F32(vec![0.5])
        );
        assert_eq!(
            cloud.attribute(CLASSIFICATION).unwrap().values,
            AttributeValues::U8(vec![7])
        );
    }

    #[test]
    fn reads_binary_with_extra_fields() {
        let mut src = b"FIELDS intensity x y z\nSIZE 2 4 4 8\nTYPE U F F F\nCOUNT 1 1 1 1\n\
WIDTH 2\nHEIGHT 1\nPOINTS 2\nDATA binary\n"
            .to_vec();
        for (i, p) in [[1.0f32, 2.0, 3.0], [4.0, 5.0, 6.0]].iter().enumerate() {
            src.extend_from_slice(&(i as u16).to_le_bytes());
            src.extend_from_slice(&p[0].to_le_bytes());
            src.extend_from_slice(&p[1].to_le_bytes());
            src.extend_from_slice(&(p[2] as f64).to_le_bytes());
        }
        let cloud = read(&src).unwrap();
        assert_eq!(cloud.positions, vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        assert!(cloud.colors.is_none());
        assert_eq!(
            cloud.attribute(INTENSITY).unwrap().values,
            AttributeValues::F32(vec![0.0, 1.0])
        );
    }

    #[test]
    fn reads_binary_compressed() {
        // Column-major x x y y z z, stored with LZF literal + back-reference runs.
        let mut raw = Vec::new();
        for v in [1.0f32, 1.0, 2.0, 2.0, 3.0, 3.0] {
            raw.extend_from_slice(&v.to_le_bytes());
        }
        // literal of 4 bytes (1.0), back-ref len 4 dist 4, literal 16 bytes.
        let mut lzf = vec![3u8];
        lzf.extend_from_slice(&raw[0..4]);
        lzf.extend_from_slice(&[(2 << 5) as u8, 3]);
        lzf.push(15);
        lzf.extend_from_slice(&raw[8..24]);
        let mut src = b"FIELDS x y z\nSIZE 4 4 4\nTYPE F F F\nWIDTH 2\nHEIGHT 1\n\
POINTS 2\nDATA binary_compressed\n"
            .to_vec();
        src.extend_from_slice(&(lzf.len() as u32).to_le_bytes());
        src.extend_from_slice(&(raw.len() as u32).to_le_bytes());
        src.extend_from_slice(&lzf);
        let cloud = read(&src).unwrap();
        assert_eq!(cloud.positions, vec![[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]]);
    }
}
