//! Merging clouds into one, and splitting a cloud by a class-like attribute.

use crate::{Attribute, AttributeValues, PointCloud};

/// Name of the attribute a merge adds: the index of the part each point
/// came from.
pub const SOURCE: &str = "source";

/// Builds one cloud from up to 256 parts, added in turn.
///
/// Attributes are the union of the parts' by name; a part without one gets
/// zeros (and an attribute that is `u8` in one part and `f32` in another
/// becomes `f32`). If any part has colors, parts without get their fill
/// color. A [`SOURCE`] attribute records the part of every point, replacing
/// one from an earlier merge.
#[derive(Debug, Default)]
pub struct Merger {
    cloud: PointCloud,
    source: Vec<u8>,
    /// Point range and fill color of each part, for colors added later.
    fills: Vec<(usize, usize, [u8; 3])>,
}

impl Merger {
    pub fn new() -> Self {
        Self::default()
    }

    /// Number of parts added so far.
    pub fn parts(&self) -> usize {
        self.fills.len()
    }

    /// Append `part`; returns false (and adds nothing) past 256 parts.
    pub fn add(&mut self, part: &PointCloud, fill: [u8; 3]) -> bool {
        let index = self.fills.len();
        if index > u8::MAX as usize {
            return false;
        }
        let (n0, n) = (self.cloud.len(), part.len());
        self.cloud.positions.extend_from_slice(&part.positions);

        match (&mut self.cloud.colors, &part.colors) {
            (Some(colors), Some(c)) => colors.extend_from_slice(c),
            (Some(colors), None) => colors.extend(std::iter::repeat_n(fill, n)),
            (None, Some(c)) => {
                let mut colors = Vec::with_capacity(n0 + n);
                for &(_, len, f) in &self.fills {
                    colors.extend(std::iter::repeat_n(f, len));
                }
                colors.extend_from_slice(c);
                self.cloud.colors = Some(colors);
            }
            (None, None) => {}
        }

        for a in part.attributes.iter().filter(|a| a.name != SOURCE) {
            let pos = match self.cloud.attributes.iter().position(|m| m.name == a.name) {
                Some(pos) => pos,
                None => {
                    let zeros = match a.values {
                        AttributeValues::F32(_) => AttributeValues::F32(vec![0.0; n0]),
                        AttributeValues::U8(_) => AttributeValues::U8(vec![0; n0]),
                    };
                    self.cloud.attributes.push(Attribute {
                        name: a.name.clone(),
                        values: zeros,
                    });
                    self.cloud.attributes.len() - 1
                }
            };
            let merged = &mut self.cloud.attributes[pos].values;
            if let (AttributeValues::U8(old), AttributeValues::F32(_)) = (&*merged, &a.values) {
                *merged = AttributeValues::F32(old.iter().map(|&v| v as f32).collect());
            }
            match (merged, &a.values) {
                (AttributeValues::F32(m), AttributeValues::F32(v)) => m.extend_from_slice(v),
                (AttributeValues::F32(m), AttributeValues::U8(v)) => {
                    m.extend(v.iter().map(|&x| x as f32))
                }
                (AttributeValues::U8(m), AttributeValues::U8(v)) => m.extend_from_slice(v),
                (AttributeValues::U8(_), AttributeValues::F32(_)) => {
                    unreachable!("converted above")
                }
            }
        }
        // Attributes this part lacks.
        for m in &mut self.cloud.attributes {
            match &mut m.values {
                AttributeValues::F32(v) if v.len() < n0 + n => v.resize(n0 + n, 0.0),
                AttributeValues::U8(v) if v.len() < n0 + n => v.resize(n0 + n, 0),
                _ => {}
            }
        }
        self.source.extend(std::iter::repeat_n(index as u8, n));
        self.fills.push((n0, n, fill));
        true
    }

    pub fn finish(mut self) -> PointCloud {
        self.cloud.attributes.push(Attribute {
            name: SOURCE.into(),
            values: AttributeValues::U8(self.source),
        });
        self.cloud
    }
}

/// The distinct values of the `u8` attribute `name`, ascending, each with
/// the indices of its points. `None` if there is no such attribute.
pub fn split_by(cloud: &PointCloud, name: &str) -> Option<Vec<(u8, Vec<usize>)>> {
    let AttributeValues::U8(values) = &cloud.attribute(name)?.values else {
        return None;
    };
    let mut groups: Vec<Vec<usize>> = vec![Vec::new(); 256];
    for (i, &v) in values.iter().enumerate() {
        groups[v as usize].push(i);
    }
    Some(
        groups
            .into_iter()
            .enumerate()
            .filter(|(_, g)| !g.is_empty())
            .map(|(v, g)| (v as u8, g))
            .collect(),
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{CLASSIFICATION, INTENSITY};

    fn cloud(n: usize, z: f64) -> PointCloud {
        PointCloud {
            positions: (0..n).map(|i| [i as f64, 0.0, z]).collect(),
            colors: None,
            attributes: Vec::new(),
        }
    }

    #[test]
    fn merges_positions_colors_and_attributes() {
        let a = cloud(3, 0.0);
        let mut b = cloud(2, 1.0);
        b.colors = Some(vec![[1, 2, 3], [4, 5, 6]]);
        b.attributes.push(Attribute {
            name: INTENSITY.into(),
            values: AttributeValues::F32(vec![7.0, 8.0]),
        });
        let mut c = cloud(1, 2.0);
        c.attributes.push(Attribute {
            name: CLASSIFICATION.into(),
            values: AttributeValues::U8(vec![2]),
        });
        let mut m = Merger::new();
        assert!(m.add(&a, [9, 9, 9]) && m.add(&b, [0, 0, 0]) && m.add(&c, [5, 5, 5]));
        let merged = m.finish();
        assert_eq!(merged.len(), 6);
        assert_eq!(merged.positions[3], [0.0, 0.0, 1.0]);
        assert_eq!(
            merged.colors.unwrap(),
            vec![
                [9, 9, 9],
                [9, 9, 9],
                [9, 9, 9],
                [1, 2, 3],
                [4, 5, 6],
                [5, 5, 5]
            ]
        );
        let get = |name: &str| {
            merged
                .attributes
                .iter()
                .find(|a| a.name == name)
                .unwrap()
                .values
                .clone()
        };
        assert_eq!(
            get(INTENSITY),
            AttributeValues::F32(vec![0.0, 0.0, 0.0, 7.0, 8.0, 0.0])
        );
        assert_eq!(
            get(CLASSIFICATION),
            AttributeValues::U8(vec![0, 0, 0, 0, 0, 2])
        );
        assert_eq!(get(SOURCE), AttributeValues::U8(vec![0, 0, 0, 1, 1, 2]));
    }

    #[test]
    fn remerging_replaces_the_source_and_widens_mixed_types() {
        let mut a = cloud(2, 0.0);
        a.attributes.push(Attribute {
            name: "x".into(),
            values: AttributeValues::U8(vec![1, 2]),
        });
        let mut b = cloud(1, 0.0);
        b.attributes.push(Attribute {
            name: "x".into(),
            values: AttributeValues::F32(vec![0.5]),
        });
        let mut first = Merger::new();
        first.add(&a, [0; 3]);
        first.add(&b, [0; 3]);
        let merged = first.finish();
        let mut again = Merger::new();
        again.add(&merged, [0; 3]);
        let twice = again.finish();
        assert_eq!(
            twice.attributes.iter().filter(|a| a.name == SOURCE).count(),
            1
        );
        assert_eq!(
            twice.attribute(SOURCE).unwrap().values,
            AttributeValues::U8(vec![0, 0, 0])
        );
        assert_eq!(
            twice.attribute("x").unwrap().values,
            AttributeValues::F32(vec![1.0, 2.0, 0.5])
        );
    }

    #[test]
    fn splits_by_a_u8_attribute() {
        let mut c = cloud(6, 0.0);
        c.attributes.push(Attribute {
            name: CLASSIFICATION.into(),
            values: AttributeValues::U8(vec![2, 6, 2, 2, 9, 6]),
        });
        let groups = split_by(&c, CLASSIFICATION).unwrap();
        assert_eq!(
            groups,
            vec![(2, vec![0, 2, 3]), (6, vec![1, 5]), (9, vec![4])]
        );
        assert!(split_by(&c, INTENSITY).is_none());
    }
}
