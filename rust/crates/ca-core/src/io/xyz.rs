//! Delimited text reader (XYZ / TXT / CSV / PTS / ASC).
//!
//! Each row starts with `x y z`, separated by whitespace, commas or
//! semicolons. Rows that do not start with three numbers (headers, comments,
//! the point count line of `.pts`) are skipped. Columns 4–6 are read as RGB
//! when the first data row has integer values in `0..=255` there; a `.pts`
//! intensity column in between is detected the same way.

use super::IoError;
use crate::PointCloud;

const FORMAT: &str = "XYZ";

fn fields(line: &str) -> impl Iterator<Item = &str> {
    line.split(|c: char| c.is_whitespace() || c == ',' || c == ';')
        .filter(|t| !t.is_empty())
}

fn numbers(line: &str) -> Vec<f64> {
    fields(line).map_while(|t| t.parse::<f64>().ok()).collect()
}

fn is_color(values: &[f64]) -> bool {
    values
        .iter()
        .all(|v| v.fract() == 0.0 && (0.0..=255.0).contains(v))
}

/// Column index of the first RGB channel, decided from the first data row.
fn color_column(first: &[f64]) -> Option<usize> {
    // x y z r g b
    if first.len() == 6 && is_color(&first[3..6]) {
        return Some(3);
    }
    // x y z intensity r g b  (Leica .pts)
    if first.len() == 7 && is_color(&first[4..7]) {
        return Some(4);
    }
    None
}

pub(crate) fn read(bytes: &[u8]) -> Result<PointCloud, IoError> {
    let text = std::str::from_utf8(bytes).map_err(|_| IoError::parse(FORMAT, "non-UTF-8 text"))?;
    let mut cloud = PointCloud::default();
    let mut color: Option<Option<usize>> = None;
    let mut values = Vec::new();
    for line in text.lines() {
        values.clear();
        values.extend(numbers(line));
        if values.len() < 3 {
            continue;
        }
        let column = *color.get_or_insert_with(|| {
            let column = color_column(&values);
            if column.is_some() {
                cloud.colors = Some(Vec::new());
            }
            column
        });
        cloud.positions.push([values[0], values[1], values[2]]);
        if let (Some(colors), Some(c)) = (cloud.colors.as_mut(), column) {
            let rgb = values
                .get(c..c + 3)
                .ok_or_else(|| IoError::parse(FORMAT, line))?;
            colors.push(std::array::from_fn(|i| rgb[i].clamp(0.0, 255.0) as u8));
        }
    }
    Ok(cloud)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reads_csv_with_header_and_rgb() {
        let cloud = read(b"x,y,z,r,g,b\n1,2,3,10,20,30\n4.5,5,6,0,0,255\n").unwrap();
        assert_eq!(cloud.positions, vec![[1.0, 2.0, 3.0], [4.5, 5.0, 6.0]]);
        assert_eq!(cloud.colors, Some(vec![[10, 20, 30], [0, 0, 255]]));
    }

    #[test]
    fn treats_float_extras_as_non_color() {
        let cloud = read(b"# normals\n1 2 3 0.0 0.7 0.7\n4 5 6 1 0 0\n").unwrap();
        assert_eq!(cloud.len(), 2);
        assert!(cloud.colors.is_none());
    }

    #[test]
    fn reads_pts_with_count_and_intensity() {
        let cloud = read(b"2\n1 2 3 -1200 255 0 0\n4 5 6 300 0 255 0\n").unwrap();
        assert_eq!(cloud.positions, vec![[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        assert_eq!(cloud.colors, Some(vec![[255, 0, 0], [0, 255, 0]]));
    }
}
