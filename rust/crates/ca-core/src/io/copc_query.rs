//! Full-density spatial hierarchy traversal with bounded pending metadata.
//! Unlike the display selector, this walks every overlapping octree level.
//! The caller supplies one page at a time and acknowledges one node at a time.

use super::IoError;
use super::copc::{CopcHeader, Entry, VoxelKey, parse_page};
use std::collections::HashSet;

const FORMAT: &str = "COPC query";

#[derive(Debug, Clone, Copy)]
pub struct QueryLimits {
    pub page_bytes: u64,
    pub compressed_node_bytes: u64,
    pub raw_node_bytes: u64,
    pub pending_entries: usize,
    pub page_depth: usize,
}

impl Default for QueryLimits {
    fn default() -> Self {
        Self {
            page_bytes: 1 << 20,
            compressed_node_bytes: 16 << 20,
            raw_node_bytes: 32 << 20,
            pending_entries: 16_384,
            page_depth: 32,
        }
    }
}

#[derive(Debug, Clone)]
struct Task {
    entry: Entry,
    ancestors: Vec<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QueryItem {
    Page { offset: u64, size: u64 },
    Node(Entry),
}

/// Inclusive world-coordinate box. A query never thins its selected nodes.
/// Data buffers belong to the caller; these limits bound each item and the
/// pending traversal, not total process RSS or caller-retained output.
pub struct CopcQuery {
    center: [f64; 3],
    halfsize: f64,
    bounds: [f64; 6],
    file_size: u64,
    total_points: u64,
    record_len: u64,
    limits: QueryLimits,
    pending: Vec<Task>,
    processed_points: u64,
}

impl CopcQuery {
    pub fn new(
        header: &CopcHeader,
        file_size: u64,
        bounds: [f64; 6],
        limits: QueryLimits,
    ) -> Result<Self, IoError> {
        if !bounds.iter().all(|v| v.is_finite()) || (0..3).any(|a| bounds[a] > bounds[a + 3]) {
            return Err(IoError::header(
                FORMAT,
                "bounds must be finite with min <= max",
            ));
        }
        // Bound extra-byte decoder models independently of the point count.
        if header.record_len() > 1024
            || header.scale().iter().any(|v| !v.is_finite() || *v == 0.0)
            || header.offset().iter().any(|v| !v.is_finite())
        {
            return Err(IoError::header(
                FORMAT,
                "query requires records <=1024 bytes and finite nonzero scales/offsets",
            ));
        }
        if !header.center.iter().all(|v| v.is_finite())
            || !header.halfsize.is_finite()
            || header.halfsize <= 0.0
            || !(header.halfsize * 2.0).is_finite()
            || (0..3).any(|a| {
                !(header.center[a] - header.halfsize).is_finite()
                    || !(header.center[a] + header.halfsize).is_finite()
            })
        {
            return Err(IoError::header(FORMAT, "invalid root cube"));
        }
        if limits.page_bytes == 0
            || limits.compressed_node_bytes == 0
            || limits.raw_node_bytes == 0
            || limits.pending_entries == 0
            || limits.page_depth == 0
            || limits.page_depth > 64
        {
            return Err(IoError::header(
                FORMAT,
                "invalid query limits (page depth must be 1..64)",
            ));
        }
        let (offset, size) = header.root_page;
        let byte_size = i32::try_from(size)
            .map_err(|_| IoError::header(FORMAT, "root page size exceeds i32"))?;
        if size == 0 && header.total_points > 0 {
            return Err(IoError::header(
                FORMAT,
                "nonempty file has an empty hierarchy",
            ));
        }
        let root = Entry {
            key: VoxelKey {
                level: 0,
                x: 0,
                y: 0,
                z: 0,
            },
            offset,
            byte_size,
            point_count: -1,
        };
        let mut query = Self {
            center: header.center,
            halfsize: header.halfsize,
            bounds,
            file_size,
            total_points: header.total_points,
            record_len: header.record_len() as u64,
            limits,
            pending: Vec::new(),
            processed_points: 0,
        };
        query.validate(root)?;
        if query.overlaps(root.key) && size > 0 {
            query.pending.push(Task {
                entry: root,
                ancestors: Vec::new(),
            });
        }
        Ok(query)
    }

    fn validate(&self, entry: Entry) -> Result<(), IoError> {
        let key = entry.key;
        if !(0..=31).contains(&key.level)
            || [key.x, key.y, key.z]
                .iter()
                .any(|&v| v < 0 || v as u64 >= 1u64 << key.level)
        {
            return Err(IoError::header(
                FORMAT,
                "invalid octree key (supported levels 0..31)",
            ));
        }
        if entry.point_count < -1
            || entry.byte_size < 0
            || (entry.point_count > 0 && entry.point_count as u64 > self.total_points)
        {
            return Err(IoError::header(
                FORMAT,
                "invalid hierarchy count or byte size",
            ));
        }
        if entry.point_count == 0 {
            if entry.offset != 0 || entry.byte_size != 0 {
                return Err(IoError::header(FORMAT, "empty node has a byte range"));
            }
            return Ok(());
        }
        let size = entry.byte_size as u64;
        if entry
            .offset
            .checked_add(size)
            .is_none_or(|end| end > self.file_size)
        {
            return Err(IoError::header(
                FORMAT,
                "hierarchy byte range is outside the file",
            ));
        }
        if entry.point_count == -1 {
            if !size.is_multiple_of(32) || (self.overlaps(key) && size > self.limits.page_bytes) {
                return Err(IoError::header(
                    FORMAT,
                    "unaligned page or page exceeds byte limit",
                ));
            }
        } else if size == 0
            || (self.overlaps(key)
                && (size > self.limits.compressed_node_bytes
                    || (entry.point_count as u64)
                        .checked_mul(self.record_len)
                        .is_none_or(|n| n > self.limits.raw_node_bytes)))
        {
            return Err(IoError::header(
                FORMAT,
                "node exceeds compressed/raw byte limit",
            ));
        }
        Ok(())
    }

    fn overlaps(&self, key: VoxelKey) -> bool {
        let edge = self.halfsize * 2.0 / 2.0f64.powi(key.level);
        [key.x, key.y, key.z].iter().enumerate().all(|(a, &index)| {
            let lo = self.center[a] - self.halfsize + index as f64 * edge;
            let hi = lo + edge;
            // Inclusive cube overlap avoids dropping points on a voxel face.
            let margin = (lo.abs() + hi.abs() + edge.abs()) * f64::EPSILON * 8.0;
            lo - margin <= self.bounds[a + 3] && hi + margin >= self.bounds[a]
        })
    }

    /// The current item stays current until supply_page/advance_node succeeds.
    /// This lets callers retry IO or cancel without losing a pending item.
    pub fn next_item(&self) -> Option<QueryItem> {
        self.pending.last().map(|task| {
            let entry = task.entry;
            if entry.point_count == -1 {
                QueryItem::Page {
                    offset: entry.offset,
                    size: entry.byte_size as u64,
                }
            } else {
                QueryItem::Node(entry)
            }
        })
    }

    /// Validate an entire page before changing traversal state.
    pub fn supply_page(&mut self, offset: u64, bytes: &[u8]) -> Result<(), IoError> {
        let task = self
            .pending
            .last()
            .ok_or_else(|| IoError::header(FORMAT, "query is complete"))?;
        if task.entry.point_count != -1
            || task.entry.offset != offset
            || bytes.len() as u64 != task.entry.byte_size as u64
        {
            return Err(IoError::header(
                FORMAT,
                "page does not match the pending range",
            ));
        }
        if task.ancestors.contains(&offset) || task.ancestors.len() >= self.limits.page_depth {
            return Err(IoError::header(
                FORMAT,
                "hierarchy page cycle or depth limit",
            ));
        }
        let scope = task.entry.key;
        let mut entries = Vec::new();
        let mut keys = HashSet::new();
        let page_entries = parse_page(bytes);
        let pointers: HashSet<_> = page_entries
            .iter()
            .filter(|e| e.point_count == -1)
            .map(|e| e.key)
            .collect();
        for entry in page_entries {
            self.validate(entry)?;
            let key = entry.key;
            let shift = key.level - scope.level;
            if shift < 0
                || key.x >> shift != scope.x
                || key.y >> shift != scope.y
                || key.z >> shift != scope.z
            {
                return Err(IoError::header(
                    FORMAT,
                    "entry is outside its page's octree scope",
                ));
            }
            if !keys.insert(key) {
                return Err(IoError::header(FORMAT, "duplicate key in hierarchy page"));
            }
            let mut parent = key.parent();
            while let Some(key) = parent {
                if pointers.contains(&key) {
                    return Err(IoError::header(
                        FORMAT,
                        "page references overlap other entries' subtrees",
                    ));
                }
                parent = key.parent();
            }
            if entry.point_count == -1
                && (entry.offset == offset || task.ancestors.contains(&entry.offset))
            {
                return Err(IoError::header(FORMAT, "hierarchy page cycle"));
            }
            if entry.point_count != 0 && self.overlaps(key) {
                entries.push(entry);
            }
        }
        if entries.len()
            > self
                .limits
                .pending_entries
                .saturating_sub(self.pending.len() - 1)
        {
            return Err(IoError::header(
                FORMAT,
                "pending hierarchy entries exceed limit",
            ));
        }
        let mut ancestors = task.ancestors.clone();
        ancestors.push(offset);
        // Stable traversal independent of hierarchy page storage order.
        entries.sort_by_key(|e| (e.key.level, e.key.x, e.key.y, e.key.z));
        self.pending.pop();
        self.pending
            .extend(entries.into_iter().rev().map(|entry| Task {
                entry,
                ancestors: ancestors.clone(),
            }));
        Ok(())
    }

    pub fn advance_node(&mut self) -> Result<(), IoError> {
        let Some(QueryItem::Node(entry)) = self.next_item() else {
            return Err(IoError::header(FORMAT, "no pending point node"));
        };
        let total = self
            .processed_points
            .checked_add(entry.point_count as u64)
            .filter(|&n| n <= self.total_points)
            .ok_or_else(|| IoError::header(FORMAT, "visited nodes exceed the LAS point count"))?;
        self.pending.pop();
        self.processed_points = total;
        Ok(())
    }

    pub fn contains(&self, point: [f64; 3]) -> bool {
        (0..3).all(|a| {
            point[a].is_finite() && point[a] >= self.bounds[a] && point[a] <= self.bounds[a + 3]
        })
    }
}

#[cfg(test)]
mod tests {
    use super::super::copc::write_minimal_copc;
    use super::*;

    fn fixture() -> Vec<u8> {
        write_minimal_copc(
            &[
                (
                    VoxelKey {
                        level: 0,
                        x: 0,
                        y: 0,
                        z: 0,
                    },
                    vec![[-2.0; 3], [0.0; 3]],
                ),
                (
                    VoxelKey {
                        level: 1,
                        x: 0,
                        y: 0,
                        z: 0,
                    },
                    vec![[-1.0; 3]],
                ),
                (
                    VoxelKey {
                        level: 1,
                        x: 1,
                        y: 1,
                        z: 1,
                    },
                    vec![[1.0; 3]],
                ),
            ],
            [0.0; 3],
            4.0,
        )
    }

    #[test]
    fn full_density_reads_all_overlapping_levels_and_retries_pending_items() {
        let file = fixture();
        let header = CopcHeader::parse(&file).unwrap();
        let mut query = CopcQuery::new(
            &header,
            file.len() as u64,
            [-3.0, -3.0, -3.0, -0.1, -0.1, -0.1],
            QueryLimits::default(),
        )
        .unwrap();
        let item = query.next_item().unwrap();
        assert_eq!(query.next_item(), Some(item));
        let QueryItem::Page { offset, size } = item else {
            panic!()
        };
        let page = &file[offset as usize..(offset + size) as usize];
        assert!(query.supply_page(offset + 1, page).is_err());
        assert_eq!(query.next_item(), Some(item));
        query.supply_page(offset, page).unwrap();
        let mut points = Vec::new();
        let mut levels = Vec::new();
        while let Some(QueryItem::Node(entry)) = query.next_item() {
            levels.push(entry.key.level);
            let chunk =
                &file[entry.offset as usize..entry.offset as usize + entry.byte_size as usize];
            let decoded = header
                .decode_node(chunk, entry.point_count as usize)
                .unwrap();
            points.extend(decoded.positions.into_iter().filter(|&p| query.contains(p)));
            query.advance_node().unwrap();
        }
        assert_eq!(levels, [0, 1]);
        assert_eq!(points, [[-2.0; 3], [-1.0; 3]]);
        assert!(query.advance_node().is_err());
        let outside = CopcQuery::new(
            &header,
            file.len() as u64,
            [10.0; 6],
            QueryLimits::default(),
        )
        .unwrap();
        assert_eq!(outside.next_item(), None);
    }

    #[test]
    fn corrupt_pages_and_limits_fail_without_advancing() {
        let file = fixture();
        let header = CopcHeader::parse(&file).unwrap();
        let (offset, size) = header.root_page;
        let original = &file[offset as usize..(offset + size) as usize];
        let limits = QueryLimits {
            pending_entries: 1,
            ..QueryLimits::default()
        };
        let mut capped = CopcQuery::new(
            &header,
            file.len() as u64,
            [-4., -4., -4., 4., 4., 4.],
            limits,
        )
        .unwrap();
        let expected = capped.next_item();
        assert!(capped.supply_page(offset, original).is_err());
        assert_eq!(capped.next_item(), expected);
        for mode in 0..4 {
            let mut page = original.to_vec();
            match mode {
                0 => page[28..32].copy_from_slice(&(-2i32).to_le_bytes()),
                1 => page[16..24].copy_from_slice(&u64::MAX.to_le_bytes()),
                2 => {
                    page[28..32].copy_from_slice(&(-1i32).to_le_bytes());
                    page[16..24].copy_from_slice(&offset.to_le_bytes());
                    page[24..28].copy_from_slice(&(size as i32).to_le_bytes());
                }
                _ => {
                    let key = page[..16].to_vec();
                    page[32..48].copy_from_slice(&key);
                }
            }
            let mut query = CopcQuery::new(
                &header,
                file.len() as u64,
                [-4., -4., -4., 4., 4., 4.],
                QueryLimits::default(),
            )
            .unwrap();
            let expected = query.next_item();
            assert!(query.supply_page(offset, &page).is_err());
            assert_eq!(query.next_item(), expected);
        }
        let limits = QueryLimits {
            page_bytes: 32,
            ..QueryLimits::default()
        };
        assert!(
            CopcQuery::new(
                &header,
                file.len() as u64,
                [-4., -4., -4., 4., 4., 4.],
                limits
            )
            .is_err()
        );
    }

    #[test]
    fn walks_child_pages_and_rejects_entries_outside_their_scope() {
        let mut file = fixture();
        let header = CopcHeader::parse(&file).unwrap();
        let root = header.root_page.0 as usize;
        let child = file[root + 32..root + 64].to_vec();
        let child_at = file.len() as u64;
        file.extend_from_slice(&child);
        file[root + 32 + 16..root + 32 + 24].copy_from_slice(&child_at.to_le_bytes());
        file[root + 32 + 24..root + 32 + 28].copy_from_slice(&32i32.to_le_bytes());
        file[root + 32 + 28..root + 32 + 32].copy_from_slice(&(-1i32).to_le_bytes());
        let mut query = CopcQuery::new(
            &header,
            file.len() as u64,
            [-3., -3., -3., -0.1, -0.1, -0.1],
            QueryLimits::default(),
        )
        .unwrap();
        query
            .supply_page(
                header.root_page.0,
                &file[root..root + header.root_page.1 as usize],
            )
            .unwrap();
        assert!(matches!(query.next_item(), Some(QueryItem::Node(_))));
        query.advance_node().unwrap();
        assert_eq!(
            query.next_item(),
            Some(QueryItem::Page {
                offset: child_at,
                size: 32
            })
        );
        let mut wrong = child.clone();
        wrong[..16].copy_from_slice(&file[root..root + 16]);
        assert!(query.supply_page(child_at, &wrong).is_err());
        query.supply_page(child_at, &child).unwrap();
        let Some(QueryItem::Node(entry)) = query.next_item() else {
            panic!()
        };
        assert_eq!(entry.key.level, 1);
        query.advance_node().unwrap();
        assert_eq!(query.next_item(), None);
    }
}
