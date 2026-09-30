//! rosbag2 recordings: a folder with a `metadata.yaml` and its files (MCAP
//! or SQLite `.db3`), or one such file; the SQLite storage read with
//! SQLite itself (the `sqlite` feature, native builds).

use crate::bag::{self, Bag, BagError, Cursor, Encoding, Message, Topic};
use std::collections::BTreeMap;
use std::io::Read;
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, BagError>;

/// A rosbag2 SQLite file: its `topics` and `messages` tables.
pub struct Sqlite {
    connection: rusqlite::Connection,
    topics: Vec<Topic>,
    /// Topic id to name.
    names: BTreeMap<i64, String>,
    /// Each message zstd-compressed on its own (`compression_mode: MESSAGE`).
    compressed: bool,
}

impl Sqlite {
    pub fn open(path: &Path, compressed_messages: bool) -> Result<Self> {
        let connection = rusqlite::Connection::open_with_flags(
            path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY | rusqlite::OpenFlags::SQLITE_OPEN_NO_MUTEX,
        )
        .map_err(|e| BagError(format!("{}: {e}", path.display())))?;
        let mut names = BTreeMap::new();
        let mut topics = Vec::new();
        {
            let mut rows = connection
                .prepare("SELECT id, name, type FROM topics ORDER BY id")
                .map_err(|e| BagError(e.to_string()))?;
            let found = rows
                .query_map([], |r| {
                    Ok((
                        r.get::<_, i64>(0)?,
                        r.get::<_, String>(1)?,
                        r.get::<_, String>(2)?,
                    ))
                })
                .map_err(|e| BagError(e.to_string()))?;
            for row in found {
                let (id, name, kind) = row.map_err(|e| BagError(e.to_string()))?;
                let count: i64 = connection
                    .query_row(
                        "SELECT COUNT(*) FROM messages WHERE topic_id = ?1",
                        [id],
                        |r| r.get(0),
                    )
                    .map_err(|e| BagError(e.to_string()))?;
                names.insert(id, name.clone());
                topics.push(Topic {
                    name,
                    kind: kind.replace("/msg/", "/"),
                    count: count as u64,
                });
            }
        }
        topics.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(Sqlite {
            connection,
            topics,
            names,
            compressed: compressed_messages,
        })
    }

    pub fn topics(&self) -> &[Topic] {
        &self.topics
    }

    /// When the first and the last message were recorded (seconds).
    pub fn time_range(&self) -> Option<(f64, f64)> {
        let (lo, hi): (Option<i64>, Option<i64>) = self
            .connection
            .query_row(
                "SELECT MIN(timestamp), MAX(timestamp) FROM messages",
                [],
                |r| Ok((r.get(0)?, r.get(1)?)),
            )
            .ok()?;
        Some((lo? as f64 * 1e-9, hi? as f64 * 1e-9))
    }

    /// Up to `limit` messages of `topics` after message id `after`, oldest first:
    /// the messages and the last id.
    fn batch(&self, topics: &[String], after: i64, limit: usize) -> Result<(Vec<Message>, i64)> {
        let ids: Vec<String> = self
            .names
            .iter()
            .filter(|(_, name)| topics.contains(name))
            .map(|(id, _)| id.to_string())
            .collect();
        if ids.is_empty() {
            return Ok((Vec::new(), after));
        }
        let sql = format!(
            "SELECT id, topic_id, timestamp, data FROM messages WHERE id > ?1 AND topic_id IN ({}) ORDER BY id LIMIT ?2",
            ids.join(",")
        );
        let mut statement = self
            .connection
            .prepare(&sql)
            .map_err(|e| BagError(e.to_string()))?;
        let rows = statement
            .query_map(rusqlite::params![after, limit as i64], |r| {
                Ok((
                    r.get::<_, i64>(0)?,
                    r.get::<_, i64>(1)?,
                    r.get::<_, i64>(2)?,
                    r.get::<_, Vec<u8>>(3)?,
                ))
            })
            .map_err(|e| BagError(e.to_string()))?;
        let mut out = Vec::new();
        let mut last = after;
        for row in rows {
            let (id, topic, stamp, mut data) = row.map_err(|e| BagError(e.to_string()))?;
            if self.compressed {
                let mut decoder = ruzstd::decoding::StreamingDecoder::new(&data[..])
                    .map_err(|e| BagError(format!("zstd: {e}")))?;
                let mut out = Vec::new();
                decoder.read_to_end(&mut out)?;
                data = out;
            }
            last = id;
            out.push(Message {
                topic: self.names[&topic].clone(),
                time: stamp as f64 * 1e-9,
                encoding: Encoding::Cdr,
                data,
            });
        }
        Ok((out, last))
    }
}

/// One file of a recording.
enum Part {
    Bag(Bag<std::fs::File>),
    Sqlite(Sqlite),
}

/// A recording: a ROS 1 bag, an MCAP file, a rosbag2 SQLite file, or a
/// rosbag2 folder of such files read one after the other.
pub struct Recording {
    parts: Vec<Part>,
    topics: Vec<Topic>,
}

/// The files a rosbag2 folder's `metadata.yaml` lists, and whether each
/// message is zstd-compressed on its own.
fn metadata(folder: &Path) -> Result<(Vec<PathBuf>, bool)> {
    let text = std::fs::read_to_string(folder.join("metadata.yaml"))?;
    let mut files = Vec::new();
    let mut in_files = false;
    let mut format = String::new();
    let mut mode = String::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if let Some(v) = trimmed.strip_prefix("compression_format:") {
            format = v.trim().trim_matches(|c| c == '"' || c == '\'').to_string();
        } else if let Some(v) = trimmed.strip_prefix("compression_mode:") {
            mode = v.trim().trim_matches(|c| c == '"' || c == '\'').to_string();
        } else if trimmed.starts_with("relative_file_paths:") {
            in_files = true;
        } else if in_files {
            if let Some(v) = trimmed.strip_prefix("- ") {
                files.push(folder.join(v.trim().trim_matches(|c| c == '"' || c == '\'')));
            } else {
                in_files = false;
            }
        }
    }
    if files.is_empty() {
        return Err(BagError(format!(
            "{}: metadata.yaml lists no files",
            folder.display()
        )));
    }
    if !format.is_empty() && format != "zstd" {
        return Err(BagError(format!(
            "{format}-compressed rosbag2 recordings are not supported"
        )));
    }
    let compressed = mode.eq_ignore_ascii_case("message");
    if mode.eq_ignore_ascii_case("file") {
        return Err(BagError(
            "a rosbag2 recording compressed file by file (.db3.zstd): decompress it first".into(),
        ));
    }
    Ok((files, compressed))
}

impl Recording {
    pub fn open(path: &str) -> Result<Self> {
        let p = Path::new(path);
        let (files, compressed) = if p.is_dir() {
            metadata(p)?
        } else {
            (vec![p.to_path_buf()], false)
        };
        let mut parts = Vec::new();
        for file in &files {
            let name = file.to_string_lossy();
            let part = if name.to_lowercase().ends_with(".db3") {
                Part::Sqlite(Sqlite::open(file, compressed)?)
            } else {
                Part::Bag(Bag::open(&name)?)
            };
            parts.push(part);
        }
        let mut totals: BTreeMap<String, Topic> = BTreeMap::new();
        for part in &parts {
            let topics = match part {
                Part::Bag(b) => b.topics(),
                Part::Sqlite(s) => s.topics(),
            };
            for t in topics {
                totals
                    .entry(t.name.clone())
                    .or_insert_with(|| Topic {
                        name: t.name.clone(),
                        kind: t.kind.clone(),
                        count: 0,
                    })
                    .count += t.count;
            }
        }
        Ok(Recording {
            parts,
            topics: totals.into_values().collect(),
        })
    }

    pub fn topics(&self) -> &[Topic] {
        &self.topics
    }

    /// When the first and the last message were recorded (seconds), over the files.
    pub fn time_range(&self) -> Option<(f64, f64)> {
        let ranges: Vec<(f64, f64)> = self
            .parts
            .iter()
            .filter_map(|p| match p {
                Part::Bag(b) => b.time_range(),
                Part::Sqlite(s) => s.time_range(),
            })
            .collect();
        let lo = ranges.iter().map(|r| r.0).min_by(f64::total_cmp)?;
        let hi = ranges.iter().map(|r| r.1).max_by(f64::total_cmp)?;
        Some((lo, hi))
    }

    /// Start reading the messages of `topics`, file by file, oldest first.
    pub fn cursor(&self, topics: &[&str]) -> RecordingCursor {
        RecordingCursor {
            topics: topics.iter().map(|t| t.to_string()).collect(),
            part: 0,
            bag: Cursor::new(topics),
            after: 0,
            queue: Vec::new(),
        }
    }
}

/// Where a read through a recording has got to (see [`Recording::cursor`]).
pub struct RecordingCursor {
    topics: Vec<String>,
    part: usize,
    bag: Cursor,
    /// The last SQLite message id read.
    after: i64,
    /// A SQLite batch, last first.
    queue: Vec<Message>,
}

/// Messages a SQLite batch holds.
const BATCH: usize = 256;

impl RecordingCursor {
    /// How far through the recording the read is, 0 to 1 (by file, and within a bag by chunk).
    pub fn progress(&self, recording: &Recording) -> f64 {
        let n = recording.parts.len().max(1) as f64;
        let within = match recording.parts.get(self.part) {
            Some(Part::Bag(b)) => self.bag.progress(b),
            _ => 0.0,
        };
        ((self.part as f64 + within) / n).min(1.0)
    }

    /// The next message, when there is one.
    pub fn next(&mut self, recording: &mut Recording) -> Option<Result<Message>> {
        loop {
            let part = recording.parts.get_mut(self.part)?;
            match part {
                Part::Bag(b) => {
                    if let Some(m) = self.bag.next(b) {
                        return Some(m);
                    }
                }
                Part::Sqlite(s) => {
                    if let Some(m) = self.queue.pop() {
                        return Some(Ok(m));
                    }
                    match s.batch(&self.topics, self.after, BATCH) {
                        Ok((mut batch, last)) if !batch.is_empty() => {
                            self.after = last;
                            batch.reverse();
                            self.queue = batch;
                            continue;
                        }
                        Ok(_) => {}
                        Err(e) => return Some(Err(e)),
                    }
                }
            }
            // On to the next file.
            self.part += 1;
            let names: Vec<&str> = self.topics.iter().map(|t| t.as_str()).collect();
            self.bag = Cursor::new(&names);
            self.after = 0;
        }
    }
}

/// Decoders shared with plain bags.
pub use bag::{decode_imu, decode_point_cloud2, ups_at};

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_sqlite_recording_is_read_file_by_file() {
        let dir = std::env::temp_dir().join(format!("ca-rosbag2-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        for (k, name) in ["part_0.db3", "part_1.db3"].iter().enumerate() {
            let c = rusqlite::Connection::open(dir.join(name)).unwrap();
            c.execute_batch(
                "CREATE TABLE topics(id INTEGER PRIMARY KEY, name TEXT, type TEXT, serialization_format TEXT, offered_qos_profiles TEXT);
                 CREATE TABLE messages(id INTEGER PRIMARY KEY, topic_id INTEGER, timestamp INTEGER, data BLOB);
                 INSERT INTO topics VALUES (1, '/imu', 'sensor_msgs/msg/Imu', 'cdr', '');
                 INSERT INTO topics VALUES (2, '/points', 'sensor_msgs/msg/PointCloud2', 'cdr', '');",
            )
            .unwrap();
            for i in 0..3 {
                let t = 1_000_000_000i64 * (10 * k as i64 + i);
                c.execute(
                    "INSERT INTO messages(topic_id, timestamp, data) VALUES (2, ?1, ?2)",
                    rusqlite::params![t, vec![k as u8, i as u8]],
                )
                .unwrap();
                c.execute(
                    "INSERT INTO messages(topic_id, timestamp, data) VALUES (1, ?1, ?2)",
                    rusqlite::params![t + 5, vec![9u8]],
                )
                .unwrap();
            }
        }
        std::fs::write(
            dir.join("metadata.yaml"),
            "rosbag2_bagfile_information:\n  version: 5\n  storage_identifier: sqlite3\n  compression_format: \"\"\n  compression_mode: \"\"\n  relative_file_paths:\n    - part_0.db3\n    - part_1.db3\n  files:\n    - path: part_0.db3\n",
        )
        .unwrap();
        let mut recording = Recording::open(dir.to_str().unwrap()).unwrap();
        let counts: Vec<(String, String, u64)> = recording
            .topics()
            .iter()
            .map(|t| (t.name.clone(), t.kind.clone(), t.count))
            .collect();
        assert_eq!(
            counts,
            [
                ("/imu".into(), "sensor_msgs/Imu".into(), 6),
                ("/points".into(), "sensor_msgs/PointCloud2".into(), 6)
            ]
        );
        assert_eq!(recording.time_range(), Some((0.0, 12.000000005)));
        let mut cursor = recording.cursor(&["/points"]);
        let mut got = Vec::new();
        while let Some(m) = cursor.next(&mut recording) {
            let m = m.unwrap();
            got.push((m.time, m.data));
        }
        assert_eq!(
            got,
            [
                (0.0, vec![0, 0]),
                (1.0, vec![0, 1]),
                (2.0, vec![0, 2]),
                (10.0, vec![1, 0]),
                (11.0, vec![1, 1]),
                (12.0, vec![1, 2])
            ]
        );
        drop(recording);
        std::fs::remove_dir_all(&dir).unwrap();
    }
}
