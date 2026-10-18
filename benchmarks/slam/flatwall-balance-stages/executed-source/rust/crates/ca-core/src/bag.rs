//! ROS bags: ROS 1 (`.bag`, format 2.0) and MCAP (ROS 2's default), read a
//! chunk at a time from any `Read + Seek` source so that bags of gigabytes
//! open, with just enough decoding for sensor_msgs PointCloud2 and Imu.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::io::{Read, Seek, SeekFrom};

#[derive(Debug, thiserror::Error)]
#[error("{0}")]
pub struct BagError(pub String);

impl From<std::io::Error> for BagError {
    fn from(e: std::io::Error) -> Self {
        BagError(e.to_string())
    }
}

type Result<T> = std::result::Result<T, BagError>;

/// A topic of the bag.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Topic {
    pub name: String,
    /// As ROS 1 names it (`sensor_msgs/PointCloud2`), for either kind of bag.
    pub kind: String,
    pub count: u64,
}

/// How a message's bytes are laid out.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Encoding {
    Ros1,
    Cdr,
}

/// A message on one of the topics asked for.
#[derive(Debug, Clone)]
pub struct Message {
    pub topic: String,
    /// When it was recorded (seconds).
    pub time: f64,
    pub encoding: Encoding,
    pub data: Vec<u8>,
}

pub const POINT_CLOUD: &str = "sensor_msgs/PointCloud2";
pub const IMU: &str = "sensor_msgs/Imu";

// --- Bytes -------------------------------------------------------------------

/// Little-endian fields of a record.
struct Fields<'a> {
    b: &'a [u8],
    o: usize,
}

impl<'a> Fields<'a> {
    fn new(b: &'a [u8]) -> Self {
        Fields { b, o: 0 }
    }

    fn need(&self, n: usize) -> Result<()> {
        if self.o + n > self.b.len() {
            return Err(BagError(
                "a record ends early (is the bag cut short?)".into(),
            ));
        }
        Ok(())
    }

    fn u16(&mut self) -> Result<u16> {
        self.need(2)?;
        self.o += 2;
        Ok(u16::from_le_bytes([self.b[self.o - 2], self.b[self.o - 1]]))
    }

    fn u32(&mut self) -> Result<u32> {
        self.need(4)?;
        self.o += 4;
        Ok(u32::from_le_bytes(
            self.b[self.o - 4..self.o].try_into().unwrap(),
        ))
    }

    fn u64(&mut self) -> Result<u64> {
        self.need(8)?;
        self.o += 8;
        Ok(u64::from_le_bytes(
            self.b[self.o - 8..self.o].try_into().unwrap(),
        ))
    }

    fn bytes(&mut self, n: usize) -> Result<&'a [u8]> {
        self.need(n)?;
        self.o += n;
        Ok(&self.b[self.o - n..self.o])
    }

    fn string(&mut self) -> Result<String> {
        let n = self.u32()? as usize;
        Ok(String::from_utf8_lossy(self.bytes(n)?).into_owned())
    }

    fn rest(&self) -> &'a [u8] {
        &self.b[self.o.min(self.b.len())..]
    }
}

fn u32_at(b: &[u8]) -> u32 {
    u32::from_le_bytes(b[..4].try_into().unwrap())
}

fn u64_at(b: &[u8]) -> u64 {
    u64::from_le_bytes(b[..8].try_into().unwrap())
}

/// Decompress a chunk as the bags name their compressions.
fn expand(compression: &str, data: &[u8], size: usize) -> Result<Vec<u8>> {
    let mut out = Vec::with_capacity(size);
    match compression {
        "" | "none" => return Ok(data.to_vec()),
        "lz4" => {
            lz4_flex::frame::FrameDecoder::new(data).read_to_end(&mut out)?;
        }
        "zstd" => {
            let mut decoder = ruzstd::decoding::StreamingDecoder::new(data)
                .map_err(|e| BagError(format!("zstd: {e}")))?;
            decoder.read_to_end(&mut out)?;
        }
        "bz2" => {
            bzip2::read::MultiBzDecoder::new(data).read_to_end(&mut out)?;
        }
        other => {
            return Err(BagError(format!(
                "{other}-compressed bags are not supported: decompress it first (e.g. rosbag decompress)"
            )));
        }
    }
    Ok(out)
}

fn read_at<R: Read + Seek>(source: &mut R, at: u64, n: usize) -> Result<Vec<u8>> {
    source.seek(SeekFrom::Start(at))?;
    let mut buf = vec![0u8; n];
    source
        .read_exact(&mut buf)
        .map_err(|_| BagError("the bag ends early (is it cut short?)".into()))?;
    Ok(buf)
}

// --- ROS 1 -------------------------------------------------------------------

/// A ROS 1 record header's `name=value` fields.
fn ros1_fields(bytes: &[u8]) -> HashMap<String, Vec<u8>> {
    let mut out = HashMap::new();
    let mut o = 0;
    while o + 4 <= bytes.len() {
        let n = u32_at(&bytes[o..]) as usize;
        let field = &bytes[(o + 4).min(bytes.len())..(o + 4 + n).min(bytes.len())];
        if let Some(eq) = field.iter().position(|&b| b == b'=') {
            out.insert(
                String::from_utf8_lossy(&field[..eq]).into_owned(),
                field[eq + 1..].to_vec(),
            );
        }
        o += 4 + n;
    }
    out
}

struct Ros1Record {
    header: HashMap<String, Vec<u8>>,
    data: Vec<u8>,
    /// Where the next record starts.
    next: u64,
}

impl Ros1Record {
    fn op(&self) -> u8 {
        self.header
            .get("op")
            .and_then(|v| v.first().copied())
            .unwrap_or(0)
    }

    fn u32(&self, name: &str) -> u32 {
        self.header
            .get(name)
            .filter(|v| v.len() >= 4)
            .map_or(0, |v| u32_at(v))
    }

    fn u64(&self, name: &str) -> u64 {
        self.header
            .get(name)
            .filter(|v| v.len() >= 8)
            .map_or(0, |v| u64_at(v))
    }

    fn time(&self, name: &str) -> f64 {
        self.header
            .get(name)
            .filter(|v| v.len() >= 8)
            .map_or(0.0, |v| {
                f64::from(u32_at(v)) + f64::from(u32_at(&v[4..])) * 1e-9
            })
    }

    fn text(&self, name: &str) -> String {
        self.header
            .get(name)
            .map(|v| String::from_utf8_lossy(v).into_owned())
            .unwrap_or_default()
    }
}

/// The record at `o` of an in-memory buffer.
fn ros1_record_in(bytes: &[u8], o: usize) -> Result<Ros1Record> {
    let mut f = Fields::new(bytes);
    f.o = o;
    let header_len = f.u32()? as usize;
    let header = ros1_fields(f.bytes(header_len)?);
    let data_len = f.u32()? as usize;
    let data = f.bytes(data_len)?.to_vec();
    Ok(Ros1Record {
        header,
        data,
        next: f.o as u64,
    })
}

/// The record at `o` of the bag, read in two steps (its header says how long its data is).
fn ros1_record_at<R: Read + Seek>(source: &mut R, o: u64) -> Result<Ros1Record> {
    let (header, data_at, data_len) = ros1_header_at(source, o)?;
    let data = read_at(source, data_at, data_len)?;
    Ok(Ros1Record {
        header,
        data,
        next: data_at + data_len as u64,
    })
}

/// The header of the record at `o` of the bag, and where its data is and how long.
fn ros1_header_at<R: Read + Seek>(
    source: &mut R,
    o: u64,
) -> Result<(HashMap<String, Vec<u8>>, u64, usize)> {
    let header_len = u32_at(&read_at(source, o, 4)?) as usize;
    let head = read_at(source, o + 4, header_len + 4)?;
    let header = ros1_fields(&head[..header_len]);
    let data_len = u32_at(&head[header_len..]) as usize;
    Ok((header, o + 8 + header_len as u64, data_len))
}

struct Ros1Chunk {
    at: u64,
    start: f64,
    end: f64,
    counts: HashMap<u32, u32>,
}

struct Ros1 {
    connections: HashMap<u32, (String, String)>,
    chunks: Vec<Ros1Chunk>,
}

fn open_ros1<R: Read + Seek>(source: &mut R, size: u64) -> Result<(Ros1, Vec<Topic>)> {
    let header = ros1_record_at(source, 13)?;
    let index_at = header.u64("index_pos");
    if header.op() != 0x03 || index_at == 0 {
        return Err(BagError(
            "the bag has no index (was recording cut short?): run rosbag reindex on it first"
                .into(),
        ));
    }
    let mut connections = HashMap::new();
    let mut chunks = Vec::new();
    let mut o = index_at;
    while o < size {
        let r = ros1_record_at(source, o)?;
        o = r.next;
        match r.op() {
            0x07 => {
                let info = ros1_fields(&r.data);
                let kind = info
                    .get("type")
                    .map(|v| String::from_utf8_lossy(v).into_owned())
                    .unwrap_or_default();
                connections.insert(r.u32("conn"), (r.text("topic"), kind));
            }
            0x06 => {
                let mut counts = HashMap::new();
                let mut k = 0;
                while k + 8 <= r.data.len() {
                    counts.insert(u32_at(&r.data[k..]), u32_at(&r.data[k + 4..]));
                    k += 8;
                }
                chunks.push(Ros1Chunk {
                    at: r.u64("chunk_pos"),
                    start: r.time("start_time"),
                    end: r.time("end_time"),
                    counts,
                });
            }
            _ => {}
        }
    }
    chunks.sort_by(|a, b| a.start.total_cmp(&b.start).then(a.at.cmp(&b.at)));
    let mut totals: BTreeMap<String, Topic> = BTreeMap::new();
    for chunk in &chunks {
        for (conn, count) in &chunk.counts {
            let Some((topic, kind)) = connections.get(conn) else {
                continue;
            };
            totals
                .entry(topic.clone())
                .or_insert_with(|| Topic {
                    name: topic.clone(),
                    kind: kind.clone(),
                    count: 0,
                })
                .count += u64::from(*count);
        }
    }
    Ok((
        Ros1 {
            connections,
            chunks,
        },
        totals.into_values().collect(),
    ))
}

// --- MCAP --------------------------------------------------------------------

struct McapRecord<'a> {
    opcode: u8,
    body: &'a [u8],
}

/// The records of an in-memory buffer.
fn mcap_records(bytes: &[u8]) -> impl Iterator<Item = McapRecord<'_>> {
    let mut o = 0;
    std::iter::from_fn(move || {
        if o + 9 > bytes.len() {
            return None;
        }
        let length = u64_at(&bytes[o + 1..]) as usize;
        let body = &bytes[(o + 9).min(bytes.len())..(o + 9 + length).min(bytes.len())];
        let record = McapRecord {
            opcode: bytes[o],
            body,
        };
        o += 9 + length;
        Some(record)
    })
}

/// A chunk record's content, expanded to the records it holds.
fn mcap_chunk_records(body: &[u8]) -> Result<Vec<u8>> {
    let mut f = Fields::new(body);
    f.u64()?;
    f.u64()?;
    let size = f.u64()? as usize;
    f.u32()?;
    let compression = f.string()?;
    let n = f.u64()? as usize;
    expand(&compression, f.bytes(n)?, size)
}

struct McapChunk {
    at: u64,
    length: u64,
    start: u64,
    end: u64,
    /// Channels the chunk index lists; None when it lists none.
    channels: Option<HashSet<u16>>,
}

#[derive(Default)]
struct Mcap {
    schemas: HashMap<u16, String>,
    /// Channel id: topic, type, encoding.
    channels: HashMap<u16, (String, String, Encoding)>,
    counts: HashMap<u16, u64>,
    chunks: Vec<McapChunk>,
}

impl Mcap {
    fn learn(&mut self, r: &McapRecord) -> Result<()> {
        let mut f = Fields::new(r.body);
        match r.opcode {
            0x03 => {
                let id = f.u16()?;
                let name = f.string()?;
                self.schemas.insert(id, name);
            }
            0x04 => {
                let id = f.u16()?;
                let schema = f.u16()?;
                let topic = f.string()?;
                let encoding = f.string()?;
                let kind = self
                    .schemas
                    .get(&schema)
                    .map(|s| s.replace("/msg/", "/"))
                    .unwrap_or_default();
                let encoding = if encoding == "ros1" {
                    Encoding::Ros1
                } else {
                    Encoding::Cdr
                };
                self.channels.insert(id, (topic, kind, encoding));
            }
            _ => {}
        }
        Ok(())
    }
}

/// Every record of an MCAP read end to end (chunks' records inside them), for
/// bags without a summary.
fn mcap_linear<R: Read + Seek>(
    source: &mut R,
    size: u64,
    mut each: impl FnMut(&McapRecord) -> Result<()>,
) -> Result<()> {
    let mut o = 8;
    while o + 9 <= size {
        let head = read_at(source, o, 9)?;
        let opcode = head[0];
        let length = u64_at(&head[1..]);
        if opcode == 0x0f || opcode == 0x02 {
            return Ok(());
        }
        let body = read_at(source, o + 9, length as usize)?;
        o += 9 + length;
        if opcode == 0x06 {
            for r in mcap_records(&mcap_chunk_records(&body)?) {
                each(&r)?;
            }
        } else {
            each(&McapRecord {
                opcode,
                body: &body,
            })?;
        }
    }
    Ok(())
}

fn open_mcap<R: Read + Seek>(source: &mut R, size: u64) -> Result<(Mcap, Vec<Topic>)> {
    let mut mcap = Mcap::default();
    if size < 37 {
        return Err(BagError("not an MCAP file".into()));
    }
    let tail = read_at(source, size - 37, 37)?;
    let summary_at = if tail[0] == 0x02 {
        u64_at(&tail[9..])
    } else {
        0
    };
    if summary_at > 0 {
        let summary = read_at(source, summary_at, (size - 37 - summary_at) as usize)?;
        for r in mcap_records(&summary) {
            mcap.learn(&r)?;
            let mut f = Fields::new(r.body);
            if r.opcode == 0x08 {
                let start = f.u64()?;
                let end = f.u64()?;
                let at = f.u64()?;
                let length = f.u64()?;
                let map_end = f.u32()? as usize + f.o;
                let mut indexed = HashSet::new();
                while f.o < map_end {
                    indexed.insert(f.u16()?);
                    f.u64()?;
                }
                mcap.chunks.push(McapChunk {
                    at,
                    length,
                    start,
                    end,
                    channels: (!indexed.is_empty()).then_some(indexed),
                });
            } else if r.opcode == 0x0b {
                f.u64()?;
                f.u16()?;
                f.u32()?;
                f.u32()?;
                f.u32()?;
                f.u32()?;
                f.u64()?;
                f.u64()?;
                let end = f.u32()? as usize + f.o;
                while f.o < end {
                    let id = f.u16()?;
                    let count = f.u64()?;
                    mcap.counts.insert(id, count);
                }
            }
        }
    }
    if mcap.chunks.is_empty() {
        // No index: count by reading it all once.
        let mut learned = Mcap::default();
        let mut counts: HashMap<u16, u64> = HashMap::new();
        mcap_linear(source, size, |r| {
            learned.learn(r)?;
            if r.opcode == 0x05 && r.body.len() >= 2 {
                *counts
                    .entry(u16::from_le_bytes([r.body[0], r.body[1]]))
                    .or_default() += 1;
            }
            Ok(())
        })?;
        mcap.schemas.extend(learned.schemas);
        mcap.channels.extend(learned.channels);
        mcap.counts = counts;
    }
    mcap.chunks
        .sort_by(|a, b| a.start.cmp(&b.start).then(a.at.cmp(&b.at)));
    let mut totals: BTreeMap<String, Topic> = BTreeMap::new();
    for (id, (topic, kind, _)) in &mcap.channels {
        totals
            .entry(topic.clone())
            .or_insert_with(|| Topic {
                name: topic.clone(),
                kind: kind.clone(),
                count: 0,
            })
            .count += mcap.counts.get(id).copied().unwrap_or(0);
    }
    Ok((mcap, totals.into_values().collect()))
}

// --- The bag -----------------------------------------------------------------

enum Format {
    Ros1(Ros1),
    Mcap(Mcap),
}

/// A bag opened for reading.
pub struct Bag<R: Read + Seek> {
    source: R,
    size: u64,
    format: Format,
    topics: Vec<Topic>,
}

impl Bag<std::fs::File> {
    pub fn open(path: &str) -> Result<Self> {
        Bag::from_reader(std::fs::File::open(path)?)
    }
}

impl<R: Read + Seek> Bag<R> {
    pub fn from_reader(mut source: R) -> Result<Self> {
        let size = source.seek(SeekFrom::End(0))?;
        let magic = read_at(&mut source, 0, 13.min(size as usize))?;
        let (format, topics) = if magic.starts_with(b"#ROSBAG V2.0\n") {
            let (r, t) = open_ros1(&mut source, size)?;
            (Format::Ros1(r), t)
        } else if magic.starts_with(b"\x89MCAP") {
            let (m, t) = open_mcap(&mut source, size)?;
            (Format::Mcap(m), t)
        } else {
            return Err(BagError(
                "not a ROS 1 bag (format 2.0) or an MCAP file".into(),
            ));
        };
        Ok(Bag {
            source,
            size,
            format,
            topics,
        })
    }

    pub fn is_ros1(&self) -> bool {
        matches!(self.format, Format::Ros1(_))
    }

    /// When the first and the last message were recorded (seconds), from the
    /// chunk index; None for a bag without one.
    pub fn time_range(&self) -> Option<(f64, f64)> {
        match &self.format {
            Format::Ros1(r) => {
                let start = r.chunks.iter().map(|c| c.start).min_by(f64::total_cmp)?;
                let end = r.chunks.iter().map(|c| c.end).max_by(f64::total_cmp)?;
                Some((start, end))
            }
            Format::Mcap(m) => {
                let start = m.chunks.iter().map(|c| c.start).min()?;
                let end = m.chunks.iter().map(|c| c.end).max()?;
                Some((start as f64 * 1e-9, end as f64 * 1e-9))
            }
        }
    }

    pub fn topics(&self) -> &[Topic] {
        &self.topics
    }

    /// The messages on `topics`, in recording order chunk by chunk.
    pub fn messages<'a>(&'a mut self, topics: &[&str]) -> Messages<'a, R> {
        Messages {
            bag: self,
            cursor: Cursor::new(topics),
        }
    }
}

/// Where a read through a bag's messages has got to: [`Cursor::next`] with
/// the bag gives the next message, so that the bag can be kept apart from
/// the read (as a Python iterator does); [`Bag::messages`] wraps both.
pub struct Cursor {
    wanted: HashSet<String>,
    chunk: usize,
    /// The current chunk's messages, last first.
    queue: Vec<Message>,
    /// For an MCAP without an index: every message, read in one go.
    linear: Option<std::vec::IntoIter<Message>>,
}

/// Iterator over a bag's messages (see [`Bag::messages`]).
pub struct Messages<'a, R: Read + Seek> {
    bag: &'a mut Bag<R>,
    cursor: Cursor,
}

impl<R: Read + Seek> Messages<'_, R> {
    /// How far through the bag's chunks the iterator is, 0 to 1.
    pub fn progress(&self) -> f64 {
        self.cursor.progress(self.bag)
    }
}

impl<R: Read + Seek> Iterator for Messages<'_, R> {
    type Item = Result<Message>;

    fn next(&mut self) -> Option<Self::Item> {
        self.cursor.next(self.bag)
    }
}

impl Cursor {
    pub fn new(topics: &[&str]) -> Self {
        Cursor {
            wanted: topics.iter().map(|t| t.to_string()).collect(),
            chunk: 0,
            queue: Vec::new(),
            linear: None,
        }
    }

    /// How far through the bag's chunks the read is, 0 to 1.
    pub fn progress<R: Read + Seek>(&self, bag: &Bag<R>) -> f64 {
        let n = match &bag.format {
            Format::Ros1(r) => r.chunks.len(),
            Format::Mcap(m) => m.chunks.len(),
        };
        if n == 0 {
            1.0
        } else {
            self.chunk as f64 / n as f64
        }
    }

    /// The next message of the topics, when there is one.
    pub fn next<R: Read + Seek>(&mut self, bag: &mut Bag<R>) -> Option<Result<Message>> {
        loop {
            if let Some(linear) = &mut self.linear {
                return linear.next().map(Ok);
            }
            if let Some(m) = self.queue.pop() {
                return Some(Ok(m));
            }
            match self.next_chunk(bag) {
                Ok(true) => {}
                Ok(false) => return None,
                Err(e) => return Some(Err(e)),
            }
        }
    }

    /// Fill the queue from the next chunk holding a wanted message; false when there is none.
    fn next_chunk<R: Read + Seek>(&mut self, bag: &mut Bag<R>) -> Result<bool> {
        loop {
            let mut found = Vec::new();
            match &bag.format {
                Format::Ros1(r) => {
                    let Some(chunk) = r.chunks.get(self.chunk) else {
                        return Ok(false);
                    };
                    self.chunk += 1;
                    let wanted: HashSet<u32> = r
                        .connections
                        .iter()
                        .filter(|(_, (topic, _))| self.wanted.contains(topic))
                        .map(|(conn, _)| *conn)
                        .collect();
                    if !chunk.counts.keys().any(|c| wanted.contains(c)) {
                        continue;
                    }
                    let (header, data_at, data_len) = ros1_header_at(&mut bag.source, chunk.at)?;
                    let chunk_record = Ros1Record {
                        header,
                        data: Vec::new(),
                        next: 0,
                    };
                    let compression = chunk_record.text("compression");
                    if compression.is_empty() || compression == "none" {
                        // Uncompressed: walk the records in place, reading only the wanted messages.
                        let end = data_at + data_len as u64;
                        let mut o = data_at;
                        while o < end {
                            let (header, at, len) = ros1_header_at(&mut bag.source, o)?;
                            o = at + len as u64;
                            let m = Ros1Record {
                                header,
                                data: Vec::new(),
                                next: o,
                            };
                            let conn = m.u32("conn");
                            if m.op() != 0x02 || !wanted.contains(&conn) {
                                continue;
                            }
                            found.push(Message {
                                topic: r.connections[&conn].0.clone(),
                                time: m.time("time"),
                                encoding: Encoding::Ros1,
                                data: read_at(&mut bag.source, at, len)?,
                            });
                        }
                    } else {
                        let data = read_at(&mut bag.source, data_at, data_len)?;
                        let records =
                            expand(&compression, &data, chunk_record.u32("size") as usize)?;
                        let mut o = 0;
                        while o < records.len() {
                            let m = ros1_record_in(&records, o)?;
                            o = m.next as usize;
                            let conn = m.u32("conn");
                            if m.op() != 0x02 || !wanted.contains(&conn) {
                                continue;
                            }
                            found.push(Message {
                                topic: r.connections[&conn].0.clone(),
                                time: m.time("time"),
                                encoding: Encoding::Ros1,
                                data: m.data,
                            });
                        }
                    }
                }
                Format::Mcap(m) => {
                    if m.chunks.is_empty() {
                        if self.linear.is_some() {
                            return Ok(false);
                        }
                        let mut all = Vec::new();
                        let wanted = &self.wanted;
                        let mut learned = Mcap::default();
                        let size = bag.size;
                        mcap_linear(&mut bag.source, size, |r| {
                            learned.learn(r)?;
                            if r.opcode == 0x05
                                && let Some(msg) = mcap_message(&learned, r.body, wanted)?
                            {
                                all.push(msg);
                            }
                            Ok(())
                        })?;
                        self.linear = Some(all.into_iter());
                        return Ok(true);
                    }
                    let Some(chunk) = m.chunks.get(self.chunk) else {
                        return Ok(false);
                    };
                    self.chunk += 1;
                    if let Some(ids) = &chunk.channels
                        && !ids.iter().any(|id| {
                            m.channels
                                .get(id)
                                .is_some_and(|(topic, _, _)| self.wanted.contains(topic))
                        })
                    {
                        continue;
                    }
                    let body = read_at(&mut bag.source, chunk.at + 9, (chunk.length - 9) as usize)?;
                    let records = mcap_chunk_records(&body)?;
                    for r in mcap_records(&records) {
                        if r.opcode == 0x05
                            && let Some(msg) = mcap_message(m, r.body, &self.wanted)?
                        {
                            found.push(msg);
                        }
                    }
                }
            }
            // Pop in ascending recording time, retaining physical record order
            // for ties (batched sensor messages often share a recording time).
            found.sort_by(|a, b| a.time.total_cmp(&b.time));
            found.reverse();
            self.queue = found;
            return Ok(true);
        }
    }
}

fn mcap_message(m: &Mcap, body: &[u8], wanted: &HashSet<String>) -> Result<Option<Message>> {
    let mut f = Fields::new(body);
    let id = f.u16()?;
    let Some((topic, _, encoding)) = m.channels.get(&id) else {
        return Ok(None);
    };
    if !wanted.contains(topic) {
        return Ok(None);
    }
    f.u32()?;
    let time = f.u64()? as f64 * 1e-9;
    f.u64()?;
    Ok(Some(Message {
        topic: topic.clone(),
        time,
        encoding: *encoding,
        data: f.rest().to_vec(),
    }))
}

// --- Messages ----------------------------------------------------------------

/// Reads ROS 1 or CDR (ROS 2) serialised fields.
struct Reader<'a> {
    b: &'a [u8],
    o: usize,
    base: usize,
    cdr: bool,
    little: bool,
}

impl<'a> Reader<'a> {
    fn new(b: &'a [u8], encoding: Encoding) -> Self {
        let cdr = encoding == Encoding::Cdr;
        Reader {
            b,
            o: if cdr { 4 } else { 0 },
            base: if cdr { 4 } else { 0 },
            cdr,
            little: !cdr || b.get(1) == Some(&1),
        }
    }

    fn align(&mut self, n: usize) {
        if self.cdr {
            self.o += (n - (self.o - self.base) % n) % n;
        }
    }

    fn take(&mut self, n: usize) -> Result<&'a [u8]> {
        if self.o + n > self.b.len() {
            return Err(BagError("a message ends early".into()));
        }
        self.o += n;
        Ok(&self.b[self.o - n..self.o])
    }

    fn u8(&mut self) -> Result<u8> {
        Ok(self.take(1)?[0])
    }

    fn u32(&mut self) -> Result<u32> {
        self.align(4);
        let b: [u8; 4] = self.take(4)?.try_into().unwrap();
        Ok(if self.little {
            u32::from_le_bytes(b)
        } else {
            u32::from_be_bytes(b)
        })
    }

    fn f64(&mut self) -> Result<f64> {
        self.align(8);
        let b: [u8; 8] = self.take(8)?.try_into().unwrap();
        Ok(if self.little {
            f64::from_le_bytes(b)
        } else {
            f64::from_be_bytes(b)
        })
    }

    fn string(&mut self) -> Result<String> {
        let n = self.u32()? as usize;
        let s = self.take(n)?;
        let s = if self.cdr && s.last() == Some(&0) {
            &s[..n - 1]
        } else {
            s
        };
        Ok(String::from_utf8_lossy(s).into_owned())
    }

    fn header_parts(&mut self) -> Result<(i64, u32, String)> {
        if !self.cdr {
            self.u32()?;
        }
        let sec = self.u32()?;
        // ROS 2 builtin_interfaces/Time uses signed seconds; ROS 1 uses u32.
        let sec = if self.cdr {
            i64::from(sec as i32)
        } else {
            i64::from(sec)
        };
        let nsec = self.u32()?;
        if nsec >= 1_000_000_000 {
            return Err(BagError("a header has nanoseconds outside [0, 1e9)".into()));
        }
        Ok((sec, nsec, self.string()?))
    }

    /// std_msgs/Header's stamp (seconds).
    fn header(&mut self) -> Result<f64> {
        let (sec, nsec, _) = self.header_parts()?;
        Ok(sec as f64 + f64::from(nsec) * 1e-9)
    }

    fn f64s<const N: usize>(&mut self) -> Result<[f64; N]> {
        let mut values = [0.0; N];
        for value in &mut values {
            *value = self.f64()?;
        }
        Ok(values)
    }
}

/// Original scalar point-time values, aligned with the finite decoded points.
/// All standard PointField integer types fit exactly in f64. Units and time
/// origin are deliberately unspecified; floating/nonfinite values are retained.
#[derive(Debug, Clone)]
pub struct RawPointTime {
    pub field: String,
    pub datatype: u8,
    pub values: Vec<f64>,
}

/// A PointCloud2 decoded.
#[derive(Debug, Clone, Default)]
pub struct PointsMessage {
    /// Exact sensor header timestamp, independent of the bag recording time.
    pub stamp_ns: i64,
    /// Floating-point header seconds, retained for existing consumers.
    pub stamp: f64,
    /// The PointCloud2 header frame, without coordinate conversion.
    pub frame_id: String,
    /// The finite points only.
    pub positions: Vec<[f64; 3]>,
    /// One per point, when the cloud has an intensity field.
    pub intensity: Option<Vec<f32>>,
    /// How far through the scan each point was taken, 0 to 1, when the cloud has a per-point time field.
    pub time: Option<Vec<f32>>,
    /// Explicitly requested source field; absent from the default decoder.
    pub raw_time: Option<RawPointTime>,
}

/// Names LiDAR drivers give a per-point time field.
const TIME_FIELDS: [&str; 7] = [
    "t",
    "time",
    "timestamp",
    "time_stamp",
    "time_offset",
    "point_time",
    "ts",
];
/// And an intensity.
const INTENSITY_FIELDS: [&str; 3] = ["intensity", "reflectivity", "i"];

fn field_value(bytes: &[u8], at: usize, datatype: u8, big: bool) -> Option<f64> {
    let v = bytes.get(at..)?;
    macro_rules! num {
        ($t:ty, $n:expr) => {{
            let b: [u8; $n] = v.get(..$n)?.try_into().ok()?;
            Some(if big {
                <$t>::from_be_bytes(b)
            } else {
                <$t>::from_le_bytes(b)
            } as f64)
        }};
    }
    match datatype {
        1 => num!(i8, 1),
        2 => num!(u8, 1),
        3 => num!(i16, 2),
        4 => num!(u16, 2),
        5 => num!(i32, 4),
        6 => num!(u32, 4),
        7 => num!(f32, 4),
        8 => num!(f64, 8),
        _ => None,
    }
}

pub fn decode_point_cloud2(data: &[u8], encoding: Encoding) -> Result<PointsMessage> {
    decode_point_cloud2_inner(data, encoding, None)
}

/// Retains an explicitly named scalar field, without normalization or unit/origin
/// inference. Invalid/missing fields and incomplete point records are errors.
/// Legacy normalized time remains independent of the requested source field.
pub fn decode_point_cloud2_with_raw_time(
    data: &[u8],
    encoding: Encoding,
    field: &str,
) -> Result<PointsMessage> {
    decode_point_cloud2_inner(data, encoding, Some(field))
}

fn validate_raw_time_layout(
    (offset, datatype, count): (usize, u8, u32),
    step: usize,
    row_step: usize,
    width: usize,
    height: usize,
    bytes: usize,
) -> Result<()> {
    let size = match datatype {
        1 | 2 => 1,
        3 | 4 => 2,
        5..=7 => 4,
        8 => 8,
        _ => return Err(BagError("unsupported raw time field datatype".into())),
    };
    if count != 1 || offset.checked_add(size).is_none_or(|end| end > step) {
        return Err(BagError(
            "raw time field must be scalar and fit within point_step".into(),
        ));
    }
    let row_width = width
        .checked_mul(step)
        .ok_or_else(|| BagError("raw time field row size overflow".into()))?;
    let extent = if width == 0 || height == 0 {
        Some(0)
    } else {
        (height - 1)
            .checked_mul(row_step)
            .and_then(|at| at.checked_add(row_width))
    };
    if row_step < row_width || extent.is_none_or(|end| end > bytes) {
        return Err(BagError(
            "raw time field needs complete, nonoverlapping point rows".into(),
        ));
    }
    Ok(())
}

fn decode_point_cloud2_inner(
    data: &[u8],
    encoding: Encoding,
    requested_time: Option<&str>,
) -> Result<PointsMessage> {
    let mut m = Reader::new(data, encoding);
    let (sec, nsec, frame_id) = m.header_parts()?;
    let height = m.u32()? as usize;
    let width = m.u32()? as usize;
    let mut layout: HashMap<String, (usize, u8)> = HashMap::new();
    let mut raw_field = None;
    for _ in 0..m.u32()? {
        let name = m.string()?;
        let offset = m.u32()? as usize;
        let datatype = m.u8()?;
        let count = m.u32()?;
        if requested_time == Some(name.as_str())
            && raw_field.replace((offset, datatype, count)).is_some()
        {
            return Err(BagError("duplicate raw time field name".into()));
        }
        layout.insert(name, (offset, datatype));
    }
    let big = m.u8()? != 0;
    let step = m.u32()? as usize;
    let row_step = m.u32()? as usize;
    let n = m.u32()? as usize;
    let bytes = m.take(n)?;
    let raw_layout = if let Some(name) = requested_time {
        let raw = raw_field.ok_or_else(|| BagError(format!("raw time field not found: {name}")))?;
        validate_raw_time_layout(raw, step, row_step, width, height, bytes.len())?;
        Some((raw.0, raw.1))
    } else {
        None
    };
    let field = |name: &str| layout.get(name).copied();
    let (Some(x), Some(y), Some(z)) = (field("x"), field("y"), field("z")) else {
        return Err(BagError("a PointCloud2 without x, y and z fields".into()));
    };
    let intensity = INTENSITY_FIELDS.iter().find_map(|f| field(f));
    let time = TIME_FIELDS.iter().find_map(|f| field(f));
    let count = width
        .checked_mul(height)
        .ok_or_else(|| BagError("PointCloud2 point count overflow".into()))?;
    let mut out = PointsMessage {
        stamp_ns: sec * 1_000_000_000 + i64::from(nsec),
        stamp: sec as f64 + f64::from(nsec) * 1e-9,
        frame_id,
        positions: Vec::with_capacity(count),
        intensity: intensity.map(|_| Vec::with_capacity(count)),
        time: time.map(|_| Vec::with_capacity(count)),
        raw_time: None,
    };
    let mut times: Vec<f64> = Vec::new();
    let shared_time = raw_layout.is_some() && raw_layout == time;
    let mut raw_values = Vec::new();
    for row in 0..height {
        for col in 0..width {
            let at = row
                .checked_mul(row_step)
                .and_then(|row| col.checked_mul(step).and_then(|col| row.checked_add(col)))
                .ok_or_else(|| BagError("PointCloud2 point offset overflow".into()))?;
            if at.checked_add(step).is_none_or(|end| end > bytes.len()) {
                break;
            }
            let get = |(offset, datatype): (usize, u8)| {
                at.checked_add(offset)
                    .and_then(|at| field_value(bytes, at, datatype, big))
            };
            let (Some(px), Some(py), Some(pz)) = (get(x), get(y), get(z)) else {
                continue;
            };
            if !(px.is_finite() && py.is_finite() && pz.is_finite()) {
                continue;
            }
            out.positions.push([px, py, pz]);
            if let (Some(values), Some(f)) = (&mut out.intensity, intensity) {
                values.push(get(f).unwrap_or(0.0) as f32);
            }
            if let Some(f) = time {
                let value = get(f);
                times.push(if shared_time {
                    value.ok_or_else(|| BagError("incomplete raw time field value".into()))?
                } else {
                    value.unwrap_or(0.0)
                });
            }
            if let Some(f) = raw_layout.filter(|_| !shared_time) {
                raw_values.push(
                    get(f).ok_or_else(|| BagError("incomplete raw time field value".into()))?,
                );
            }
        }
    }
    if let Some(fractions) = &mut out.time {
        let (lo, hi) = times
            .iter()
            .fold((f64::MAX, f64::MIN), |(lo, hi), &t| (lo.min(t), hi.max(t)));
        if hi > lo {
            fractions.extend(times.iter().map(|t| ((t - lo) / (hi - lo)) as f32));
        } else {
            out.time = None;
        }
    }
    if let (Some(name), Some((_, datatype))) = (requested_time, raw_layout) {
        out.raw_time = Some(RawPointTime {
            field: name.to_owned(),
            datatype,
            values: if shared_time { times } else { raw_values },
        });
    }
    Ok(out)
}

/// An Imu message decoded.
#[derive(Debug, Clone, Copy)]
pub struct ImuMessage {
    pub stamp: f64,
    /// World up in the IMU's frame from its orientation, when it gives one.
    pub up: Option<[f64; 3]>,
    pub acceleration: [f64; 3],
}

/// ROS measurement availability, distinct from a zero (unknown) covariance.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MeasurementStatus {
    Missing,
    Unknown,
    Known,
    Invalid,
}

impl MeasurementStatus {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Missing => "missing",
            Self::Unknown => "unknown",
            Self::Known => "known",
            Self::Invalid => "invalid",
        }
    }

    pub fn is_usable(self) -> bool {
        matches!(self, Self::Known | Self::Unknown)
    }
}

/// Validates without modifying the original covariance or measurement.
fn measurement_status(values: &[f64], covariance: &[f64; 9]) -> MeasurementStatus {
    use MeasurementStatus::*;
    if covariance[0] == -1.0 {
        return Missing;
    }
    if !values.iter().chain(covariance).all(|x| x.is_finite()) {
        return Invalid;
    }
    let scale = covariance.iter().fold(0.0_f64, |s, x| s.max(x.abs()));
    if scale == 0.0 {
        return Unknown;
    }
    let c = covariance.map(|x| x / scale);
    let tolerance = 1e-12;
    if [0, 4, 8].iter().any(|&i| c[i] < 0.0)
        || [(1, 3), (2, 6), (5, 7)]
            .iter()
            .any(|&(i, j)| (c[i] - c[j]).abs() > tolerance)
        || [(0, 4, 1), (0, 8, 2), (4, 8, 5)]
            .iter()
            .any(|&(i, j, k)| c[i] * c[j] - c[k] * c[k] < -tolerance)
    {
        return Invalid;
    }
    let determinant = c[0] * (c[4] * c[8] - c[5] * c[7]) - c[1] * (c[3] * c[8] - c[5] * c[6])
        + c[2] * (c[3] * c[7] - c[4] * c[6]);
    if determinant < -tolerance {
        Invalid
    } else {
        Known
    }
}

/// An unmodified sensor_msgs/Imu, including exact header time and validity metadata.
#[derive(Debug, Clone)]
pub struct RawImuMessage {
    pub stamp_ns: i64,
    pub stamp: f64,
    pub frame_id: String,
    /// Quaternion x, y, z, w; no normalization is applied.
    pub orientation: [f64; 4],
    pub orientation_covariance: [f64; 9],
    pub angular_velocity: [f64; 3],
    pub angular_velocity_covariance: [f64; 9],
    pub acceleration: [f64; 3],
    pub acceleration_covariance: [f64; 9],
}

impl RawImuMessage {
    pub fn orientation_status(&self) -> MeasurementStatus {
        let status = measurement_status(&self.orientation, &self.orientation_covariance);
        if status.is_usable()
            && (self.orientation.iter().map(|x| x * x).sum::<f64>() - 1.0).abs() > 1e-3
        {
            MeasurementStatus::Invalid
        } else {
            status
        }
    }

    pub fn angular_velocity_status(&self) -> MeasurementStatus {
        measurement_status(&self.angular_velocity, &self.angular_velocity_covariance)
    }

    pub fn acceleration_status(&self) -> MeasurementStatus {
        measurement_status(&self.acceleration, &self.acceleration_covariance)
    }
}

pub fn decode_imu_raw(data: &[u8], encoding: Encoding) -> Result<RawImuMessage> {
    let mut m = Reader::new(data, encoding);
    let (sec, nsec, frame_id) = m.header_parts()?;
    Ok(RawImuMessage {
        stamp_ns: sec * 1_000_000_000 + i64::from(nsec),
        stamp: sec as f64 + f64::from(nsec) * 1e-9,
        frame_id,
        orientation: m.f64s()?,
        orientation_covariance: m.f64s()?,
        angular_velocity: m.f64s()?,
        angular_velocity_covariance: m.f64s()?,
        acceleration: m.f64s()?,
        acceleration_covariance: m.f64s()?,
    })
}

pub fn decode_imu(data: &[u8], encoding: Encoding) -> Result<ImuMessage> {
    let raw = decode_imu_raw(data, encoding)?;
    let [x, y, z, w] = raw.orientation;
    // Preserve the established gravity-direction API's orientation predicate.
    let usable =
        raw.orientation_covariance[0] != -1.0 && (x * x + y * y + z * z + w * w - 1.0).abs() < 0.1;
    // The last row of the orientation's matrix: world z seen in the IMU's frame.
    let up = usable.then_some([
        2.0 * (x * z - w * y),
        2.0 * (y * z + w * x),
        1.0 - 2.0 * (x * x + y * y),
    ]);
    Ok(ImuMessage {
        stamp: raw.stamp,
        up,
        acceleration: raw.acceleration,
    })
}

/// A pose from a trajectory topic.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PoseMessage {
    pub stamp: f64,
    pub position: [f64; 3],
    /// x, y, z, w.
    pub orientation: [f64; 4],
}

pub const POSE_STAMPED: &str = "geometry_msgs/PoseStamped";
pub const ODOMETRY: &str = "nav_msgs/Odometry";
pub const TF: &str = "tf2_msgs/TFMessage";

fn pose_of(m: &mut Reader) -> Result<([f64; 3], [f64; 4])> {
    let position = [m.f64()?, m.f64()?, m.f64()?];
    let orientation = [m.f64()?, m.f64()?, m.f64()?, m.f64()?];
    Ok((position, orientation))
}

/// A geometry_msgs/PoseStamped.
pub fn decode_pose_stamped(data: &[u8], encoding: Encoding) -> Result<PoseMessage> {
    let mut m = Reader::new(data, encoding);
    let stamp = m.header()?;
    let (position, orientation) = pose_of(&mut m)?;
    Ok(PoseMessage {
        stamp,
        position,
        orientation,
    })
}

/// A nav_msgs/Odometry: its pose.
pub fn decode_odometry(data: &[u8], encoding: Encoding) -> Result<PoseMessage> {
    let mut m = Reader::new(data, encoding);
    let stamp = m.header()?;
    m.string()?;
    let (position, orientation) = pose_of(&mut m)?;
    Ok(PoseMessage {
        stamp,
        position,
        orientation,
    })
}

/// A tf2_msgs/TFMessage: the transforms whose child frame is `frame`, as poses.
pub fn decode_tf(data: &[u8], encoding: Encoding, frame: &str) -> Result<Vec<PoseMessage>> {
    let mut m = Reader::new(data, encoding);
    let mut out = Vec::new();
    for _ in 0..m.u32()? {
        let stamp = m.header()?;
        let child = m.string()?;
        let (position, orientation) = pose_of(&mut m)?;
        if child == frame {
            out.push(PoseMessage {
                stamp,
                position,
                orientation,
            });
        }
    }
    Ok(out)
}

/// Per time in `times`, the up direction (unit, in the IMU's frame) from the
/// IMU messages: its orientation when the nearest message within `window`
/// seconds has one, else the mean acceleration within `window` either side
/// (at rest an accelerometer reads straight up); None where there is none.
pub fn ups_at(imu: &[ImuMessage], times: &[f64], window: f64) -> Vec<Option<[f64; 3]>> {
    let mut sorted: Vec<ImuMessage> = imu.to_vec();
    sorted.sort_by(|a, b| a.stamp.total_cmp(&b.stamp));
    let stamps: Vec<f64> = sorted.iter().map(|m| m.stamp).collect();
    let lower_bound = |t: f64| stamps.partition_point(|&s| s < t);
    times
        .iter()
        .map(|&t| {
            if sorted.is_empty() {
                return None;
            }
            let mut k = lower_bound(t).min(sorted.len() - 1);
            if k > 0 && (stamps[k - 1] - t).abs() < (stamps[k] - t).abs() {
                k -= 1;
            }
            if (stamps[k] - t).abs() > window {
                return None;
            }
            let up = sorted[k].up.unwrap_or_else(|| {
                let mut sum = [0.0; 3];
                let mut j = lower_bound(t - window);
                while j < sorted.len() && stamps[j] <= t + window {
                    for (s, a) in sum.iter_mut().zip(sorted[j].acceleration) {
                        *s += a;
                    }
                    j += 1;
                }
                sum
            });
            let norm = (up[0] * up[0] + up[1] * up[1] + up[2] * up[2]).sqrt();
            (norm.is_finite() && norm > 1e-6).then(|| up.map(|v| v / norm))
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    /// Serialises ROS 2 messages (CDR, little-endian) the way a bag holds them.
    struct Cdr(Vec<u8>);

    impl Cdr {
        fn new() -> Self {
            Cdr(vec![0, 1, 0, 0])
        }

        fn align(&mut self, n: usize) {
            while !(self.0.len() - 4).is_multiple_of(n) {
                self.0.push(0);
            }
        }

        fn u8(mut self, v: u8) -> Self {
            self.0.push(v);
            self
        }

        fn u32(mut self, v: u32) -> Self {
            self.align(4);
            self.0.extend(v.to_le_bytes());
            self
        }

        fn f64s(mut self, values: &[f64]) -> Self {
            for v in values {
                self.align(8);
                self.0.extend(v.to_le_bytes());
            }
            self
        }

        fn string(mut self, s: &str) -> Self {
            self = self.u32(s.len() as u32 + 1);
            self.0.extend(s.bytes());
            self.0.push(0);
            self
        }

        fn raw(mut self, b: &[u8]) -> Self {
            self = self.u32(b.len() as u32);
            self.0.extend(b);
            self
        }

        fn header(self, time: f64) -> Self {
            self.u32(time as u32)
                .u32(((time % 1.0) * 1e9).round() as u32)
                .string("lidar")
        }
    }

    fn point_cloud(points: &[[f32; 4]], time: f64) -> Vec<u8> {
        let mut data = Vec::new();
        for p in points {
            for v in p {
                data.extend(v.to_le_bytes());
            }
        }
        let mut m = Cdr::new()
            .header(time)
            .u32(1)
            .u32(points.len() as u32)
            .u32(4);
        for (k, name) in ["x", "y", "z", "intensity"].iter().enumerate() {
            m = m.string(name).u32(4 * k as u32).u8(7).u32(1);
        }
        m.u8(0)
            .u32(16)
            .u32(16 * points.len() as u32)
            .raw(&data)
            .u8(1)
            .0
    }

    fn imu(yaw: f64, time: f64) -> Vec<u8> {
        Cdr::new()
            .header(time)
            .f64s(&[0.0, 0.0, (yaw / 2.0).sin(), (yaw / 2.0).cos()])
            .f64s(&[0.0; 9 + 3 + 9])
            .f64s(&[0.0, 0.0, 9.8])
            .f64s(&[0.0; 9])
            .0
    }

    #[test]
    fn raw_imu_retains_exact_time_and_all_fields() {
        let data = Cdr::new()
            .u32(1_700_000_000)
            .u32(123)
            .string("imu/data")
            .f64s(&[0.0, 0.0, 0.0, 1.0])
            .f64s(&[0.0; 9])
            .f64s(&[0.1, -0.2, 0.3])
            .f64s(&[0.01, 0.0, 0.0, 0.0, 0.02, 0.0, 0.0, 0.0, 0.03])
            .f64s(&[1.0, 2.0, 9.81])
            .f64s(&[-1.0; 9])
            .0;
        let raw = decode_imu_raw(&data, Encoding::Cdr).unwrap();
        assert_eq!(raw.stamp_ns, 1_700_000_000_000_000_123);
        assert_eq!(raw.frame_id, "imu/data");
        assert_eq!(raw.angular_velocity, [0.1, -0.2, 0.3]);
        assert_eq!(raw.angular_velocity_covariance[8], 0.03);
        assert_eq!(raw.acceleration, [1.0, 2.0, 9.81]);
        assert_eq!(raw.orientation_status(), MeasurementStatus::Unknown);
        assert_eq!(raw.angular_velocity_status(), MeasurementStatus::Known);
        assert_eq!(raw.acceleration_status(), MeasurementStatus::Missing);
        assert!(decode_imu_raw(&data[..data.len() - 1], Encoding::Cdr).is_err());
        assert_eq!(
            decode_imu(&data, Encoding::Cdr).unwrap().up,
            Some([0.0, 0.0, 1.0])
        );
    }

    #[test]
    fn raw_imu_handles_signed_ros2_seconds_and_rejects_bad_nanoseconds() {
        let mut data = imu(0.0, 0.0);
        data[4..8].copy_from_slice(&(-1_i32).to_le_bytes());
        data[8..12].copy_from_slice(&999_999_999_u32.to_le_bytes());
        let raw = decode_imu_raw(&data, Encoding::Cdr).unwrap();
        assert_eq!(raw.stamp_ns, -1);
        data[8..12].copy_from_slice(&1_000_000_000_u32.to_le_bytes());
        assert!(decode_imu_raw(&data, Encoding::Cdr).is_err());
    }

    #[test]
    fn imu_covariance_status_does_not_confuse_missing_unknown_or_invalid() {
        use MeasurementStatus::*;
        assert_eq!(measurement_status(&[1.0; 3], &[0.0; 9]), Unknown);
        assert_eq!(measurement_status(&[f64::NAN; 3], &[-1.0; 9]), Missing);
        assert_eq!(measurement_status(&[f64::NAN; 3], &[0.0; 9]), Invalid);
        for covariance in [
            [-2.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 2.0, 0.0, 2.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, 0.5, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0],
            [1.0, -0.9, -0.9, -0.9, 1.0, -0.9, -0.9, -0.9, 1.0],
        ] {
            assert_eq!(measurement_status(&[1.0; 3], &covariance), Invalid);
        }
        assert_eq!(measurement_status(&[1.0; 3], &[1.0; 9]), Known);
        let mut raw = decode_imu_raw(&imu(0.0, 1.0), Encoding::Cdr).unwrap();
        raw.orientation = [0.0; 4];
        assert_eq!(raw.orientation_status(), Invalid);
        raw.orientation_covariance[0] = -1.0;
        assert_eq!(raw.orientation_status(), Missing);
    }

    /// An MCAP (unchunked, without a summary) of `messages` on channels 0 (points) and 1 (imu).
    fn mcap(messages: &[(u16, f64, Vec<u8>)]) -> Vec<u8> {
        let record = |op: u8, body: &[u8]| {
            let mut r = vec![op];
            r.extend((body.len() as u64).to_le_bytes());
            r.extend(body);
            r
        };
        let string = |s: &str| {
            let mut b = (s.len() as u32).to_le_bytes().to_vec();
            b.extend(s.bytes());
            b
        };
        let magic = b"\x89MCAP0\r\n";
        let mut out = magic.to_vec();
        out.extend(record(0x01, &[string("ros2"), string("test")].concat()));
        for (id, (topic, kind)) in [
            ("/points", "sensor_msgs/msg/PointCloud2"),
            ("/imu", "sensor_msgs/msg/Imu"),
        ]
        .iter()
        .enumerate()
        {
            let id = id as u16;
            out.extend(record(
                0x03,
                &[
                    (id + 1).to_le_bytes().to_vec(),
                    string(kind),
                    string("ros2msg"),
                    0u32.to_le_bytes().to_vec(),
                ]
                .concat(),
            ));
            out.extend(record(
                0x04,
                &[
                    id.to_le_bytes().to_vec(),
                    (id + 1).to_le_bytes().to_vec(),
                    string(topic),
                    string("cdr"),
                    0u32.to_le_bytes().to_vec(),
                ]
                .concat(),
            ));
        }
        for (k, (channel, time, data)) in messages.iter().enumerate() {
            let ns = (time * 1e9).round() as u64;
            out.extend(record(
                0x05,
                &[
                    channel.to_le_bytes().to_vec(),
                    (k as u32).to_le_bytes().to_vec(),
                    ns.to_le_bytes().to_vec(),
                    ns.to_le_bytes().to_vec(),
                    data.clone(),
                ]
                .concat(),
            ));
        }
        out.extend(record(0x0f, &0u32.to_le_bytes()));
        out.extend(record(0x02, &[0u8; 20]));
        out.extend(magic);
        out
    }

    /// A ROS 1 bag (format 2.0) with one chunk of `messages` on connection 0 (points) and 1 (imu).
    fn ros1_bag(messages: &[(u32, f64, Vec<u8>)], compression: &str) -> Vec<u8> {
        let field = |name: &str, value: &[u8]| {
            let mut f = ((name.len() + 1 + value.len()) as u32)
                .to_le_bytes()
                .to_vec();
            f.extend(name.bytes());
            f.push(b'=');
            f.extend(value);
            f
        };
        let time = |t: f64| {
            [
                (t as u32).to_le_bytes(),
                (((t % 1.0) * 1e9).round() as u32).to_le_bytes(),
            ]
            .concat()
        };
        let record = |header: &[Vec<u8>], data: &[u8]| {
            let header = header.concat();
            let mut r = (header.len() as u32).to_le_bytes().to_vec();
            r.extend(&header);
            r.extend((data.len() as u32).to_le_bytes());
            r.extend(data);
            r
        };
        // The ROS 1 messages: CDR data works for the test's decoders too, so the
        // chunk holds CDR bytes tagged as ROS 1; the reader does not look inside.
        let mut chunk_records = Vec::new();
        let mut counts: HashMap<u32, u32> = HashMap::new();
        for (conn, t, data) in messages {
            chunk_records.extend(record(
                &[
                    field("op", &[2]),
                    field("conn", &conn.to_le_bytes()),
                    field("time", &time(*t)),
                ],
                data,
            ));
            *counts.entry(*conn).or_default() += 1;
        }
        let compressed = match compression {
            "lz4" => {
                let mut enc = lz4_flex::frame::FrameEncoder::new(Vec::new());
                std::io::Write::write_all(&mut enc, &chunk_records).unwrap();
                enc.finish().unwrap()
            }
            "bz2" => {
                let mut enc =
                    bzip2::write::BzEncoder::new(Vec::new(), bzip2::Compression::default());
                std::io::Write::write_all(&mut enc, &chunk_records).unwrap();
                enc.finish().unwrap()
            }
            _ => chunk_records.clone(),
        };
        let mut out = b"#ROSBAG V2.0\n".to_vec();
        let header_at = out.len();
        // The bag header is padded to a fixed size so that index_pos can be filled in.
        let header_record = |index_pos: u64| {
            let fields = [
                field("op", &[3]),
                field("index_pos", &index_pos.to_le_bytes()),
                field("conn_count", &2u32.to_le_bytes()),
                field("chunk_count", &1u32.to_le_bytes()),
            ]
            .concat();
            let mut r = (fields.len() as u32).to_le_bytes().to_vec();
            r.extend(&fields);
            r.extend(0u32.to_le_bytes());
            r
        };
        out.extend(header_record(0));
        let chunk_at = out.len() as u64;
        out.extend(record(
            &[
                field("op", &[5]),
                field("compression", compression.as_bytes()),
                field("size", &(chunk_records.len() as u32).to_le_bytes()),
            ],
            &compressed,
        ));
        let index_at = out.len() as u64;
        for (conn, (topic, kind)) in [
            ("/points", "sensor_msgs/PointCloud2"),
            ("/imu", "sensor_msgs/Imu"),
        ]
        .iter()
        .enumerate()
        {
            let info = [
                field("topic", topic.as_bytes()),
                field("type", kind.as_bytes()),
                field("md5sum", b"0"),
                field("message_definition", b""),
            ]
            .concat();
            out.extend(record(
                &[
                    field("op", &[7]),
                    field("conn", &(conn as u32).to_le_bytes()),
                    field("topic", topic.as_bytes()),
                ],
                &info,
            ));
        }
        let mut count_data = Vec::new();
        for (conn, n) in &counts {
            count_data.extend(conn.to_le_bytes());
            count_data.extend(n.to_le_bytes());
        }
        out.extend(record(
            &[
                field("op", &[6]),
                field("ver", &1u32.to_le_bytes()),
                field("chunk_pos", &chunk_at.to_le_bytes()),
                field("start_time", &time(messages[0].1)),
                field("end_time", &time(messages.last().unwrap().1)),
                field("count", &(counts.len() as u32).to_le_bytes()),
            ],
            &count_data,
        ));
        let header = header_record(index_at);
        out[header_at..header_at + header.len()].copy_from_slice(&header);
        out
    }

    fn scans() -> Vec<(u16, f64, Vec<u8>)> {
        (0..3)
            .flat_map(|k| {
                let t = 100.0 + 0.1 * k as f64;
                let points: Vec<[f32; 4]> =
                    (0..5).map(|i| [i as f32, k as f32, 1.0, 0.5]).collect();
                [
                    (1, t - 0.01, imu(0.1 * k as f64, t - 0.01)),
                    (0, t, point_cloud(&points, t)),
                ]
            })
            .collect()
    }

    fn check<R: Read + Seek>(mut bag: Bag<R>) {
        let names: Vec<(String, String, u64)> = bag
            .topics()
            .iter()
            .map(|t| (t.name.clone(), t.kind.clone(), t.count))
            .collect();
        assert_eq!(
            names,
            [
                ("/imu".into(), IMU.into(), 3),
                ("/points".into(), POINT_CLOUD.into(), 3)
            ]
        );
        let messages: Vec<Message> = bag
            .messages(&["/points", "/imu"])
            .map(|m| m.unwrap())
            .collect();
        assert_eq!(messages.len(), 6);
        let mut clouds = Vec::new();
        let mut imus = Vec::new();
        for m in &messages {
            match m.topic.as_str() {
                "/points" => clouds.push(decode_point_cloud2(&m.data, m.encoding).unwrap()),
                _ => imus.push(decode_imu(&m.data, m.encoding).unwrap()),
            }
        }
        assert_eq!(clouds.len(), 3);
        assert_eq!(clouds[2].positions[4], [4.0, 2.0, 1.0]);
        assert_eq!(clouds[2].intensity.as_deref(), Some(&[0.5f32; 5][..]));
        assert!(clouds[2].time.is_none());
        assert!((clouds[1].stamp - 100.1).abs() < 1e-6);
        let ups = ups_at(&imus, &[100.2], 0.5);
        let up = ups[0].unwrap();
        assert!((up[2] - 1.0).abs() < 1e-9 && up[0].abs() < 1e-9);
        // Only one topic asked for.
        assert_eq!(bag.messages(&["/imu"]).count(), 3);
    }

    #[test]
    fn an_mcap_is_read() {
        check(Bag::from_reader(Cursor::new(mcap(&scans()))).unwrap());
    }

    #[test]
    fn a_ros1_bag_is_read_compressed_or_not() {
        let messages: Vec<(u32, f64, Vec<u8>)> = scans()
            .into_iter()
            .map(|(c, t, d)| (u32::from(c), t, d))
            .collect();
        for compression in ["none", "lz4", "bz2"] {
            let bytes = ros1_bag(&messages, compression);
            let mut bag = Bag::from_reader(Cursor::new(bytes)).unwrap();
            assert!(bag.is_ros1());
            // The messages are CDR bytes tagged as ROS 1 by the bag: decode them as CDR.
            let found: Vec<PointsMessage> = bag
                .messages(&["/points"])
                .map(|m| decode_point_cloud2(&m.unwrap().data, Encoding::Cdr).unwrap())
                .collect();
            assert_eq!(found.len(), 3, "{compression}");
            assert_eq!(found[1].positions[0], [0.0, 1.0, 1.0]);
        }
    }

    #[test]
    fn equal_recording_times_preserve_physical_order_and_topic_filtering() {
        // Physical order differs from timestamp order. The four records at
        // time 1 must keep their order, including across requested topics.
        let messages = vec![
            (1, 2.0, vec![4]),
            (0, 1.0, vec![90]),
            (1, 1.0, vec![0]),
            (0, 1.0, vec![91]),
            (1, 1.0, vec![1]),
            (1, 3.0, vec![5]),
        ];
        for compression in ["none", "lz4", "bz2"] {
            let mut bag = Bag::from_reader(Cursor::new(ros1_bag(&messages, compression))).unwrap();
            for (topics, expected) in [
                (vec!["/points", "/imu"], vec![90, 0, 91, 1, 4, 5]),
                (vec!["/imu"], vec![0, 1, 4, 5]),
                (vec!["/points"], vec![90, 91]),
            ] {
                let found: Vec<_> = bag.messages(&topics).map(|m| m.unwrap()).collect();
                assert!(found.windows(2).all(|w| w[0].time <= w[1].time));
                assert_eq!(
                    found.iter().map(|m| m.data[0]).collect::<Vec<_>>(),
                    expected,
                    "{compression}, {topics:?}"
                );
            }
        }
    }

    #[test]
    fn a_ros1_point_cloud_decodes_with_a_time_field() {
        // ROS 1 layout: seq, stamp, frame_id (no NUL), no alignment.
        let mut b = Vec::new();
        b.extend(7u32.to_le_bytes());
        b.extend(12u32.to_le_bytes());
        b.extend(500_000_000u32.to_le_bytes());
        b.extend(3u32.to_le_bytes());
        b.extend(b"os1");
        b.extend(1u32.to_le_bytes());
        b.extend(3u32.to_le_bytes());
        b.extend(4u32.to_le_bytes());
        for (k, name) in ["x", "y", "z", "t"].iter().enumerate() {
            b.extend((name.len() as u32).to_le_bytes());
            b.extend(name.bytes());
            b.extend((4 * k as u32).to_le_bytes());
            b.push(if *name == "t" { 6 } else { 7 });
            b.extend(1u32.to_le_bytes());
        }
        b.push(0);
        b.extend(16u32.to_le_bytes());
        b.extend(48u32.to_le_bytes());
        b.extend(48u32.to_le_bytes());
        for (p, t) in [
            ([1.0f32, 2.0, 3.0], 0u32),
            ([f32::NAN, 0.0, 0.0], 50),
            ([4.0, 5.0, 6.0], 100),
        ] {
            for v in p {
                b.extend(v.to_le_bytes());
            }
            b.extend(t.to_le_bytes());
        }
        b.push(0);
        let cloud = decode_point_cloud2(&b, Encoding::Ros1).unwrap();
        assert_eq!(cloud.stamp_ns, 12_500_000_000);
        assert_eq!(cloud.frame_id, "os1");
        assert!((cloud.stamp - 12.5).abs() < 1e-9);
        assert_eq!(cloud.positions, [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        assert_eq!(cloud.time.as_deref(), Some(&[0.0f32, 1.0][..]));
        assert!(cloud.intensity.is_none());
        assert!(cloud.raw_time.is_none());
        let retained = decode_point_cloud2_with_raw_time(&b, Encoding::Ros1, "t").unwrap();
        assert_eq!(retained.positions, cloud.positions);
        assert_eq!(retained.time, cloud.time);
        let raw = retained.raw_time.unwrap();
        assert_eq!(raw.field, "t");
        assert_eq!(raw.datatype, 6);
        assert_eq!(raw.values, [0.0, 100.0]);
    }

    #[test]
    fn explicit_point_time_retains_custom_nonfinite_constant_and_empty_values() {
        let values = [0.5f32, -0.0, f32::NAN, f32::INFINITY];
        let points: Vec<_> = values.iter().map(|&v| [1.0, 2.0, 3.0, v]).collect();
        let data = point_cloud(&points, 0.0);
        let cloud = decode_point_cloud2_with_raw_time(&data, Encoding::Cdr, "intensity").unwrap();
        assert!(cloud.time.is_none());
        let raw = cloud.raw_time.unwrap();
        assert_eq!(raw.field, "intensity");
        assert_eq!(raw.datatype, 7);
        assert_eq!(
            raw.values.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
            values.map(|v| f64::from(v).to_bits())
        );
        for points in [
            vec![[1.0, 2.0, 3.0, 7.5], [4.0, 5.0, 6.0, 7.5]],
            vec![[f32::NAN, 2.0, 3.0, 7.5]],
        ] {
            let data = point_cloud(&points, 0.0);
            let cloud =
                decode_point_cloud2_with_raw_time(&data, Encoding::Cdr, "intensity").unwrap();
            assert!(cloud.time.is_none());
            assert_eq!(
                cloud.raw_time.unwrap().values,
                vec![7.5; cloud.positions.len()]
            );
        }
    }

    #[test]
    fn explicit_point_time_rejects_ambiguous_or_incomplete_layouts() {
        let make = |datatype, count, offset, row_step, bytes, duplicate| {
            let mut m = Cdr::new()
                .header(0.0)
                .u32(1)
                .u32(1)
                .u32(4 + u32::from(duplicate));
            for (k, name) in ["x", "y", "z"].iter().enumerate() {
                m = m.string(name).u32(k as u32 * 4).u8(7).u32(1);
            }
            m = m.string("t").u32(offset).u8(datatype).u32(count);
            if duplicate {
                m = m.string("t").u32(offset).u8(datatype).u32(count);
            }
            m.u8(0).u32(16).u32(row_step).raw(&vec![0; bytes]).u8(1).0
        };
        for (datatype, count, offset, row_step, bytes, duplicate) in [
            (0, 1, 12, 16, 16, false),
            (9, 1, 12, 16, 16, false),
            (6, 0, 12, 16, 16, false),
            (6, 2, 12, 16, 16, false),
            (6, 1, 13, 16, 16, false),
            (6, 1, u32::MAX, 16, 16, false),
            (6, 1, 12, 15, 16, false),
            (6, 1, 12, 16, 15, false),
            (6, 1, 12, 16, 16, true),
        ] {
            let data = make(datatype, count, offset, row_step, bytes, duplicate);
            assert!(decode_point_cloud2_with_raw_time(&data, Encoding::Cdr, "t").is_err());
        }
        let data = make(6, 1, 12, 16, 16, false);
        assert!(decode_point_cloud2_with_raw_time(&data, Encoding::Cdr, "absent").is_err());
        assert!(field_value(&[0; 4], usize::MAX, 6, false).is_none());
    }

    #[test]
    fn point_cloud_header_retains_integer_nanoseconds_and_signed_cdr_seconds() {
        for (sec, nsec) in [
            (i32::MIN, 0u32),
            (-1, 999_999_999),
            (0, 0),
            (1_700_000_000, 123),
            (i32::MAX, 999_999_999),
        ] {
            let mut data = point_cloud(&[[1.0, 2.0, 3.0, 0.5]], 0.0);
            data[4..8].copy_from_slice(&sec.to_le_bytes());
            data[8..12].copy_from_slice(&nsec.to_le_bytes());
            let cloud = decode_point_cloud2(&data, Encoding::Cdr).unwrap();
            assert_eq!(
                cloud.stamp_ns,
                i64::from(sec) * 1_000_000_000 + i64::from(nsec)
            );
            assert_eq!(cloud.stamp, f64::from(sec) + f64::from(nsec) * 1e-9);
            assert_eq!(cloud.frame_id, "lidar");
            assert_eq!(cloud.positions, [[1.0, 2.0, 3.0]]);
        }
        let mut data = point_cloud(&[], 0.0);
        data[8..12].copy_from_slice(&1_000_000_000u32.to_le_bytes());
        assert!(
            decode_point_cloud2(&data, Encoding::Cdr)
                .unwrap_err()
                .0
                .contains("nanoseconds")
        );
    }
}
