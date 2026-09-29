/**
 * ROS bags in the browser: ROS 1 (`.bag`, format 2.0) and MCAP (ROS 2's
 * default), read a slice at a time from a File so that bags of gigabytes
 * open, with just enough decoding for sensor_msgs PointCloud2 and Imu.
 */
import { decompress as zstd } from "fzstd";

export type Encoding = "ros1" | "cdr";

export interface BagTopic {
  topic: string;
  /** As ROS 1 names it (`sensor_msgs/PointCloud2`), for either kind of bag. */
  type: string;
  count: number;
}

export interface BagMessage {
  topic: string;
  /** When it was recorded (seconds). */
  time: number;
  data: Uint8Array;
  encoding: Encoding;
}

export interface Bag {
  format: "ros1" | "mcap";
  topics: BagTopic[];
  /** The messages on `topics` in recording order (chunk by chunk), reporting the share of the bag read. */
  messages(topics: Set<string>, progress?: (fraction: number) => void): AsyncGenerator<BagMessage>;
}

export const POINT_CLOUD = "sensor_msgs/PointCloud2";
export const IMU = "sensor_msgs/Imu";

export const isBag = (name: string) => /\.(bag|mcap)$/i.test(name);

const text = new TextDecoder();

/** Reads a Blob in windows, so that many small records cost few reads. */
class Source {
  private start = 0;
  private window = new Uint8Array(0);
  private readonly blob: Blob;
  private readonly span: number;

  constructor(blob: Blob, span = 8 << 20) {
    this.blob = blob;
    this.span = span;
  }

  get size(): number {
    return this.blob.size;
  }

  async bytes(at: number, length: number): Promise<Uint8Array> {
    if (at < this.start || at + length > this.start + this.window.length) {
      const end = Math.min(this.blob.size, at + Math.max(length, this.span));
      this.window = new Uint8Array(await this.blob.slice(at, end).arrayBuffer());
      this.start = at;
      if (this.window.length < length) throw new Error("the bag ends early (is it cut short?)");
    }
    return this.window.subarray(at - this.start, at - this.start + length);
  }
}

const view = (b: Uint8Array) => new DataView(b.buffer, b.byteOffset, b.byteLength);

// --- Decompression -----------------------------------------------------------

/** An LZ4 frame (as ROS 1's roslz4 and MCAP write them) of `size` bytes once expanded. */
export function lz4Frame(src: Uint8Array, size: number): Uint8Array {
  const v = view(src);
  if (v.getUint32(0, true) !== 0x184d2204) throw new Error("not an LZ4 frame");
  const flags = src[4];
  let o = 6 + (flags & 0x08 ? 8 : 0) + (flags & 0x01 ? 4 : 0) + 1;
  const blockChecksum = (flags & 0x10) !== 0;
  const out = new Uint8Array(size);
  let n = 0;
  for (;;) {
    const word = v.getUint32(o, true);
    o += 4;
    if (word === 0) break;
    const length = word & 0x7fffffff;
    if (word & 0x80000000) {
      out.set(src.subarray(o, o + length), n);
      n += length;
    } else {
      n = lz4Block(src, o, o + length, out, n);
    }
    o += length + (blockChecksum ? 4 : 0);
  }
  return n === size ? out : out.subarray(0, n);
}

/** Decode one LZ4 block of `src[start..end)` into `out` at `n` (earlier blocks may be referred to); the new end. */
function lz4Block(src: Uint8Array, start: number, end: number, out: Uint8Array, n: number): number {
  let o = start;
  while (o < end) {
    const token = src[o++];
    let literals = token >> 4;
    if (literals === 15) {
      let b;
      do literals += b = src[o++];
      while (b === 255);
    }
    out.set(src.subarray(o, o + literals), n);
    n += literals;
    o += literals;
    if (o >= end) break;
    const offset = src[o] | (src[o + 1] << 8);
    o += 2;
    let length = token & 15;
    if (length === 15) {
      let b;
      do length += b = src[o++];
      while (b === 255);
    }
    length += 4;
    // Byte by byte: a match may overlap what it copies.
    for (let k = n - offset, stop = n + length; n < stop; ) out[n++] = out[k++];
  }
  return n;
}

/** Decompressors by a bag's name for them; `bz2` is added by whoever has one (the Rust core's). */
export const decompressors: Record<string, (data: Uint8Array, size: number) => Uint8Array> = {
  lz4: lz4Frame,
  zstd: (data, size) => zstd(data, new Uint8Array(size)),
};

function expand(compression: string, data: Uint8Array, size: number): Uint8Array {
  if (compression === "" || compression === "none") return data;
  const decompress = decompressors[compression];
  if (!decompress) {
    throw new Error(`${compression}-compressed bags are not supported: decompress it first (e.g. rosbag decompress)`);
  }
  return decompress(data, size);
}

// --- ROS 1 -------------------------------------------------------------------

/** A ROS 1 record header's `name=value` fields. */
function fields(bytes: Uint8Array): Map<string, Uint8Array> {
  const v = view(bytes);
  const out = new Map<string, Uint8Array>();
  for (let o = 0; o + 4 <= bytes.length; ) {
    const length = v.getUint32(o, true);
    const field = bytes.subarray(o + 4, o + 4 + length);
    const eq = field.indexOf(61);
    out.set(text.decode(field.subarray(0, eq)), field.subarray(eq + 1));
    o += 4 + length;
  }
  return out;
}

interface Ros1Record {
  header: Map<string, Uint8Array>;
  data: Uint8Array;
  /** Where the next record starts. */
  next: number;
}

/** The record at `o` of an in-memory buffer. */
function ros1Record(bytes: Uint8Array, o: number): Ros1Record {
  const v = view(bytes);
  const headerLength = v.getUint32(o, true);
  const header = fields(bytes.subarray(o + 4, o + 4 + headerLength));
  const at = o + 4 + headerLength;
  const dataLength = v.getUint32(at, true);
  return { header, data: bytes.subarray(at + 4, at + 4 + dataLength), next: at + 4 + dataLength };
}

/** The record at `o` of a bag, read in two steps (its header says how long its data is). */
async function ros1RecordAt(source: Source, o: number): Promise<Ros1Record> {
  const headerLength = view(await source.bytes(o, 4)).getUint32(0, true);
  const head = await source.bytes(o, 4 + headerLength + 4);
  const header = fields(head.subarray(4, 4 + headerLength));
  const dataLength = view(head).getUint32(4 + headerLength, true);
  const data = await source.bytes(o + 8 + headerLength, dataLength);
  return { header, data, next: o + 8 + headerLength + dataLength };
}

const u32 = (b: Uint8Array | undefined) => (b ? view(b).getUint32(0, true) : 0);
const u64 = (b: Uint8Array | undefined) => (b ? Number(view(b).getBigUint64(0, true)) : 0);
const ros1Time = (b: Uint8Array | undefined) => (b ? view(b).getUint32(0, true) + view(b).getUint32(4, true) * 1e-9 : 0);
const op = (r: Ros1Record) => r.header.get("op")?.[0];

async function openRos1(source: Source): Promise<Bag> {
  const bagHeader = await ros1RecordAt(source, 13);
  const indexAt = u64(bagHeader.header.get("index_pos"));
  if (op(bagHeader) !== 0x03 || indexAt === 0) {
    throw new Error("the bag has no index (was recording cut short?): run rosbag reindex on it first");
  }
  const connections = new Map<number, { topic: string; type: string }>();
  const chunks: { at: number; start: number; counts: Map<number, number> }[] = [];
  for (let o = indexAt; o < source.size; ) {
    const r = await ros1RecordAt(source, o);
    o = r.next;
    if (op(r) === 0x07) {
      const info = fields(r.data);
      connections.set(u32(r.header.get("conn")), {
        topic: text.decode(r.header.get("topic")),
        type: text.decode(info.get("type")),
      });
    } else if (op(r) === 0x06) {
      const counts = new Map<number, number>();
      const v = view(r.data);
      for (let k = 0; k + 8 <= r.data.length; k += 8) counts.set(v.getUint32(k, true), v.getUint32(k + 4, true));
      chunks.push({ at: u64(r.header.get("chunk_pos")), start: ros1Time(r.header.get("start_time")), counts });
    }
  }
  chunks.sort((a, b) => a.start - b.start || a.at - b.at);
  const totals = new Map<string, BagTopic>();
  for (const chunk of chunks) {
    for (const [conn, count] of chunk.counts) {
      const c = connections.get(conn);
      if (!c) continue;
      const t = totals.get(c.topic) ?? { topic: c.topic, type: c.type, count: 0 };
      t.count += count;
      totals.set(c.topic, t);
    }
  }
  return {
    format: "ros1",
    topics: [...totals.values()],
    async *messages(topics, progress) {
      const wanted = new Set([...connections].filter(([, c]) => topics.has(c.topic)).map(([conn]) => conn));
      for (const [k, chunk] of chunks.entries()) {
        progress?.(k / chunks.length);
        if (![...chunk.counts.keys()].some((conn) => wanted.has(conn))) continue;
        const r = await ros1RecordAt(source, chunk.at);
        const compression = text.decode(r.header.get("compression"));
        const records = expand(compression, r.data, u32(r.header.get("size")));
        const found: BagMessage[] = [];
        for (let o = 0; o < records.length; ) {
          const m = ros1Record(records, o);
          o = m.next;
          if (op(m) !== 0x02) continue;
          const conn = u32(m.header.get("conn"));
          if (!wanted.has(conn)) continue;
          found.push({ topic: connections.get(conn)!.topic, time: ros1Time(m.header.get("time")), data: m.data, encoding: "ros1" });
        }
        found.sort((a, b) => a.time - b.time);
        yield* found;
      }
      progress?.(1);
    },
  };
}

// --- MCAP --------------------------------------------------------------------

/** Reads the fields of an MCAP record's content. */
class Fields {
  o = 0;
  private readonly b: Uint8Array;
  private readonly v: DataView;

  constructor(b: Uint8Array) {
    this.b = b;
    this.v = view(b);
  }

  u16(): number {
    this.o += 2;
    return this.v.getUint16(this.o - 2, true);
  }

  u32(): number {
    this.o += 4;
    return this.v.getUint32(this.o - 4, true);
  }

  u64(): number {
    this.o += 8;
    return Number(this.v.getBigUint64(this.o - 8, true));
  }

  bytes(length: number): Uint8Array {
    this.o += length;
    return this.b.subarray(this.o - length, this.o);
  }

  string(): string {
    return text.decode(this.bytes(this.u32()));
  }

  /** A map's byte length, skipped. */
  skipMap(): void {
    this.o += this.u32();
  }

  rest(): Uint8Array {
    return this.b.subarray(this.o);
  }
}

interface McapRecord {
  opcode: number;
  body: Uint8Array;
}

/** The records of an in-memory buffer. */
function* mcapRecords(bytes: Uint8Array): Generator<McapRecord> {
  const v = view(bytes);
  for (let o = 0; o + 9 <= bytes.length; ) {
    const length = Number(v.getBigUint64(o + 1, true));
    yield { opcode: bytes[o], body: bytes.subarray(o + 9, o + 9 + length) };
    o += 9 + length;
  }
}

interface McapChannel {
  topic: string;
  type: string;
  encoding: Encoding;
}

async function openMcap(source: Source): Promise<Bag> {
  const schemas = new Map<number, string>();
  const channels = new Map<number, McapChannel>();
  const counts = new Map<number, number>();
  const chunks: { at: number; length: number; start: number; channels: Set<number> | null }[] = [];

  const learn = (r: McapRecord) => {
    const f = new Fields(r.body);
    if (r.opcode === 0x03) {
      const id = f.u16();
      schemas.set(id, f.string());
    } else if (r.opcode === 0x04) {
      const id = f.u16();
      const schema = f.u16();
      const topic = f.string();
      const encoding = f.string();
      const type = (schemas.get(schema) ?? "").replace("/msg/", "/");
      channels.set(id, { topic, type, encoding: encoding === "ros1" ? "ros1" : "cdr" });
    }
  };

  const tail = await source.bytes(source.size - 37, 37);
  const footer = new Fields(tail.subarray(9));
  const summaryAt = tail[0] === 0x02 ? footer.u64() : 0;
  if (summaryAt > 0) {
    const summary = await source.bytes(summaryAt, source.size - 37 - summaryAt);
    for (const r of mcapRecords(summary)) {
      learn(r);
      const f = new Fields(r.body);
      if (r.opcode === 0x08) {
        const start = f.u64();
        f.u64();
        const at = f.u64();
        const length = f.u64();
        const indexed = new Set<number>();
        const end = f.u32() + f.o;
        while (f.o < end) {
          indexed.add(f.u16());
          f.u64();
        }
        chunks.push({ at, length, start, channels: indexed.size ? indexed : null });
      } else if (r.opcode === 0x0b) {
        f.u64();
        f.u16();
        f.u32();
        f.u32();
        f.u32();
        f.u32();
        f.u64();
        f.u64();
        const end = f.u32() + f.o;
        while (f.o < end) {
          const id = f.u16();
          counts.set(id, f.u64());
        }
      }
    }
  }

  /** Every record after the magic, reading the file end to end; chunks' records inside them. */
  async function* linear(progress?: (fraction: number) => void): AsyncGenerator<McapRecord> {
    for (let o = 8; o + 9 <= source.size; ) {
      const head = await source.bytes(o, 9);
      const opcode = head[0];
      const length = Number(view(head).getBigUint64(1, true));
      if (opcode === 0x0f || opcode === 0x02) return;
      const body = await source.bytes(o + 9, length);
      o += 9 + length;
      progress?.(o / source.size);
      if (opcode === 0x06) yield* mcapRecords(chunkRecords(body));
      else yield { opcode, body };
    }
  }

  if (chunks.length === 0) {
    // No index: count by reading it all once.
    for await (const r of linear()) {
      learn(r);
      if (r.opcode === 0x05) {
        const id = view(r.body).getUint16(0, true);
        counts.set(id, (counts.get(id) ?? 0) + 1);
      }
    }
  }
  chunks.sort((a, b) => a.start - b.start || a.at - b.at);

  const totals = new Map<string, BagTopic>();
  for (const [id, c] of channels) {
    const t = totals.get(c.topic) ?? { topic: c.topic, type: c.type, count: 0 };
    t.count += counts.get(id) ?? 0;
    totals.set(c.topic, t);
  }

  const message = (body: Uint8Array, wanted: Set<number>): BagMessage | null => {
    const f = new Fields(body);
    const id = f.u16();
    if (!wanted.has(id)) return null;
    f.u32();
    const time = f.u64() * 1e-9;
    f.u64();
    const c = channels.get(id)!;
    return { topic: c.topic, time, data: f.rest(), encoding: c.encoding };
  };

  return {
    format: "mcap",
    topics: [...totals.values()],
    async *messages(topics, progress) {
      const wanted = new Set([...channels].filter(([, c]) => topics.has(c.topic)).map(([id]) => id));
      if (chunks.length === 0) {
        for await (const r of linear(progress)) {
          learn(r);
          const m = r.opcode === 0x05 ? message(r.body, wanted) : null;
          if (m) yield m;
        }
        return;
      }
      for (const [k, chunk] of chunks.entries()) {
        progress?.(k / chunks.length);
        if (chunk.channels && ![...chunk.channels].some((id) => wanted.has(id))) continue;
        const body = await source.bytes(chunk.at + 9, chunk.length - 9);
        const found: BagMessage[] = [];
        for (const r of mcapRecords(chunkRecords(body))) {
          const m = r.opcode === 0x05 ? message(r.body, wanted) : null;
          if (m) found.push(m);
        }
        found.sort((a, b) => a.time - b.time);
        yield* found;
      }
      progress?.(1);
    },
  };
}

/** A chunk record's content, expanded to the records it holds. */
function chunkRecords(body: Uint8Array): Uint8Array {
  const f = new Fields(body);
  f.u64();
  f.u64();
  const size = f.u64();
  f.u32();
  const compression = f.string();
  const records = f.bytes(f.u64());
  return expand(compression, records, size);
}

export async function openBag(file: Blob): Promise<Bag> {
  const source = new Source(file);
  const magic = await source.bytes(0, 13);
  if (text.decode(magic) === "#ROSBAG V2.0\n") return openRos1(source);
  if (magic[0] === 0x89 && text.decode(magic.subarray(1, 5)) === "MCAP") return openMcap(source);
  throw new Error("not a ROS 1 bag (format 2.0) or an MCAP file");
}

// --- Messages ----------------------------------------------------------------

/** Reads ROS 1 or CDR (ROS 2) serialised fields. */
class Message {
  private o: number;
  private readonly base: number;
  private readonly little: boolean;
  private readonly v: DataView;
  private readonly b: Uint8Array;
  private readonly cdr: boolean;

  constructor(b: Uint8Array, cdr: boolean) {
    this.b = b;
    this.cdr = cdr;
    this.v = view(b);
    this.base = this.o = cdr ? 4 : 0;
    this.little = cdr ? b[1] === 1 : true;
  }

  private align(n: number): void {
    if (this.cdr) this.o += (n - ((this.o - this.base) % n)) % n;
  }

  u8(): number {
    return this.b[this.o++];
  }

  u32(): number {
    this.align(4);
    this.o += 4;
    return this.v.getUint32(this.o - 4, this.little);
  }

  f64(): number {
    this.align(8);
    this.o += 8;
    return this.v.getFloat64(this.o - 8, this.little);
  }

  bytes(length: number): Uint8Array {
    this.o += length;
    return this.b.subarray(this.o - length, this.o);
  }

  string(): string {
    const s = this.bytes(this.u32());
    return text.decode(this.cdr && s.at(-1) === 0 ? s.subarray(0, -1) : s);
  }

  /** std_msgs/Header's stamp (seconds). */
  header(): number {
    if (!this.cdr) this.u32();
    const sec = this.u32();
    const stamp = sec + this.u32() * 1e-9;
    this.string();
    return stamp;
  }
}

export interface PointsMessage {
  stamp: number;
  /** Three per point, finite ones only. */
  xyz: Float32Array;
  /** One per point, when the cloud has an `intensity` field. */
  intensity: Float32Array | null;
  /** How far through the scan each point was taken, 0 to 1, when the cloud has a per-point time field. */
  time: Float32Array | null;
}

/** Names LiDAR drivers give a per-point time field. */
const TIME_FIELDS = ["t", "time", "timestamp", "time_stamp", "time_offset", "point_time", "ts"];

export function decodePointCloud2(data: Uint8Array, encoding: Encoding): PointsMessage {
  const m = new Message(data, encoding === "cdr");
  const stamp = m.header();
  const height = m.u32();
  const width = m.u32();
  const layout = new Map<string, { offset: number; type: number }>();
  for (let k = m.u32(); k > 0; k--) {
    const name = m.string();
    const offset = m.u32();
    const type = m.u8();
    m.u32();
    layout.set(name, { offset, type });
  }
  const bigEndian = m.u8() !== 0;
  const step = m.u32();
  const rowStep = m.u32();
  const bytes = m.bytes(m.u32());
  const v = view(bytes);
  const reader = (name: string) => {
    const f = layout.get(name);
    if (!f) return null;
    const le = !bigEndian;
    const o = f.offset;
    switch (f.type) {
      case 1:
        return (at: number) => v.getInt8(at + o);
      case 2:
        return (at: number) => v.getUint8(at + o);
      case 3:
        return (at: number) => v.getInt16(at + o, le);
      case 4:
        return (at: number) => v.getUint16(at + o, le);
      case 5:
        return (at: number) => v.getInt32(at + o, le);
      case 6:
        return (at: number) => v.getUint32(at + o, le);
      case 7:
        return (at: number) => v.getFloat32(at + o, le);
      case 8:
        return (at: number) => v.getFloat64(at + o, le);
      default:
        return null;
    }
  };
  const [x, y, z] = ["x", "y", "z"].map(reader);
  if (!x || !y || !z) throw new Error("a PointCloud2 without x, y and z fields");
  const intensity = reader("intensity");
  const timeField = TIME_FIELDS.find((name) => layout.has(name));
  const time = timeField ? reader(timeField) : null;
  const xyz = new Float32Array(width * height * 3);
  const values = intensity ? new Float32Array(width * height) : null;
  const times = time ? new Float64Array(width * height) : null;
  let n = 0;
  for (let row = 0; row < height; row++) {
    for (let col = 0; col < width; col++) {
      const at = row * rowStep + col * step;
      if (at + step > bytes.length) break;
      const px = x(at);
      const py = y(at);
      const pz = z(at);
      if (!Number.isFinite(px) || !Number.isFinite(py) || !Number.isFinite(pz)) continue;
      xyz[3 * n] = px;
      xyz[3 * n + 1] = py;
      xyz[3 * n + 2] = pz;
      if (values) values[n] = intensity!(at);
      if (times) times[n] = time!(at);
      n++;
    }
  }
  let fractions: Float32Array | null = null;
  if (times && n > 1) {
    let lo = Infinity;
    let hi = -Infinity;
    for (let i = 0; i < n; i++) {
      lo = Math.min(lo, times[i]);
      hi = Math.max(hi, times[i]);
    }
    if (hi > lo) fractions = Float32Array.from(times.subarray(0, n), (t) => (t - lo) / (hi - lo));
  }
  return { stamp, xyz: xyz.subarray(0, 3 * n), intensity: values ? values.subarray(0, n) : null, time: fractions };
}

export interface ImuMessage {
  stamp: number;
  /** World up in the IMU's frame from its orientation, when it gives one. */
  up: [number, number, number] | null;
  acceleration: [number, number, number];
}

export function decodeImu(data: Uint8Array, encoding: Encoding): ImuMessage {
  const m = new Message(data, encoding === "cdr");
  const stamp = m.header();
  const [x, y, z, w] = [m.f64(), m.f64(), m.f64(), m.f64()];
  const covariance = m.f64();
  for (let k = 0; k < 8 + 3 + 9; k++) m.f64();
  const acceleration: [number, number, number] = [m.f64(), m.f64(), m.f64()];
  const usable = covariance !== -1 && Math.abs(x * x + y * y + z * z + w * w - 1) < 0.1;
  // The last row of the orientation's matrix: world z seen in the IMU's frame.
  const up: [number, number, number] | null = usable
    ? [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]
    : null;
  return { stamp, up, acceleration };
}

/**
 * Per time in `times`, the up direction (unit, in the IMU's frame) from the
 * IMU messages: its orientation when the nearest message within `window`
 * seconds has one, else the mean acceleration within `window` either side
 * (at rest an accelerometer reads straight up); null where there is none.
 */
export function upsAt(imu: ImuMessage[], times: number[], window = 0.5): ([number, number, number] | null)[] {
  const sorted = imu.slice().sort((a, b) => a.stamp - b.stamp);
  const stamps = sorted.map((m) => m.stamp);
  const lowerBound = (t: number) => {
    let lo = 0;
    let hi = stamps.length;
    while (lo < hi) {
      const mid = (lo + hi) >> 1;
      if (stamps[mid] < t) lo = mid + 1;
      else hi = mid;
    }
    return lo;
  };
  return times.map((t) => {
    if (sorted.length === 0) return null;
    let k = Math.min(lowerBound(t), sorted.length - 1);
    if (k > 0 && Math.abs(stamps[k - 1] - t) < Math.abs(stamps[k] - t)) k--;
    if (Math.abs(stamps[k] - t) > window) return null;
    let up = sorted[k].up;
    if (!up) {
      const sum = [0, 0, 0];
      for (let j = lowerBound(t - window); j < sorted.length && stamps[j] <= t + window; j++) {
        for (let a = 0; a < 3; a++) sum[a] += sorted[j].acceleration[a];
      }
      up = sum as [number, number, number];
    }
    const norm = Math.hypot(...up);
    return Number.isFinite(norm) && norm > 1e-6 ? (up.map((v) => v / norm) as [number, number, number]) : null;
  });
}
