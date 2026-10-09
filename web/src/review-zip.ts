/** Bounded ZIP records shared by generated-map reviews and workspace snapshots. */
export const REVIEW_LIMIT = 64 * 1024 * 1024;
export const MANIFEST_LIMIT = 10 * 1024 * 1024;
const HEADER_LIMIT = MANIFEST_LIMIT;
export interface Member {
  path: string;
  bytes: number;
  compressed: number;
  method: number;
  flags: number;
  offset: number;
}
const decoder = new TextDecoder("utf-8", { fatal: true });
function path(value: unknown): string {
  if (
    typeof value !== "string" ||
    !value ||
    value.startsWith("/") ||
    value.includes("\\") ||
    value.split("/").some((p) => !p || p === "." || p === "..")
  )
    throw new Error("Unsafe review member path");
  return value;
}
async function range(
  file: Blob,
  start: number,
  length: number,
): Promise<DataView> {
  if (
    !Number.isSafeInteger(start) ||
    start < 0 ||
    length < 0 ||
    start + length > file.size
  )
    throw new Error("Invalid ZIP member bounds");
  return new DataView(await file.slice(start, start + length).arrayBuffer());
}

export async function indexReviewZip(
  file: File,
  signal: AbortSignal,
): Promise<Map<string, Member>> {
  if (file.size < 22 || file.size > REVIEW_LIMIT + HEADER_LIMIT)
    throw new Error(
      "Review ZIP exceeds the 64 MiB browser limit or is incomplete; use the CLI for larger packages",
    );
  const tailStart = Math.max(0, file.size - 65557);
  const tail = await range(file, tailStart, file.size - tailStart);
  let end = -1;
  for (let i = tail.byteLength - 22; i >= 0; i--)
    if (
      tail.getUint32(i, true) === 0x06054b50 &&
      i + 22 + tail.getUint16(i + 20, true) === tail.byteLength
    ) {
      end = i;
      break;
    }
  if (end < 0 || tail.getUint16(end + 4, true) || tail.getUint16(end + 6, true))
    throw new Error("Unsupported review ZIP directory");
  const count = tail.getUint16(end + 10, true),
    size = tail.getUint32(end + 12, true),
    start = tail.getUint32(end + 16, true);
  if (
    count > 129 ||
    count !== tail.getUint16(end + 8, true) ||
    start + size !== tailStart + end
  )
    throw new Error("Unsupported or excessive review ZIP directory");
  const directory = await range(file, start, size);
  const result = new Map<string, Member>();
  let offset = 0,
    total = 0;
  for (let i = 0; i < count; i++) {
    signal.throwIfAborted();
    if (offset + 46 > size || directory.getUint32(offset, true) !== 0x02014b50)
      throw new Error("Invalid ZIP directory member");
    const flags = directory.getUint16(offset + 8, true),
      method = directory.getUint16(offset + 10, true);
    const nameLength = directory.getUint16(offset + 28, true),
      extra = directory.getUint16(offset + 30, true),
      comment = directory.getUint16(offset + 32, true);
    const next = offset + 46 + nameLength + extra + comment;
    if (next > size) throw new Error("Incomplete ZIP directory member");
    const name = path(
      decoder.decode(
        new Uint8Array(
          directory.buffer,
          directory.byteOffset + offset + 46,
          nameLength,
        ),
      ),
    );
    const mode = (directory.getUint32(offset + 38, true) >>> 16) & 0o170000;
    if (
      result.has(name) ||
      flags & 1 ||
      ![0, 8].includes(method) ||
      ![0, 0o100000].includes(mode) ||
      directory.getUint16(offset + 34, true)
    )
      throw new Error(
        "Duplicate, nonregular, encrypted or unsupported ZIP member",
      );
    const bytes = directory.getUint32(offset + 24, true),
      compressed = directory.getUint32(offset + 20, true),
      local = directory.getUint32(offset + 42, true);
    total += bytes;
    if (
      total > REVIEW_LIMIT ||
      local >= start ||
      compressed > start - local ||
      (name === "manifest.json" && bytes > HEADER_LIMIT)
    )
      throw new Error("Review contents exceed browser limits or ZIP bounds");
    result.set(name, {
      path: name,
      bytes,
      compressed,
      method,
      flags,
      offset: local,
    });
    offset = next;
  }
  if (offset !== size || !result.has("manifest.json"))
    throw new Error("Review ZIP has no complete manifest directory");
  // Reject overlapping local records and local names/flags that disagree with the directory.
  const sorted = [...result.values()].sort((a, b) => a.offset - b.offset);
  for (const [i, member] of sorted.entries()) {
    const local = await range(file, member.offset, 30);
    const length = local.getUint16(26, true),
      extra = local.getUint16(28, true);
    if (
      local.getUint32(0, true) !== 0x04034b50 ||
      local.getUint16(6, true) !== member.flags ||
      local.getUint16(8, true) !== member.method
    )
      throw new Error("ZIP local record differs from directory");
    const name = decoder.decode(
      new Uint8Array((await range(file, member.offset + 30, length)).buffer),
    );
    if (name !== member.path)
      throw new Error("ZIP local member name differs from directory");
    member.offset += 30 + length + extra;
    if (member.offset + member.compressed > (sorted[i + 1]?.offset ?? start))
      throw new Error("Overlapping ZIP records");
  }
  return result;
}

export async function readReviewMember(
  file: File,
  member: Member,
  signal: AbortSignal,
): Promise<Uint8Array<ArrayBuffer>> {
  let stream = file
    .slice(member.offset, member.offset + member.compressed)
    .stream();
  if (member.method === 8) {
    if (typeof DecompressionStream === "undefined")
      throw new Error(
        "This browser cannot decompress review ZIPs; extract and open the files manually",
      );
    stream = stream.pipeThrough(new DecompressionStream("deflate-raw"));
  }
  const reader = stream.getReader(),
    chunks: Uint8Array[] = [];
  let size = 0;
  try {
    while (true) {
      signal.throwIfAborted();
      const { value, done } = await reader.read();
      if (done) break;
      size += value.byteLength;
      if (size > member.bytes)
        throw new Error("Decompressed member exceeds its recorded byte limit");
      chunks.push(value);
    }
    if (size !== member.bytes)
      throw new Error("ZIP member size differs from manifest");
  } finally {
    await reader.cancel();
    reader.releaseLock();
  }
  const data = new Uint8Array(size);
  let offset = 0;
  for (const chunk of chunks) {
    data.set(chunk, offset);
    offset += chunk.byteLength;
  }
  return data;
}

/** Store bounded blobs in one ZIP without extra dependencies or lossy conversion. */
export async function writeReviewZip(
  entries: [string, Blob][],
  signal: AbortSignal,
): Promise<Blob> {
  const names = new Set(entries.map(([name]) => path(name)));
  if (
    names.size !== entries.length ||
    entries.length > 129 ||
    entries.reduce((n, [, b]) => n + b.size, 0) > REVIEW_LIMIT
  )
    throw new Error("Snapshot exceeds the 64 MiB content or member limit");
  const table = new Uint32Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i;
    for (let j = 0; j < 8; j++) c = (c >>> 1) ^ (c & 1 ? 0xedb88320 : 0);
    table[i] = c;
  }
  const encoder = new TextEncoder(),
    locals: BlobPart[] = [],
    directory: BlobPart[] = [];
  let offset = 0,
    directoryBytes = 0;
  for (const [name, blob] of entries) {
    signal.throwIfAborted();
    const text = encoder.encode(name),
      bytes = new Uint8Array(await blob.arrayBuffer());
    if (text.length > 65535)
      throw new Error("Snapshot member name is too long");
    let crc = 0xffffffff;
    for (let i = 0; i < bytes.length; i++) {
      if (i % 1048576 === 0) signal.throwIfAborted();
      crc = (crc >>> 8) ^ table[(crc ^ bytes[i]) & 255];
    }
    crc = (crc ^ 0xffffffff) >>> 0;
    const local = new Uint8Array(30),
      l = new DataView(local.buffer),
      central = new Uint8Array(46),
      c = new DataView(central.buffer);
    l.setUint32(0, 0x04034b50, true);
    l.setUint16(4, 20, true);
    l.setUint16(6, 0x800, true);
    l.setUint16(12, 0x21, true);
    l.setUint32(14, crc, true);
    l.setUint32(18, bytes.length, true);
    l.setUint32(22, bytes.length, true);
    l.setUint16(26, text.length, true);
    c.setUint32(0, 0x02014b50, true);
    c.setUint16(4, 20, true);
    c.setUint16(6, 20, true);
    c.setUint16(8, 0x800, true);
    c.setUint16(14, 0x21, true);
    c.setUint32(16, crc, true);
    c.setUint32(20, bytes.length, true);
    c.setUint32(24, bytes.length, true);
    c.setUint16(28, text.length, true);
    c.setUint32(42, offset, true);
    locals.push(local, text, blob);
    directory.push(central, text);
    offset += local.length + text.length + bytes.length;
    directoryBytes += central.length + text.length;
  }
  const end = new Uint8Array(22),
    e = new DataView(end.buffer);
  e.setUint32(0, 0x06054b50, true);
  e.setUint16(8, entries.length, true);
  e.setUint16(10, entries.length, true);
  e.setUint32(12, directoryBytes, true);
  e.setUint32(16, offset, true);
  signal.throwIfAborted();
  return new Blob([...locals, ...directory, end], { type: "application/zip" });
}
