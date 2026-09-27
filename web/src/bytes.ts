// Byte ranges of a local file or a remote one (HTTP range requests), so a
// COPC file can be read node by node without fetching all of it.

export type ByteSource = { file: File } | { url: string };

export async function readRange(source: ByteSource, offset: number, length: number): Promise<Uint8Array> {
  if ("file" in source) {
    return new Uint8Array(await source.file.slice(offset, offset + length).arrayBuffer());
  }
  const response = await fetch(source.url, { headers: { Range: `bytes=${offset}-${offset + length - 1}` } });
  if (!response.ok) throw new Error(`HTTP ${response.status}`);
  const bytes = new Uint8Array(await response.arrayBuffer());
  if (response.status === 206) return bytes;
  // The server ignored the range and sent the whole file.
  if (bytes.length >= offset + length) return bytes.subarray(offset, offset + length);
  throw new Error("the server does not support range requests");
}
