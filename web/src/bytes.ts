// Bounded byte ranges. Full download fallback is unsafe for large sources.
export type ByteSource = { file: File } | { url: string };
export const MAX_RANGE_BYTES = 64 * 1024 * 1024;

function validateRange(offset: number, length: number, bounded = true): void {
  if (!Number.isSafeInteger(offset) || !Number.isSafeInteger(length) || offset < 0 || length < 0 ||
      !Number.isSafeInteger(offset + length) || (bounded && length > MAX_RANGE_BYTES)) {
    throw new Error("invalid byte range or range exceeds 64 MiB");
  }
}

/** Check headers before reading, then consume at most the declared span. */
export async function readRangeResponse(response: Response, offset: number, length: number): Promise<{ bytes: Uint8Array; total: number }> {
  validateRange(offset, length);
  const reject = async (message: string): Promise<never> => {
    await response.body?.cancel();
    throw new Error(message);
  };
  if (response.status !== 206) return reject(`HTTP byte ranges require 206, received ${response.status}`);
  const match = /^bytes (\d+)-(\d+)\/(\d+)$/.exec(response.headers.get("content-range") ?? "");
  if (!match) return reject("missing or invalid Content-Range; the server must expose this header for CORS");
  const [start, stop, total] = match.slice(1).map(Number);
  if (![start, stop, total].every(Number.isSafeInteger) || total <= offset || start !== offset || stop !== Math.min(offset + length, total) - 1) {
    return reject("Content-Range does not match the requested byte span");
  }
  const expected = stop - start + 1;
  const encoding = response.headers.get("content-encoding");
  if (encoding && encoding.toLowerCase() !== "identity") return reject("encoded HTTP range responses are unsupported");
  const size = response.headers.get("content-length");
  if (size !== null && (!/^\d+$/.test(size) || Number(size) !== expected)) return reject("Content-Length does not match Content-Range");
  if (!response.body) return reject("missing HTTP range body");
  const reader = response.body.getReader();
  const bytes = new Uint8Array(expected);
  let at = 0;
  try {
    for (;;) {
      const { done, value } = await reader.read();
      if (done) break;
      if (at + value.length > expected) throw new Error("HTTP range body exceeds its declared span");
      bytes.set(value, at);
      at += value.length;
    }
    if (at !== expected) throw new Error("HTTP range body is truncated");
    return { bytes, total };
  } catch (error) {
    await reader.cancel();
    throw error;
  } finally {
    reader.releaseLock();
  }
}

export async function readRange(source: ByteSource, offset: number, length: number, signal?: AbortSignal): Promise<Uint8Array> {
  validateRange(offset, length, !("file" in source));
  if (length === 0) return new Uint8Array();
  if ("file" in source) return new Uint8Array(await source.file.slice(offset, offset + length).arrayBuffer());
  const response = await fetch(source.url, { signal, cache: "no-store", headers: { Range: `bytes=${offset}-${offset + length - 1}` } });
  return (await readRangeResponse(response, offset, length)).bytes;
}
