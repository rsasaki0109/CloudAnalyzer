/** Content identities without retaining a second copy of a large source file. */
export type SourceReference =
  | { kind: "file"; name: string; size: number; algorithm: "sha256-chunks-v1"; digest: string }
  | { kind: "http"; name: string; size: number; url: string; etag: string };

const CHUNK = 8 * 1024 * 1024;
const fingerprints = new WeakMap<File, SourceReference>();
export const sourceName = (file: File) => file.webkitRelativePath || file.name;
const hex = (bytes: ArrayBuffer) => [...new Uint8Array(bytes)].map(b => b.toString(16).padStart(2, "0")).join("");

export async function referenceFile(file: File, signal?: AbortSignal): Promise<SourceReference> {
  signal?.throwIfAborted();
  const cached = fingerprints.get(file);
  if (cached) return cached;
  const chunks: Uint8Array[] = [];
  for (let start = 0; start < file.size; start += CHUNK) {
    signal?.throwIfAborted();
    chunks.push(new Uint8Array(await crypto.subtle.digest("SHA-256", await file.slice(start, start + CHUNK).arrayBuffer())));
  }
  // Include the algorithm, byte length and chunk length in the root digest.
  const prefix = new TextEncoder().encode(`sha256-chunks-v1:${CHUNK}:${file.size}:`);
  const root = new Uint8Array(prefix.length + chunks.length * 32);
  root.set(prefix);
  chunks.forEach((chunk, i) => root.set(chunk, prefix.length + i * 32));
  const digest = hex(await crypto.subtle.digest("SHA-256", root));
  signal?.throwIfAborted();
  const reference: SourceReference = { kind: "file", name: sourceName(file), size: file.size, algorithm: "sha256-chunks-v1", digest };
  fingerprints.set(file, reference);
  return reference;
}

export function parseSource(value: unknown): SourceReference {
  if (!value || typeof value !== "object") throw new Error("Invalid source reference");
  const source = value as Record<string, unknown>;
  if (typeof source.name !== "string" || !source.name || !Number.isSafeInteger(source.size) || Number(source.size) < 0) throw new Error("Invalid source name or size");
  if (source.kind === "file" && source.algorithm === "sha256-chunks-v1" && typeof source.digest === "string" && /^[0-9a-f]{64}$/.test(source.digest)) {
    return { kind: "file", name: source.name, size: Number(source.size), algorithm: source.algorithm, digest: source.digest };
  }
  if (source.kind === "http" && typeof source.url === "string" && /^https?:\/\//.test(source.url) && typeof source.etag === "string" && /^"[^"\r\n]*"$/.test(source.etag)) {
    return { kind: "http", name: source.name, size: Number(source.size), url: source.url, etag: source.etag };
  }
  throw new Error("Invalid source content identity");
}

export async function matchesFile(file: File, source: SourceReference, signal?: AbortSignal): Promise<boolean> {
  if (source.kind !== "file" || file.size !== source.size) return false;
  const actual = await referenceFile(file, signal);
  return actual.kind === "file" && actual.digest === source.digest;
}
