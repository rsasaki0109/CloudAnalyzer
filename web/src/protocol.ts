// Messages exchanged between the UI thread and the WASM worker.

export type Vec3 = [number, number, number];

export interface LoadedCloud {
  id: number;
  name: string;
  count: number;
  /** Interleaved xyz, already shifted by the session's global shift. */
  positions: Float32Array;
  /** Interleaved rgb, or null when the file carries no colors. */
  colors: Uint8Array | null;
  /** [minX, minY, minZ, maxX, maxY, maxZ] in original coordinates. */
  bounds: number[];
  shift: Vec3;
}

export interface C2cStats {
  count: number;
  min: number;
  max: number;
  mean: number;
  rms: number;
  stdDev: number;
  median: number;
}

export interface C2cOutput {
  distances: Float32Array;
  stats: C2cStats;
  millis: number;
  /** Number of WASM workers the computation was split across. */
  workers: number;
}

export type Request =
  | { kind: "load"; name: string; bytes: ArrayBuffer }
  | { kind: "c2c"; compared: number; reference: number }
  | { kind: "remove"; id: number };

export type Response =
  | { ok: true; value: unknown }
  | { ok: false; error: string };
