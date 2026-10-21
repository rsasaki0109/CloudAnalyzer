/** Opt-in host timings use performance.now(); GPU durations use a separate clock. */
export interface GpuVerificationTiming {
  /** Includes wasm-bindgen argument imports/copies, not just the native search. */
  verifyCallMillis: number;
  distanceExportMillis: number;
}

export interface GpuLaneTiming extends GpuVerificationTiming {
  first: number; count: number; lane: number; resident: boolean;
  queryExportMillis: number;
  /** Includes dispatch, scheduling, verification, export and result transfer. */
  roundTripMillis: number;
}

export interface GpuBatchTiming {
  first: number; count: number; queryExportMillis: number;
  /** Host queue.writeBuffer call; not device transfer duration. */
  uploadCallMillis?: number;
  /** Map/copy/unmap of candidates and timestamps; includes queue/scheduling wait. */
  readbackMillis?: number;
  /** Within-pass timestamp delta only; can be quantized to zero. */
  shaderMillis?: number;
}

export interface GpuC2cDiagnostics {
  timestampQuery: "enabled" | "unsupported" | "unavailable";
  gpu: GpuBatchTiming[];
  verification: GpuLaneTiming[];
  gpuBatchesOmitted: number;
  verificationBatchesOmitted: number;
}

// Small requested batches must not turn diagnostic replies into unbounded traces.
export const GPU_DIAGNOSTIC_BATCH_LIMIT = 256;

/** Called only on the opt-in path. Always release the WASM result on export failure. */
export function timeGpuVerification(verify: () => {
  takeDistances(): Float64Array; readonly certified: number; free(): void;
}) {
  const started = performance.now();
  const verified = verify();
  try {
    const verifiedAt = performance.now();
    const distances = verified.takeDistances();
    const exportedAt = performance.now();
    return { distances, certified: verified.certified, timing: {
      verifyCallMillis: verifiedAt - started, distanceExportMillis: exportedAt - verifiedAt,
    } };
  } finally { verified.free(); }
}
