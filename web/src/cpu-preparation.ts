import type { Cloud } from "./wasm/ca_wasm.js";
import { releaseOn, runOn, runOnGeneration, workerGeneration } from "./pool";

type Owner = "c2c" | "map-quality";
export interface CpuSliceInput {
  reference: Float64Array;
  queries: Float64Array;
  indices: Uint32Array | null;
}

let nextJob = 0;
let current: CpuPreparation | undefined;

/** One idle CPU preparation across C2C and Map quality, plus active leases. */
export class CpuPreparation {
  readonly id = ++nextJob;
  readonly workers: number;
  readonly inputBytes: number;
  readonly generations: (number | undefined)[];
  private readonly expected: number[];
  private readonly ready: Promise<unknown>[];
  private readonly prepared: boolean[];
  private users = 0;
  private retired = false;
  private disposed = false;
  indices: (Uint32Array | null)[];

  constructor(readonly owner: Owner, readonly compared: Cloud, readonly reference: Cloud,
    readonly layout: string, inputs: CpuSliceInput[]) {
    this.workers = inputs.length;
    this.indices = inputs.map(input => input.indices);
    this.expected = inputs.map(input => input.queries.length / 3);
    this.prepared = inputs.map(() => false);
    this.inputBytes = inputs.reduce((sum, input) => sum + input.reference.byteLength +
      input.queries.byteLength + (input.indices?.byteLength ?? 0), 0);
    this.ready = inputs.map(async ({ reference, queries }, lane) => {
      await runOn(lane, { kind: "cloud-prepare", job: this.id, reference, queries });
      this.prepared[lane] = true;
    });
    this.generations = inputs.map((_, lane) => workerGeneration(lane));
    for (const ready of this.ready) void ready.catch(() => {});
  }

  retire(): void {
    this.retired = true;
    if (current === this) current = undefined;
    this.release();
  }

  private release(): void {
    if (!this.retired || this.users || this.disposed) return;
    this.disposed = true;
    for (let lane = 0; lane < this.workers; lane++) {
      void releaseOn(lane, { kind: "cloud-release", job: this.id }).catch(() => {});
    }
    this.indices = [];
  }

  /** Search every slice afresh, drain failures, and suppress late callbacks. */
  async run(done: (lane: number, values: Float64Array) => void, simd?: boolean, pipeline = true): Promise<void> {
    this.users++;
    let closed = false;
    try {
      const results = await Promise.allSettled(this.ready.map(async (ready, lane) => {
        try {
          const guard = <T>(pending: Promise<T>) => pending.catch(error => { closed = true; throw error; });
          const preparation = guard(ready);
          const search = () => runOnGeneration(lane, this.generations[lane], { kind: "cloud-prepared", job: this.id, simd });
          // Same-worker messages are ordered; cloud-prepare installs its state
          // synchronously after WASM initialization. Enqueue cold searches now
          // so a busy main worker does not delay them until the preparation ack.
          const distances = guard((pipeline || this.prepared[lane] ? search() : preparation.then(search)).then(values => {
            if (values.length !== this.expected[lane]) throw new Error("Incomplete C2C worker result");
            return values;
          }));
          // Drain both messages even if one fails. Preparation failure must
          // reject the job even when an independently delivered result succeeds.
          const outcomes = await Promise.allSettled([preparation, distances]);
          const failed = outcomes.find(outcome => outcome.status === "rejected");
          if (failed?.status === "rejected") throw failed.reason;
          const values = (outcomes[1] as PromiseFulfilledResult<Float64Array>).value;
          if (!closed) done(lane, values);
        } catch (error) { closed = true; throw error; }
      }));
      const failed = results.find(result => result.status === "rejected");
      if (failed?.status === "rejected") throw failed.reason;
    } catch (error) {
      this.retire();
      throw error;
    } finally {
      closed = true;
      this.users--;
      this.release();
    }
  }
}

/** A different workflow/layout or a lost worker replaces the idle group. */
export function findCpuPreparation(owner: Owner, compared: Cloud, reference: Cloud,
  layout: string): CpuPreparation | undefined {
  if (current?.owner === owner && current.compared === compared && current.reference === reference &&
    current.layout === layout && current.generations.every((generation, lane) =>
      generation !== undefined && generation === workerGeneration(lane))) return current;
  current?.retire();
  return undefined;
}

export function prepareCpuSlices(owner: Owner, compared: Cloud, reference: Cloud,
  layout: string, inputs: CpuSliceInput[]): CpuPreparation {
  current?.retire();
  return current = new CpuPreparation(owner, compared, reference, layout, inputs);
}

/** Coordinates/order changed or either source is being removed. */
export function invalidateCpuPreparation(cloud: Cloud): void {
  if (current?.compared === cloud || current?.reference === cloud) current.retire();
}
