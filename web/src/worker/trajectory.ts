// Trajectory files and ATE/RPE evaluation.

import { evaluateTrajectory, TrajectoryData } from "../wasm/ca_wasm.js";
import type { TrajectoryEvaluation, TrajectoryPoses, Request } from "../protocol";

/** A trajectory file's poses, or null when the file is not a trajectory. */
export async function readTrajectory(file: File): Promise<TrajectoryPoses | null> {
  // TUM and CSV share extensions with point clouds: judge by the first line.
  const format = TrajectoryData.detect(file.name, await file.slice(0, 4096).text());
  if (!format) return null;
  const data = TrajectoryData.parse(await file.text(), format);
  const poses: TrajectoryPoses = {
    format: format as TrajectoryPoses["format"],
    timestamps: data.timestamps(),
    positions: data.positions(),
    orientations: data.orientations() ?? null,
  };
  data.free();
  return poses;
}
export function evaluate(req: Extract<Request, { kind: "trajectory-eval" }>): TrajectoryEvaluation {
  const data = (p: TrajectoryPoses) => new TrajectoryData(p.timestamps, p.positions, p.orientations ?? new Float64Array());
  const [estimate, reference] = [data(req.estimate), data(req.reference)];
  try {
    const out = evaluateTrajectory(estimate, reference, req.maxTimeDelta, req.alignment, req.delta, req.deltaUnit);
    const stats = (name: string) => {
      const s = out.stats(name);
      if (!s) return null;
      const [count, rmse, mean, median, std, min, max] = s;
      return { count, rmse, mean, median, std, min, max };
    };
    const ateRotation = out.values("ate_rotation");
    const value: TrajectoryEvaluation = {
      timestamps: out.timestamps(),
      estimate: out.estimate(),
      reference: out.reference(),
      ate: out.values("ate")!,
      ateRotation: ateRotation ?? null,
      stats: {
        ate: stats("ate")!,
        ateRotation: stats("ate_rotation"),
        rpe: stats("rpe"),
        rpeRotation: stats("rpe_rotation"),
        rpePercent: stats("rpe_percent"),
      },
      matrix: Array.from(out.matrix()),
      scale: out.scale,
      endpointDrift: out.endpointDrift,
      referenceLength: out.referenceLength,
      estimateLength: out.estimateLength,
    };
    out.free();
    return value;
  } finally {
    estimate.free();
    reference.free();
  }
}
