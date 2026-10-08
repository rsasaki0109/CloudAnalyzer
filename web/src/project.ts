import { parseReviews, type LaneReview } from "./lane-review";
import { parseSession, type Session } from "./session";
import { parseSource, type SourceReference } from "./source-reference";
import type { PoseGraphFiles } from "./protocol";

export type GraphLoadOptions = Omit<PoseGraphFiles, "graph" | "scans">;
export interface SavedGraphSource {
  graph: SourceReference | null;
  scans: SourceReference[];
  options: GraphLoadOptions;
  first: number;
  nodeIds: number[];
}
export interface SavedPoseGraph {
  name: string;
  snapshot: string;
  sources: SavedGraphSource[];
  sessions: { name: string; first: number; count: number }[];
  imu: { nodes: number[]; ups: number[] };
}
export interface ControlSetting { value: string | boolean; cloud?: string }
export interface Project {
  app: "CloudAnalyzer Project";
  version: 1;
  session: Session;
  vectorMap: string;
  poseGraph: SavedPoseGraph | null;
  settings: Record<string, ControlSetting>;
  reviews: LaneReview[];
}

const object = (v: unknown): Record<string, unknown> => {
  if (!v || typeof v !== "object" || Array.isArray(v)) throw new Error("Invalid project object");
  return v as Record<string, unknown>;
};
const text = (v: unknown): string => {
  if (typeof v !== "string") throw new Error("Invalid project text");
  return v;
};
const numbers = (v: unknown): number[] => {
  if (!Array.isArray(v) || !v.every(n => typeof n === "number" && Number.isFinite(n))) throw new Error("Invalid project numbers");
  return v;
};
const integer = (v: unknown): number => {
  if (!Number.isSafeInteger(v) || Number(v) < 0) throw new Error("Invalid project index");
  return Number(v);
};

export function parseProject(input: unknown): Project {
  const p = object(input);
  if (p.app !== "CloudAnalyzer Project" || p.version !== 1) throw new Error("Unsupported CloudAnalyzer project version");
  const session = parseSession(p.session);
  if (session.clouds.some(c => !c.source)) throw new Error("Project cloud is missing its source identity");
  const vectorMap = text(p.vectorMap);
  object(JSON.parse(vectorMap));
  const settings: Record<string, ControlSetting> = {};
  for (const [id, value] of Object.entries(object(p.settings))) {
    const setting = object(value);
    if (!/^[a-z][a-zA-Z0-9-]*$/.test(id) || (typeof setting.value !== "string" && typeof setting.value !== "boolean")) throw new Error("Invalid project setting");
    settings[id] = { value: setting.value, cloud: setting.cloud === undefined ? undefined : text(setting.cloud) };
  }
  let poseGraph: SavedPoseGraph | null = null;
  if (p.poseGraph !== null) {
    const graph = object(p.poseGraph);
    const snapshot = text(graph.snapshot);
    const native = object(JSON.parse(snapshot));
    if (native.version !== 1 || !Array.isArray(object(native.graph).nodes)) throw new Error("Invalid saved pose graph");
    const count = (object(native.graph).nodes as unknown[]).length;
    if (!Array.isArray(graph.sources) || !Array.isArray(graph.sessions)) throw new Error("Invalid graph source list");
    const sources = graph.sources.map(s => {
      const source = object(s), options = object(source.options), odometry = object(options.odometry);
      if (!Array.isArray(source.scans)) throw new Error("Invalid scan references");
      const nodeIds = numbers(source.nodeIds), first = integer(source.first);
      if (first + nodeIds.length > count || !nodeIds.every(Number.isSafeInteger)) throw new Error("Invalid source node range");
      const finite = (v: unknown, minimum = 0) => {
        if (typeof v !== "number" || !Number.isFinite(v) || v < minimum) throw new Error("Invalid graph loading option");
        return v;
      };
      const extrinsic = options.extrinsic === null ? null : numbers(options.extrinsic);
      if (extrinsic && extrinsic.length !== 16) throw new Error("Invalid scan transform");
      const result: SavedGraphSource = {
        graph: source.graph === null ? null : parseSource(source.graph),
        scans: source.scans.map(parseSource), first, nodeIds,
        options: { voxel: finite(options.voxel), displayPoints: integer(options.displayPoints), sigmaT: finite(options.sigmaT, Number.MIN_VALUE), sigmaRDeg: finite(options.sigmaRDeg, Number.MIN_VALUE), extrinsic,
          odometry: { minRange: finite(odometry.minRange), maxRange: finite(odometry.maxRange), keyframeSpacing: finite(odometry.keyframeSpacing), deskew: odometry.deskew === true } },
      };
      if ([result.graph, ...result.scans].some(ref => ref && ref.kind !== "file")) throw new Error("Pose graph sources must be local files");
      return result;
    });
    const sessions = graph.sessions.map(s => {
      const value = object(s), first = integer(value.first), size = integer(value.count);
      if (first + size > count) throw new Error("Invalid graph session range");
      return { name: text(value.name), first, count: size };
    });
    const imu = object(graph.imu), nodes = numbers(imu.nodes), ups = numbers(imu.ups);
    if (ups.length !== nodes.length * 3 || !nodes.every(n => Number.isSafeInteger(n) && n >= 0 && n < count)) throw new Error("Invalid saved IMU directions");
    poseGraph = { name: text(graph.name), snapshot, sources, sessions, imu: { nodes, ups } };
  }
  return { app: "CloudAnalyzer Project", version: 1, session, vectorMap, poseGraph, settings, reviews: parseReviews(p.reviews ?? []) };
}

export function projectSources(project: Project): SourceReference[] {
  return [
    ...project.session.clouds.flatMap(c => c.source ? [c.source] : []),
    ...project.poseGraph?.sources.flatMap(s => [...(s.graph ? [s.graph] : []), ...s.scans]) ?? [],
  ];
}
