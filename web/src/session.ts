// Saved or shared app state: the view, and per cloud how it is shown. The
// data itself is not included; clouds are found again by URL (fetched) or by
// file name (the user opens the same files).

export type Vec3 = [number, number, number];

export interface SessionCloud {
  name: string;
  /** Where the cloud was loaded from, if it came from a URL. */
  url?: string;
  visible: boolean;
  mode: "rgb" | "solid" | "intensity" | "classification" | "c2c" | "normal" | "shade";
  solid: Vec3;
  /** ICP transforms applied to it, oldest first (row-major 4x4). */
  transforms: number[][];
  /** A C2C / C2M distance to recompute: against the cloud or mesh of this name. */
  distance?: { reference: string; signed: boolean };
}

export interface Session {
  app: "CloudAnalyzer Web";
  version: 1;
  /** In original (unshifted) coordinates. */
  camera?: { position: Vec3; target: Vec3 };
  pointSize: number;
  edl: boolean;
  edlStrength: number;
  pointBudget: number;
  ramp: string;
  range: { lo: number; hi: number } | null;
  hiddenClasses: number[];
  /** Clipping box in original coordinates, if clipping is on. */
  clip: { min: Vec3; max: Vec3 } | null;
  /** Cross-section line (x, y in original coordinates) and half its band width. */
  profile?: { line: [number, number][]; halfWidth: number } | null;
  /** Text labels at points (original coordinates). */
  labels?: { position: Vec3; text: string }[];
  /** Viewport background colour (#rrggbb). */
  background?: string;
  pointSizeMode?: "fixed" | "adaptive";
  /** Saved camera views, in original coordinates. */
  views?: { name: string; position: Vec3; target: Vec3 }[];
  clouds: SessionCloud[];
}

const MODES = ["rgb", "solid", "intensity", "classification", "c2c", "normal", "shade"] as const;

function fail(what: string): never {
  throw new Error(`Not a CloudAnalyzer session: ${what}`);
}

const isNumber = (v: unknown): v is number => typeof v === "number" && Number.isFinite(v);

function vec3(v: unknown, what: string): Vec3 {
  if (!Array.isArray(v) || v.length !== 3 || !v.every(isNumber)) fail(what);
  return [v[0], v[1], v[2]];
}

function number(v: unknown, fallback: number): number {
  return isNumber(v) ? v : fallback;
}

function parseProfile(v: unknown): Session["profile"] {
  if (typeof v !== "object" || v === null) return null;
  const p = v as Record<string, unknown>;
  if (!Array.isArray(p.line) || !isNumber(p.halfWidth) || p.halfWidth <= 0) return null;
  const line = p.line.filter(
    (xy): xy is [number, number] => Array.isArray(xy) && xy.length === 2 && xy.every(isNumber),
  );
  return line.length >= 2 ? { line, halfWidth: p.halfWidth } : null;
}

/** Check a parsed JSON value and fill in defaults. */
export function parseSession(json: unknown): Session {
  if (typeof json !== "object" || json === null) fail("not an object");
  const o = json as Record<string, unknown>;
  if (o.app !== "CloudAnalyzer Web") fail("unknown app");
  if (o.version !== 1) fail(`unsupported version ${String(o.version)}`);
  const camera = o.camera as Record<string, unknown> | undefined;
  const range = o.range as Record<string, unknown> | null | undefined;
  const clip = o.clip as Record<string, unknown> | null | undefined;
  if (!Array.isArray(o.clouds)) fail("no clouds");
  return {
    app: "CloudAnalyzer Web",
    version: 1,
    camera: camera ? { position: vec3(camera.position, "camera"), target: vec3(camera.target, "camera") } : undefined,
    pointSize: number(o.pointSize, 2),
    edl: o.edl !== false,
    edlStrength: number(o.edlStrength, 1),
    pointBudget: number(o.pointBudget, 3_000_000),
    ramp: typeof o.ramp === "string" ? o.ramp : "Blue > Green > Yellow > Red",
    range: range && isNumber(range.lo) && isNumber(range.hi) ? { lo: range.lo, hi: range.hi } : null,
    hiddenClasses: Array.isArray(o.hiddenClasses)
      ? o.hiddenClasses.filter((c): c is number => Number.isInteger(c) && c >= 0 && c < 256)
      : [],
    clip: clip ? { min: vec3(clip.min, "clip"), max: vec3(clip.max, "clip") } : null,
    profile: parseProfile(o.profile),
    background: typeof o.background === "string" && /^#[0-9a-f]{6}$/i.test(o.background) ? o.background : undefined,
    pointSizeMode: o.pointSizeMode === "adaptive" ? "adaptive" : "fixed",
    views: Array.isArray(o.views)
      ? o.views.flatMap((v: unknown) => {
          const r = (v ?? {}) as Record<string, unknown>;
          return typeof r.name === "string"
            ? [{ name: r.name, position: vec3(r.position, "view"), target: vec3(r.target, "view") }]
            : [];
        })
      : [],
    labels: Array.isArray(o.labels)
      ? o.labels.flatMap((l: unknown) => {
          const r = (l ?? {}) as Record<string, unknown>;
          return typeof r.text === "string" ? [{ position: vec3(r.position, "label"), text: r.text }] : [];
        })
      : [],
    clouds: o.clouds.map((c: unknown) => {
      if (typeof c !== "object" || c === null) fail("cloud");
      const r = c as Record<string, unknown>;
      if (typeof r.name !== "string") fail("cloud name");
      const distance = r.distance as Record<string, unknown> | undefined;
      return {
        name: r.name,
        url: typeof r.url === "string" ? r.url : undefined,
        visible: r.visible !== false,
        mode: MODES.includes(r.mode as (typeof MODES)[number]) ? (r.mode as SessionCloud["mode"]) : "solid",
        solid: Array.isArray(r.solid) ? vec3(r.solid, "color") : [235, 235, 235],
        transforms: Array.isArray(r.transforms)
          ? r.transforms.map((m: unknown) => {
              if (!Array.isArray(m) || m.length !== 16 || !m.every(isNumber)) fail("transform");
              return m as number[];
            })
          : [],
        distance:
          distance && typeof distance.reference === "string"
            ? { reference: distance.reference, signed: distance.signed === true }
            : undefined,
      };
    }),
  };
}

/** Session as URL-safe base64 of its JSON, for a link's hash. */
export function encodeSession(session: Session): string {
  const bytes = new TextEncoder().encode(JSON.stringify(session));
  let binary = "";
  for (const b of bytes) binary += String.fromCharCode(b);
  return btoa(binary).replace(/\+/g, "-").replace(/\//g, "_").replace(/=+$/, "");
}

export function decodeSession(text: string): Session {
  const base64 = text.replace(/-/g, "+").replace(/_/g, "/");
  const binary = atob(base64 + "=".repeat((4 - (base64.length % 4)) % 4));
  const bytes = Uint8Array.from(binary, (ch) => ch.charCodeAt(0));
  return parseSession(JSON.parse(new TextDecoder().decode(bytes)));
}

/** File name for a URL: its last path segment, decoded. */
export function nameFromUrl(url: string): string {
  const path = new URL(url, location.href).pathname;
  const last = path.split("/").filter(Boolean).at(-1) ?? "cloud";
  try {
    return decodeURIComponent(last);
  } catch {
    return last;
  }
}
