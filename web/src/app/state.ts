/** State shared by the panels: the viewer, the open clouds and the distance display. */

import * as THREE from "three";
import type { RampName } from "../colormap";
import type { LodNode } from "../lod";
import type { C2cOutput, LoadedCloud, Vec3 } from "../protocol";
import { Viewer } from "../viewer";
import { $ } from "./dom";

export type ColorMode =
  | "rgb"
  | "solid"
  | "intensity"
  | "classification"
  | "c2c"
  | "normal"
  | "shade"
  | "opacity"
  | "scalar";

/** Where a cloud came from: sessions can restore file and URL clouds. */
export type Origin = ({ kind: "file"; file?: File } | { kind: "url"; url: string; file?: File; size?: number; etag?: string } | { kind: "derived" }) & { loadMaxPoints?: number };

export interface Entry {
  cloud: LoadedCloud;
  nodes: LodNode[];
  solid: Vec3;
  mode: ColorMode;
  visible: boolean;
  c2c?: C2cOutput & { referenceName: string };
  /** Scalar fields fetched or computed so far, by name. */
  fields: Map<string, Float32Array>;
  /** The field the "scalar" color mode shows. */
  field?: { name: string; values: Float32Array; stats: C2cOutput["stats"] };
  /** Transforms applied by ICP, newest last, for undo. */
  transforms: number[][];
  origin: Origin;
}

export const viewer = new Viewer($("viewport"));

export const entries = new Map<number, Entry>();

/** How distances are colored, and which cloud's distances the colorbar shows. */
export const display: {
  ramp: RampName;
  range: { lo: number; hi: number } | null;
  activeC2c: number | null;
} = { ramp: "Blue > Green > Yellow > Red", range: null, activeC2c: null };

/** ASPRS class codes currently hidden in every cloud. */
export const hiddenClasses = new Set<number>();

export const isMesh = (entry: Entry) => entry.cloud.kind === "mesh";

export const clouds = () => [...entries.values()].filter((e) => !isMesh(e));

export function findByName(name: string): Entry | undefined {
  return [...entries.values()].find((e) => e.cloud.name === name);
}

let sceneShift: Vec3 | null = null;

/**
 * Offset between original and render coordinates: the shift of the first
 * cloud opened, kept until the list is empty so drawn things never move.
 * Each cloud is drawn relative to its own shift and placed at
 * `toRender(cloud.shift)`; three.js combines that (float64) placement with
 * the camera on the CPU, so far-apart clouds all stay precise.
 */
export function globalShift(): Vec3 {
  if (entries.size === 0) sceneShift = null;
  return sceneShift ?? [0, 0, 0];
}

/** Add an entry to the list; the first one sets the global shift. */
export function putEntry(entry: Entry): void {
  if (entries.size === 0) sceneShift = entry.cloud.shift;
  entries.set(entry.cloud.id, entry);
}

/** Original coordinates to render coordinates. */
export function toRender(v: readonly number[]): THREE.Vector3 {
  const shift = globalShift();
  return new THREE.Vector3(v[0] - shift[0], v[1] - shift[1], v[2] - shift[2]);
}

/** Render coordinates to original coordinates. */
export function toOriginal(v: THREE.Vector3): Vec3 {
  const shift = globalShift();
  return [v.x + shift[0], v.y + shift[1], v.z + shift[2]];
}

export function hideEntry(entry: Entry): void {
  entry.visible = false;
  viewer.setVisible(entry.cloud.id, false);
}

/** A list of listeners. */
export class Signal<A extends unknown[] = []> {
  private listeners: ((...args: A) => void)[] = [];
  add(listener: (...args: A) => void): void {
    this.listeners.push(listener);
  }
  emit(...args: A): void {
    for (const listener of this.listeners) listener(...args);
  }
}

/** The cloud list changed: panels refresh their cloud pickers. */
export const listChanged = new Signal();
/** A cloud was removed or its points replaced: point references to it are stale. */
export const pointsInvalidated = new Signal<[cloudId: number]>();
/** The shown distance result or its colors changed. */
export const distanceChanged = new Signal();
