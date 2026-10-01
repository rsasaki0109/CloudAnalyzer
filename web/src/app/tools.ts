/**
 * Interactive tools that take over clicks on the view (measuring, labeling,
 * drawing a profile line…). At most one is active; without one, a click
 * picks a point and a double click centres the view on it.
 */

import { pointAt } from "../api";
import type { Vec3 } from "../protocol";
import { typing } from "./dom";
import { viewer } from "./state";
import type * as THREE from "three";

export interface Tool {
  click(x: number, y: number): void | Promise<void>;
  /** Return true to capture a primary-pointer drag instead of orbiting. */
  pointerDown?(x: number, y: number): boolean;
  pointerMove?(x: number, y: number): void;
  pointerUp?(x: number, y: number): void | Promise<void>;
  pointerCancel?(): void;
  doubleClick?(x: number, y: number): void;
  /** A key while active; return true if handled. Unhandled Escape leaves the tool. */
  key?(e: KeyboardEvent): boolean;
  /** Called when the tool becomes active. */
  enter?(): void;
  /** Called when the tool is left, to reset its state and UI. */
  exit(): void;
}

export interface PickedPoint {
  cloudId: number;
  index: number;
  /** Render (shifted) position, for drawing. */
  render: THREE.Vector3;
  /** Exact original coordinates. */
  exact: Vec3;
}

let active: Tool | null = null;

/** What clicks, double clicks and Escape do without a tool. */
export const idle: {
  click(x: number, y: number): void | Promise<void>;
  doubleClick(x: number, y: number): void;
  escape(): void;
} = {
  click: () => {},
  doubleClick: (x, y) => {
    const hit = viewer.pick(x, y);
    if (hit) viewer.centerOn(hit.position);
  },
  escape: () => {},
};

export function activeTool(): Tool | null {
  return active;
}

/** Switch to `tool` (null for none), leaving the current one. */
export function setTool(tool: Tool | null): void {
  if (active === tool) return;
  viewer.cancelToolDrag();
  const previous = active;
  active = tool;
  previous?.exit();
  tool?.enter?.();
  document.getElementById("viewport")!.classList.toggle("measuring", tool !== null);
}

/** Switch `tool` on, or off if it is active. */
export function toggleTool(tool: Tool): void {
  setTool(active === tool ? null : tool);
}

/** The point under the cursor with its exact coordinates, or null. */
export async function pickPoint(x: number, y: number): Promise<PickedPoint | null> {
  const hit = viewer.pick(x, y);
  if (!hit) return null;
  try {
    const exact = await pointAt(hit.cloudId, hit.index);
    return { cloudId: hit.cloudId, index: hit.index, render: hit.position, exact };
  } catch {
    return null; // the cloud was removed while we were asking
  }
}

viewer.onClick = (x, y) => void (active ?? idle).click(x, y);
viewer.onToolPointerDown = (x, y) => active?.pointerDown?.(x, y) ?? false;
viewer.onToolPointerMove = (x, y) => active?.pointerMove?.(x, y);
viewer.onToolPointerUp = (x, y) => void active?.pointerUp?.(x, y);
viewer.onToolPointerCancel = () => active?.pointerCancel?.();
viewer.onDoubleClick = (x, y) => {
  if (active) active.doubleClick?.(x, y);
  else idle.doubleClick(x, y);
};

/** Single-key shortcuts that toggle tools, e.g. "m" for measuring. */
const shortcuts = new Map<string, Tool>();
export function shortcut(key: string, tool: Tool): void {
  shortcuts.set(key.toLowerCase(), tool);
}

window.addEventListener("keydown", (e) => {
  if (typing(e)) return;
  if (active?.key?.(e)) return;
  const tool = shortcuts.get(e.key.toLowerCase());
  if (tool && !e.ctrlKey && !e.metaKey && !e.altKey) {
    toggleTool(tool);
  } else if (e.key === "Escape") {
    if (active) setTool(null);
    else idle.escape();
  }
});
