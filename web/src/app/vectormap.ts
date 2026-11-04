/**
 * Vector map panel: roads drawn over the clouds become lanes (both
 * directions, shared boundaries, connected pieces), joined through
 * junctions with turning lanes and given stop lines, traffic lights and
 * crosswalks; the map is checked for Autoware and saved as
 * lanelet2_map.osm with map_projector_info.yaml. The map lives in the
 * worker (vectormap-rs); this module draws its view and turns clicks into
 * vectormap commands.
 *
 * The panel is split into the modules below. Their top-level code runs in
 * this order, as it did when they were one file, because the modules import
 * each other's names from here rather than directly, and this module loads
 * the panel's dependencies first, in the order the single file imported them.
 */

import "../memory-budget";
import "../project-change";
import "../lane-review";
import "three";
import "three/examples/jsm/lines/LineSegments2.js";
import "three/examples/jsm/lines/LineSegmentsGeometry.js";
import "../api";
import "../selection-queue";
import "./entries";
import "./colors";
import "./history";
import "./dom";
import "./state";
import "./trajectory";
import "./tools";
import "./vectormap-geometry";
import "./road-build-failure";
import "../lane-surface-overlap";

export * from "./vectormap-types";
export * from "./vectormap-scene";
export * from "./vectormap-evidence";
export * from "./vectormap-junction";
export * from "./vectormap-signal";
export * from "./vectormap-crosswalk";
export * from "./vectormap-discovery";
export * from "./vectormap-build";
export * from "./vectormap-reviews";
export * from "./vectormap-edits";
export * from "./vectormap-quality";
export * from "./vectormap-lanes";
export * from "./vectormap-road";
export * from "./vectormap-relations";
export * from "./vectormap-features";
export * from "./vectormap-boundary";
export * from "./vectormap-files";
