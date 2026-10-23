import { locateDistanceRegions } from "./distance-regions";
self.onmessage = event => {
  try {
    const { positions, distances, shift, threshold, cell, minPoints } = event.data;
    self.postMessage({ result: locateDistanceRegions(positions, distances, shift, threshold, cell, minPoints) });
  } catch (error) { self.postMessage({ error: error instanceof Error ? error.message : String(error) }); }
};
