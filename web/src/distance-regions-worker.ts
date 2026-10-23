import { inspectDistanceCells, locateDistanceRegions } from "./distance-regions";
self.onmessage = event => {
  try {
    const { positions, distances, shift, threshold, cell, minPoints, inspection } = event.data;
    self.postMessage({ result: inspection ? inspectDistanceCells(positions, distances, shift, threshold, cell, minPoints, inspection.rank, inspection.cell) :
      locateDistanceRegions(positions, distances, shift, threshold, cell, minPoints) });
  } catch (error) { self.postMessage({ error: error instanceof Error ? error.message : String(error) }); }
};
