import { analyzeCoverage } from "./map-coverage";
self.onmessage = (event: MessageEvent) => {
  try {
    const { positions, shift, cell, minPoints } = event.data;
    const grid = analyzeCoverage(positions, shift, cell, minPoints);
    self.postMessage({ grid }, { transfer: [grid.counts.buffer, grid.categories.buffer] });
  } catch (error) {
    self.postMessage({ error: error instanceof Error ? error.message : String(error) });
  }
};
