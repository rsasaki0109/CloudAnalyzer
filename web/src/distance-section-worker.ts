import { compareSections } from "./distance-section";
self.onmessage = event => {
  try {
    const { source, reference, settings } = event.data, result = compareSections(source, reference, settings);
    self.postMessage({ result }, { transfer: [result.source.along.buffer, result.source.positions.buffer, result.reference.along.buffer, result.reference.positions.buffer] });
  } catch (error) { self.postMessage({ error: error instanceof Error ? error.message : String(error) }); }
};
