// Entry point: each module under app/ wires up its own panel when imported.
import "./app/tasks";
import "./app/memory";
import "./app/layout";
import "./app/display";
import "./app/picking";
import "./app/distance";
import "./app/align";
import "./app/icp";
import "./app/classes";
import "./app/volume";
import "./app/raster";
import "./app/mesh";
import "./app/processing";
import "./app/segment";
import "./app/clip";
import "./app/copc-box";
import "./app/profile";
import "./app/image";
import "./app/report";
import "./app/scalars";
import "./app/shapes";
import "./app/posegraph";
import "./app/trajectory";
import "./app/vectormap";
import "./app/mapping-review";
import { runDemo } from "./app/demos";
import { errorText, setStatus } from "./app/dom";
import { renderList } from "./app/entries";
import { loadUrls } from "./app/loading";
import { applySession } from "./app/session";
import { decodeSession } from "./session";

renderList();

/** `#session=…` restores a shared view, `?demo=…` runs a demo, `?url=…` (repeatable) opens clouds. */
async function startFromLink(): Promise<void> {
  const encoded = new URLSearchParams(location.hash.slice(1)).get("session");
  const params = new URLSearchParams(location.search);
  const urls = params.getAll("url");
  const demo = params.get("demo");
  try {
    if (encoded) await applySession(decodeSession(encoded));
    else if (demo) await runDemo(demo);
    else if (urls.length) await loadUrls(urls);
  } catch (err) {
    setStatus(`Link: ${errorText(err)}`, true);
  }
}
void startFromLink();
