// Entry point: each module under app/ wires up its own panel when imported.
import "./app/tasks";
import "./app/layout";
import "./app/display";
import "./app/picking";
import "./app/distance";
import "./app/icp";
import "./app/classes";
import "./app/volume";
import "./app/processing";
import "./app/clip";
import "./app/profile";
import "./app/image";
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
