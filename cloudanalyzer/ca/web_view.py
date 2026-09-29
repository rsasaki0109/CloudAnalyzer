"""Open results in the CloudAnalyzer web app: `ca web-view` and the MCP tool ``view_link``.

The web app runs in the browser and cannot read local files by itself, so the files are
served from this machine, on 127.0.0.1 only and only the files named, with the headers a
page on another origin needs to fetch them (CORS and private network access), and the
app is opened with one ``?url=`` per file.
"""

from __future__ import annotations

import http.server
import socket
import threading
import urllib.parse
from pathlib import Path

APP_URL = "https://rsasaki0109.github.io/CloudAnalyzer/app/"
# What the app opens from a link: clouds and meshes, and TUM trajectories. KITTI pose text
# (.txt) is left out: the app would take it for a point cloud.
VIEWABLE = {".ply", ".pcd", ".las", ".laz", ".e57", ".xyz", ".obj", ".stl", ".tum", ".splat"}


def viewable_files(paths: list[str]) -> list[Path]:
    """The files to show: each file given, and the viewable files in each folder given."""
    out: list[Path] = []
    for raw in paths:
        p = Path(raw)
        if p.is_dir():
            out += sorted(f for f in p.iterdir() if f.is_file() and f.suffix.lower() in VIEWABLE)
        elif p.is_file():
            out.append(p)
        else:
            raise FileNotFoundError(raw)
    if not out:
        raise ValueError("nothing to show: no .ply, .pcd, .las/.laz, .tum ... among " + ", ".join(paths))
    return out


class _Handler(http.server.BaseHTTPRequestHandler):
    files: dict[str, Path] = {}

    def log_message(self, format: str, *args) -> None:  # noqa: A002 - quiet
        pass

    def _headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "Range")
        self.send_header("Access-Control-Expose-Headers", "Content-Range, Content-Length")
        self.send_header("Access-Control-Allow-Private-Network", "true")
        self.send_header("Cache-Control", "no-store")

    def do_OPTIONS(self) -> None:  # noqa: N802 - http.server's naming
        self.send_response(204)
        self._headers()
        self.end_headers()

    def do_GET(self) -> None:  # noqa: N802
        path = self.files.get(urllib.parse.unquote(self.path.split("?")[0]))
        if path is None or not path.is_file():
            self.send_response(404)
            self._headers()
            self.end_headers()
            return
        size = path.stat().st_size
        start, end = 0, size - 1
        ranged = self.headers.get("Range", "")
        if ranged.startswith("bytes="):
            first, _, last = ranged[6:].partition("-")
            if first.isdigit():
                start = int(first)
                end = min(int(last), size - 1) if last.isdigit() else size - 1
        partial = bool(ranged) and (start, end) != (0, size - 1)
        self.send_response(206 if partial else 200)
        self._headers()
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(end - start + 1))
        self.send_header("Accept-Ranges", "bytes")
        if partial:
            self.send_header("Content-Range", f"bytes {start}-{end}/{size}")
        self.end_headers()
        with open(path, "rb") as f:
            f.seek(start)
            left = end - start + 1
            while left > 0:
                chunk = f.read(min(1 << 20, left))
                if not chunk:
                    break
                self.wfile.write(chunk)
                left -= len(chunk)


class Viewer:
    """A server for some files and the web app link that opens them."""

    def __init__(self, paths: list[str], port: int = 0, app: str = APP_URL):
        files = viewable_files(paths)
        names: dict[str, Path] = {}
        for k, f in enumerate(files):
            # A short unique path per file keeps its name (the app names the cloud after it).
            names[f"/{k}/{urllib.parse.quote(f.name)}"] = f.resolve()
        handler = type("Handler", (_Handler,), {"files": {urllib.parse.unquote(k): v for k, v in names.items()}})
        self.server = http.server.ThreadingHTTPServer(("127.0.0.1", port), handler)
        self.port = self.server.server_address[1]
        base = f"http://127.0.0.1:{self.port}"
        query = "&".join("url=" + urllib.parse.quote(base + k, safe="") for k in names)
        self.link = f"{app}?{query}"
        self.files = files

    def serve_in_background(self) -> "Viewer":
        threading.Thread(target=self.server.serve_forever, daemon=True).start()
        return self

    def serve_forever(self) -> None:
        self.server.serve_forever()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])
