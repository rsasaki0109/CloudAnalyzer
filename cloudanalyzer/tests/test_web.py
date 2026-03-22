"""Tests for ca.web module."""

import json
import threading
import time
from urllib.request import urlopen

from ca.web import serve, _make_handler, _VIEWER_HTML


class TestWebHandler:
    def test_serves_html(self, sample_pcd_file):
        data_json = json.dumps({"positions": [0, 0, 0], "filename": "test.pcd"})
        handler_cls = _make_handler(_VIEWER_HTML, data_json)

        from http.server import HTTPServer
        server = HTTPServer(('127.0.0.1', 0), handler_cls)
        port = server.server_address[1]
        t = threading.Thread(target=server.handle_request, daemon=True)
        t.start()

        resp = urlopen(f"http://127.0.0.1:{port}/")
        html = resp.read().decode()
        assert "CloudAnalyzer" in html
        server.server_close()

    def test_serves_data(self, sample_pcd_file):
        data_json = json.dumps({"positions": [1.0, 2.0, 3.0], "filename": "test.pcd"})
        handler_cls = _make_handler(_VIEWER_HTML, data_json)

        from http.server import HTTPServer
        server = HTTPServer(('127.0.0.1', 0), handler_cls)
        port = server.server_address[1]
        t = threading.Thread(target=server.handle_request, daemon=True)
        t.start()

        resp = urlopen(f"http://127.0.0.1:{port}/data.json")
        data = json.loads(resp.read())
        assert data["positions"] == [1.0, 2.0, 3.0]
        server.server_close()
