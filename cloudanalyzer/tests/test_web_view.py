"""`ca web-view`: results served from this machine for the web app, opened with one link."""

import urllib.error
import urllib.parse
import urllib.request

import pytest

from ca.web_view import Viewer, viewable_files


def _results(tmp_path):
    out = tmp_path / "fixed"
    out.mkdir()
    (out / "map.ply").write_bytes(b"ply\n" + bytes(range(256)) * 8)
    (out / "run.tum").write_text("0 0 0 0 0 0 0 1\n")
    (out / "poses_kitti.txt").write_text("1 0 0 0 0 1 0 0 0 0 1 0\n")  # the app would take .txt for a cloud
    (out / "report.json").write_text("{}")
    (out / "lanelet2_map.osm").write_text("<osm version='0.6'/>")
    return out


def test_the_viewable_files_of_a_folder(tmp_path):
    out = _results(tmp_path)
    assert [f.name for f in viewable_files([str(out)])] == ["lanelet2_map.osm", "map.ply", "run.tum"]
    assert [f.name for f in viewable_files([str(out / "report.json")])] == ["report.json"]
    with pytest.raises(FileNotFoundError):
        viewable_files([str(tmp_path / "missing")])
    with pytest.raises(ValueError, match="nothing to show"):
        viewable_files([str(tmp_path / "fixed" / "..")])


def test_the_link_opens_the_served_files(tmp_path):
    out = _results(tmp_path)
    viewer = Viewer([str(out)], app="https://example.org/app/").serve_in_background()
    try:
        link = urllib.parse.urlparse(viewer.link)
        assert viewer.link.startswith("https://example.org/app/?url=")
        urls = urllib.parse.parse_qs(link.query)["url"]
        assert [u.rsplit("/", 1)[1] for u in urls] == ["lanelet2_map.osm", "map.ply", "run.tum"]
        assert all(u.startswith(f"http://127.0.0.1:{viewer.port}/") for u in urls)
        size = (out / "map.ply").stat().st_size
        # A range, as the app asks for a file's first bytes.
        r = urllib.request.urlopen(urllib.request.Request(urls[1], headers={"Range": "bytes=0-9"}))
        assert r.status == 206 and r.read() == (out / "map.ply").read_bytes()[:10]
        assert r.headers["Content-Range"] == f"bytes 0-9/{size}"
        assert r.headers["Access-Control-Allow-Origin"] == "*"
        # The whole file, and the preflight a page elsewhere sends first.
        assert urllib.request.urlopen(urls[1]).read() == (out / "map.ply").read_bytes()
        assert urllib.request.urlopen(urls[0]).read() == (out / "lanelet2_map.osm").read_bytes()
        pre = urllib.request.urlopen(urllib.request.Request(urls[0], method="OPTIONS"))
        assert pre.status == 204 and pre.headers["Access-Control-Allow-Private-Network"] == "true"
        # Only the files named are served.
        with pytest.raises(urllib.error.HTTPError) as err:
            urllib.request.urlopen(f"http://127.0.0.1:{viewer.port}/0/report.json")
        assert err.value.code == 404
    finally:
        viewer.close()
