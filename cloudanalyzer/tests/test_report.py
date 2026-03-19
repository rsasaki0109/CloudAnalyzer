"""Tests for ca.report module."""

import json

from ca.report import make_json, save_json, make_markdown


SAMPLE_STATS = {"mean": 0.05, "median": 0.04, "max": 0.12, "min": 0.001, "std": 0.02}


class TestMakeJson:
    def test_structure(self):
        data = make_json(1000, 2000, 0.95, 0.01, SAMPLE_STATS)
        assert data["source_points"] == 1000
        assert data["target_points"] == 2000
        assert data["fitness"] == 0.95
        assert data["rmse"] == 0.01
        assert data["distance_stats"] == SAMPLE_STATS

    def test_no_registration(self):
        data = make_json(100, 200, None, None, SAMPLE_STATS)
        assert data["fitness"] is None
        assert data["rmse"] is None


class TestSaveJson:
    def test_writes_valid_json(self, tmp_path):
        data = make_json(100, 200, 0.9, 0.02, SAMPLE_STATS)
        path = tmp_path / "out.json"
        save_json(data, str(path))
        loaded = json.loads(path.read_text())
        assert loaded == data

    def test_creates_parent_dirs(self, tmp_path):
        path = tmp_path / "sub" / "dir" / "out.json"
        save_json({"test": 1}, str(path))
        assert path.exists()


class TestMakeMarkdown:
    def test_contains_sections(self, tmp_path):
        data = make_json(100, 200, 0.95, 0.01, SAMPLE_STATS)
        path = tmp_path / "report.md"
        make_markdown(data, str(path))
        content = path.read_text()
        assert "# CloudAnalyzer Report" in content
        assert "## Registration" in content
        assert "## Distance Stats" in content
        assert "## Point Counts" in content
        assert "0.9500" in content  # fitness
        assert "Source: 100" in content

    def test_no_registration_section(self, tmp_path):
        data = make_json(100, 200, None, None, SAMPLE_STATS)
        path = tmp_path / "report.md"
        make_markdown(data, str(path))
        content = path.read_text()
        assert "## Registration" not in content
        assert "## Distance Stats" in content
