"""Tests for baseline history management."""

import json

import pytest

from ca.baseline_history import discover_history, list_baselines, rotate_history, save_baseline


def _write_summary(path, passed=True, project="test"):
    data = {
        "config_path": str(path),
        "project": project,
        "summary": {"passed": passed, "failed_check_ids": []},
        "checks": [],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return str(path)


class TestDiscoverHistory:
    def test_empty_dir(self, tmp_path):
        assert discover_history(str(tmp_path)) == []

    def test_nonexistent_dir(self, tmp_path):
        assert discover_history(str(tmp_path / "nope")) == []

    def test_finds_json_files_sorted(self, tmp_path):
        _write_summary(tmp_path / "baseline-20260301.json")
        _write_summary(tmp_path / "baseline-20260315.json")
        _write_summary(tmp_path / "baseline-20260310.json")

        result = discover_history(str(tmp_path))

        assert len(result) == 3
        assert "20260301" in result[0]
        assert "20260310" in result[1]
        assert "20260315" in result[2]

    def test_skips_invalid_json(self, tmp_path):
        _write_summary(tmp_path / "good.json")
        (tmp_path / "bad.json").write_text("not json", encoding="utf-8")

        result = discover_history(str(tmp_path))
        assert len(result) == 1


class TestSaveBaseline:
    def test_saves_with_timestamp(self, tmp_path):
        summary = _write_summary(tmp_path / "summary.json")
        history_dir = tmp_path / "history"

        dest = save_baseline(summary, str(history_dir))

        assert history_dir.exists()
        assert "baseline-" in dest
        assert json.loads(open(dest).read())["project"] == "test"

    def test_saves_with_custom_label(self, tmp_path):
        summary = _write_summary(tmp_path / "summary.json")
        history_dir = tmp_path / "history"

        dest = save_baseline(summary, str(history_dir), label="v1.0")

        assert "baseline-v1.0.json" in dest

    def test_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            save_baseline(str(tmp_path / "nope.json"), str(tmp_path / "history"))


class TestRotateHistory:
    def test_keeps_newest(self, tmp_path):
        for i in range(5):
            _write_summary(tmp_path / f"baseline-{i:04d}.json")

        removed = rotate_history(str(tmp_path), keep=3)

        assert len(removed) == 2
        remaining = discover_history(str(tmp_path))
        assert len(remaining) == 3
        assert "0002" in remaining[0]

    def test_noop_when_under_limit(self, tmp_path):
        _write_summary(tmp_path / "baseline-0001.json")

        removed = rotate_history(str(tmp_path), keep=5)
        assert removed == []

    def test_rejects_keep_zero(self, tmp_path):
        with pytest.raises(ValueError, match="keep must be"):
            rotate_history(str(tmp_path), keep=0)


class TestListBaselines:
    def test_lists_with_metadata(self, tmp_path):
        _write_summary(tmp_path / "baseline-a.json", passed=True)
        _write_summary(tmp_path / "baseline-b.json", passed=False)

        entries = list_baselines(str(tmp_path))

        assert len(entries) == 2
        assert entries[0]["passed"] is True
        assert entries[1]["passed"] is False
        assert entries[0]["name"] == "baseline-a.json"
