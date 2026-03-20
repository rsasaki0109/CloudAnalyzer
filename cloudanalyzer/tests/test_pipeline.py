"""Tests for ca.pipeline module."""

from ca.pipeline import run_pipeline


class TestPipeline:
    def test_basic(self, source_and_target_files, tmp_path):
        src, tgt = source_and_target_files
        output = str(tmp_path / "pipeline_out.pcd")
        result = run_pipeline(src, tgt, output, voxel_size=0.1)
        assert result["filter"]["removed"] >= 0
        assert result["downsample"]["output"] > 0
        assert result["evaluation"]["chamfer"] >= 0
        assert 0 <= result["evaluation"]["auc"] <= 1.0

    def test_custom_params(self, source_and_target_files, tmp_path):
        src, tgt = source_and_target_files
        output = str(tmp_path / "pipeline_out.pcd")
        result = run_pipeline(
            src, tgt, output,
            voxel_size=0.05, nb_neighbors=10, std_ratio=3.0,
            thresholds=[0.01, 0.1, 0.5],
        )
        assert len(result["evaluation"]["f1_scores"]) == 3
