"""Audit the frozen real-data comparison without running external estimators."""
import importlib.util
from pathlib import Path


def test_saved_nclt_baseline_scores_and_identities():
    root = Path(__file__).resolve().parents[2]
    path = root/'benchmarks/slam/nclt-odometry-baselines/verify.py'
    spec = importlib.util.spec_from_file_location('nclt_baselines_verifier', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    module.verify()
