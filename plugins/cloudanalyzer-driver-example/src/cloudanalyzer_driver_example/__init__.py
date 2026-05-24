"""Example third-party SLAM driver for ``ca slam-run``.

This package exists to demonstrate the
``cloudanalyzer.slam_run_drivers`` entry-point contract on a real,
pip-installable, MIT-licensed package. It is intentionally *not* shipped
inside the cloudanalyzer monorepo's ``ca.experiments.slam_run/`` slice —
the whole point is to prove the plugin pathway works for code that does
not live in CloudAnalyzer's own import graph.
"""

from cloudanalyzer_driver_example.driver import Open3DICPSlamDriver

__all__ = ["Open3DICPSlamDriver"]
__version__ = "0.1.0"
