"""Report generation (JSON, Markdown and HTML).

The builders live in ``ca.reports``, one module per report kind; this module
re-exports every name so ``from ca.report import ...`` keeps working.
"""

from ca.reports.common import (
    make_json,
    save_json,
    make_markdown,
    _gate_for_item,
    _html_command_block,
    _html_summary_action,
    _compression_for_item,
    _quality_vs_size_plot_path,
    _trajectory_overlay_plot_path,
    _trajectory_error_plot_path,
    _run_f1_plot_path,
    _format_optional_float,
)
from ca.reports.batch import (
    make_batch_summary,
    make_batch_markdown,
    make_batch_html,
    save_batch_report,
)
from ca.reports.ground import (
    make_ground_markdown,
    make_ground_html,
    save_ground_report,
)
from ca.reports.image import (
    _format_optional_metric,
    make_image_markdown,
    make_image_html,
    save_image_report,
    make_rendered_markdown,
    make_rendered_html,
    save_rendered_report,
)
from ca.reports.detection import (
    make_detection_markdown,
    make_detection_html,
    save_detection_report,
    make_tracking_markdown,
    make_tracking_html,
    save_tracking_report,
)
from ca.reports.trajectory import (
    _format_optional_ratio,
    _trajectory_gate_for_item,
    _TRAJECTORY_LOW_COVERAGE_THRESHOLD,
    make_trajectory_markdown,
    make_trajectory_html,
    save_trajectory_report,
)
from ca.reports.trajectory_batch import (
    make_trajectory_batch_summary,
    make_trajectory_batch_markdown,
    make_trajectory_batch_html,
    save_trajectory_batch_report,
)
from ca.reports.run import (
    make_run_markdown,
    make_run_html,
    save_run_report,
)
from ca.reports.run_batch import (
    _run_batch_gate_for_item,
    make_run_batch_summary,
    make_run_batch_markdown,
    make_run_batch_html,
    save_run_batch_report,
)

__all__ = [
    "make_json",
    "save_json",
    "make_markdown",
    "_gate_for_item",
    "_html_command_block",
    "_html_summary_action",
    "_compression_for_item",
    "_quality_vs_size_plot_path",
    "_trajectory_overlay_plot_path",
    "_trajectory_error_plot_path",
    "_run_f1_plot_path",
    "_format_optional_float",
    "make_batch_summary",
    "make_batch_markdown",
    "make_batch_html",
    "save_batch_report",
    "make_ground_markdown",
    "make_ground_html",
    "save_ground_report",
    "_format_optional_metric",
    "make_image_markdown",
    "make_image_html",
    "save_image_report",
    "make_rendered_markdown",
    "make_rendered_html",
    "save_rendered_report",
    "make_detection_markdown",
    "make_detection_html",
    "save_detection_report",
    "make_tracking_markdown",
    "make_tracking_html",
    "save_tracking_report",
    "_format_optional_ratio",
    "_trajectory_gate_for_item",
    "_TRAJECTORY_LOW_COVERAGE_THRESHOLD",
    "make_trajectory_markdown",
    "make_trajectory_html",
    "save_trajectory_report",
    "make_trajectory_batch_summary",
    "make_trajectory_batch_markdown",
    "make_trajectory_batch_html",
    "save_trajectory_batch_report",
    "make_run_markdown",
    "make_run_html",
    "save_run_report",
    "_run_batch_gate_for_item",
    "make_run_batch_summary",
    "make_run_batch_markdown",
    "make_run_batch_html",
    "save_run_batch_report",
]
