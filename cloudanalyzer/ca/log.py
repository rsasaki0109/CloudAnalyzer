"""Logging configuration for CloudAnalyzer."""

import logging
import sys


logger = logging.getLogger("ca")


def setup_logging(verbose: bool = False, quiet: bool = False) -> None:
    """Configure logging level and format.

    Args:
        verbose: Enable DEBUG level logging.
        quiet: Suppress all non-error output.
    """
    if quiet:
        level = logging.ERROR
    elif verbose:
        level = logging.DEBUG
    else:
        level = logging.INFO

    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.handlers.clear()
    logger.addHandler(handler)
    logger.setLevel(level)
