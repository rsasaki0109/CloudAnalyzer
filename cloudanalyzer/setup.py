"""Setup shim — metadata is in pyproject.toml."""
from setuptools import setup, find_packages

setup(
    name="cloudanalyzer",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "open3d>=0.17.0",
        "numpy>=1.24.0",
        "typer>=0.9.0",
        "matplotlib>=3.7.0",
        "PyYAML>=6.0",
    ],
    entry_points={
        "console_scripts": [
            "ca=cli.main:main",
        ],
    },
)
