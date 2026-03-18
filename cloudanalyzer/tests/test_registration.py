"""Tests for ca.registration module."""

import numpy as np
import pytest

from ca.registration import register, SUPPORTED_METHODS


class TestRegister:
    def test_gicp_returns_tuple(self, simple_pcd, shifted_pcd):
        transformed, fitness, rmse = register(simple_pcd, shifted_pcd, method="gicp")
        assert 0.0 <= fitness <= 1.0
        assert rmse >= 0.0
        assert len(transformed.points) == len(simple_pcd.points)

    def test_icp_returns_tuple(self, simple_pcd, shifted_pcd):
        transformed, fitness, rmse = register(simple_pcd, shifted_pcd, method="icp")
        assert 0.0 <= fitness <= 1.0
        assert rmse >= 0.0

    def test_identical_clouds_high_fitness(self, simple_pcd, identical_pcd):
        _, fitness, rmse = register(simple_pcd, identical_pcd, method="gicp")
        assert fitness > 0.8
        assert rmse < 0.1

    def test_unsupported_method(self, simple_pcd, shifted_pcd):
        with pytest.raises(ValueError, match="Unsupported method"):
            register(simple_pcd, shifted_pcd, method="ndt")

    def test_supported_methods(self):
        assert SUPPORTED_METHODS == {"icp", "gicp"}

    def test_case_insensitive(self, simple_pcd, shifted_pcd):
        transformed, fitness, rmse = register(simple_pcd, shifted_pcd, method="GICP")
        assert 0.0 <= fitness <= 1.0
