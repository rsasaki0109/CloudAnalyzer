"""Accumulator allocation stays bounded by occupied stream cells."""

import numpy as np
import pytest

import ca.core.map_evaluate as maps
from ca.io import PointAccumulator


@pytest.mark.parametrize('repetitions', [1, 2, 20])
def test_repeated_cells_allocate_once_and_keep_update_identity(repetitions, monkeypatch):
    created, updates = [], []
    constructor, update = PointAccumulator.__init__, PointAccumulator.update

    def traced_constructor(self, *args, **kwargs):
        constructor(self, *args, **kwargs)
        created.append(self)

    def traced_update(self, values):
        updates.append((self, values.copy()))
        return update(self, values)

    monkeypatch.setattr(PointAccumulator, '__init__', traced_constructor)
    monkeypatch.setattr(PointAccumulator, 'update', traced_update)
    points = np.array([[1.1, 0., 0.], [.2, 0., 0.], [1.2, 0., 0.], [.1, 0., 0.]])

    def chunks():
        for _ in range(repetitions):
            yield np.empty((0, 3))
            yield points

    voxels = maps.build_voxel_gaussians_from_chunks(chunks(), 1.)
    assert list(voxels) == [(0, 0, 0), (1, 0, 0)]
    assert len(created) == len(voxels) == 2
    assert len(updates) == 2 * repetitions
    for index, (accumulator, batch) in enumerate(updates):
        assert accumulator is created[index % 2]
        np.testing.assert_array_equal(batch, points[[1, 3] if index % 2 == 0 else [0, 2]])
    assert all(voxel.num_points == 2 * repetitions for voxel in voxels.values())


def test_zero_count_cell_keeps_its_original_accumulator(monkeypatch):
    created, updated = [], []
    constructor, update = PointAccumulator.__init__, PointAccumulator.update

    def traced_constructor(self, *args, **kwargs):
        constructor(self, *args, **kwargs)
        created.append(self)

    def traced_update(self, values):
        updated.append(self)
        return update(self, values)

    monkeypatch.setattr(PointAccumulator, '__init__', traced_constructor)
    monkeypatch.setattr(PointAccumulator, 'update', traced_update)
    with np.errstate(invalid='ignore'):
        voxels = maps.build_voxel_gaussians_from_chunks([np.array([[np.nan, 0., 0.]])] * 3, 1.)
    assert len(created) == len(voxels) == 1
    assert len(updated) == 3 and all(value is created[0] for value in updated)
    assert next(iter(voxels.values())).num_points == 0
