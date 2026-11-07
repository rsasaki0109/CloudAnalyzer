"""Bounded regular-file inventories used before Docker preparation."""
import os
import stat
from pathlib import Path
from types import SimpleNamespace

import pytest

from ca import glim_run as runner


def test_inventory_is_sorted_and_hashes_nested_files(tmp_path):
    (tmp_path/'z').mkdir()
    (tmp_path/'z'/'b').write_bytes(b'b')
    (tmp_path/'a').write_bytes(b'a')
    entries = runner.inventory(tmp_path)
    assert [entry['relative_path'] for entry in entries] == ['a', 'z/b']
    assert all(entry['bytes'] == 1 and len(entry['sha256']) == 64 for entry in entries)
    assert runner.inventory(tmp_path/'a')[0]['relative_path'] == 'a'


@pytest.mark.parametrize('special', ['fifo', 'symlink', 'directory_symlink'])
def test_rejects_special_files_before_hashing(tmp_path, monkeypatch, special):
    (tmp_path/'config.json').write_text('{}')
    item = tmp_path/'special'
    if special == 'fifo':
        os.mkfifo(item)
    else:
        item.symlink_to(tmp_path if special == 'directory_symlink' else tmp_path/'config.json')
    monkeypatch.setattr(runner, '_artifact', lambda *a: pytest.fail('must reject before hashing'))
    with pytest.raises(ValueError, match='symlink|regular files'):
        runner.inventory(tmp_path)
    with pytest.raises(ValueError, match='symlink|regular files'):
        runner.inventory(item)


def test_limit_counts_directories_and_stops_scanning(tmp_path, monkeypatch):
    monkeypatch.setattr(runner, 'MAX_INVENTORY_ENTRIES', 2)
    for name in ('a', 'b', 'c'):
        (tmp_path/name).mkdir()
    real_scandir = os.scandir
    opened = []
    def scan(path):
        opened.append(path)
        return real_scandir(path)
    monkeypatch.setattr(runner.os, 'scandir', scan)
    with pytest.raises(ValueError, match='exceeds 2'):
        runner.inventory(tmp_path)
    assert opened == [tmp_path]


def test_scan_error_is_not_silently_omitted(tmp_path, monkeypatch):
    def inaccessible(path):
        raise PermissionError('unreadable input directory')
    monkeypatch.setattr(runner.os, 'scandir', inaccessible)
    with pytest.raises(PermissionError, match='unreadable'):
        runner.inventory(tmp_path)


def test_fifo_config_rejected_before_docker_or_output_creation(tmp_path, monkeypatch):
    bag = tmp_path/'input.mcap'; bag.write_bytes(b'fixture')
    config = tmp_path/'settings'; config.mkdir()
    (config/'config.json').write_text('{}')
    os.mkfifo(config/'pipe')
    monkeypatch.setattr(runner.subprocess, 'check_output', lambda *a, **k: pytest.fail('Docker called'))
    out = tmp_path/'run'
    with pytest.raises(ValueError, match='regular files'):
        runner.run_glim(str(bag), str(config), str(out), 'test:local')
    assert not out.exists()


@pytest.mark.parametrize('mode', [stat.S_IFSOCK, stat.S_IFCHR, stat.S_IFBLK])
def test_rejects_nonregular_stat_types(tmp_path, monkeypatch, mode):
    monkeypatch.setattr(Path, 'lstat', lambda self: SimpleNamespace(st_mode=mode))
    with pytest.raises(ValueError, match='regular files'):
        runner.inventory(tmp_path)
