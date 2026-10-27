"""Tests for the public benchmark pack builder."""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPTS_ROOT = REPO_ROOT / "scripts"
if str(SCRIPTS_ROOT) not in sys.path:
    sys.path.insert(0, str(SCRIPTS_ROOT))

import build_public_benchmark_pack as builder  # noqa: E402
from build_public_benchmark_pack import build_public_benchmark_pack  # noqa: E402
from ca.core import load_check_suite, run_check_suite  # noqa: E402


@pytest.mark.parametrize('failure', ['timeout', 'http503', 'truncated', 'reset'])
def test_bunny_download_retries_transient_failures_without_partial_data(tmp_path, monkeypatch, failure):
    import io
    import urllib.error
    import public_benchmark_assets as assets
    class Response(io.BytesIO):
        headers = {'Content-Length':'8'}
    calls, delays = [], []
    def open_url(url, *, timeout):
        calls.append((url, timeout))
        if len(calls) == 1:
            if failure == 'timeout':
                raise urllib.error.URLError(TimeoutError('fixture timeout'))
            if failure == 'http503':
                raise urllib.error.HTTPError(url, 503, 'unavailable', {}, None)
            if failure == 'reset':
                raise ConnectionResetError('fixture connection reset')
            return Response(b'partial')
        return Response(b'complete')
    monkeypatch.setattr(assets.urllib.request, 'urlopen', open_url)
    monkeypatch.setattr(assets.time, 'sleep', delays.append)
    path = tmp_path/'bunny.tar.gz'
    assets._download_bunny_archive(path)
    assert path.read_bytes() == b'complete'
    assert calls == [(assets.BUNNY_ARCHIVE_URL, 30)] * 2
    assert delays == [1]


def test_bunny_download_exhaustion_removes_partial_archive(tmp_path, monkeypatch):
    import io
    import public_benchmark_assets as assets
    class Response(io.BytesIO):
        headers = {'Content-Length':'100'}
    calls, delays = [], []
    def open_url(url, *, timeout):
        calls.append(timeout)
        return Response(b'partial')
    monkeypatch.setattr(assets.urllib.request, 'urlopen', open_url)
    monkeypatch.setattr(assets.time, 'sleep', delays.append)
    path = tmp_path/'bunny.tar.gz'
    with pytest.raises(RuntimeError, match='after 3 attempts'):
        assets._download_bunny_archive(path)
    assert calls == [30, 30, 30]
    assert delays == [1, 2]
    assert not path.exists()


def test_bunny_http404_is_not_retried(tmp_path, monkeypatch):
    import urllib.error
    import public_benchmark_assets as assets
    def open_url(url, *, timeout):
        raise urllib.error.HTTPError(url, 404, 'missing', {}, None)
    monkeypatch.setattr(assets.urllib.request, 'urlopen', open_url)
    monkeypatch.setattr(assets.time, 'sleep', lambda _: pytest.fail('permanent error retried'))
    with pytest.raises(urllib.error.HTTPError) as error:
        assets._download_bunny_archive(tmp_path/'bunny.tar.gz')
    assert error.value.code == 404


def test_bunny_local_write_failure_is_not_retried(tmp_path, monkeypatch):
    import io
    import public_benchmark_assets as assets
    def fail_copy(*args):
        raise PermissionError('fixture disk error')
    monkeypatch.setattr(assets.urllib.request, 'urlopen', lambda *a, **k: io.BytesIO(b'data'))
    monkeypatch.setattr(assets.shutil, 'copyfileobj', fail_copy)
    monkeypatch.setattr(assets.time, 'sleep', lambda _: pytest.fail('local failure retried'))
    with pytest.raises(PermissionError, match='disk error'):
        assets._download_bunny_archive(tmp_path/'bunny.tar.gz')


def test_bunny_mesh_download_uses_retrying_transport(tmp_path, monkeypatch):
    import io
    import tarfile
    import public_benchmark_assets as assets
    mesh_path = tmp_path/'fixture.ply'
    assets.o3d.io.write_triangle_mesh(str(mesh_path), assets.o3d.geometry.TriangleMesh.create_box())
    archive_bytes = io.BytesIO()
    with tarfile.open(fileobj=archive_bytes, mode='w:gz') as archive:
        archive.add(mesh_path, arcname='bunny/reconstruction/bun_zipper.ply')
    payload = archive_bytes.getvalue()
    class Response(io.BytesIO):
        headers = {'Content-Length':str(len(payload))}
    calls = []
    def open_url(url, *, timeout):
        calls.append((url, timeout))
        return Response(payload)
    monkeypatch.setattr(assets.urllib.request, 'urlopen', open_url)
    mesh = assets.download_bunny_mesh()
    assert len(mesh.vertices) == 8
    assert calls == [(assets.BUNNY_ARCHIVE_URL, 30)]


def _bunny_archive(path, members):
    import io
    import tarfile
    with tarfile.open(path, 'w:gz') as archive:
        for name, kind, payload in members:
            info = tarfile.TarInfo(name)
            info.type = kind
            if kind == tarfile.REGTYPE:
                info.size = len(payload)
                archive.addfile(info, io.BytesIO(payload))
            else:
                info.linkname = 'target.ply'
                archive.addfile(info)


def test_bunny_extraction_ignores_archive_paths_and_preserves_mesh_bytes(tmp_path):
    import tarfile
    import public_benchmark_assets as assets
    archive = tmp_path/'input.tar.gz'
    output_dir = tmp_path/'output'; output_dir.mkdir()
    outside = tmp_path/'outside'; outside.write_bytes(b'original')
    _bunny_archive(archive, [
        ('../outside', tarfile.REGTYPE, b'overwrite'),
        (str(outside), tarfile.REGTYPE, b'overwrite'),
        ('unused-link', tarfile.SYMTYPE, b''),
        (assets.BUNNY_MESH_MEMBER, tarfile.REGTYPE, b'exact mesh bytes'),
    ])
    target = output_dir/'mesh.ply'
    assets._extract_bunny_mesh(archive, target)
    assert target.read_bytes() == b'exact mesh bytes'
    assert outside.read_bytes() == b'original'
    assert list(output_dir.iterdir()) == [target]


@pytest.mark.parametrize('case', ['missing', 'duplicate', 'symlink', 'hardlink', 'directory', 'empty', 'oversize', 'entries'])
def test_bunny_extraction_rejects_invalid_mesh_without_output(tmp_path, monkeypatch, case):
    import tarfile
    import public_benchmark_assets as assets
    entry = (assets.BUNNY_MESH_MEMBER, tarfile.REGTYPE, b'mesh')
    members = [entry]
    if case == 'missing':
        members = [('other.ply', tarfile.REGTYPE, b'mesh')]
    elif case == 'duplicate':
        members = [entry, entry]
    elif case in ('symlink', 'hardlink', 'directory'):
        kind = {'symlink':tarfile.SYMTYPE, 'hardlink':tarfile.LNKTYPE, 'directory':tarfile.DIRTYPE}[case]
        members = [(assets.BUNNY_MESH_MEMBER, kind, b'')]
    elif case == 'empty':
        members = [(assets.BUNNY_MESH_MEMBER, tarfile.REGTYPE, b'')]
    elif case == 'oversize':
        monkeypatch.setattr(assets, 'MAX_BUNNY_MESH_BYTES', 3)
    else:
        monkeypatch.setattr(assets, 'MAX_BUNNY_ARCHIVE_ENTRIES', 1)
        members += [('other.ply', tarfile.REGTYPE, b'other')]
    archive = tmp_path/'input.tar.gz'
    _bunny_archive(archive, members)
    target = tmp_path/'mesh.ply'
    with pytest.raises(RuntimeError, match='Stanford Bunny'):
        assets._extract_bunny_mesh(archive, target)
    assert not target.exists()


def _synthetic_source_points() -> np.ndarray:
    """Create a deterministic nontrivial 3D shape for offline tests."""
    angles = np.linspace(0.0, 2.0 * np.pi, 80, endpoint=False)
    heights = np.linspace(-0.6, 0.6, 18)
    points: list[list[float]] = []
    for z in heights:
        radius = 0.75 - (0.2 * abs(z))
        for angle in angles:
            wobble = 1.0 + (0.08 * np.sin(angle * 3.0))
            x = radius * wobble * np.cos(angle)
            y = radius * wobble * np.sin(angle)
            points.append([x, y, z + (0.04 * np.sin(angle * 2.0))])
    return np.asarray(points, dtype=float)


@pytest.mark.parametrize('verify_digest', [False, True])
def test_local_bunny_archive_builds_offline_and_records_exact_identity(tmp_path, monkeypatch, verify_digest):
    import hashlib
    import tarfile
    import public_benchmark_assets as assets
    mesh_path = tmp_path/'mesh.ply'
    assets.o3d.io.write_triangle_mesh(str(mesh_path), assets.o3d.geometry.TriangleMesh.create_sphere(resolution=20))
    archive = tmp_path/'local.tar.gz'
    _bunny_archive(archive, [(assets.BUNNY_MESH_MEMBER, tarfile.REGTYPE, mesh_path.read_bytes())])
    raw = archive.read_bytes()
    def forbid(*args, **kwargs):
        pytest.fail('local archive mode attempted a download')
    monkeypatch.setattr(builder, 'download_bunny_mesh', forbid)
    monkeypatch.setattr(urllib.request, 'urlopen', forbid)
    output = tmp_path/'pack'
    digest = hashlib.sha256(raw).hexdigest()
    result = build_public_benchmark_pack(output, bunny_archive=archive,
                                         bunny_sha256=digest.upper() if verify_digest else None)
    assert result['source'] == 'local-bunny-archive'
    manifest = json.loads((output/'manifest.json').read_text())
    expected_identity = {'sha256':digest, 'bytes':len(raw), 'origin':'local', 'authenticity_verified':False}
    if verify_digest:
        expected_identity['expected_sha256'] = digest
    assert manifest['source_archive'] == expected_identity
    assert archive.read_bytes() == raw
    assert 'authenticity' in (output/'ATTRIBUTION.md').read_text()
    assert '--bunny-archive' in (output/'README.md').read_text()
    for entry in manifest['configs']:
        rerun = run_check_suite(load_check_suite(str(output/entry['config_path'])))
        assert rerun['summary']['passed'] == entry['expected_pass']


@pytest.mark.parametrize('case', ['inside_output', 'synthetic', 'custom', 'missing', 'symlink', 'oversize', 'invalid'])
def test_invalid_local_archive_preserves_existing_output(tmp_path, monkeypatch, case):
    import public_benchmark_assets as assets
    output = tmp_path/'pack'; output.mkdir()
    sentinel = output/'keep'; sentinel.write_text('keep')
    archive = tmp_path/'local.tar.gz'; archive.write_bytes(b'invalid archive')
    options = {}
    if case == 'inside_output':
        archive = sentinel
    elif case == 'synthetic':
        options['fixture'] = 'synthetic'
    elif case == 'custom':
        options['source_points'] = _synthetic_source_points()
    elif case == 'missing':
        archive = tmp_path/'missing'
    elif case == 'symlink':
        link = tmp_path/'link'; link.symlink_to(archive); archive = link
    elif case == 'oversize':
        monkeypatch.setattr(assets, 'MAX_LOCAL_BUNNY_ARCHIVE_BYTES', 1)
    import tarfile
    with pytest.raises((ValueError, tarfile.TarError)):
        build_public_benchmark_pack(output, bunny_archive=archive, **options)
    assert sentinel.read_text() == 'keep'


def test_cli_forwards_local_archive_and_rejects_synthetic(monkeypatch):
    calls = []
    def capture(output_dir, **options):
        assert output_dir == Path('benchmarks/public/local-bunny')
        calls.append(options)
        return {'output_dir':str(output_dir), 'pack_name':'test', 'source':'local-bunny-archive', 'manifest':{'configs':[]}}
    monkeypatch.setattr(builder, 'build_public_benchmark_pack', capture)
    monkeypatch.setattr(sys, 'argv', ['build', '--bunny-archive', 'local.tar.gz'])
    builder.main()
    assert calls == [{'fixture':'stanford-bunny', 'bunny_archive':Path('local.tar.gz')}]
    monkeypatch.setattr(sys, 'argv', ['build', '--fixture', 'synthetic', '--bunny-archive', 'local.tar.gz'])
    with pytest.raises(SystemExit) as error:
        builder.main()
    assert error.value.code == 2
    assert len(calls) == 1


@pytest.mark.parametrize('digest', ['0'*64, '', 'g'*64, 'a'*63, 'a'*65])
def test_bunny_digest_failure_precedes_extraction_and_output_changes(tmp_path, monkeypatch, digest):
    import public_benchmark_assets as assets
    archive = tmp_path/'input.tar.gz'; archive.write_bytes(b'not even an archive')
    output = tmp_path/'pack'; output.mkdir()
    sentinel = output/'keep'; sentinel.write_text('keep')
    monkeypatch.setattr(assets, '_extract_bunny_mesh', lambda *args: pytest.fail('unverified archive extracted'))
    with pytest.raises(ValueError, match='SHA-256'):
        build_public_benchmark_pack(output, bunny_archive=archive, bunny_sha256=digest)
    assert sentinel.read_text() == 'keep'


def test_digest_without_archive_never_downloads(tmp_path, monkeypatch):
    monkeypatch.setattr(builder, 'download_bunny_mesh', lambda: pytest.fail('unexpected download'))
    with pytest.raises(ValueError, match='requires bunny_archive'):
        build_public_benchmark_pack(tmp_path/'output', bunny_sha256='a'*64)
    assert not (tmp_path/'output').exists()
    monkeypatch.setattr(sys, 'argv', ['build', '--bunny-sha256', 'a'*64])
    with pytest.raises(SystemExit) as error:
        builder.main()
    assert error.value.code == 2


def test_cli_forwards_expected_bunny_digest(monkeypatch):
    calls = []
    def capture(output_dir, **options):
        calls.append(options)
        return {'output_dir':str(output_dir), 'pack_name':'test', 'source':'local-bunny-archive', 'manifest':{'configs':[]}}
    monkeypatch.setattr(builder, 'build_public_benchmark_pack', capture)
    monkeypatch.setattr(sys, 'argv', ['build', '--bunny-archive', 'local.tar.gz', '--bunny-sha256', 'a'*64])
    builder.main()
    assert calls == [{'fixture':'stanford-bunny', 'bunny_archive':Path('local.tar.gz'), 'bunny_sha256':'a'*64}]


class TestPublicBenchmarkPack:
    def test_synthetic_builder_runs_offline_and_detects_regressions(self, tmp_path: Path, monkeypatch):
        output_dir = tmp_path / "public-pack"

        def forbid_download(*args, **kwargs):
            pytest.fail("Synthetic fixtures must not download geometry")

        monkeypatch.setattr(builder, "download_bunny_mesh", forbid_download)
        monkeypatch.setattr(urllib.request, "urlretrieve", forbid_download)
        monkeypatch.setattr(urllib.request, "urlopen", forbid_download)
        result = build_public_benchmark_pack(
            output_dir,
            fixture="synthetic",
        )

        manifest_path = output_dir / "manifest.json"
        assert manifest_path.exists()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

        assert result["output_dir"] == str(output_dir.resolve())
        assert result["source"] == manifest["source"] == "synthetic-smoke"
        assert manifest["pack"] == "public-benchmark-pack"
        assert [item["id"] for item in manifest["configs"]] == [
            "suite-pass",
            "suite-regression",
        ]

        pass_entry = manifest["configs"][0]
        regression_entry = manifest["configs"][1]
        assert pass_entry["expected_pass"] is True
        assert regression_entry["expected_pass"] is False
        assert pass_entry["failed_check_ids"] == []
        assert set(regression_entry["failed_check_ids"]) == {
            "mapping-postprocess",
            "localization-run",
            "perception-output",
            "integrated-run",
        }

        assert (output_dir / "README.md").exists()
        assert (output_dir / "ATTRIBUTION.md").exists()
        assert (output_dir / "baselines" / "mapping_ref.pcd").exists()
        assert (output_dir / "outputs" / "mapping_pass.pcd").exists()
        assert (output_dir / "outputs" / "mapping_fail.pcd").exists()
        assert (output_dir / "expected" / "suite-pass.summary.json").exists()
        assert (output_dir / "expected" / "suite-regression.summary.json").exists()
        assert (output_dir / "reports" / "pass" / "mapping-postprocess.html").exists()
        assert (output_dir / "reports" / "regression" / "integrated-run.html").exists()
        assert (output_dir / "results" / "pass" / "integrated-run.json").exists()
        for name in ["README.md", "ATTRIBUTION.md"]:
            text = (output_dir / name).read_text(encoding="utf-8")
            assert "procedural surface" in text
            assert "Stanford" not in text
        assert "--fixture synthetic" in (output_dir / "README.md").read_text(encoding="utf-8")

    def test_generated_configs_match_manifest_expectations(self, tmp_path: Path):
        output_dir = tmp_path / "public-pack"
        build_public_benchmark_pack(
            output_dir,
            source_points=_synthetic_source_points(),
            seed=19,
        )
        manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["source"] == "custom"
        for name in ["README.md", "ATTRIBUTION.md"]:
            assert "Stanford" not in (output_dir / name).read_text(encoding="utf-8")

        for item in manifest["configs"]:
            config_path = output_dir / item["config_path"]
            rerun = run_check_suite(load_check_suite(str(config_path)))
            assert rerun["summary"]["passed"] is item["expected_pass"]
            assert rerun["summary"]["failed_check_ids"] == item["failed_check_ids"]


def test_synthetic_source_is_small_finite_and_deterministic():
    points = builder._synthetic_smoke_points()
    assert points.shape == (1440, 3)
    assert np.isfinite(points).all()
    assert (np.ptp(points, axis=0) > 1.0).all()
    np.testing.assert_array_equal(points, builder._synthetic_smoke_points())


def test_default_download_failure_is_not_replaced_by_synthetic(tmp_path: Path, monkeypatch):
    output_dir = tmp_path / "existing"
    output_dir.mkdir()
    sentinel = output_dir / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")

    def fail_download():
        raise TimeoutError("Stanford unavailable")

    monkeypatch.setattr(builder, "download_bunny_mesh", fail_download)
    with pytest.raises(TimeoutError, match="Stanford unavailable"):
        build_public_benchmark_pack(output_dir)
    assert sentinel.read_text(encoding="utf-8") == "keep"


def test_default_fixture_retains_stanford_download_and_attribution(tmp_path: Path, monkeypatch):
    downloads = []

    def fake_download():
        downloads.append(True)
        return SimpleNamespace(vertices=_synthetic_source_points())

    monkeypatch.setattr(builder, "download_bunny_mesh", fake_download)
    result = build_public_benchmark_pack(tmp_path / "stanford-pack")
    assert downloads == [True]
    assert result["source"] == result["manifest"]["source"] == "stanford-bunny"
    for name in ["README.md", "ATTRIBUTION.md"]:
        text = (tmp_path / "stanford-pack" / name).read_text(encoding="utf-8")
        assert "Stanford Bunny" in text
        assert builder.BUNNY_SOURCE_PAGE in text
        assert builder.BUNNY_ARCHIVE_URL in text
        assert "synthetic" not in text


@pytest.mark.parametrize("fixture,custom", [("invalid", False), ("synthetic", True)])
def test_invalid_source_selection_preserves_existing_output(tmp_path: Path, fixture, custom):
    sentinel = tmp_path / "keep.txt"
    sentinel.write_text("keep", encoding="utf-8")
    with pytest.raises(ValueError):
        build_public_benchmark_pack(
            tmp_path, fixture=fixture,
            source_points=_synthetic_source_points() if custom else None,
        )
    assert sentinel.read_text(encoding="utf-8") == "keep"


@pytest.mark.parametrize(
    "arguments,fixture,output",
    [
        ([], "stanford-bunny", "benchmarks/public/stanford-bunny-mini"),
        (["--fixture", "synthetic"], "synthetic", "benchmarks/public/self-qa-synthetic"),
        (["--fixture", "synthetic", "--output", "my-pack"], "synthetic", "my-pack"),
    ],
)
def test_cli_selects_fixture_and_separate_default_output(monkeypatch, capsys, arguments, fixture, output):
    calls = []

    def capture_build(output_dir, *, fixture):
        calls.append((output_dir, fixture))
        return {"output_dir": str(output_dir), "pack_name": "test", "source": fixture, "manifest": {"configs": []}}

    monkeypatch.setattr(builder, "build_public_benchmark_pack", capture_build)
    monkeypatch.setattr(sys, "argv", ["build_public_benchmark_pack.py", *arguments])
    builder.main()
    assert calls == [(Path(output), fixture)]
    assert f"Source:       {fixture}" in capsys.readouterr().out


def test_self_qa_uses_offline_fixture_with_matching_config():
    workflow = yaml.safe_load((REPO_ROOT / ".github/workflows/self-qa.yml").read_text(encoding="utf-8"))
    steps = workflow["jobs"]["qa"]["steps"]
    build_step = next(step for step in steps if "build_public_benchmark_pack.py" in step.get("run", ""))
    action_step = next(step for step in steps if step.get("uses") == "./cloudanalyzer-action")
    assert "--fixture synthetic" in build_step["run"]
    assert "--output benchmarks/public/self-qa-synthetic" in build_step["run"]
    assert action_step["with"]["config"] == "benchmarks/public/self-qa-synthetic/configs/suite-pass.cloudanalyzer.yaml"
