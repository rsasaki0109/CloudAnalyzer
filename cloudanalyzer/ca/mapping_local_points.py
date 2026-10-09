"""Explicit XY-column replacement with exact retention of outside PLY records."""
from __future__ import annotations

import hashlib
import json
import math
import re
from pathlib import Path
from typing import Any, cast

import numpy as np

from ca import mapping_job as jobs, mapping_retry as retries, mapping_frames as frames
from ca.mapping_patch import _checks

PROTOCOL: dict[str, Any] = {"model": "full_fusion_candidate_spliced_into_explicit_xy_column",
    "maximum_side_m": 20., "maximum_points_per_map": 2_000_000, "maximum_map_bytes": 128_000_000,
    "boundary": "lower_inclusive_upper_exclusive", "all_heights": True,
    "outside_record_order_and_bytes_preserved": True, "independent_accuracy_established": False}


def records(path: Path) -> tuple[bytes, np.ndarray]:
    """Read the canonical scalar binary PLY written by posegraph_fix.write_ply."""
    if path.stat().st_size > PROTOCOL["maximum_map_bytes"]:
        raise ValueError("local updates require bounded canonical binary PLY maps")
    with path.open('rb') as stream:
        lines: list[bytes] = []
        while sum(map(len, lines)) < 16_384:
            line = stream.readline(16_384)
            lines.append(line)
            if line == b'end_header\n': break
            if not line: raise ValueError("local updates require canonical binary PLY")
        else:
            raise ValueError("local PLY header exceeds its bound")
        if lines[:2] != [b'ply\n', b'format binary_little_endian 1.0\n'] or lines[3:6] != [
                b'property double x\n', b'property double y\n', b'property double z\n']:
            raise ValueError("local updates require canonical binary PLY with double XYZ")
        match = re.fullmatch(rb'element vertex ([0-9]+)\n', lines[2])
        if match is None: raise ValueError("local updates require one vertex element")
        count = int(match[1])
        if not 1 <= count <= PROTOCOL['maximum_points_per_map']:
            raise ValueError("local update point count exceeds its bound")
        fields = []
        for line in lines[6:-1]:
            match = re.fullmatch(rb'property float ([A-Za-z_][A-Za-z_0-9]*)\n', line)
            if match is None: raise ValueError("unsupported local PLY attribute schema")
            fields.append(match[1].decode('ascii'))
        if len(fields) > 16 or len(set(fields + ['x','y','z'])) != len(fields) + 3:
            raise ValueError("invalid local PLY attribute schema")
        dtype = np.dtype([('x','<f8'), ('y','<f8'), ('z','<f8'), *[(k,'<f4') for k in fields]])
        header = b''.join(lines)
        if path.stat().st_size != len(header) + count * dtype.itemsize:
            raise ValueError("local PLY records do not match the header")
        rows = np.fromfile(stream, dtype=dtype, count=count)
    if any(not np.isfinite(rows[k]).all() for k in rows.dtype.names or ()):
        raise ValueError("local point maps require finite coordinates and attributes")
    return header, rows


def mask(rows: np.ndarray, box: list[float]) -> np.ndarray:
    return (rows['x'] >= box[0]) & (rows['y'] >= box[1]) & (rows['x'] < box[2]) & (rows['y'] < box[3])


def inside_geometry(points: Any, box: list[float]) -> bool:
    xy = np.asarray(points, dtype=float)[:, :2]
    return bool(np.isfinite(xy).all() and np.all(xy >= box[:2]) and np.all(xy <= box[2:]))


def _box(values: Any, voxel: float) -> list[float]:
    if (not isinstance(values, list) or len(values) != 4 or any(type(v) not in (int, float)
            or not math.isfinite(v) or abs(v) > 1e6 for v in values)):
        raise ValueError("bounds_xy needs four finite [xmin,ymin,xmax,ymax] coordinates")
    if not math.isfinite(voxel) or voxel <= 0:
        raise ValueError("local updates need a fixed positive map voxel size")
    if any(values[i] >= values[i+2] for i in (0,1)):
        raise ValueError("local update bounds must have positive requested sides")
    def cell(v: float, upper: bool) -> int:
        scaled = v / voxel
        if math.isclose(scaled, round(scaled), abs_tol=1e-8, rel_tol=0): return round(scaled)
        return math.ceil(scaled) if upper else math.floor(scaled)
    box = [cell(v, i >= 2) * voxel for i,v in enumerate(values)]
    if any(not 0 < box[i+2] - box[i] <= PROTOCOL['maximum_side_m'] + 1e-8 for i in (0,1)):
        raise ValueError("local update box must have positive sides of at most 20 m after voxel alignment")
    return box


def _read(artifact: dict[str, Any]) -> dict[str, Any]:
    jobs._verify(artifact)
    return cast(dict[str, Any], json.loads(Path(artifact['path']).read_text()))


def validate_region(gap_evidence: dict[str, Any], cid: int, gap_ids: Any, bounds_xy: Any) -> list[float]:
    gaps = _read(gap_evidence); retries._verify_inputs(gaps['inputs'])
    if (gaps['candidate_id'] != cid or not isinstance(gap_ids, list) or not 1 <= len(gap_ids) <= 8
        or any(type(i) is not int for i in gap_ids) or len(set(gap_ids)) != len(gap_ids)
        or not set(gap_ids) <= {g['id'] for g in gaps['gaps']}):
        raise ValueError("choose 1..8 distinct inspected gap IDs for a local point update")
    box = _box(bounds_xy, gaps['pointcloud_options']['map_voxel_m'])
    for gap in (g for g in gaps['gaps'] if g['id'] in gap_ids):
        if not any(inside_geometry([p['trajectory']], box) for p in gap['profiles']
                   if gap['from_m'] <= p['station_m'] <= gap['to_m']):
            raise ValueError("each selected gap needs an inspected trajectory profile inside the update box")
    return box


def preview(root: Path, cid: int, gap_evidence: dict[str, Any], frame_evidence: dict[str, Any],
            gap_ids: Any, bounds_xy: Any) -> dict[str, Any]:
    with jobs._locked(root):
        job = jobs._load(root); jobs._inputs(job)
        retries._parent(job, cid)
        gaps, saved = _read(gap_evidence), _read(frame_evidence)
        retries._verify_inputs(gaps['inputs']); retries._verify_inputs(saved['inputs'])
        if gaps['candidate_id'] != cid or saved['candidate_id'] != cid or saved['inputs']['gap_evidence'] != gap_evidence:
            raise ValueError("local point preview needs matching gap and unused-frame observations")
        if saved['protocol'] != frames.PROTOCOL:
            raise ValueError("unused-frame consistency protocol changed")
        box = validate_region(gap_evidence, cid, gap_ids, bounds_xy)
        inputs = {**saved['inputs'], 'frame_evidence': frame_evidence, 'gap_evidence': gap_evidence}
        request = {'candidate_id': cid, 'gap_ids': gap_ids, 'bounds_xy': bounds_xy}
        suffix = hashlib.sha256(json.dumps(request, sort_keys=True).encode()).hexdigest()[:16]
        path = root / f'local-points-{cid:02d}-{suffix}.json'
        if path.exists():
            report = _read(jobs._artifact(path))
            if report['inputs'] != inputs or report['request'] != request or report['protocol'] != PROTOCOL:
                raise ValueError("local point preview inputs or protocol changed")
        else:
            _, base = records(Path(inputs['pointcloud_map']['path']))
            selected = mask(base, box)
            holds = ['no_outside_baseline_points'] if selected.all() else []
            eligible = []
            module = jobs.core(); assert module is not None
            for row in saved['frames']:
                if not row['eligible'] or not any(g['gap_id'] in gap_ids for g in row['gap_observations']): continue
                jobs._verify(row['raw_frame'])
                xyz = frames._align(np.asarray(module.read(row['raw_frame']['path'])['positions']), np.asarray(row['pose_hypothesis']))
                count = int(np.count_nonzero((xyz[:,0] >= box[0]) & (xyz[:,1] >= box[1]) & (xyz[:,0] < box[2]) & (xyz[:,1] < box[3])))
                if count >= frames.PROTOCOL['minimum_gap_returns']:
                    eligible.append({'frame_id': row['frame_id'], 'raw_returns_inside': count})
            if not eligible: holds.append('no_eligible_unused_frames_observe_box')
            report = {'schema': 'cloudanalyzer.local_point_preview.v1', 'request': request, 'inputs': inputs,
                'protocol': PROTOCOL, 'effective_bounds_xy': box, 'pointcloud_options': saved['pointcloud_options'],
                'inside_points': int(selected.sum()), 'outside_points': int((~selected).sum()),
                'outside_records_sha256': hashlib.sha256(base[~selected].tobytes()).hexdigest(),
                'record_dtype': base.dtype.descr, 'eligible_frames': eligible, 'holds': holds,
                'note': 'Explicit XY column at all heights, voxel-aligned outwards. The full fusion candidate is still generated; only its inside records replace baseline inside records. Outside record bytes and relative order remain fixed, including every scalar attribute. No local speedup or independent accuracy is established. Retained HD geometry and four source audits must not regress.'}
            jobs._inputs(job); retries._verify_inputs(inputs); jobs._save(path, report)
        retries._verify_inputs(report['inputs'])
        return {'file': jobs._artifact(path), 'candidate_id': cid, 'gap_ids': gap_ids,
            'requested_bounds_xy': bounds_xy, 'effective_bounds_xy': report['effective_bounds_xy'],
            'inside_points': report['inside_points'], 'outside_points': report['outside_points'],
            'outside_records_sha256': report['outside_records_sha256'],
            'eligible_frame_ids': [r['frame_id'] for r in report['eligible_frames']],
            'holds': report['holds'],
            'protocol': report['protocol'], 'note': report['note']}


def validate(evidence: dict[str, Any], frame_evidence: dict[str, Any], cid: int, frame_ids: Any) -> dict[str, Any]:
    report = _read(evidence); retries._verify_inputs(report['inputs'])
    frames.validate(frame_evidence, frame_ids)
    if report['holds']:
        raise ValueError("local point preview is held: " + ', '.join(report['holds']))
    if report['protocol'] != PROTOCOL or report['request']['candidate_id'] != cid or report['inputs']['frame_evidence'] != frame_evidence:
        raise ValueError("local point preview belongs to another decision or protocol")
    if not set(frame_ids) <= {r['frame_id'] for r in report['eligible_frames']}:
        raise ValueError("choose inspected eligible frames observing the local box")
    return report


def bounds(job: dict[str, Any]) -> list[float] | None:
    point = job.get('pointcloud') or {}
    if 'local_update_report' not in point.get('files', {}): return None
    report = _read(point['files']['local_update_report'])
    if not report['passes'] or report['updated_map'] != point['files']['map']:
        raise ValueError("local update report does not describe the accepted point map")
    return cast(list[float], report['effective_bounds_xy'])


def apply(child: Path, parent: dict[str, Any], preview_file: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    """Keep failed update evidence; let the caller decide whether the child is ready."""
    preview = _read(preview_file); retries._verify_inputs(preview['inputs'])
    base_file = preview['inputs']['pointcloud_map']
    trial_file = jobs._artifact(result['outputs']['map'])
    header, base = records(Path(base_file['path'])); _, trial = records(Path(trial_file['path']))
    if base.dtype != trial.dtype:
        raise ValueError("local candidate changed the point attribute schema")
    box = preview['effective_bounds_xy']; a, b = mask(base, box), mask(trial, box)
    if not b.any(): raise ValueError("local fusion candidate has no inside points")
    if int(a.sum()) != preview['inside_points'] or hashlib.sha256(base[~a].tobytes()).hexdigest() != preview['outside_records_sha256']:
        raise ValueError("local baseline records changed after preview")
    merged = np.concatenate((base[~a], trial[b]))
    if len(merged) > PROTOCOL['maximum_points_per_map']:
        raise ValueError("local merged point count exceeds its bound")
    output = child / 'local_map.ply'
    header = re.sub(rb'element vertex [0-9]+\n', f'element vertex {len(merged)}\n'.encode(), header, count=1)
    with output.open('wb') as stream:
        stream.write(header); stream.write(merged.tobytes())
    _, reopened = records(output)
    outside = reopened[~mask(reopened, box)]
    if outside.dtype != base.dtype or outside.tobytes() != base[~a].tobytes():
        raise ValueError("local point export changed outside coordinates, attributes or record order")
    module = jobs.core(); assert module is not None
    audits = {key: json.loads(module.audit_vector_map_quality_details(str(output), parent['files'][fkey]['path']))
              for key,fkey in (('editable','editable_map'),('reopened_osm','map'))}
    audits['ground_consensus'] = {key: json.loads(module.audit_vector_map_ground_consensus_details(str(output), parent['files'][fkey]['path']))
                                for key,fkey in (('editable','editable_map'),('reopened_osm','map'))}
    audits_path, checks_path = child/'local-point-audits.json', child/'local-point-checks.json'
    jobs._save(audits_path, audits)
    before = _read(parent['quality_report'])
    old_ir = _read(parent['files']['editable_map'])
    checks = _checks(before, audits, {l['id'] for l in old_ir['lanes']}, set())
    jobs._save(checks_path, checks)
    report = {'schema': 'cloudanalyzer.local_point_update.v1', 'protocol': PROTOCOL, 'preview': preview_file,
        'baseline_map': base_file, 'full_fusion_candidate': trial_file, 'updated_map': jobs._artifact(output),
        'effective_bounds_xy': box, 'before_inside_points': int(a.sum()), 'after_inside_points': int(b.sum()),
        'outside_points': len(outside), 'outside_records_sha256': hashlib.sha256(outside.tobytes()).hexdigest(),
        'outside_records_bit_identical': True, 'record_dtype': base.dtype.descr,
        'baseline_hd_files': parent['files'], 'baseline_hd_audits': parent['quality_report'],
        'retained_hd_source_audits': jobs._artifact(audits_path), 'retained_hd_source_checks': jobs._artifact(checks_path),
        'passes': checks['passes'], 'holds': checks['holds'], 'point_map_total': len(merged),
        'full_candidate_generation_still_required': True, 'automatic_adoption': False}
    report_path = child/'local-point-update.json'; jobs._save(report_path, report)
    retries._verify_inputs(preview['inputs']); jobs._verify(preview_file); jobs._verify(trial_file)
    return {'report': jobs._artifact(report_path), 'checks': jobs._artifact(checks_path), 'audits': jobs._artifact(audits_path),
        'map': report['updated_map'], 'full_fusion_candidate': trial_file,
        'passes': checks['passes'], 'holds': checks['holds'], 'effective_bounds_xy': box,
        'outside_records_bit_identical': True, 'outside_points': len(outside), 'map_points': len(merged)}
