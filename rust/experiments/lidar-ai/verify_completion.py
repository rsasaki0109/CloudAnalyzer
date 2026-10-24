"""Independent stdlib check of a fixed Rust near-surface completion result.

Optional annotations score the already frozen masks only. They never guide the
completion rule. The ordinary evaluate_removal CLI remains nearest-only.
"""
from array import array
import argparse
from collections import defaultdict
import json
import math
from pathlib import Path
import struct
import sys

from evaluate_removal import (CLASSES, MAX_POINTS, bounded_read, check, evaluate,
                              projected_owners, score_annotations, sha, validate_native_result)

RADIUS = .15
SEPARATION = .02
SEED = struct.unpack('<f', struct.pack('<f', .99))[0]
CONFLICT = struct.unpack('<f', struct.pack('<f', .9))[0]
LIMIT = 256


def project_scan(scan: bytes):
    """Independent nearest-owner projection shared by the scalar checks."""
    check(bool(scan) and len(scan) % 16 == 0 and len(scan) <= MAX_POINTS * 16, 'invalid single-scan XYZI records')
    points = list(struct.iter_unpack('<4f', scan))
    owners, pixels, ranges = [-1] * 131072, [-1] * len(points), [math.inf] * 131072
    low, up = math.radians(-25), math.radians(3)
    for i, (x,y,z,intensity) in enumerate(points):
        if not all(map(math.isfinite, (x,y,z,intensity))): continue
        check(0 <= intensity <= 1, 'invalid source remission')
        r = math.hypot(math.hypot(x,y),z)
        if not 1e-6 <= r <= 100: continue
        pitch = math.asin(z/r)
        if not low <= pitch <= up: continue
        col = min(2047, math.floor(.5 * (-math.atan2(y,x)/math.pi+1) * 2048))
        row = min(63, math.floor((up-pitch)/(up-low)*64))
        p = row*2048+col; pixels[i]=p
        if r < ranges[p]: owners[p],ranges[p]=i,r
    return points, owners, pixels, ranges


def reference_input(scan: bytes) -> bytes:
    """Rebuild the native little-endian f32 input, including f32 rounding steps."""
    points, owners, _, ranges = project_scan(scan)
    f32 = lambda value: struct.unpack('<f', struct.pack('<f', value))[0]
    mean = tuple(map(f32, (12.12, 10.88, .23, -1.04, .21)))
    std = tuple(map(f32, (12.32, 11.47, 6.91, .86, .16)))
    values = array('f', [0]) * (5 * 131072)
    for pixel, owner in enumerate(owners):
        if owner < 0: continue
        for channel, value in enumerate((f32(ranges[pixel]), *points[owner])):
            values[channel * 131072 + pixel] = f32(f32(value - mean[channel]) / std[channel])
    if sys.byteorder != 'little': values.byteswap()
    return values.tobytes()


def reference_completion(scan: bytes, tensor: bytes, baseline: bytes, targets=(1,), threshold=.9, *, class_counts=None) -> tuple[bytes, dict]:
    """Scalar independent projection/grid oracle, including the finite work cap."""
    check(type(threshold) in (int, float) and math.isfinite(threshold) and 0 <= threshold <= 1, 'invalid baseline threshold')
    check(bool(targets) and all(type(c) is int and 0 < c < len(CLASSES) for c in targets), 'invalid target classes')
    points, owners, pixels, _ = project_scan(scan)
    threshold = struct.unpack('<f', struct.pack('<f', threshold))[0]
    check(len(baseline) == len(points), 'source/mask budget mismatch')
    check(len(tensor) == 20 * 64 * 2048 * 4, 'invalid probability tensor shape')
    values = array('f'); values.frombytes(tensor)
    if sys.byteorder != 'little': values.byteswap()
    classes, scores = {}, {}
    grid = defaultdict(list)
    cell = lambda p: tuple(math.floor(v/RADIUS) for v in p[:3])
    check(all(v in (0,1) for v in baseline), 'invalid baseline mask')
    for pix, i in enumerate(owners):
        probabilities = [values[c*131072+pix] for c in range(20)]
        check(all(math.isfinite(v) and 0 <= v <= 1 for v in probabilities) and abs(sum(probabilities)-1) <= .002, 'invalid softmax')
        if i < 0: continue
        c = max(range(20), key=probabilities.__getitem__)
        classes[i],scores[i]=c,probabilities[c]
        check(bool(baseline[i]) == (c in targets and scores[i] >= threshold), 'baseline mask does not match probabilities')
        if scores[i] >= CONFLICT: grid[cell(points[i])].append(i)
    check(all(not selected or i in classes for i,selected in enumerate(baseline)), 'baseline selects an unowned return')
    if class_counts is not None:
        actual_counts = [0] * len(CLASSES)
        for predicted in classes.values(): actual_counts[predicted] += 1
        check(class_counts == actual_counts, 'baseline class counts do not match probabilities')
    distance = lambda a,b: sum((float(a[k])-float(b[k]))**2 for k in range(3))
    result = bytearray(baseline)
    counts = dict(distance_rejected=0,crowded_rejected=0,conflict_rejected=0,support_rejected=0)
    for i, point in enumerate(points):
        pix = pixels[i]
        if pix < 0 or i in classes: continue
        first = owners[pix]
        if first < 0 or not baseline[first] or scores[first] < SEED: continue
        if distance(point,points[first]) > RADIUS**2: counts['distance_rejected']+=1;continue
        x,y,z=cell(point); neighbors=[];crowded=False
        for dx in range(-1,2):
            if crowded: break
            for dy in range(-1,2):
                if crowded: break
                for dz in range(-1,2):
                    bucket=grid.get((x+dx,y+dy,z+dz),[])
                    if len(neighbors)+len(bucket)>LIMIT: crowded=True;break
                    neighbors.extend(bucket)
        if crowded: counts['crowded_rejected']+=1;continue
        # Inspect all neighbors, even after finding support.
        close=[j for j in neighbors if distance(point,points[j]) <= RADIUS**2]
        if any(classes[j]!=classes[first] for j in close): counts['conflict_rejected']+=1;continue
        supported=any(j!=first and baseline[j] and scores[j]>=SEED and distance(points[first],points[j])>=SEPARATION**2 for j in close)
        if supported: result[i]=1
        else: counts['support_rejected']+=1
    counts.update(original_removed=sum(baseline), added=sum(result)-sum(baseline),total_removed=sum(result),retained=len(points)-sum(result))
    return bytes(result),counts


def verify(scan: bytes, base: Path, result: Path, labels: bytes | None = None, pairing: dict | None = None) -> dict:
    report_raw=bounded_read(base/'report.json',65536);report=json.loads(report_raw)
    check(isinstance(report, dict), 'native baseline report must be a JSON object')
    check(report['input_sha256']==sha(scan) and report['class_names']==list(CLASSES), 'native baseline source/classes differ')
    baseline=bounded_read(base/'candidate-mask.u8',MAX_POINTS)
    validate_native_result(scan, baseline, bounded_read(base/'retained.bin',MAX_POINTS*16),
                           bounded_read(base/'candidates.bin',MAX_POINTS*16), report)
    tensor=bounded_read(base/'probabilities.f32',20*131072*4)
    original_input=bounded_read(base/'input.f32',5*131072*4)
    check(original_input == reference_input(scan), 'baseline debug input does not match this raw scan')
    metadata=json.loads(bounded_read(result/'completion-report.json',65536))
    check(isinstance(metadata, dict) and isinstance(metadata.get('base'), dict), 'completion report/base must be JSON objects')
    check(metadata.get('app')=='CloudAnalyzer LiDAR surface completion experiment' and metadata.get('version')==1, 'unsupported completion report')
    check(metadata['input_sha256']==sha(scan) and metadata['source_modified'] is False, 'completion source differs')
    check(metadata['base']['report_sha256']==sha(report_raw) and metadata['base']['probabilities_sha256']==sha(tensor)
          and metadata['base']['input_tensor_sha256']==sha(original_input) and metadata['base']['candidate_mask_sha256']==sha(baseline), 'completion baseline hashes differ')
    check(all(metadata['base'][key] == report[key] for key in ('onnx_sha256', 'weights_sha256'))
          and all(metadata['base'][key] == report['selection'][key] for key in ('target_classes', 'threshold')),
          'completion model/selection differs from the native baseline')
    policy=metadata['policy']
    check(policy==dict(id='near-surface-two-seeds-v1',radius_m=RADIUS,min_seed_separation_m=SEPARATION,
                      seed_score=SEED,conflict_score=CONFLICT,max_neighborhood_owners=LIMIT), 'unsupported completion policy')
    targets=tuple(CLASSES.index(name) for name in report['selection']['target_classes'])
    expected,counts=reference_completion(scan,tensor,baseline,targets,report['selection']['threshold'],
                                        class_counts=report['selection']['class_counts'])
    actual=bounded_read(result/'completed-mask.u8',MAX_POINTS)
    check(actual==expected and metadata['counts']==counts, 'Rust/Python completion mask or rejection counts differ')
    added=bytes(a and not b for a,b in zip(actual,baseline))
    check(bounded_read(result/'added-mask.u8',MAX_POINTS)==added and metadata['added_mask_sha256']==sha(added), 'added mask differs')
    check(metadata['completed_mask_sha256']==sha(actual),'completed mask hash differs')
    kept=b''.join(scan[i*16:(i+1)*16] for i,v in enumerate(actual) if not v)
    removed=b''.join(scan[i*16:(i+1)*16] for i,v in enumerate(actual) if v)
    check(bounded_read(result/'retained.bin',MAX_POINTS*16)==kept and bounded_read(result/'candidates.bin',MAX_POINTS*16)==removed,'completion changes source records')
    check(metadata['retained_sha256']==sha(kept) and metadata['candidate_sha256']==sha(removed),'completion output hashes differ')
    output={'app':'CloudAnalyzer surface completion check','version':1,'independent_mask_and_counts_exact':True,
            'baseline_metadata_and_records_checked':True,'preprocessing_byte_exact':True,
            'source_record_split_exact':True,'baseline_report_sha256':sha(report_raw),'completion_report_sha256':sha(bounded_read(result/'completion-report.json',65536)),
            'counts':counts,'hashes':{'source':sha(scan),'probabilities':sha(tensor),'mask':sha(actual)}}
    if labels is not None:
        check(pairing is not None, 'explicit annotation pairing required')
        baseline_audit=evaluate(scan,labels,baseline,bounded_read(base/'retained.bin',MAX_POINTS*16),
                                 bounded_read(base/'candidates.bin',MAX_POINTS*16),report,pairing)
        classified,_=projected_owners(scan)
        completed=score_annotations(scan,labels,actual,kept,removed,classified,report['selection']['target_classes'],report['selection']['threshold'],pairing)
        output['baseline_audit']=baseline_audit;output['completed_audit']=completed
        delta={key:completed['full_scan'][key]-baseline_audit['full_scan'][key] for key in ['tp','fp','fn','tn']}
        delta['ignored_removed']=completed['ignored_removed']-baseline_audit['ignored_removed']
        output['delta']=delta
        output['diagnostic_gate_passed']=delta['tp']>0 and delta['fp']==0 and delta['ignored_removed']==0
        output['limits']=['Completed masks can exceed the counterfactual nearest-only recall ceiling; it remains labelled as that baseline ceiling',
                          'Small validation sample, not independent held-out model accuracy; no automatic adoption']
    return output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for key in ['scan','base','result','out']:parser.add_argument('--'+key,type=Path,required=True)
    parser.add_argument('--labels',type=Path);parser.add_argument('--pairing',type=Path)
    args=parser.parse_args()
    try:
        check(not args.out.exists(),'output must be a new JSON file')
        check(bool(args.labels)==bool(args.pairing),'use --labels and --pairing together')
        record=verify(bounded_read(args.scan,MAX_POINTS*16),args.base,args.result,
                      bounded_read(args.labels,MAX_POINTS*4) if args.labels else None,
                      json.loads(bounded_read(args.pairing,65536)) if args.pairing else None)
        with args.out.open('x',encoding='utf-8',newline='\n') as file:file.write(json.dumps(record,indent=2,allow_nan=False)+'\n')
        print(json.dumps({k:v for k,v in record.items() if k not in ['baseline_audit','completed_audit']},allow_nan=False))
    except (OSError,ValueError,KeyError,TypeError) as error:parser.exit(1,f'Completion check failed: {error}\n')


if __name__=='__main__':main()
