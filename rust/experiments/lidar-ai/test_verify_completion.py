"""Synthetic geometry/accounting cases for the independent completion oracle."""
from array import array
import json
import math
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from evaluate_removal import projected_owners, sha
from verify_completion import (CONFLICT, LIMIT, RADIUS, SEED, SEPARATION,
                               reference_completion, reference_input, verify)
from test_evaluate_removal import fixture


def case(points, predictions):
    raw=b''.join(struct.pack('<4f',*p) for p in points)
    values=array('f',[0])*(20*131072)
    values[:131072]=array('f',[1])*131072
    owners={}
    for i,p in enumerate(points):
        x,y,z,_=p;r=math.hypot(math.hypot(x,y),z)
        row=min(63,math.floor((math.radians(3)-math.asin(z/r))/math.radians(28)*64))
        col=min(2047,math.floor(.5*(-math.atan2(y,x)/math.pi+1)*2048))
        pixel=row*2048+col
        if pixel not in owners or r<owners[pixel][1]:owners[pixel]=(i,r)
    baseline=bytearray(len(points))
    for pix,(i,_) in owners.items():
        cls,score=predictions.get(i,(0,1))
        if cls:
            values[pix]=1-score;values[cls*131072+pix]=score
            if cls==1 and score>=.9:baseline[i]=1
    if sys.byteorder!='little':values.byteswap()
    return raw,values.tobytes(),bytes(baseline)


class CompletionOracleTests(unittest.TestCase):
    def test_pair_adds_close_returns_without_propagating_a_chain(self):
        points=[(10,0,0,.5),(10.02,0,0,.5),(10,.05,0,.5),(10.14,0,0,.5),(10.27,0,0,.5)]
        raw,tensor,baseline=case(points,{0:(1,.999),2:(1,.999)})
        mask,counts=reference_completion(raw,tensor,baseline)
        self.assertEqual(mask,bytes([1,1,1,1,0]))
        self.assertEqual(counts['added'],2)

    def test_conflicting_owner_vetoes_and_baseline_owners_stay_selected(self):
        points=[(10,0,0,.5),(10.02,0,0,.5),(10,.05,0,.5),(10,-.06,0,.5)]
        raw,tensor,baseline=case(points,{0:(1,.999),2:(1,.999),3:(9,.95)})
        mask,counts=reference_completion(raw,tensor,baseline)
        self.assertEqual(mask,baseline);self.assertEqual(counts['conflict_rejected'],1)

    def test_invalid_empty_pixel_softmax_and_unowned_baseline_selection_reject(self):
        raw,tensor,baseline=case([(10,0,0,.5),(10.02,0,0,.5)],{0:(1,.999)})
        bad=bytearray(tensor);bad[:4]=struct.pack('<f',float('nan'))
        with self.assertRaisesRegex(ValueError,'softmax'):reference_completion(raw,bytes(bad),baseline)
        with self.assertRaisesRegex(ValueError,'unowned'):reference_completion(raw,tensor,bytes([1,1]))

    def test_f32_baseline_threshold_boundary_matches_the_native_contract(self):
        raw,tensor,baseline=case([(10,0,0,.5),(10.02,0,0,.5)],{0:(1,.9)})
        mask,counts=reference_completion(raw,tensor,baseline)
        self.assertEqual(mask,baseline);self.assertEqual(counts['added'],0)

    def test_invalid_scan_threshold_and_targets_reject_before_completion(self):
        raw,tensor,baseline=case([(10,0,0,.5)],{0:(1,.999)})
        for scan,targets,threshold in [(b'',(1,),.9),(raw[:-1],(1,),.9),
                                      (raw,(0,),.9),(raw,(20,),.9),(raw,(),.9),
                                      (raw,(1,),float('nan')),(raw,(1,),1.00000001)]:
            with self.subTest(targets=targets,threshold=threshold,bytes=len(scan)):
                with self.assertRaises(ValueError):
                    reference_completion(scan,tensor,baseline,targets,threshold)

    def test_frozen_completion_packet_preserves_both_failed_cases_and_separate_groups(self):
        from evaluate_removal import sha
        packet=Path(__file__).resolve().parents[3]/'benchmarks/lidar-ai/near-surface-completion'
        read=lambda name: json.loads((packet/name).read_text(encoding='utf-8'))
        protocol,frozen,summary,checks=read('protocol.json'),read('frozen-completion.json'),read('summary.json'),read('checks.json')
        self.assertEqual(sha((packet/'protocol.json').read_bytes()),frozen['protocol_sha256'])
        self.assertEqual(frozen['reference_inputs'],[]);self.assertEqual(len(checks),6)
        self.assertFalse(summary['diagnostic_gate_passed_all_frames'])
        self.assertEqual(summary['decision'],'experimental_only_not_adopted_as_default')
        expected={'development':protocol['development_frames'],'additional':protocol['frames']}
        for group in summary['groups']:
            rows=[r for r in checks if r['group']==group['group']]
            self.assertEqual(sorted(r['frame'] for r in rows),sorted(expected[group['group']]))
            for key in ['tp','fp','fn','tn','ignored_removed']:
                self.assertEqual(sum(r['check']['delta'][key] for r in rows),group['delta'][key])
            self.assertEqual(group['gate_passed_all_frames'],all(r['check']['diagnostic_gate_passed'] for r in rows))
        failed=[]
        for row in checks:
            result=row['check'];delta=result['delta'];before=result['baseline_audit'];after=result['completed_audit']
            self.assertTrue(result['independent_mask_and_counts_exact'] and result['source_record_split_exact'])
            for key in ['tp','fp','fn','tn']:
                self.assertEqual(delta[key],after['full_scan'][key]-before['full_scan'][key])
            self.assertEqual(delta['tp']+delta['fp']+delta['ignored_removed'],result['counts']['added'])
            passed=delta['tp']>0 and delta['fp']==0 and delta['ignored_removed']==0
            self.assertEqual(passed,result['diagnostic_gate_passed'])
            if not passed:failed.append(row['frame'])
        self.assertEqual(sorted(failed),['000000','001000'])


class CompletionVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.base = self.root/'base'; self.base.mkdir()
        self.result = self.root/'completion'; self.result.mkdir()
        points = [(10,0,0,.5),(10.02,0,0,.5),(10,.05,0,.5)]
        self.scan, self.tensor, self.baseline = case(points,{0:(1,.999),2:(1,.999)})
        native = fixture(points=points,raw_ids=[10]*3,mask=list(self.baseline))
        self.report = native['report']
        self.report['weights_sha256'] = '1'*64
        self.report['selection']['threshold'] = struct.unpack('<f',struct.pack('<f',.9))[0]
        self.input = reference_input(self.scan)
        self.baseline_kept, self.baseline_removed = native['retained'], native['candidates']
        self.mask, self.counts = reference_completion(self.scan,self.tensor,self.baseline)
        self.metadata = dict(app='CloudAnalyzer LiDAR surface completion experiment',version=1,source_modified=False,
            policy=dict(id='near-surface-two-seeds-v1',radius_m=RADIUS,min_seed_separation_m=SEPARATION,
                        seed_score=SEED,conflict_score=CONFLICT,max_neighborhood_owners=LIMIT))
        self.write_files()

    def write_files(self):
        report_raw = json.dumps(self.report).encode()
        for name, data in [('report.json',report_raw),('candidate-mask.u8',self.baseline),
                           ('input.f32',self.input),('probabilities.f32',self.tensor),
                           ('retained.bin',self.baseline_kept),('candidates.bin',self.baseline_removed)]:
            (self.base/name).write_bytes(data)
        added = bytes(a and not b for a,b in zip(self.mask,self.baseline))
        kept = b''.join(self.scan[i*16:i*16+16] for i,v in enumerate(self.mask) if not v)
        removed = b''.join(self.scan[i*16:i*16+16] for i,v in enumerate(self.mask) if v)
        self.metadata.update(input_sha256=sha(self.scan),counts=self.counts,
            completed_mask_sha256=sha(self.mask),added_mask_sha256=sha(added),
            retained_sha256=sha(kept),candidate_sha256=sha(removed),
            base=dict(report_sha256=sha(report_raw),probabilities_sha256=sha(self.tensor),
                input_tensor_sha256=sha(self.input),candidate_mask_sha256=sha(self.baseline),
                onnx_sha256=self.report['onnx_sha256'],weights_sha256=self.report['weights_sha256'],
                target_classes=self.report['selection']['target_classes'],threshold=self.report['selection']['threshold']))
        for name,data in [('completed-mask.u8',self.mask),('added-mask.u8',added),
                          ('retained.bin',kept),('candidates.bin',removed),
                          ('completion-report.json',json.dumps(self.metadata).encode())]:
            (self.result/name).write_bytes(data)

    def test_label_free_verification_checks_records_without_claiming_accuracy(self):
        result = verify(self.scan,self.base,self.result)
        self.assertEqual(result['counts']['added'],1)
        self.assertTrue(result['source_record_split_exact'])
        self.assertTrue(result['baseline_metadata_and_records_checked'])
        self.assertTrue(result['preprocessing_byte_exact'])
        self.assertNotIn('diagnostic_gate_passed',result)
        self.assertNotIn('completed_audit',result)

    def test_rehashed_baseline_with_wrong_model_profile_counts_or_records_rejects_without_labels(self):
        original = json.dumps(self.report)
        for mutate in [lambda r:r.update(model='other'),
                       lambda r:r['sensor_profile'].update(width=1024),
                       lambda r:r['projection'].update(projected=1),
                       lambda r:r['selection'].update(removed=0),
                       lambda r:r['selection'].update(class_counts=[2]+[0]*19),
                       lambda r:r.update(retained_sha256='0'*64)]:
            self.report = json.loads(original); mutate(self.report); self.write_files()
            with self.assertRaises(ValueError): verify(self.scan,self.base,self.result)
        self.report = json.loads(original)
        self.baseline_kept = self.baseline_kept[:-1]+b'X'
        self.report['retained_sha256'] = sha(self.baseline_kept)
        self.write_files()
        with self.assertRaisesRegex(ValueError,'exact source split'): verify(self.scan,self.base,self.result)

    def test_rehashed_debug_tensor_substitution_rejects_without_labels(self):
        self.input = bytes(len(self.input)); self.write_files()
        with self.assertRaisesRegex(ValueError,'debug input'): verify(self.scan,self.base,self.result)
        source = self.root/'scan.bin'; source.write_bytes(self.scan)
        output = self.root/'check.json'
        process = subprocess.run([sys.executable,str(Path(__file__).with_name('verify_completion.py')),
            '--scan',str(source),'--base',str(self.base),'--result',str(self.result),'--out',str(output)],
            capture_output=True,text=True)
        self.assertNotEqual(process.returncode,0)
        self.assertIn('debug input',process.stderr)
        self.assertFalse(output.exists())
        self.assertEqual(source.read_bytes(),self.scan)

    def test_completion_model_and_selection_bindings_must_match_without_labels(self):
        for key,value in [('weights_sha256','0'*64),('onnx_sha256','0'*64),('threshold',.8),('target_classes',['truck'])]:
            self.write_files(); self.metadata['base'][key]=value
            (self.result/'completion-report.json').write_text(json.dumps(self.metadata))
            with self.subTest(key=key):
                with self.assertRaisesRegex(ValueError,'model/selection'): verify(self.scan,self.base,self.result)

    @unittest.skipUnless(os.environ.get('CA_LIDAR_COMPLETE_BIN'),'set CA_LIDAR_COMPLETE_BIN for CPU Rust/Python integration')
    def test_actual_cpu_rust_output_matches_independent_verifier_without_labels(self):
        source = self.root/'scan.bin'; source.write_bytes(self.scan)
        output = self.root/'native'
        process = subprocess.run([os.environ['CA_LIDAR_COMPLETE_BIN'],'--scan',str(source),
                                  '--base',str(self.base),'--out',str(output)],capture_output=True,text=True)
        self.assertEqual(process.returncode,0,process.stderr)
        result = verify(self.scan,self.base,output)
        self.assertEqual(result['counts'],self.counts)
        self.assertEqual((output/'completed-mask.u8').read_bytes(),self.mask)


if __name__=='__main__':unittest.main()
