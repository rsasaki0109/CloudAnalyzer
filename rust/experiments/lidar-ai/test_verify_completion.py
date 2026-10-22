"""Synthetic geometry/accounting cases for the independent completion oracle."""
from array import array
import json
import math
from pathlib import Path
import struct
import sys
import unittest

from evaluate_removal import projected_owners
from verify_completion import reference_completion


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


if __name__=='__main__':unittest.main()
