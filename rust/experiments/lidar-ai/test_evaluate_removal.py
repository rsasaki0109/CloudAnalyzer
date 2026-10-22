"""Small fabricated labels test accounting only; no model-accuracy evidence."""
import json
from pathlib import Path
import struct
import subprocess
import sys
import tempfile
import unittest

from evaluate_removal import CLASSES, CHECKPOINT, ONNX, evaluate, projected_owners, sha


def fixture(raw_ids=None, mask=None, points=None):
    points = points or [(10, 0, 0, .5), (20, 0, 0, .5), (0, 10, 0, .5), (0, -10, 0, .5), (-10, 0, 0, .5)]
    source = b"".join(struct.pack("<4f", *p) for p in points)
    raw_ids = raw_ids or [10 | (7 << 16), 10 | (7 << 16), 40, 252 | (9 << 16), 99]
    labels = b"".join(struct.pack("<I", v) for v in raw_ids)
    mask = bytes([1, 0, 1, 0, 0] if mask is None else mask)
    retained = b"".join(source[i*16:(i+1)*16] for i, v in enumerate(mask) if not v)
    candidates = b"".join(source[i*16:(i+1)*16] for i, v in enumerate(mask) if v)
    _, projection = projected_owners(source)
    counts = [0] * 20; counts[1] = projection["projected"]
    pairing = dict(app="CloudAnalyzer SemanticKITTI pairing", version=1, sequence="08", frame="000000", split="validation",
                   point_order="original-kitti-xyzi", label_format="semantickitti-uint32-le", scan_sha256=sha(source), labels_sha256=sha(labels),
                   scan_member="dataset/sequences/08/velodyne/000000.bin", labels_member="dataset/sequences/08/labels/000000.label")
    report = dict(experimental=True, model="SalsaNext SemanticKITTI20", backend="wgpu", device_selection="discrete_gpu_0", source_modified=False,
                  source_weight_sha256=CHECKPOINT, onnx_sha256=ONNX, class_names=list(CLASSES), input_sha256=sha(source), mask_bytes=len(mask),
                  retained_sha256=sha(retained), candidate_sha256=sha(candidates), projection=projection,
                  sensor_profile=dict(height=64, width=2048, fov_up_deg=3, fov_down_deg=-25, max_range_m=100, remission="unchanged normalized0..1"),
                  selection=dict(target_classes=["car"], threshold=.9, removed=sum(mask), retained=len(mask)-sum(mask),
                                 unclassified_retained=len(mask)-projection["projected"], class_counts=counts))
    return dict(scan=source, labels=labels, mask=mask, retained=retained, candidates=candidates, report=report, pairing=pairing)


class RemovalAuditTests(unittest.TestCase):
    def test_known_confusion_includes_occluded_cars_and_decodes_instance_bits(self):
        result = evaluate(**fixture())
        self.assertEqual(result["full_scan"], dict(tp=1, fp=1, fn=2, tn=0, precision=.5, recall=1/3, iou=.25, non_target_false_removal_rate=1.))
        self.assertEqual(result["projected_owners_only"]["fn"], 1)
        self.assertEqual(result["target_unclassified_retained"], 1)
        self.assertEqual(result["full_scan_recall_ceiling_from_nearest_only_ownership"], 2/3)
        self.assertEqual(result["false_removals_by_raw_class"], {"40": 1})
        self.assertEqual(result["ignored_retained"], 1)
        self.assertEqual([(v["instance_id"], v["points"], v["retained"]) for v in result["target_instances"]], [(7, 2, 1), (9, 1, 1)])

    def test_ignored_removals_are_reported_and_not_mislabelled_as_known_false_positives(self):
        result = evaluate(**fixture(raw_ids=[99, 0, 1, 52, 40], mask=[1, 0, 1, 0, 0]))
        self.assertEqual((result["ignored_removed"], result["ignored_retained"]), (2, 2))
        self.assertEqual(result["full_scan"]["tn"], 1)
        self.assertIsNone(result["full_scan"]["precision"])
        self.assertIsNone(result["full_scan"]["recall"])
        self.assertEqual(result["precision_if_all_ignored_removals_are_errors"], 0.)

    def test_no_removals_never_become_perfect_precision(self):
        result = evaluate(**fixture(mask=[0] * 5))
        self.assertIsNone(result["full_scan"]["precision"])
        self.assertEqual(result["full_scan"]["recall"], 0.)
        json.dumps(result, allow_nan=False)

    def test_same_length_label_substitution_and_wrong_sequence_pairing_are_rejected(self):
        value = fixture(); value["labels"] = value["labels"][4:] + value["labels"][:4]
        with self.assertRaisesRegex(ValueError, "labels SHA256"):
            evaluate(**value)
        for key, changed in [("split", "train"), ("labels_member", "sequences/00/labels/000000.label"), ("frame", "000001")]:
            value = fixture(); value["pairing"][key] = changed
            with self.assertRaises(ValueError):
                evaluate(**value)

    def test_unknown_labels_bad_masks_and_invalid_model_metadata_fail_closed(self):
        with self.assertRaisesRegex(ValueError, "unknown SemanticKITTI"):
            evaluate(**fixture(raw_ids=[12345] * 5))
        with self.assertRaisesRegex(ValueError, "unclassified"):
            evaluate(**fixture(mask=[0, 1, 0, 0, 0]))
        for mutate in [lambda v: v.update(mask=b"\x02" * 5), lambda v: v.update(retained=b"bad"),
                       lambda v: v["report"]["selection"].update(threshold=float("nan")),
                       lambda v: v["report"]["sensor_profile"].update(height=32)]:
            value=fixture(); mutate(value)
            with self.assertRaises(ValueError):
                evaluate(**value)

    def test_invalid_and_out_of_view_target_returns_stay_in_full_scan_false_negatives(self):
        points=[(10,0,0,.5),(float('nan'),0,0,.5),(0,0,10,.5),(200,0,0,.5),(0,10,0,.5)]
        result=evaluate(**fixture(raw_ids=[10]*5, mask=[1,0,0,0,0], points=points))
        self.assertEqual(result["full_scan"]["fn"],4)
        self.assertEqual(result["target_unclassified_retained"],3)
        self.assertEqual(result["full_scan_recall_ceiling_from_nearest_only_ownership"],.4)

    def test_cli_never_overwrites_an_existing_report_or_source(self):
        with tempfile.TemporaryDirectory() as directory:
            p=Path(directory); out=p/'existing.json'; out.write_bytes(b'keep')
            script=Path(__file__).with_name('evaluate_removal.py')
            result=subprocess.run([sys.executable,str(script),'--scan',str(p/'scan.bin'),'--labels',str(p/'scan.label'),
                '--pairing',str(p/'pairing.json'),'--result',str(p),'--out',str(out)],capture_output=True)
            self.assertNotEqual(result.returncode,0)
            self.assertEqual(out.read_bytes(),b'keep')
            self.assertIn(b'new JSON file',result.stderr)


if __name__ == "__main__":
    unittest.main()
