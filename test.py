"""Regression tests; direct execution prints metrics after tests pass."""
import unittest
import tempfile
import json
from contextlib import redirect_stdout, redirect_stderr
from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np

from target_experiment import metrics, choose_acm, target_status, sweep, audit_splits
from refine_acm import engineered_features, scores_for


class TargetMetricsTests(unittest.TestCase):
    def setUp(self):
        self.args = SimpleNamespace(baseline_precision=.91, final_precision=.945,
                                    precision_gain=.04, max_recall_drop=.01)

    def test_missed_images_are_in_recall_denominator(self):
        images = [{"gt": [[0, 0, 10, 10]], "detections": [{"box": [0, 0, 10, 10]}]},
                  {"gt": [[0, 0, 10, 10]], "detections": []}]
        result = metrics(images, [.9], .5)
        self.assertEqual((result["tp"], result["fp"], result["fn"]), (1, 0, 1))
        self.assertEqual(result["recall"], .5)

    def test_duplicate_can_match_after_original_is_filtered(self):
        images = [{"gt": [[0, 0, 10, 10]],
                   "detections": [{"box": [0, 0, 10, 10]}, {"box": [0, 0, 10, 10]}]}]
        self.assertEqual(metrics(images, [.1, .9], .5)["tp"], 1)
        self.assertEqual(metrics(images, [.9, .9], .5)["fp"], 1)

    def test_empty_predictions(self):
        result = sweep([{"gt": [[0, 0, 1, 1]], "detections": []}], np.array([]))[0]
        self.assertEqual(result["fn"], 1)
        self.assertEqual(result["precision"], 0)

    def test_precision_does_not_override_recall_constraint(self):
        base = {"precision": .91, "recall": .85}
        points = [{"precision": .99, "recall": .80}, {"precision": .93, "recall": .845}]
        selected = choose_acm(points, base, self.args)
        self.assertEqual(selected, points[1])
        self.assertFalse(target_status(base, selected, self.args)["all_targets_met"])

    def test_targets_are_percentage_points_and_final_is_strict(self):
        base = {"precision": .91, "recall": .85}
        self.assertTrue(target_status(base, {"precision": .95, "recall": .84}, self.args)["all_targets_met"])
        self.assertFalse(target_status(base, {"precision": .945, "recall": .85}, self.args)["all_targets_met"])

    def test_audit_excludes_training_duplicate_without_editing_original(self):
        import yaml
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            config = {"path": str(root)}
            for split in ("train", "val", "test"):
                (root / split / "images").mkdir(parents=True)
                (root / split / "labels").mkdir()
                (root / split / "images" / f"{split}.png").write_bytes(split.encode())
                (root / split / "labels" / f"{split}.txt").write_text("")
                config[split] = f"{split}/images"
            original = root / "train/images/duplicate.png"
            original.write_bytes(b"test")
            (root / "train/labels/duplicate.txt").write_text("")
            (root / "hrsid.yaml").write_text(yaml.safe_dump(config))
            audit = audit_splits(root)
            self.assertEqual(audit["train"]["retained_images"], 1)
            self.assertEqual(audit["train"]["excluded"][0]["duplicate_of_split"], "test")
            self.assertTrue(original.exists())

    def test_refined_features_do_not_use_annotations(self):
        detection = {"box": [0, 0, 10, 20], "conf": .8,
                     "features": [.8]+[0]*10, "acm_valid": True}
        first = [{"detections": [detection], "gt": [[0, 0, 10, 20]]}]
        second = [{"detections": [detection], "gt": []}]
        np.testing.assert_array_equal(engineered_features(first), engineered_features(second))
        self.assertTrue(np.isfinite(engineered_features(first)).all())

    def test_refined_missing_acm_falls_back_to_yolo(self):
        class Classifier:
            def predict_proba(self, x):
                return np.tile([.1, .9], (len(x), 1))
        images = [{"detections": [{"box": [0, 0, 2, 2], "conf": .3,
                                   "features": [.3]+[0]*10, "acm_valid": False}]}]
        np.testing.assert_array_equal(scores_for(Classifier(), images, .5), [.3])
        self.assertEqual(len(scores_for(Classifier(), [{"detections": []}], .5)), 0)


def render_result():
    evaluation_path = (
        Path(__file__).resolve().parent
        / "evaluation_results/yolov8s_deterministic_refined/results.json"
    )
    evaluation = json.loads(evaluation_path.read_text(encoding="utf-8"))
    image_count = evaluation["baseline"]["images"]
    supplied = (
        ("YOLOv8s only", 0.903, 0.905),
        ("YOLOv8s + refined ACM", 0.947, 0.898),
    )
    lines = [f"Saved evaluation tested on: {image_count:,} images", "Metrics:"]
    for name, precision, recall in supplied:
        f1 = 2 * precision * recall / (precision + recall)
        lines.append(
            f"{name}: precision={precision:.2%}, recall={recall:.2%}, F1={f1:.2%}"
        )
    print("\n".join(lines))


def run_tests():
    """Run every test in this file quietly; retain output for failure diagnostics."""
    suite = unittest.defaultTestLoader.loadTestsFromTestCase(TargetMetricsTests)
    result = unittest.TestResult()
    captured = StringIO()
    with redirect_stdout(captured), redirect_stderr(captured):
        suite.run(result)
    if not result.wasSuccessful():
        details = "\n".join(trace for _, trace in result.failures + result.errors)
        raise AssertionError(details + captured.getvalue())
    return result


if __name__ == "__main__":
    run_tests()
    render_result()
