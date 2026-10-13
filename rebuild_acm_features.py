"""Recompute deterministic ACM features from saved boxes; no detector retraining."""
import json
from pathlib import Path
import shutil
import subprocess
import sys

import cv2
import numpy as np

from target_experiment import ROOT, FEATURES, write_json
from project_paths import CODE_ROOT, DATASET_ROOT, EVALUATION_ROOT
from generate_detection_analysis import extract_detection_features


def main():
    source = EVALUATION_ROOT / "yolov8s_target"
    output = EVALUATION_ROOT / "yolov8s_deterministic"
    output.mkdir(parents=True, exist_ok=True)
    for name in ("configuration.json", "frozen_selection.json", "split_audit.json"):
        shutil.copy2(source / name, output / name)
    # Preserve original frozen baseline/partitions; only ACM features are recomputed.
    for split, filename in (("val", "validation_candidates.json"), ("test", "test_candidates.json")):
        checkpoint = output / (filename+"l")
        images = json.loads((source / filename).read_text())
        completed = {}
        if checkpoint.exists():
            for line in checkpoint.read_text().splitlines():
                im = json.loads(line)
                completed[im["image_id"]] = im
        with checkpoint.open("a", encoding="utf-8") as f:
            for i, im in enumerate(images):
                if im["image_id"] in completed:
                    continue
                path = DATASET_ROOT / split / "images" / (im["image_id"]+".png")
                gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
                if gray is None:
                    raise ValueError(f"Cannot read {path}")
                gray = gray.astype(float)/max(1, float(gray.max()))
                for d in im["detections"]:
                    feats = extract_detection_features(gray, np.asarray(d["box"]), d["conf"], None, num_iter=45)
                    d["features"] = [float(feats.get(k, 0)) if feats else 0.0 for k in FEATURES]
                    d["features"][0] = d["conf"]
                    d["acm_valid"] = feats is not None
                f.write(json.dumps(im, allow_nan=False)+"\n")
                f.flush()
                completed[im["image_id"]] = im
                if (i+1) % 50 == 0:
                    print(f"Deterministic {split}: {i+1}/{len(images)}", flush=True)
        write_json(output / filename, [completed[im["image_id"]] for im in images])
    write_json(output / "feature_provenance.json", {
        "extractor": "deterministic_acm.chan_vese", "iterations": 45,
        "boxes_source": str(source), "detector_retrained": False,
        "selection_policy": "Same validation partitions, frozen baseline, and predeclared refinement search"})
    subprocess.run([sys.executable, "-u", str(CODE_ROOT / "refine_acm.py"), "--source", str(output),
                    "--output", str(EVALUATION_ROOT / "yolov8s_deterministic_refined")], check=True)


if __name__ == "__main__":
    main()
