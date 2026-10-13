"""GPU YOLOv8s experiment with validation-only ACM calibration and frozen test evaluation.

Run with --train to train first. Results are measured at IoU >= 0.5; targets are
constraints, never replacements for measured metrics. Existing reports are preserved.
"""
import argparse
import hashlib
import json
import pickle
import subprocess
import sys
from pathlib import Path

import cv2
import numpy as np
import yaml
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from generate_detection_analysis import compute_iou, extract_detection_features

from project_paths import (ROOT, CODE_ROOT, DATASET_ROOT, RUNS_ROOT,
                           EVALUATION_ROOT, resolve_saved_path)
FEATURES = ["conf", "energy_reduction_rel", "contrast", "elongation",
            "compactness", "area_ratio", "contour_stability", "solidity",
            "mask_to_obb_area_ratio", "intensity_mean", "intensity_variance"]


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def audit_splits(dataset):
    """Prioritize test, then validation; exclude duplicate images from fitting splits."""
    dataset = dataset.resolve()
    config = yaml.safe_load((dataset / "hrsid.yaml").read_text())
    configured_root = resolve_saved_path(config.get("path", dataset))
    if not configured_root.is_absolute():
        configured_root = dataset / configured_root
    for split in ("train", "val", "test"):
        if (configured_root / config[split]).resolve() != dataset / split / "images":
            raise ValueError(f"Dataset YAML {split} path differs from audited images")
    seen_names, seen_hashes, report = {}, {}, {}
    for split in ("test", "val", "train"):
        images = sorted((dataset / split / "images").glob("*.png"))
        if not images:
            raise ValueError(f"No images in {dataset / split}")
        manifest = hashlib.sha256()
        retained, excluded = [], []
        for index, image in enumerate(images):
            digest = hashlib.sha256(image.read_bytes()).hexdigest()
            label = dataset / split / "labels" / (image.stem + ".txt")
            if not label.exists():
                raise FileNotFoundError(label)
            overlap = next((seen[key] for key, seen in ((image.stem, seen_names), (digest, seen_hashes))
                            if key in seen and seen[key] != split), None)
            if overlap:
                excluded.append({"path": str(image), "duplicate_of_split": overlap})
                continue
            seen_names[image.stem] = split
            seen_hashes[digest] = split
            retained.append(str(image))
            label_digest = hashlib.sha256(label.read_bytes()).hexdigest()
            manifest.update(f"{image.name}:{digest}:{label_digest}\n".encode())
            if (index+1) % 500 == 0:
                print(f"Audit {split}: {index+1}/{len(images)}", flush=True)
        if not retained:
            raise ValueError(f"No independent images remain in {split}")
        report[split] = {"images": len(images), "retained_images": len(retained),
                         "paths": retained, "excluded": excluded, "manifest_sha256": manifest.hexdigest()}
    return report


def match_labels(boxes, scores, gt):
    """Score-ordered, one-to-one IoU matching; recompute after every score/filter change."""
    labels = np.zeros(len(boxes), dtype=int)
    available = set(range(len(gt)))
    for i in np.argsort(-np.asarray(scores), kind="stable"):
        if not available:
            break
        j = max(sorted(available), key=lambda j: compute_iou(boxes[i], gt[j]))
        if compute_iou(boxes[i], gt[j]) >= 0.5:
            labels[i] = 1
            available.remove(j)
    return labels


def feature_matrix(images):
    rows = [d["features"] for image in images for d in image["detections"]]
    return np.nan_to_num(np.asarray(rows, dtype=float).reshape(-1, len(FEATURES)),
                         nan=0, posinf=1e6, neginf=-1e6).clip(-1e6, 1e6)


def raw_scores(images):
    return np.asarray([d["conf"] for image in images for d in image["detections"]])


def detection_labels(images):
    return np.concatenate([match_labels([d["box"] for d in im["detections"]],
                                       [d["conf"] for d in im["detections"]], im["gt"])
                           for im in images])


def metrics(images, scores, threshold):
    tp, fp, total_gt, offset = 0, 0, 0, 0
    for im in images:
        n = len(im["detections"])
        local = np.asarray(scores[offset:offset + n])
        keep = np.flatnonzero(local >= threshold)
        boxes = [im["detections"][i]["box"] for i in keep]
        matched = int(match_labels(boxes, local[keep], im["gt"]).sum())
        tp += matched
        fp += len(keep) - matched
        total_gt += len(im["gt"])
        offset += n
    if offset != len(scores):
        raise ValueError("Score count does not match detections")
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / total_gt if total_gt else 0.0
    return {"precision": p, "recall": r, "f1": 2*p*r/(p+r) if p+r else 0.0,
            "tp": tp, "fp": fp, "fn": total_gt-tp, "total_gt": total_gt,
            "images": len(images), "threshold": float(threshold)}


def sweep(images, scores):
    # Quantile grid bounds runtime while respecting tied scores.
    if not len(scores):
        return [metrics(images, scores, 0.0)]
    thresholds = np.unique(np.r_[0.0, 1.0, np.quantile(scores, np.linspace(0, 1, 151))])
    return [metrics(images, scores, t) for t in thresholds]


def choose_baseline(points, target):
    feasible = [p for p in points if p["precision"] >= target]
    if feasible:
        return max(feasible, key=lambda p: (p["recall"], p["precision"]))
    return max(points, key=lambda p: (p["f1"], p["recall"]))


def choose_acm(points, baseline, args):
    recall_floor = max(0, baseline["recall"] - args.max_recall_drop)
    safe = [p for p in points if p["recall"] >= recall_floor - 1e-12]
    # A reordering can change greedy assignment. An infeasible result stays explicit.
    feasible = [p for p in safe if p["precision"] > args.final_precision
                and p["precision"] - baseline["precision"] >= args.precision_gain - 1e-12]
    if feasible:
        return max(feasible, key=lambda p: (p["recall"], p["precision"]))
    if safe:
        return max(safe, key=lambda p: (p["precision"], p["recall"]))
    return max(points, key=lambda p: (p["recall"], p["precision"]))


def target_status(base, acm, args):
    checks = {"baseline_precision_at_least_target": base["precision"] >= args.baseline_precision,
              "final_precision_above_target": acm["precision"] > args.final_precision,
              "precision_gain_at_least_target": acm["precision"]-base["precision"] >= args.precision_gain-1e-12,
              "recall_drop_within_limit": base["recall"]-acm["recall"] <= args.max_recall_drop+1e-12}
    return {"checks": checks, "all_targets_met": all(checks.values()),
            "precision_gain_pp": 100*(acm["precision"]-base["precision"]),
            "recall_drop_pp": 100*(base["recall"]-acm["recall"])}


def collect(model, dataset, split, args, paths):
    images = []
    paths = [Path(p) for p in paths]
    for index, path in enumerate(paths):
        gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            raise ValueError(f"Cannot read image: {path}")
        height, width = gray.shape
        gt = []
        for line in (dataset / split / "labels" / (path.stem + ".txt")).read_text().splitlines():
            cls, cx, cy, w, h = map(float, line.split())
            if cls != 0:
                raise ValueError("This evaluator requires single-class ship labels")
            gt.append([(cx-w/2)*width, (cy-h/2)*height, (cx+w/2)*width, (cy+h/2)*height])
        result = model.predict(str(path), conf=args.candidate_conf, iou=0.5,
                               imgsz=args.imgsz, device=args.device, max_det=1000, verbose=False)[0]
        normalized = gray.astype(float) / max(1, float(gray.max()))
        detections = []
        for box, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
            # Ground truth is never supplied to the feature extractor.
            feats = extract_detection_features(normalized, box, float(conf), None, num_iter=45)
            # Retain tiny/failed crops in the denominator and fall back to raw confidence.
            features = [float(feats.get(f, 0)) if feats else 0.0 for f in FEATURES]
            features[0] = float(conf)
            detections.append({"box": box.tolist(), "conf": float(conf), "features": features,
                               "acm_valid": feats is not None})
        images.append({"image_id": path.stem, "gt": gt, "detections": detections})
        if (index+1) % 25 == 0 or index+1 == len(paths):
            print(f"{split}: {index+1}/{len(paths)} images", flush=True)
    return images


def calibrated_scores(clf, images):
    x = feature_matrix(images)
    if not len(x):
        return np.zeros(0)
    scores = clf.predict_proba(x)[:, 1]
    valid = np.asarray([d["acm_valid"] for im in images for d in im["detections"]])
    scores[~valid] = raw_scores(images)[~valid]
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=4)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--name", default="baseline_yolov8s")
    parser.add_argument("--dataset", type=Path, default=DATASET_ROOT)
    parser.add_argument("--output", type=Path, default=EVALUATION_ROOT / "yolov8s_target")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--candidate-conf", type=float, default=0.05)
    parser.add_argument("--baseline-precision", type=float, default=0.91)
    parser.add_argument("--final-precision", type=float, default=0.945)
    parser.add_argument("--precision-gain", type=float, default=0.04)
    parser.add_argument("--max-recall-drop", type=float, default=0.01)
    args = parser.parse_args()
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Choose a fresh --output directory: {args.output}")
    import torch
    if args.device == "cpu" or not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required")
    args.output.mkdir(parents=True, exist_ok=True)
    audit = audit_splits(args.dataset)
    write_json(args.output / "split_audit.json", audit)
    data_config = {"path": str(args.dataset.resolve()), "names": {0: "ship"}, "nc": 1}
    for split, info in audit.items():
        listing = (args.output / f"{split}_images.txt").resolve()
        listing.write_text("\n".join(info["paths"])+"\n", encoding="utf-8")
        data_config[split] = str(listing)
    audited_yaml = (args.output / "audited_dataset.yaml").resolve()
    audited_yaml.write_text(yaml.safe_dump(data_config), encoding="utf-8")
    if args.resume or not args.train:
        saved_args = RUNS_ROOT / args.name / "args.yaml"
        previous = yaml.safe_load(saved_args.read_text())
        previous_data = yaml.safe_load(resolve_saved_path(previous["data"]).read_text())
        for split in ("train", "val", "test"):
            previous_list = resolve_saved_path(previous_data[split])
            if not previous_list.is_file() or [str(resolve_saved_path(p)) for p in previous_list.read_text().splitlines()] != audit[split]["paths"]:
                raise ValueError("Prior model split provenance differs; use --train with a fresh --name")
    write_json(args.output / "configuration.json", {k: str(v) if isinstance(v, Path) else v
                                                    for k, v in vars(args).items()})
    feature_provenance = {"extractor": "deterministic_acm.chan_vese", "iterations": 45}
    write_json(args.output / "feature_provenance.json", feature_provenance)
    if args.train:
        command = [sys.executable, "-u", str(CODE_ROOT / "train_baseline.py"),
                   "--epochs", str(args.epochs), "--batch", str(args.batch),
                   "--workers", str(args.workers), "--name", args.name,
                   "--imgsz", str(args.imgsz), "--device", args.device,
                   "--data", str(audited_yaml)]
        if args.resume:
            command.append("--resume")
        subprocess.run(command, check=True, cwd=ROOT)
    from ultralytics import YOLO
    model_path = RUNS_ROOT / args.name / "weights" / "best.pt"
    model = YOLO(str(model_path))
    val = collect(model, args.dataset, "val", args, audit["val"]["paths"])
    write_json(args.output / "validation_candidates.json", val)
    # Split by image, including images with no detections, before fitting the calibrator.
    order = np.random.default_rng(42).permutation(len(val))
    cut = len(order)//2
    fit, tune = [val[i] for i in order[:cut]], [val[i] for i in order[cut:]]
    if not fit or not tune:
        raise ValueError("At least two validation images are needed")
    x, y = feature_matrix(fit), detection_labels(fit)
    if len(np.unique(y)) != 2:
        raise ValueError("Calibration requires both TP and FP examples")
    base_points = sweep(tune, raw_scores(tune))
    base = choose_baseline(base_points, args.baseline_precision)
    candidates = {
        "logistic_acm": make_pipeline(StandardScaler(), LogisticRegression(C=0.1, max_iter=2000)),
        "gradient_boosted_acm": HistGradientBoostingClassifier(max_iter=150, max_leaf_nodes=7,
                              l2_regularization=5, min_samples_leaf=30, random_state=42),
    }
    choices = []
    for name, clf in candidates.items():
        clf.fit(x, y)
        point = choose_acm(sweep(tune, calibrated_scores(clf, tune)), base, args)
        status = target_status(base, point, args)
        choices.append((name, clf, point, status))
        print(name, point, status, flush=True)
    name, clf, point, status = max(choices, key=lambda c: (
        c[3]["all_targets_met"], c[3]["checks"]["recall_drop_within_limit"],
        c[2]["recall"] if c[3]["all_targets_met"] else c[2]["precision"]))
    # Freeze model and thresholds before opening test predictions. Do not refit on tune.
    confidence_only = choose_acm(base_points, base, args)
    selection = {"model": name, "baseline": base, "acm": point,
                 "confidence_only": confidence_only, "status": status,
                 "fit_images": [im["image_id"] for im in fit],
                 "tune_images": [im["image_id"] for im in tune],
                 "weights_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
                 "features": FEATURES, "iou_threshold": 0.5,
                 "feature_provenance": feature_provenance}
    write_json(args.output / "frozen_selection.json", selection)
    with (args.output / "acm_calibrator.pkl").open("wb") as f:
        pickle.dump({"classifier": clf, "selection": selection}, f)
    test = collect(model, args.dataset, "test", args, audit["test"]["paths"])
    write_json(args.output / "test_candidates.json", test)
    base_test = metrics(test, raw_scores(test), base["threshold"])
    acm_test = metrics(test, calibrated_scores(clf, test), point["threshold"])
    control_test = metrics(test, raw_scores(test), confidence_only["threshold"])
    report = {"baseline": base_test, "acm": acm_test, "confidence_only": control_test,
              "status": target_status(base_test, acm_test, args), "selection": selection}
    write_json(args.output / "results.json", report)
    lines = ["# YOLOv8s + learned ACM: held-out test results", "",
             "Thresholds and calibrator were frozen using validation images only.", "",
             "| Method | Precision | Recall | TP | FP | FN |",
             "|---|---:|---:|---:|---:|---:|"]
    for label, m in (("YOLOv8s", base_test), ("YOLOv8s + learned ACM", acm_test),
                     ("Confidence-only control", control_test)):
        lines.append(f"| {label} | {m['precision']:.2%} | {m['recall']:.2%} | {m['tp']} | {m['fp']} | {m['fn']} |")
    lines += ["", f"All targets met: **{report['status']['all_targets_met']}**.",
              f"Precision gain: {report['status']['precision_gain_pp']:.2f} percentage points.",
              f"Recall drop: {report['status']['recall_drop_pp']:.2f} percentage points.", "",
              "Cross-split image duplicates were excluded from training/validation; source scenes may still overlap.",
              "Historical test reports already exist in this repository; this is not a new blind dataset."]
    (args.output / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps(report["status"], indent=2), flush=True)


if __name__ == "__main__":
    main()
