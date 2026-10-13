"""Bounded validation-only search over learned ACM scores and YOLO blending.

Uses cached detections from target_experiment.py; never chooses settings on test.
The model/threshold selection is saved before the test cache is opened.
"""
import argparse
import json
import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from project_paths import EVALUATION_ROOT

from target_experiment import (ROOT, feature_matrix, raw_scores, detection_labels,
                               sweep, choose_acm, target_status, metrics, write_json)


def engineered_features(images):
    x = feature_matrix(images)
    confidence = np.clip(x[:, :1], 1e-5, 1-1e-5)
    geometry = []
    for im in images:
        for detection in im["detections"]:
            x1, y1, x2, y2 = detection["box"]
            w, h = max(1, x2-x1), max(1, y2-y1)
            geometry.append([np.log(w*h), np.log(max(w, h)/min(w, h)),
                             float(detection["acm_valid"])])
    transformed = np.sign(x[:, 1:])*np.log1p(np.abs(x[:, 1:]))
    return np.column_stack([np.log(confidence/(1-confidence)), transformed,
                            transformed*confidence, np.asarray(geometry).reshape(-1, 3)])


def scores_for(classifier, images, alpha):
    raw = raw_scores(images)
    if not len(raw):
        return raw
    probs = classifier.predict_proba(engineered_features(images))[:, 1]
    valid = np.asarray([d["acm_valid"] for im in images for d in im["detections"]])
    scores = alpha*raw + (1-alpha)*probs
    scores[~valid] = raw[~valid]
    return scores


def evaluate_frozen(source, output):
    """Finish test evaluation of an existing frozen selection without refitting."""
    with (output / "acm_calibrator.pkl").open("rb") as f:
        state = pickle.load(f)
    selection = json.loads((output / "frozen_selection.json").read_text())
    if state["selection"] != selection:
        raise ValueError("Saved classifier and frozen selection differ")
    original = json.loads((source / "frozen_selection.json").read_text())
    if original["weights_sha256"] != selection["weights_sha256"]:
        raise ValueError("Detector weights differ between caches")
    config = SimpleNamespace(**json.loads((source / "configuration.json").read_text()))
    best = selection["selected"]
    test = json.loads((source / "test_candidates.json").read_text())
    baseline = metrics(test, raw_scores(test), selection["baseline"]["threshold"])
    acm = metrics(test, scores_for(state["classifier"], test, best["yolo_weight"]), best["metrics"]["threshold"])
    report = {"baseline": baseline, "acm": acm, "status": target_status(baseline, acm, config),
              "selection": best, "note": selection["note"]}
    if (output / "results.json").exists():
        raise FileExistsError("Test results already exist")
    write_json(output / "results.json", report)
    print(json.dumps(report, indent=2), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=EVALUATION_ROOT / "yolov8s_target")
    parser.add_argument("--output", type=Path, default=EVALUATION_ROOT / "yolov8s_refined")
    parser.add_argument("--evaluate-frozen", action="store_true",
                        help="Evaluate saved selection without repeating calibration")
    args = parser.parse_args()
    if args.evaluate_frozen:
        evaluate_frozen(args.source, args.output)
        return
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Choose a fresh output directory: {args.output}")
    config = SimpleNamespace(**json.loads((args.source / "configuration.json").read_text()))
    original = json.loads((args.source / "frozen_selection.json").read_text())
    validation = json.loads((args.source / "validation_candidates.json").read_text())
    lookup = {im["image_id"]: im for im in validation}
    fit = [lookup[i] for i in original["fit_images"]]
    tune = [lookup[i] for i in original["tune_images"]]
    x, y = engineered_features(fit), detection_labels(fit)
    base = metrics(tune, raw_scores(tune), original["baseline"]["threshold"])
    args.output.mkdir(parents=True)
    choices, classifiers = [], {}
    # Eight predeclared models, four blends each; no search over test results.
    for c in (0.01, 0.1, 1.0, 10.0):
        classifiers[f"logistic_C{c}"] = make_pipeline(StandardScaler(), LogisticRegression(C=c, max_iter=3000))
    for leaves in (3, 7):
        for regularization in (5, 20):
            classifiers[f"boost_leaves{leaves}_reg{regularization}"] = HistGradientBoostingClassifier(
                max_iter=100, max_leaf_nodes=leaves, l2_regularization=regularization,
                min_samples_leaf=30, random_state=42)
    for name, classifier in classifiers.items():
        classifier.fit(x, y)
        for alpha in (0.0, 0.25, 0.5, 0.75):
            point = choose_acm(sweep(tune, scores_for(classifier, tune, alpha)), base, config)
            choice = {"model": name, "yolo_weight": alpha, "metrics": point,
                      "status": target_status(base, point, config)}
            choices.append(choice)
        print(name, "validation search complete", flush=True)
    best = max(choices, key=lambda c: (c["status"]["all_targets_met"],
        c["status"]["checks"]["recall_drop_within_limit"],
        c["metrics"]["recall"] if c["status"]["all_targets_met"] else c["metrics"]["precision"],
        c["metrics"]["recall"]))
    selection = {"selected": best, "baseline": base, "candidates": choices,
                 "fit_images": original["fit_images"], "tune_images": original["tune_images"],
                 "weights_sha256": original["weights_sha256"],
                 "note": "Exploratory refinement on the existing validation split; no test tuning."}
    provenance = args.source / "feature_provenance.json"
    if provenance.exists():
        selection["feature_provenance"] = json.loads(provenance.read_text())
    write_json(args.output / "frozen_selection.json", selection)
    classifier = classifiers[best["model"]]
    with (args.output / "acm_calibrator.pkl").open("wb") as f:
        pickle.dump({"classifier": classifier, "selection": selection}, f)
    print("Selection frozen:", best, flush=True)
    # All settings are fixed above this line; test data is opened only now.
    if not (args.source / "test_candidates.json").exists():
        print("Test extraction is still pending. Run --evaluate-frozen once its cache exists.", flush=True)
        return
    evaluate_frozen(args.source, args.output)


if __name__ == "__main__":
    main()
