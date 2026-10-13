"""Predict ships on unlabeled images with the trained YOLOv8s + saved ACM calibrator."""
import argparse
import hashlib
import json
import pickle
from pathlib import Path

import cv2
import numpy as np

from target_experiment import ROOT, FEATURES, calibrated_scores
from project_paths import RUNS_ROOT, EVALUATION_ROOT
from generate_detection_analysis import extract_detection_features
from refine_acm import scores_for


def predict_image(model, classifier, selection, path, device="0", imgsz=800):
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot read image {path}")
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE).astype(float)
    gray /= max(1, gray.max())
    result = model.predict(str(path), conf=0.05, iou=0.5, imgsz=imgsz,
                           device=device, max_det=1000, verbose=False)[0]
    detections = []
    for box, conf in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
        feats = extract_detection_features(gray, box, float(conf), None, num_iter=45)
        features = [float(feats.get(f, 0)) if feats else 0.0 for f in FEATURES]
        features[0] = float(conf)
        detections.append({"box": box.tolist(), "conf": float(conf), "features": features,
                           "acm_valid": feats is not None})
    images = [{"image_id": path.stem, "detections": detections}]
    if "selected" in selection:
        selected = selection["selected"]
        scores = scores_for(classifier, images, selected["yolo_weight"])
        threshold = selected["metrics"]["threshold"]
    else:
        scores = calibrated_scores(classifier, images)
        threshold = selection["acm"]["threshold"]
    output = []
    for d, score in zip(detections, scores):
        if score < threshold:
            continue
        output.append({"box_xyxy": d["box"], "yolo_confidence": d["conf"],
                       "acm_score": float(score), "class": "ship"})
        x1, y1, x2, y2 = map(int, d["box"])
        cv2.rectangle(img, (x1, y1), (x2, y2), (40, 220, 40), 2)
        cv2.putText(img, f"ship {score:.2f}", (x1, max(15, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (40, 220, 40), 1)
    return img, {"image": str(path), "threshold": threshold, "detections": output}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", type=Path, help="Image file or directory")
    parser.add_argument("--model", type=Path, default=RUNS_ROOT / "baseline_yolov8s/weights/best.pt")
    parser.add_argument("--calibrator", type=Path,
                        default=EVALUATION_ROOT / "yolov8s_deterministic_refined/acm_calibrator.pkl")
    parser.add_argument("--output", type=Path, default=ROOT / "predicted_examples/yolov8s_acm")
    parser.add_argument("--device", default="0")
    parser.add_argument("--imgsz", type=int, default=800)
    args = parser.parse_args()
    # Only load locally generated, trusted pickle artifacts.
    with args.calibrator.open("rb") as f:
        state = pickle.load(f)
    if state["selection"].get("feature_provenance", {}).get("extractor") != "deterministic_acm.chan_vese":
        raise ValueError("This inference command requires a calibrator fitted with deterministic ACM features")
    if hashlib.sha256(args.model.read_bytes()).hexdigest() != state["selection"]["weights_sha256"]:
        raise ValueError("Calibrator and detector weights do not match")
    paths = sorted(p for p in args.source.iterdir() if p.suffix.lower() in (".png", ".jpg", ".jpeg", ".tif")) if args.source.is_dir() else [args.source]
    if not paths:
        raise ValueError("No input images found")
    args.output.mkdir(parents=True, exist_ok=True)
    from ultralytics import YOLO
    model = YOLO(str(args.model))
    for path in paths:
        image_path = args.output / (path.stem + "_acm.png")
        json_path = args.output / (path.stem + "_acm.json")
        if image_path.exists() or json_path.exists():
            raise FileExistsError(f"Output exists for {path.name}; choose a fresh --output")
        image, predictions = predict_image(model, state["classifier"], state["selection"],
                                            path, args.device, args.imgsz)
        if not cv2.imwrite(str(image_path), image):
            raise OSError(f"Could not write {image_path}")
        json_path.write_text(json.dumps(predictions, indent=2), encoding="utf-8")
        print(f"{path.name}: {len(predictions['detections'])} ships", flush=True)


if __name__ == "__main__":
    main()
