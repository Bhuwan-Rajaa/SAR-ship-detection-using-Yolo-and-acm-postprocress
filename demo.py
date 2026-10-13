"""Run the trained detector on one image and save a YOLOv8s/ACM comparison.

Examples:
    python demo.py
    python demo.py path/to/image.png
    python demo.py path/to/image.png --show
"""
import argparse
from datetime import datetime
import hashlib
import json
import pickle
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from predict_yolov8s_acm import predict_image

from project_paths import ROOT, DATASET_ROOT, RUNS_ROOT, EVALUATION_ROOT


def add_title(image, title):
    panel = cv2.copyMakeBorder(image, 50, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25, 25, 25))
    scale = max(.35, min(.7, image.shape[1] / 1100))
    cv2.putText(panel, title, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), 1)
    return panel


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", nargs="?", type=Path, help="Image path; defaults to an included HRSID example")
    parser.add_argument("--model", type=Path, default=RUNS_ROOT / "baseline_yolov8s/weights/best.pt")
    parser.add_argument("--calibrator", type=Path,
                        default=EVALUATION_ROOT / "yolov8s_deterministic_refined/acm_calibrator.pkl")
    parser.add_argument("--device", default="0", help="GPU index (default: 0); use cpu for CPU inference")
    parser.add_argument("--output", type=Path, help="Output directory; defaults to a new timestamped demo folder")
    parser.add_argument("--show", action="store_true", help="Display the comparison; press any key to close")
    args = parser.parse_args()

    if args.image is None:
        examples = sorted((DATASET_ROOT / "test/images").glob("*.png"))
        if not examples:
            parser.error("No included example found. Provide an image path.")
        args.image = examples[0]
    args.image = args.image.resolve()
    if not args.image.is_file():
        parser.error(f"Image not found: {args.image}")
    original = cv2.imread(str(args.image))
    if original is None:
        parser.error(f"Cannot read image: {args.image}")

    # Load the locally generated classifier and verify its detector/extractor provenance.
    with args.calibrator.open("rb") as f:
        state = pickle.load(f)
    selection = state["selection"]
    if selection.get("feature_provenance", {}).get("extractor") != "deterministic_acm.chan_vese":
        raise ValueError("Calibrator must use the current deterministic ACM extractor")
    if hashlib.sha256(args.model.read_bytes()).hexdigest() != selection["weights_sha256"]:
        raise ValueError("Detector weights do not match the calibrator")

    output = args.output or ROOT / "demo_results" / datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    output = output.resolve()
    destinations = [output / name for name in ("yolo_only.png", "yolo_acm.png", "comparison.png", "detections.json")]
    if any(path.exists() for path in destinations):
        raise FileExistsError(f"Demo outputs already exist in {output}; choose a fresh --output")
    output.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.model))
    baseline_threshold = selection["baseline"]["threshold"]
    result = model.predict(str(args.image), conf=baseline_threshold, iou=.5, imgsz=800,
                           device=args.device, max_det=1000, verbose=False)[0]
    baseline = original.copy()
    yolo_detections = []
    for box, confidence in zip(result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box)
        cv2.rectangle(baseline, (x1, y1), (x2, y2), (255, 170, 50), 2)
        cv2.putText(baseline, f"ship {confidence:.2f}", (x1, max(15, y1-5)),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, (255, 170, 50), 1)
        yolo_detections.append({"box_xyxy": box.tolist(), "confidence": float(confidence), "class": "ship"})

    acm_image, acm_predictions = predict_image(model, state["classifier"], selection,
                                              args.image, device=args.device, imgsz=800)
    comparison = np.hstack([
        add_title(original, "Input image"),
        add_title(baseline, f"YOLOv8s: {len(yolo_detections)} detections"),
        add_title(acm_image, f"YOLOv8s + refined ACM: {len(acm_predictions['detections'])} detections"),
    ])
    for path, image in zip(destinations[:3], (baseline, acm_image, comparison)):
        if not cv2.imwrite(str(path), image):
            raise OSError(f"Could not save {path}")
    report = {"image": str(args.image), "model": str(args.model), "device": args.device,
              "yolo_only": {"threshold": baseline_threshold, "detections": yolo_detections},
              "yolo_acm": acm_predictions}
    destinations[3].write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")

    print(f"Image: {args.image.name}")
    print(f"YOLOv8s: {len(yolo_detections)} detections")
    print(f"YOLOv8s + refined ACM: {len(acm_predictions['detections'])} detections")
    print(f"Saved comparison: {destinations[2]}")
    print(f"Saved detection details: {destinations[3]}")
    if args.show:
        preview = comparison
        if preview.shape[1] > 1600:
            ratio = 1600 / preview.shape[1]
            preview = cv2.resize(preview, None, fx=ratio, fy=ratio)
        cv2.imshow("Ship detection demo", preview)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
