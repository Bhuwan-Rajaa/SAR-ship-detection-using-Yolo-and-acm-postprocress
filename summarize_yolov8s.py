"""Generate measured YOLOv8s comparison charts and deterministic prediction examples."""
import json
import pickle

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from target_experiment import ROOT, raw_scores, metrics, match_labels
from refine_acm import scores_for
from project_paths import DATASET_ROOT, EVALUATION_ROOT, RUNS_ROOT


def main():
    source = EVALUATION_ROOT / "yolov8s_target"
    refined = EVALUATION_ROOT / "yolov8s_deterministic_refined"
    features = EVALUATION_ROOT / "yolov8s_deterministic"
    output = EVALUATION_ROOT / "yolov8s_summary"
    output.mkdir(parents=True, exist_ok=True)
    original = json.loads((source / "results.json").read_text())
    final = json.loads((refined / "results.json").read_text())
    test = json.loads((features / "test_candidates.json").read_text())
    with (refined / "acm_calibrator.pkl").open("rb") as f:
        state = pickle.load(f)
    raw = raw_scores(test)
    scores = scores_for(state["classifier"], test, final["selection"]["yolo_weight"])
    baseline_threshold = final["baseline"]["threshold"]
    acm_threshold = final["acm"]["threshold"]
    # Verify published numbers against the saved boxes, including undetected images.
    assert metrics(test, raw, baseline_threshold) == final["baseline"]
    assert metrics(test, scores, acm_threshold) == final["acm"]
    rows = [("YOLOv8s", final["baseline"]), ("Confidence only", original["confidence_only"]),
            ("Refined ACM", final["acm"])]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), layout="constrained")
    colors = ["#456990", "#9ba8b5", "#25866f"]
    for axis, metric, title in zip(axes, ("precision", "recall", "f1"), ("Precision", "Recall", "F1")):
        values = [100*m[metric] for _, m in rows]
        bars = axis.bar(range(len(rows)), values, color=colors)
        axis.bar_label(bars, fmt="%.2f", padding=3)
        axis.set_xticks(range(len(rows)), [name for name, _ in rows], rotation=25, ha="right")
        axis.set_ylim(0, 100)
        axis.set_title(title + " (%)")
        axis.spines[["top", "right"]].set_visible(False)
    axes[0].axhline(94.5, color="#a33", linestyle="--", linewidth=1, label="94.5% target")
    axes[0].legend(fontsize=8)
    fig.savefig(output / "test_comparison.png", dpi=180)
    plt.close(fig)
    training = pd.read_csv(RUNS_ROOT / "baseline_yolov8s/results.csv")
    training.columns = training.columns.str.strip()
    fig, axis = plt.subplots(figsize=(9, 4), layout="constrained")
    for column, label in (("metrics/precision(B)", "Validation precision"),
                           ("metrics/recall(B)", "Validation recall"),
                           ("metrics/mAP50(B)", "Validation mAP50")):
        axis.plot(training["epoch"], training[column]*100, label=label)
    axis.set(xlabel="Epoch", ylabel="Percent", title="YOLOv8s training: validation metrics")
    axis.legend()
    fig.savefig(output / "training_curves.png", dpi=180)
    plt.close(fig)
    # Six fixed, evenly spaced images; selection is independent of successes/failures.
    example_ids = []
    offsets = np.r_[0, np.cumsum([len(im["detections"]) for im in test])]
    for index in np.linspace(0, len(test)-1, 6, dtype=int):
        im = test[index]
        example_ids.append(im["image_id"])
        image = cv2.imread(str(DATASET_ROOT / "test/images" / (im["image_id"]+".png")))
        if image is None:
            raise ValueError(f"Missing image: {im['image_id']}")
        panels = []
        for title, values, threshold in (("YOLOv8s", raw, baseline_threshold),
                                         ("YOLOv8s + refined ACM", scores, acm_threshold)):
            canvas = image.copy()
            local = values[offsets[index]:offsets[index+1]]
            keep = np.flatnonzero(local >= threshold)
            boxes = [im["detections"][i]["box"] for i in keep]
            labels = match_labels(boxes, local[keep], im["gt"])
            for box in im["gt"]:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(canvas, (x1,y1), (x2,y2), (255, 220, 50), 1)
            for box, label in zip(boxes, labels):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(canvas, (x1,y1), (x2,y2), (40,220,40) if label else (0,100,255), 2)
            canvas = cv2.copyMakeBorder(canvas, 65, 0, 0, 0, cv2.BORDER_CONSTANT, value=(25,25,25))
            cv2.putText(canvas, f"{title}: TP {sum(labels)} / FP {len(labels)-sum(labels)}",
                        (12,25), cv2.FONT_HERSHEY_SIMPLEX, .65, (255,255,255), 1)
            cv2.putText(canvas, "Green: TP | Orange: FP | Cyan: ground truth", (12,50),
                        cv2.FONT_HERSHEY_SIMPLEX, .5, (255,255,255), 1)
            panels.append(canvas)
        cv2.imwrite(str(output / f"example_{index:04d}.png"), np.hstack(panels))
    checks = final["status"]["checks"]
    precision_met = checks["final_precision_above_target"] and checks["precision_gain_at_least_target"]
    conclusion = (f"Precision targets: {'met' if precision_met else 'not met'}. "
                  f"Recall-loss limit: {'met' if checks['recall_drop_within_limit'] else 'not met'}.")
    lines = ["# YOLOv8s and ACM: completed experiment", "", f"**{conclusion}**", "",
             "Evaluated on all 1,962 test images and 5,922 ships at matching IoU >= 0.5.", "",
             "| Method | Precision | Recall | F1 | TP | FP | FN |",
             "|---|---:|---:|---:|---:|---:|---:|"]
    for name, m in rows:
        lines.append(f"| {name} | {m['precision']:.2%} | {m['recall']:.2%} | {m['f1']:.2%} | {m['tp']} | {m['fp']} | {m['fn']} |")
    lines += ["", "![Test comparison](test_comparison.png)", "",
              f"Refined ACM precision gain: **{final['status']['precision_gain_pp']:.2f} percentage points**.",
              f"Recall loss: **{final['status']['recall_drop_pp']:.2f} percentage points**.", "",
              f"Refined ACM removed {final['baseline']['fp']-final['acm']['fp']} false positives and",
              f"lost {final['baseline']['tp']-final['acm']['tp']} true positives relative to baseline.",
              f"Precision difference versus confidence-only control: {100*(final['acm']['precision']-original['confidence_only']['precision']):+.2f} percentage points.",
              f"True-positive difference versus confidence-only control: {final['acm']['tp']-original['confidence_only']['tp']:+d}.", "",
              "## Protocol", "",
              "YOLOv8s completed 100 GPU epochs at 800-pixel resolution. Calibration used 272 validation",
              "images for fitting and 273 for threshold/model selection. The refinement compared eight",
              "predeclared classifiers and four blend weights using the same validation partition.",
              f"Selected model: {final['selection']['model']}; YOLO blend weight: {final['selection']['yolo_weight']}.",
              "The model was selected on validation; no test thresholds were tuned.",
              "This is an exploratory validation search, not an independent confirmatory experiment.", "",
              "## Reproducibility correction", "",
              "An inference smoke test found that the original ACM curvature-smoothing cycle used",
              "module-global state, making scores depend on prior detections. The extractor now uses",
              "an explicit local iteration phase. Validation and test features were rebuilt from the",
              "same saved YOLO boxes, and the same predeclared validation search was repeated.",
              "The table uses the corrected model. Earlier ACM results are retained as archived artifacts.", "",
              f"YOLO threshold: {baseline_threshold:.8f}. Refined ACM threshold: {acm_threshold:.8f}.",
              "The confidence-only control also used a threshold selected on validation.",
              "One train image and one validation image duplicated test images and were excluded from",
              "the fitting file lists. Source scenes can still overlap. Historical test results already",
              "exist in this repository; these test images are not a newly collected blind dataset.", "",
              "Training validation metrics below use Ultralytics' operating-point convention; the table",
              "above uses explicitly frozen thresholds and therefore has different precision/recall.", "",
              "![Training curves](training_curves.png)", "", "## Prediction examples", "",
              "Six evenly spaced test images, selected independently of their outcomes:", ""]
    for index, name in zip(np.linspace(0, len(test)-1, 6, dtype=int), example_ids):
        lines += [f"{name}", f"![Example {index}](example_{index:04d}.png)", ""]
    lines += ["## Reuse", "", "```powershell", "python predict_yolov8s_acm.py path/to/image.png",
              "```", "", "Outputs include annotated images and JSON detections. No labels are needed.",
              "The prediction command checks that detector weights match the calibration artifact."]
    (output / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(output / "report.md")


if __name__ == "__main__":
    main()
