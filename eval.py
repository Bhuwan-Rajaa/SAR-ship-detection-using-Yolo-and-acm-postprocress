"""Plot precision/recall metrics and calculated F1, without model evaluation.

Run: python eval.py
Outputs: evaluation_results/yolov8_results/ (PNG, SVG, and JSON).
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


METRICS = {
    "YOLOv8s only": {"precision": 0.903, "recall": 0.905},
    "YOLOv8s + refined ACM": {"precision": 0.947, "recall": 0.898},
}
COLORS = ["#426b9b", "#228879"]
PROVENANCE = "Precision and recall metrics; F1 calculated as 2PR / (P + R)."


def prepare_metrics():
    result = {}
    for name, supplied in METRICS.items():
        precision, recall = supplied["precision"], supplied["recall"]
        if not (0 <= precision <= 1 and 0 <= recall <= 1):
            raise ValueError("Precision and recall must be between 0 and 1")
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        result[name] = {**supplied, "f1": f1}
    return result


def draw_metric(axis, metrics, key, title):
    values = [row[key] for row in metrics.values()]
    bars = axis.bar(range(len(metrics)), values, color=COLORS, width=0.55)
    axis.bar_label(bars, labels=[f"{value:.2%}" for value in values], padding=6, fontsize=12)
    axis.set_xticks(range(len(metrics)), ["YOLOv8s\nonly", "YOLOv8s +\nrefined ACM"])
    axis.set_ylim(0, 1.05)
    axis.set_yticks([0, .2, .4, .6, .8, 1])
    axis.yaxis.set_major_formatter(PercentFormatter(1))
    axis.set_title(title, fontsize=14, fontweight="bold", pad=14)
    axis.set_axisbelow(True)
    axis.grid(axis="y", alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)


def save_figure(figure, output, name):
    for extension in ("png", "svg"):
        figure.savefig(output / f"{name}.{extension}", dpi=220, bbox_inches="tight")
    plt.close(figure)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "evaluation_results/yolov8_results")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    metrics = prepare_metrics()
    items = [("precision", "Precision"), ("recall", "Recall"), ("f1", "F1 score")]

    figure, axes = plt.subplots(1, 3, figsize=(12, 4.8))
    figure.suptitle("YOLOv8s vs. refined ACM — metrics", fontsize=16, fontweight="bold")
    for axis, (key, title) in zip(axes, items):
        draw_metric(axis, metrics, key, title)
    figure.text(.5, .02, PROVENANCE, ha="center", fontsize=9, color="#555555")
    figure.tight_layout(rect=(0, .07, 1, .93))
    save_figure(figure, args.output, "metrics_comparison")

    for key, title in items:
        figure, axis = plt.subplots(figsize=(6, 5))
        draw_metric(axis, metrics, key, title)
        figure.text(.5, .02, "Metrics; F1 derived from precision and recall.",
                    ha="center", fontsize=8, color="#555555")
        figure.tight_layout(rect=(0, .07, 1, 1))
        save_figure(figure, args.output, key)

    (args.output / "metrics.json").write_text(json.dumps({
        "provenance": PROVENANCE,
        "model_evaluation_performed": False,
        "metrics": metrics,
    }, indent=2) + "\n", encoding="utf-8")
    for name, row in metrics.items():
        print(f"{name}: precision={row['precision']:.2%}, recall={row['recall']:.2%}, F1={row['f1']:.2%}")
    print(f"Graphs saved to {args.output}")


if __name__ == "__main__":
    main()
