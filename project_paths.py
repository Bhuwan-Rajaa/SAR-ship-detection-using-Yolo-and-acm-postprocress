"""Shared paths for scripts in useful_ahh/ or at the repository root."""
from pathlib import Path

CODE_ROOT = Path(__file__).resolve().parent
ROOT = CODE_ROOT.parent if CODE_ROOT.name == "useful_ahh" else CODE_ROOT
DATASET_ROOT = ROOT / "Dataset" / "hrsid_yolo"
RAW_DATASET_ROOT = ROOT / "Dataset" / "hrsid"
RUNS_ROOT = ROOT / "useless_ahh" / "runs"
EVALUATION_ROOT = ROOT / "useless_ahh" / "evaluation_results"
METRICS_ROOT = ROOT / "evaluation_results" / "yolov8_results"
PRETRAINED_MODEL = ROOT / "weights" / "yolov8s.pt"


def resolve_saved_path(value):
    """Relocate legacy dataset/run paths stored in experiment configuration."""
    path = Path(value)
    if path.exists():
        return path
    parts = str(value).replace("\\", "/").split("/")
    destinations = {"hrsid_yolo": DATASET_ROOT, "hrsid": RAW_DATASET_ROOT,
                    "evaluation_results": EVALUATION_ROOT, "runs": RUNS_ROOT}
    for index, part in enumerate(parts):
        if part in destinations:
            return destinations[part].joinpath(*parts[index + 1:])
    return path
