"""
train_baseline.py
=================
Train YOLOv8s baseline for ship detection on HRSID dataset.

Optimized for RTX 4050 Laptop GPU (6GB VRAM).

Usage:
  python train_baseline.py
  python train_baseline.py --epochs 50 --batch 8  # override defaults
"""

import argparse
from pathlib import Path
import torch
from ultralytics import YOLO


# --- Configuration -----------------------------------------------------------

from project_paths import ROOT, DATASET_ROOT, RUNS_ROOT, PRETRAINED_MODEL
DATASET_YAML = DATASET_ROOT / "hrsid.yaml"
PROJECT_DIR = RUNS_ROOT
EXPERIMENT_NAME = "baseline_yolov8s"


def parse_args():
    parser = argparse.ArgumentParser(description="Train YOLOv8s on HRSID")
    parser.add_argument("--model", type=str, default=str(PRETRAINED_MODEL),
                        help="Pretrained model to start from (default: yolov8s.pt)")
    parser.add_argument("--data", type=Path, default=DATASET_YAML)
    parser.add_argument("--device", type=str, default="0", help="CUDA GPU index")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=100,
                        help="Number of training epochs (default: 100)")
    parser.add_argument("--batch", type=int, default=4,
                        help="Batch size (default: 4 for YOLOv8s on 6GB VRAM)")
    parser.add_argument("--imgsz", type=int, default=800,
                        help="Input image size (default: 800, native HRSID resolution)")
    parser.add_argument("--patience", type=int, default=25,
                        help="Early stopping patience (default: 25)")
    parser.add_argument("--workers", type=int, default=4,
                        help="DataLoader workers (default: 4)")
    parser.add_argument("--resume", action="store_true",
                        help="Resume training from last checkpoint")
    parser.add_argument("--name", type=str, default=EXPERIMENT_NAME,
                        help="Experiment name (default: baseline_yolov8s)")
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("YOLOv8s Ship Detection - Baseline Training")
    print("=" * 60)
    print(f"  Model:      {args.model}")
    print(f"  Dataset:    {args.data}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch}")
    print(f"  Image size: {args.imgsz}")
    print(f"  Patience:   {args.patience}")
    print(f"  Workers:    {args.workers}")
    print(f"  Project:    {PROJECT_DIR}")
    print(f"  Name:       {args.name}")
    print("=" * 60)

    # Verify dataset config exists
    if not args.data.exists():
        raise FileNotFoundError(
            f"Dataset YAML not found: {args.data}\n"
            "Run convert_hrsid.py first to create the YOLO-format dataset."
        )

    if args.device == "cpu" or not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU training is required; CUDA is unavailable or CPU was selected.")
    run_dir = PROJECT_DIR / args.name
    if not args.resume and run_dir.exists():
        raise FileExistsError(f"Run already exists: {run_dir}. Use --resume or a new --name.")

    # Load model
    if args.resume:
        # Resume from last checkpoint
        last_ckpt = PROJECT_DIR / args.name / "weights" / "last.pt"
        if not last_ckpt.exists():
            raise FileNotFoundError(f"No checkpoint found at: {last_ckpt}")
        model = YOLO(str(last_ckpt))
        print(f"\nResuming from: {last_ckpt}")
    else:
        model = YOLO(args.model)
        print(f"\nLoaded pretrained model: {args.model}")

    # Train
    results = model.train(
        data=str(args.data),
        device=args.device,
        seed=args.seed,
        deterministic=True,
        resume=args.resume,
        amp=True,
        cos_lr=True,
        close_mosaic=10,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        patience=args.patience,
        workers=args.workers,
        project=str(PROJECT_DIR),
        name=args.name,
        exist_ok=True,

        # --- Optimizer ---
        optimizer="SGD",
        lr0=0.01,
        lrf=0.01,       # Final LR = lr0 * lrf
        momentum=0.937,
        weight_decay=0.0005,

        # --- Augmentation (adapted for grayscale SAR) ---
        hsv_h=0.0,      # No hue augmentation (grayscale)
        hsv_s=0.0,      # No saturation augmentation (grayscale)
        hsv_v=0.1,      # Slight brightness jitter (useful for SAR intensity variation)
        degrees=15.0,    # Slight rotation (ships can be at any angle)
        translate=0.1,
        scale=0.5,
        fliplr=0.5,
        flipud=0.5,     # Vertical flip is valid for SAR overhead imagery
        mosaic=1.0,
        mixup=0.1,

        # --- Other ---
        save=True,
        save_period=10,  # Save checkpoint every 10 epochs
        plots=True,      # Generate training plots
        verbose=True,
    )

    print("\n" + "=" * 60)
    print("TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best model: {PROJECT_DIR / args.name / 'weights' / 'best.pt'}")
    print(f"  Results:    {PROJECT_DIR / args.name}")


if __name__ == "__main__":
    main()
