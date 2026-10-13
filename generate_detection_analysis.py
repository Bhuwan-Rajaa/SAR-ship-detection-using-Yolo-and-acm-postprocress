"""
generate_detection_analysis.py
==============================
Extracts detailed per-detection Chan-Vese Active Contour Model (ACM) features
for YOLO detections, matched against ground-truth labels and masks.

Generates a CSV containing:
- image_id: Image filename stem
- conf: YOLO detection confidence score
- is_tp: 1 if True Positive (IoU >= 0.5 with GT), 0 if False Positive
- initial_energy: Chan-Vese energy at iteration 0 (checkerboard init)
- final_energy: Chan-Vese energy after contour evolution
- energy_reduction: initial_energy - final_energy
- energy_reduction_rel: (initial - final) / initial
- contrast: Foreground-to-background intensity contrast ratio
- mask_to_obb_area_ratio: Ratio of mask area to minimum-area rotated bounding box (OBB)
- contour_stability: Jaccard similarity between contour at N-10 and N iterations
- gt_mask_iou: Intersection-over-Union with ground-truth polygon mask (0.0 if FP)
- area_ratio: Mask area / bbox area
- compactness: 4 * pi * area / perimeter^2
- solidity: area / convex hull area
- elongation: major axis / minor axis
- intensity_mean: Mean foreground pixel intensity
- intensity_variance: Variance of foreground pixel intensities
- segment_area: Total foreground pixels
"""

import argparse
import json
import csv
import time
import numpy as np
import cv2
from pathlib import Path
from ultralytics import YOLO
from skimage.segmentation import checkerboard_level_set
from deterministic_acm import chan_vese
from scipy.ndimage import median_filter


def compute_iou(box1, box2):
    """Compute IoU between two [x1, y1, x2, y2] boxes."""
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])

    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = max(0, box1[2] - box1[0]) * max(0, box1[3] - box1[1])
    area2 = max(0, box2[2] - box2[0]) * max(0, box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0.0


def match_detections(det_boxes, gt_boxes, iou_thresh=0.5):
    """
    Greedy matching between detection boxes and GT boxes.
    Returns:
        matches: dict mapping det_idx -> gt_idx (or None if FP)
    """
    matches = {i: None for i in range(len(det_boxes))}
    if len(det_boxes) == 0 or len(gt_boxes) == 0:
        return matches

    iou_matrix = np.zeros((len(det_boxes), len(gt_boxes)))
    for i, det in enumerate(det_boxes):
        for j, gt in enumerate(gt_boxes):
            iou_matrix[i, j] = compute_iou(det, gt)

    used_gt = set()
    while True:
        if iou_matrix.size == 0:
            break
        max_val = iou_matrix.max()
        if max_val < iou_thresh:
            break
        det_idx, gt_idx = np.unravel_index(iou_matrix.argmax(), iou_matrix.shape)
        matches[det_idx] = gt_idx
        used_gt.add(gt_idx)
        iou_matrix[det_idx, :] = -1
        iou_matrix[:, gt_idx] = -1

    return matches


def compute_chan_vese_energy(image_crop, mask, lambda1=1.0, lambda2=1.5, mu=0.1):
    """
    Calculate Chan-Vese energy functional:
    E = lambda1 * integral_inside (I - c1)^2 + lambda2 * integral_outside (I - c2)^2 + mu * Perimeter
    """
    m = mask.astype(bool)
    n1 = int(np.sum(m))
    n0 = int(np.sum(~m))

    c1 = float(np.mean(image_crop[m])) if n1 > 0 else 0.0
    c2 = float(np.mean(image_crop[~m])) if n0 > 0 else 0.0

    fit1 = float(lambda1 * np.sum((image_crop[m] - c1) ** 2)) if n1 > 0 else 0.0
    fit2 = float(lambda2 * np.sum((image_crop[~m] - c2) ** 2)) if n0 > 0 else 0.0

    # Perimeter via morphological gradient
    eroded = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8))
    perimeter = float(np.sum(mask.astype(np.uint8) - eroded))

    return fit1 + fit2 + mu * perimeter, c1, c2


def extract_detection_features(
    gray_img,
    bbox,
    conf,
    gt_ann,
    num_iter=45,
    bbox_padding=0.5,
    lambda1=1.0,
    lambda2=1.5,
    mu=0.1,
):
    """
    Runs Chan-Vese segmentation and extracts all required analytical features.
    """
    img_h, img_w = gray_img.shape
    x1, y1, x2, y2 = bbox.astype(int)
    bw = x2 - x1
    bh = y2 - y1

    if bw < 3 or bh < 3:
        return None

    # Padded crop
    pad_x = int(bw * bbox_padding)
    pad_y = int(bh * bbox_padding)
    cx1 = max(0, x1 - pad_x)
    cy1 = max(0, y1 - pad_y)
    cx2 = min(img_w, x2 + pad_x)
    cy2 = min(img_h, y2 + pad_y)

    crop = gray_img[cy1:cy2, cx1:cx2]
    if crop.size == 0 or crop.shape[0] < 3 or crop.shape[1] < 3:
        return None

    crop = median_filter(crop, size=3)

    # Initial level set (checkerboard)
    init_ls = checkerboard_level_set(crop.shape, 6)
    init_energy, _, _ = compute_chan_vese_energy(crop, init_ls, lambda1, lambda2, mu)

    # Run N - 10 iterations
    n_pre = max(10, num_iter - 10)
    mask_pre = chan_vese(
        crop,
        num_iter=n_pre,
        init_level_set=init_ls,
        lambda1=lambda1,
        lambda2=lambda2,
    )

    # Run remaining 10 iterations
    mask_final = chan_vese(
        crop,
        num_iter=10,
        init_level_set=mask_pre,
        start_iteration=n_pre,
        lambda1=lambda1,
        lambda2=lambda2,
    )

    # Stability: Jaccard similarity between mask_pre and mask_final
    inter = np.sum((mask_pre == 1) & (mask_final == 1))
    union = np.sum((mask_pre == 1) | (mask_final == 1))
    stability = float(inter / (union + 1e-6))

    # Polarize labels so foreground is brighter (SAR ships are bright scatterers)
    m_fg = mask_final == 1
    m_bg = mask_final == 0
    mean_fg = float(crop[m_fg].mean()) if m_fg.sum() > 0 else 0.0
    mean_bg = float(crop[m_bg].mean()) if m_bg.sum() > 0 else 0.0

    if mean_fg < mean_bg:
        mask_final = 1 - mask_final
        mean_fg, mean_bg = mean_bg, mean_fg

    # Compute final energy
    final_energy, c1, c2 = compute_chan_vese_energy(crop, mask_final, lambda1, lambda2, mu)
    energy_reduction = float(init_energy - final_energy)
    energy_reduction_rel = float(energy_reduction / (init_energy + 1e-8))

    # Foreground-background intensity contrast
    contrast = float(mean_fg / (mean_bg + 1e-6))

    # Morphological descriptors
    seg_pixels = int(np.sum(mask_final))
    area_ratio = float(seg_pixels / (bw * bh)) if (bw * bh) > 0 else 0.0

    # Contours & OBB
    contours, _ = cv2.findContours(
        mask_final.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    mask_to_obb_area_ratio = 0.0
    compactness = 0.0
    solidity = 0.0
    elongation = 1.0

    if contours:
        largest = max(contours, key=cv2.contourArea)
        area_c = cv2.contourArea(largest)
        perim_c = cv2.arcLength(largest, True)

        if perim_c > 0:
            compactness = float(4.0 * np.pi * area_c / (perim_c ** 2))

        # Convex hull for solidity
        hull = cv2.convexHull(largest)
        hull_area = cv2.contourArea(hull)
        if hull_area > 0:
            solidity = float(area_c / hull_area)

        # MinAreaRect for OBB
        if len(largest) >= 5:
            rect = cv2.minAreaRect(largest)
            (rx, ry), (rw, rh), angle = rect
            obb_area = float(rw * rh)
            if obb_area > 0:
                mask_to_obb_area_ratio = float(area_c / obb_area)
            if min(rw, rh) > 0:
                elongation = float(max(rw, rh) / min(rw, rh))

    # Pixel intensity statistics
    fg_pixels = crop[mask_final == 1]
    if len(fg_pixels) > 0:
        intensity_mean = float(np.mean(fg_pixels))
        intensity_var = float(np.var(fg_pixels))
    else:
        intensity_mean = 0.0
        intensity_var = 0.0

    # Ground-truth mask IoU (where matched to GT)
    gt_mask_iou = 0.0
    if gt_ann is not None and "segmentation" in gt_ann:
        gt_crop_mask = np.zeros(crop.shape, dtype=np.uint8)
        for poly in gt_ann["segmentation"]:
            if len(poly) >= 6:
                pts = np.array(poly).reshape(-1, 2)
                # Translate to crop coordinates
                pts_crop = pts - np.array([cx1, cy1])
                cv2.fillPoly(gt_crop_mask, [pts_crop.astype(np.int32)], 1)

        m_inter = np.sum((mask_final == 1) & (gt_crop_mask == 1))
        m_union = np.sum((mask_final == 1) | (gt_crop_mask == 1))
        gt_mask_iou = float(m_inter / (m_union + 1e-6))

    return {
        "conf": float(conf),
        "is_tp": 1 if gt_ann is not None else 0,
        "initial_energy": init_energy,
        "final_energy": final_energy,
        "energy_reduction": energy_reduction,
        "energy_reduction_rel": energy_reduction_rel,
        "contrast": contrast,
        "mask_to_obb_area_ratio": mask_to_obb_area_ratio,
        "contour_stability": stability,
        "gt_mask_iou": gt_mask_iou,
        "area_ratio": area_ratio,
        "compactness": compactness,
        "solidity": solidity,
        "elongation": elongation,
        "intensity_mean": intensity_mean,
        "intensity_variance": intensity_var,
        "segment_area": seg_pixels,
    }


def main():
    from project_paths import DATASET_ROOT, RAW_DATASET_ROOT, RUNS_ROOT, EVALUATION_ROOT
    parser = argparse.ArgumentParser(description="Generate per-detection ACM analysis CSV")
    parser.add_argument("--split", type=str, default="val", choices=["val", "test"])
    parser.add_argument("--model", type=str, default=str(RUNS_ROOT / "baseline_yolov8s/weights/best.pt"))
    parser.add_argument("--dataset", type=Path, default=DATASET_ROOT)
    parser.add_argument("--annotations", type=Path, default=RAW_DATASET_ROOT / "annotations")
    parser.add_argument("--imgsz", type=int, default=800)
    parser.add_argument("--device", type=str, default="0")
    parser.add_argument("--conf", type=float, default=0.10, help="YOLO confidence cutoff for candidate extraction")
    parser.add_argument("--iou", type=float, default=0.5, help="NMS IoU threshold")
    parser.add_argument("--num_iter", type=int, default=45, help="ACM iterations")
    parser.add_argument("--max_images", type=int, default=None, help="Optional limit for rapid testing")
    parser.add_argument("--output", type=str, default=None)
    args = parser.parse_args()

    # Paths
    split_dir = args.dataset / args.split / "images"
    if args.split == "val":
        coco_json = args.annotations / "train2017.json"
    else:
        coco_json = args.annotations / "test2017.json"

    if args.output is None:
        out_csv = EVALUATION_ROOT / "yolov8s_detection_analysis" / f"{args.split}_detection_analysis.csv"
    else:
        out_csv = Path(args.output)
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading COCO annotations from {coco_json.name}...")
    with open(coco_json, "r") as f:
        coco = json.load(f)

    # Build image filename -> list of annotations map
    id_to_file = {img["id"]: img["file_name"] for img in coco["images"]}
    file_to_anns = {}
    for ann in coco["annotations"]:
        fn = id_to_file.get(ann["image_id"])
        if fn:
            file_to_anns.setdefault(fn, []).append(ann)

    print(f"Loading YOLO model from {args.model}...")
    model = YOLO(args.model)

    image_paths = sorted(split_dir.glob("*.png"))
    if args.max_images:
        image_paths = image_paths[:args.max_images]
    print(f"Processing {len(image_paths)} images from {args.split} split...")

    records = []
    t0 = time.time()

    for idx, img_p in enumerate(image_paths):
        stem = img_p.stem
        fn = img_p.name

        img = cv2.imread(str(img_p), cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        h, w = img.shape
        gray = img.astype(np.float64)
        if gray.max() > 0:
            gray /= gray.max()

        # Run YOLO inference
        results = model.predict(source=str(img_p), conf=args.conf, iou=args.iou, imgsz=args.imgsz, device=args.device, verbose=False)
        r = results[0]
        if len(r.boxes) == 0:
            continue

        det_boxes = r.boxes.xyxy.cpu().numpy()
        det_confs = r.boxes.conf.cpu().numpy()

        # Load GT for this image
        anns = file_to_anns.get(fn, [])
        gt_boxes = []
        for a in anns:
            bx, by, bw, bh = a["bbox"]
            gt_boxes.append([bx, by, bx + bw, by + bh])
        gt_boxes = np.array(gt_boxes) if gt_boxes else np.zeros((0, 4))

        # Match detections to GT
        matches = match_detections(det_boxes, gt_boxes, iou_thresh=0.5)

        for d_i, bbox in enumerate(det_boxes):
            c = float(det_confs[d_i])
            gt_idx = matches[d_i]
            gt_ann = anns[gt_idx] if gt_idx is not None else None

            feats = extract_detection_features(
                gray, bbox, c, gt_ann, num_iter=args.num_iter
            )
            if feats is not None:
                record = {"image_id": stem}
                record.update(feats)
                records.append(record)

        if (idx + 1) % 50 == 0 or (idx + 1) == len(image_paths):
            elapsed = time.time() - t0
            print(f"  [{idx+1}/{len(image_paths)}] Extracted {len(records)} detections ({elapsed:.1f}s)")

    # Save to CSV
    if records:
        with open(out_csv, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        print(f"\nSuccessfully wrote {len(records)} detection records to:\n  {out_csv}")
    else:
        print("No detections found.")


if __name__ == "__main__":
    main()
