#!/usr/bin/env python3
"""
Compute COCO mAP from:
  - ground-truth YOLO labels in labels_dir (one .txt per image, YOLO normalized: cls xc yc w h)
  - predictions exported by EdgeVisionRT in preds_dir (one .txt per image)
    format per line: cls xc yc w h conf

This script is intentionally lightweight: it converts YOLO -> COCO in-memory,
then runs pycocotools COCOeval.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def list_images(images_dir: Path) -> list[Path]:
    paths = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in IMG_EXTS]
    paths.sort(key=lambda x: x.name)
    return paths


def read_yolo_labels(label_file: Path) -> list[tuple[int, float, float, float, float]]:
    # Returns: (cls, xc, yc, w, h) normalized
    if not label_file.exists():
        return []
    out = []
    with label_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 5:
                continue
            cls = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:5])
            out.append((cls, xc, yc, w, h))
    return out


def read_yolo_preds(pred_file: Path) -> list[tuple[int, float, float, float, float, float]]:
    # Returns: (cls, xc, yc, w, h, conf) normalized
    if not pred_file.exists():
        return []
    out = []
    with pred_file.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 6:
                continue
            cls = int(float(parts[0]))
            xc, yc, w, h = map(float, parts[1:5])
            conf = float(parts[5])
            out.append((cls, xc, yc, w, h, conf))
    return out


def yolo_norm_to_coco_bbox(xc: float, yc: float, w: float, h: float, img_w: int, img_h: int) -> list[float]:
    bw = w * img_w
    bh = h * img_h
    x1 = (xc * img_w) - bw / 2.0
    y1 = (yc * img_h) - bh / 2.0

    # Clip to image bounds
    x1 = max(0.0, min(float(img_w - 1), x1))
    y1 = max(0.0, min(float(img_h - 1), y1))
    bw = max(0.0, min(float(img_w), bw))
    bh = max(0.0, min(float(img_h), bh))
    return [x1, y1, bw, bh]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--labels-dir", required=True)
    ap.add_argument("--preds-dir", required=True)
    ap.add_argument("--out-metrics-csv", required=True)
    ap.add_argument("--num-classes", type=int, default=1)
    args = ap.parse_args()

    images_dir = Path(args.images_dir)
    labels_dir = Path(args.labels_dir)
    preds_dir = Path(args.preds_dir)
    out_csv = Path(args.out_metrics_csv)

    if not images_dir.is_dir():
        raise SystemExit(f"images-dir not found: {images_dir}")
    if not labels_dir.is_dir():
        raise SystemExit(f"labels-dir not found: {labels_dir}")
    if not preds_dir.is_dir():
        raise SystemExit(f"preds-dir not found: {preds_dir}")

    image_paths = list_images(images_dir)
    if not image_paths:
        raise SystemExit(f"No images found in {images_dir}")

    # Build COCO GT
    gt = {
        "images": [],
        "categories": [{"id": i + 1, "name": f"class_{i}"} for i in range(args.num_classes)],
        "annotations": [],
    }

    ann_id = 1
    image_id_map = {}  # filename -> COCO image id

    for coco_i, img_path in enumerate(image_paths, start=1):
        img = Image.open(img_path)
        w, h = img.size
        img_id = coco_i
        image_id_map[img_path.name] = img_id

        gt["images"].append(
            {"id": img_id, "file_name": img_path.name, "width": w, "height": h}
        )

        label_file = labels_dir / f"{img_path.stem}.txt"
        labels = read_yolo_labels(label_file)
        for (cls, xc, yc, bw, bh) in labels:
            if cls < 0 or cls >= args.num_classes:
                continue
            x, y, abs_w, abs_h = yolo_norm_to_coco_bbox(xc, yc, bw, bh, w, h)
            area = float(abs_w * abs_h)
            gt["annotations"].append(
                {
                    "id": ann_id,
                    "image_id": img_id,
                    "category_id": int(cls) + 1,
                    "bbox": [x, y, abs_w, abs_h],
                    "area": area,
                    "iscrowd": 0,
                }
            )
            ann_id += 1

    gt_json_path = out_csv.parent / "gt_coco_tmp.json"
    dt_json_path = out_csv.parent / "dt_coco_tmp.json"
    with gt_json_path.open("w", encoding="utf-8") as f:
        json.dump(gt, f)

    # Build COCO detections
    dt = []
    for img_path in image_paths:
        img_id = image_id_map[img_path.name]
        img = Image.open(img_path)
        w, h = img.size

        pred_file = preds_dir / f"{img_path.stem}.txt"
        preds = read_yolo_preds(pred_file)
        for (cls, xc, yc, pw, ph, conf) in preds:
            if cls < 0 or cls >= args.num_classes:
                continue
            x, y, abs_w, abs_h = yolo_norm_to_coco_bbox(xc, yc, pw, ph, w, h)
            dt.append(
                {
                    "image_id": img_id,
                    "category_id": int(cls) + 1,
                    "bbox": [x, y, abs_w, abs_h],
                    "score": float(conf),
                }
            )

    with dt_json_path.open("w", encoding="utf-8") as f:
        json.dump(dt, f)

    coco_gt = COCO(str(gt_json_path))
    coco_dt = coco_gt.loadRes(str(dt_json_path))
    coco_eval = COCOeval(coco_gt, coco_dt, iouType="bbox")
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()

    stats = coco_eval.stats
    # stats:
    # [0] AP@[.5:.95], [1] AP@.5, [2] AP@.75, [3] AP_small, [4] AP_medium, [5] AP_large,
    # [6] AR@1, [7] AR@10, [8] AR@100, [9] AR_small, [10] AR_medium, [11] AR_large
    results = {
        "mAP_50_95": float(stats[0]),
        "mAP_50": float(stats[1]),
        "mAP_75": float(stats[2]),
        "AP_small": float(stats[3]),
        "AP_medium": float(stats[4]),
        "AP_large": float(stats[5]),
        "AR_max100": float(stats[8]),
    }

    # Rough P/R estimate from COCO precision/recall tensors
    # Precision tensor shape: [TxRxKxAxM]
    # We'll average valid precision values.
    precision = coco_eval.eval.get("precision")
    recall = coco_eval.eval.get("recall")
    if precision is not None:
        valid_prec = precision[precision >= 0]
        p_mean = float(valid_prec.mean()) if valid_prec.size else 0.0
    else:
        p_mean = 0.0
    if recall is not None:
        valid_rec = recall[recall >= 0]
        r_mean = float(valid_rec.mean()) if valid_rec.size else 0.0
    else:
        r_mean = 0.0
    f1 = (2 * p_mean * r_mean / (p_mean + r_mean)) if (p_mean + r_mean) > 0 else 0.0
    results["precision_mean"] = p_mean
    results["recall_mean"] = r_mean
    results["f1_mean"] = f1

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8") as f:
        f.write("metric,value\n")
        for k, v in results.items():
            f.write(f"{k},{v:.6f}\n")

    print(f"Saved metrics to: {out_csv}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

