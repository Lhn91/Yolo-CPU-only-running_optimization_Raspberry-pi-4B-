#!/bin/bash
set -e

# Usage:
#   ./run_images_visdrone_eval.sh /home/lhn/Visdrone/test
# Exports:
#   - image detail csv:  <EdgeVisionRT>/Visdrone_image_detail.csv
#   - image summary csv: <EdgeVisionRT>/Visdrone_image_summary.csv
#   - predictions dir:    <EdgeVisionRT>/preds_yolo
#   - mAP metrics csv:   <EdgeVisionRT>/Visdrone_map_metrics.csv

DATASET_ROOT="${1:-/home/lhn/Visdrone/test}"
EDGE_DIR="$(cd "$(dirname "$0")" && pwd)"

IMG_DIR="${DATASET_ROOT}/images"
LAB_DIR="${DATASET_ROOT}/labels"

PARAM_PATH="${EDGE_DIR}/models/721_M4_BaoLong.ncnn.param"
BIN_PATH="${EDGE_DIR}/models/721_M4_BaoLong.ncnn.bin"

DETAIL_CSV="${EDGE_DIR}/Visdrone_image_detail.csv"
SUMMARY_CSV="${EDGE_DIR}/Visdrone_image_summary.csv"
PREDS_DIR="${EDGE_DIR}/preds_yolo"
MAP_CSV="${EDGE_DIR}/Visdrone_map_metrics.csv"

if [ ! -d "${IMG_DIR}" ]; then
  echo "Error: images folder not found: ${IMG_DIR}"
  exit 1
fi
if [ ! -d "${LAB_DIR}" ]; then
  echo "Error: labels folder not found: ${LAB_DIR}"
  exit 1
fi
if [ ! -f "${PARAM_PATH}" ]; then
  echo "Error: param not found: ${PARAM_PATH}"
  exit 1
fi
if [ ! -f "${BIN_PATH}" ]; then
  echo "Error: bin not found: ${BIN_PATH}"
  exit 1
fi

MAX_IMAGES=$(python3 - <<PY
import glob, os
img_dir = r"${IMG_DIR}"
imgs = []
for ext in ["jpg","jpeg","png","bmp","webp"]:
    imgs += glob.glob(os.path.join(img_dir, f"*.{ext}"))
print(len(imgs))
PY
)

if [ "${MAX_IMAGES}" -le 0 ]; then
  echo "Error: no images found under ${IMG_DIR}"
  exit 1
fi

echo "Running EdgeVisionRT on ${MAX_IMAGES} images..."
export OMP_NUM_THREADS=4

mkdir -p "${PREDS_DIR}"

sudo -n true 2>/dev/null || true

cd "${EDGE_DIR}"
./build/yolo_inference \
  --images-dir "${IMG_DIR}" \
  --param "${PARAM_PATH}" \
  --bin "${BIN_PATH}" \
  --frames "${MAX_IMAGES}" \
  --warmup 30 \
  --output-detail "${DETAIL_CSV}" \
  --output-summary "${SUMMARY_CSV}" \
  --preds-dir "${PREDS_DIR}"

echo "Computing mAP (COCOeval) from exported YOLO preds..."
python3 scripts/eval_map_images_from_yolo_preds.py \
  --images-dir "${IMG_DIR}" \
  --labels-dir "${LAB_DIR}" \
  --preds-dir "${PREDS_DIR}" \
  --out-metrics-csv "${MAP_CSV}" \
  --num-classes 1

echo "Done."
echo "Detail CSV:  ${DETAIL_CSV}"
echo "Summary CSV: ${SUMMARY_CSV}"
echo "mAP CSV:     ${MAP_CSV}"

