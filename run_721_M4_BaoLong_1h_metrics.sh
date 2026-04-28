#!/bin/bash
# Dedicated 1-hour camera benchmark for 721_M4_BaoLong.ncnn
# Example:
#   chmod +x run_721_M4_BaoLong_1h_metrics.sh
#   ./run_721_M4_BaoLong_1h_metrics.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "${SCRIPT_DIR}"

CAMERA_DEVICE="${CAMERA_DEVICE:-/dev/video0}"
DURATION_SEC="${DURATION_SEC:-3600}"
SNAPSHOT_SEC="${SNAPSHOT_SEC:-900}"
WARMUP_FRAMES="${WARMUP_FRAMES:-30}"
RUN_TAG="$(date +%Y%m%d_%H%M%S)"
RESULTS_DIR="${SCRIPT_DIR}/benchmark_runs/721_M4_BaoLong"
FRAME_CSV="${RESULTS_DIR}/${RUN_TAG}_frame_metrics.csv"
INTERVAL_CSV="${RESULTS_DIR}/${RUN_TAG}_interval_metrics.csv"
LOG_FILE="${RESULTS_DIR}/${RUN_TAG}_console.log"

mkdir -p "${RESULTS_DIR}"

echo "Dang build lai project..."
./build.sh clean && ./build.sh

echo "Dang dung giao dien Desktop (lightdm)..."
sudo systemctl stop lightdm 2>/dev/null || true

echo "Cap quyen Framebuffer cho LCD..."
sudo chmod 666 /dev/fb0 2>/dev/null || true

echo "Chay camera benchmark 1 gio voi model 721_M4_BaoLong.ncnn"
echo "Camera: ${CAMERA_DEVICE}"
echo "Frame CSV: ${FRAME_CSV}"
echo "Interval CSV: ${INTERVAL_CSV}"
echo "Console log: ${LOG_FILE}"
echo ""

./run.sh \
  cam "${CAMERA_DEVICE}" \
  fb \
  int8 \
  model 721_M4_BaoLong.ncnn \
  --frames 0 \
  --warmup "${WARMUP_FRAMES}" \
  --duration-sec "${DURATION_SEC}" \
  --snapshot-sec "${SNAPSHOT_SEC}" \
  --output "${FRAME_CSV}" \
  --interval-output "${INTERVAL_CSV}" 2>&1 | tee "${LOG_FILE}"
