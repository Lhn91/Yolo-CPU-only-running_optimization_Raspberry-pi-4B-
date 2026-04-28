# CPU Evaluation – EdgeVisionRT

Monitors CPU/system metrics **while** running YOLO object detection on a
Raspberry Pi camera feed.

## Quick start (on the Pi)

```bash
cd EdgeVisionRT/CPU_evaluation
chmod +x run_eval_cpu.sh
./run_eval_cpu.sh
```

Press **Ctrl+C** to stop. All CSVs are saved under `results/<timestamp>/`.

## What gets recorded

| File | Source | Contents |
|------|--------|----------|
| `cpu_metrics.csv` | `cpu_monitor.py` | Temperature, per-core freq & usage, memory, throttle flags |
| `frame_metrics.csv` | C++ `yolo_inference` | Per-frame latency breakdown (capture/preprocess/inference/postprocess) |
| `interval_metrics.csv` | C++ `yolo_inference` | Windowed stats: FPS, P50/P95/P99 latency, memory, temperature |

## Configuration (environment variables)

| Variable | Default | Description |
|----------|---------|-------------|
| `MONITOR_INTERVAL` | `5` | Seconds between CPU samples |
| `SNAPSHOT_SEC` | `5` | C++ interval-output period |
| `DURATION_SEC` | `0` | Total run time (0 = until Ctrl+C) |
| `CAMERA_DEVICE` | `/dev/video0` | V4L2 camera device |
| `MODEL_NAME` | `721_M4_BaoLong.ncnn` | NCNN model base name |

Example – run for 1 hour with 15-minute snapshots:

```bash
DURATION_SEC=3600 SNAPSHOT_SEC=900 MONITOR_INTERVAL=900 ./run_eval_cpu.sh
```

## Standalone monitor

You can run `cpu_monitor.py` independently:

```bash
python3 cpu_monitor.py --output my_metrics.csv --interval 5
```
