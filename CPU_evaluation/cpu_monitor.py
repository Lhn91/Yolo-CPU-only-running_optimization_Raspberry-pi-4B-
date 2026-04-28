#!/usr/bin/env python3
"""
CPU & System Metrics Monitor for Raspberry Pi.

Periodically samples CPU usage, temperature, frequency, memory, and
throttling status, then appends a row to a CSV file.

Only uses the Python standard library — no pip packages required.

Usage:
    python3 cpu_monitor.py --output metrics.csv --interval 5
    python3 cpu_monitor.py --help
"""

import argparse
import csv
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

_running = True


def _signal_handler(signum, frame):
    global _running
    _running = False


# ── Readers ──────────────────────────────────────────────────────────────

def read_cpu_temperature() -> float:
    """Read SoC temperature in °C from sysfs, fallback to vcgencmd."""
    try:
        with open("/sys/class/thermal/thermal_zone0/temp") as f:
            return int(f.read().strip()) / 1000.0
    except (OSError, ValueError):
        pass
    try:
        out = subprocess.check_output(
            ["vcgencmd", "measure_temp"], stderr=subprocess.DEVNULL, timeout=2
        ).decode()
        return float(out.split("=")[1].split("'")[0])
    except Exception:
        return float("nan")


def read_cpu_frequencies():
    """Return current frequency (MHz) for each CPU core."""
    freqs = []
    core = 0
    while True:
        path = f"/sys/devices/system/cpu/cpu{core}/cpufreq/scaling_cur_freq"
        try:
            with open(path) as f:
                freqs.append(int(f.read().strip()) / 1000.0)
        except OSError:
            break
        core += 1
    if not freqs:
        try:
            out = subprocess.check_output(
                ["vcgencmd", "measure_clock", "arm"], stderr=subprocess.DEVNULL, timeout=2
            ).decode()
            hz = int(out.strip().split("=")[1])
            freqs.append(hz / 1_000_000.0)
        except Exception:
            pass
    return freqs


def read_proc_stat() -> dict:
    """Parse /proc/stat and return per-core + total idle/total jiffies."""
    result = {}
    try:
        with open("/proc/stat") as f:
            for line in f:
                if not line.startswith("cpu"):
                    break
                parts = line.split()
                name = parts[0]
                values = list(map(int, parts[1:]))
                idle = values[3] + (values[4] if len(values) > 4 else 0)
                total = sum(values)
                result[name] = {"idle": idle, "total": total}
    except OSError:
        pass
    return result


def compute_cpu_usage(prev, curr):
    """Compute CPU usage percentage from two /proc/stat snapshots."""
    usage = {}
    for name in curr:
        if name not in prev:
            continue
        d_total = curr[name]["total"] - prev[name]["total"]
        d_idle = curr[name]["idle"] - prev[name]["idle"]
        if d_total > 0:
            usage[name] = (1.0 - d_idle / d_total) * 100.0
        else:
            usage[name] = 0.0
    return usage


def read_memory_info():
    """Return memory stats in MB from /proc/meminfo."""
    info = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                parts = line.split()
                key = parts[0].rstrip(":")
                if key in ("MemTotal", "MemFree", "MemAvailable", "Buffers", "Cached"):
                    info[key] = int(parts[1])
    except OSError:
        return {"total_mb": 0, "used_mb": 0, "available_mb": 0, "usage_pct": 0}

    total = info.get("MemTotal", 0)
    available = info.get("MemAvailable", 0)
    used = total - available
    return {
        "total_mb": round(total / 1024.0, 1),
        "used_mb": round(used / 1024.0, 1),
        "available_mb": round(available / 1024.0, 1),
        "usage_pct": round(used / total * 100, 1) if total else 0.0,
    }


def read_throttled() -> str:
    """Return vcgencmd get_throttled hex string (e.g. '0x0')."""
    try:
        out = subprocess.check_output(
            ["vcgencmd", "get_throttled"], stderr=subprocess.DEVNULL, timeout=2
        ).decode().strip()
        return out.split("=")[1] if "=" in out else out
    except Exception:
        return "n/a"


def read_governor() -> str:
    try:
        with open("/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor") as f:
            return f.read().strip()
    except OSError:
        return "n/a"


# ── Main loop ────────────────────────────────────────────────────────────

def build_csv_header(num_cores):
    header = ["timestamp", "elapsed_sec", "cpu_temp_c"]
    for i in range(num_cores):
        header.append(f"cpu_freq_mhz_core{i}")
    for i in range(num_cores):
        header.append(f"cpu_usage_pct_core{i}")
    header += [
        "cpu_usage_pct_total",
        "mem_total_mb", "mem_used_mb", "mem_available_mb", "mem_usage_pct",
        "throttled_hex", "governor",
    ]
    return header


def collect_row(
    start_time: float,
    prev_stat: dict,
    num_cores: int,
):
    """Collect one sample and return (row_values, new_proc_stat)."""
    now = time.time()
    elapsed = round(now - start_time, 3)
    timestamp = time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(now))

    temp = read_cpu_temperature()
    freqs = read_cpu_frequencies()
    cur_stat = read_proc_stat()
    usage = compute_cpu_usage(prev_stat, cur_stat)
    mem = read_memory_info()
    throttled = read_throttled()
    governor = read_governor()

    row = [timestamp, elapsed, round(temp, 2)]

    for i in range(num_cores):
        row.append(round(freqs[i], 1) if i < len(freqs) else "")

    for i in range(num_cores):
        core_name = f"cpu{i}"
        row.append(round(usage.get(core_name, 0.0), 1))

    row.append(round(usage.get("cpu", 0.0), 1))
    row += [mem["total_mb"], mem["used_mb"], mem["available_mb"], mem["usage_pct"]]
    row += [throttled, governor]

    return row, cur_stat


def detect_num_cores() -> int:
    n = 0
    while os.path.exists(f"/sys/devices/system/cpu/cpu{n}"):
        n += 1
    return max(n, 1)


def main():
    parser = argparse.ArgumentParser(description="CPU & system metrics monitor for Raspberry Pi")
    parser.add_argument("-o", "--output", required=True, help="Output CSV path")
    parser.add_argument("-i", "--interval", type=float, default=5.0,
                        help="Sampling interval in seconds (default: 5)")
    parser.add_argument("--cores", type=int, default=0,
                        help="Number of CPU cores (0 = auto-detect)")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)

    num_cores = args.cores if args.cores > 0 else detect_num_cores()
    header = build_csv_header(num_cores)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    prev_stat = read_proc_stat()
    time.sleep(0.1)

    start_time = time.time()
    sample_count = 0

    print(f"[cpu_monitor] Writing to {out_path}")
    print(f"[cpu_monitor] Interval: {args.interval}s | Cores: {num_cores}")
    print(f"[cpu_monitor] Press Ctrl+C or send SIGTERM to stop.\n")

    with open(out_path, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(header)
        csvfile.flush()

        while _running:
            row, prev_stat = collect_row(start_time, prev_stat, num_cores)
            writer.writerow(row)
            csvfile.flush()
            sample_count += 1

            if sample_count % 12 == 1:
                print(f"[cpu_monitor] #{sample_count}  temp={row[2]}°C  "
                      f"cpu_total={row[3 + num_cores * 2]}%  "
                      f"mem={row[3 + num_cores * 2 + 2]}/{row[3 + num_cores * 2 + 1]} MB")

            next_tick = start_time + sample_count * args.interval
            sleep_dur = next_tick - time.time()
            if sleep_dur > 0:
                time.sleep(sleep_dur)

    print(f"\n[cpu_monitor] Stopped. {sample_count} samples written to {out_path}")


if __name__ == "__main__":
    main()
