import time
import math
import random
import requests

API = "http://127.0.0.1:8000/metrics"  # đổi thành http://<ip-pi>:8000 nếu API chạy máy khác

t0 = time.time()

while True:
    t = time.time() - t0

    # FPS dao động quanh 18–22 + nhiễu nhỏ
    fps = 20 + 2 * math.sin(t / 5) + random.uniform(-0.6, 0.6)

    # Latency tương quan nghịch với FPS (giả lập)
    latency_ms = max(25.0, 900.0 / max(fps, 1.0) + random.uniform(-3, 3))

    # Nhiệt độ tăng nhẹ theo thời gian + dao động
    cpu_temp_c = 62 + 0.02 * t + random.uniform(-0.4, 0.4)

    cpu_percent = min(95.0, 35 + 0.5 * t + random.uniform(-2, 2))
    ram_percent = min(90.0, 30 + random.uniform(-1, 1))

    # Số detect: đôi khi 0, đôi khi nhiều
    detect_count = random.choices([0, 1, 2, 3, 5], weights=[0.35, 0.25, 0.2, 0.15, 0.05])[0]

    camera_status = "ok" if random.random() > 0.02 else "error"

    payload = {
        "fps": float(fps),
        "latency_ms": float(latency_ms),
        "cpu_temp_c": float(cpu_temp_c),
        "cpu_percent": float(cpu_percent),
        "ram_percent": float(ram_percent),
        "detect_count": int(detect_count),
        "camera_status": camera_status,
    }

    try:
        r = requests.post(API, json=payload, timeout=1.0)
        print(r.status_code, payload)
    except Exception as e:
        print("POST failed:", e)

    time.sleep(1.0)  # 1 Hz; chỉnh 0.5 hoặc 2 tùy bạn