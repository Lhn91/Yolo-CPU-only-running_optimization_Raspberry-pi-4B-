# Raspberry Pi Dashboard

Ứng dụng **FastAPI** thu thập metric từ Raspberry Pi (hoặc máy gửi dữ liệu khác) qua HTTP, lưu vào **MySQL**, và hiển thị dashboard web (Chart.js) cập nhật theo chu kỳ.

## Yêu cầu

- Python 3.10+ (khuyến nghị)

## Cài đặt

```bash
python -m venv venv
# Windows:
venv\Scripts\activate
# Linux / macOS:
# source venv/bin/activate

pip install -r requirements.txt
```

## Cấu hình MySQL

1. Tạo database:

```sql
CREATE DATABASE pi_dashboard CHARACTER SET utf8mb4 COLLATE utf8mb4_unicode_ci;
```

2. Thiết lập biến môi trường `DB_URL` trước khi chạy server:

- Windows PowerShell:

```powershell
$env:DB_URL="mysql+pymysql://root:password@127.0.0.1:3306/pi_dashboard"
```

- Linux / macOS:

```bash
export DB_URL="mysql+pymysql://root:password@127.0.0.1:3306/pi_dashboard"
```

Script mẫu `fake_metrics.py` dùng thêm thư viện `requests`:

```bash
pip install requests
```

## Chạy server API + dashboard

Từ thư mục gốc của repo (cùng cấp với thư mục `app/`):

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

- Dashboard: [http://127.0.0.1:8000/](http://127.0.0.1:8000/)
- Kiểm tra nhanh: [http://127.0.0.1:8000/health](http://127.0.0.1:8000/health)

Dùng `--host 0.0.0.0` để máy khác trong mạng LAN (hoặc Raspberry Pi) có thể gọi API bằng IP của máy chủ.

## Gửi dữ liệu thử (máy local)

Trong một terminal khác (vẫn kích hoạt venv, đã cài `requests`):

```bash
python fake_metrics.py
```

Sửa biến `API` trong `fake_metrics.py` thành URL đúng, ví dụ:

- Server chạy trên chính máy này: `http://127.0.0.1:8000/metrics`
- Server chạy trên PC khác / VPS: `http://192.168.x.x:8000/metrics`

## Raspberry Pi gửi metric thật

1. Cài Python và `pip install requests` (và tùy chọn `psutil` nếu đọc CPU/RAM).
2. Sao chép một script kiểu `fake_metrics.py`, giữ nguyên **POST JSON** tới `POST /metrics`.
3. Đặt `API = "http://<IP_máy_chạy_uvicorn>:8000/metrics"`.
4. Trong vòng lặp (`while True`), đọc số liệu thật (nhiệt độ, CPU, v.v.) rồi gửi `payload` cùng các field như bảng dưới; `time.sleep(...)` quyết định tần suất (ví dụ 1 giây).

**Lưu ý:** Nếu API chạy trên **PC**, Pi phải trỏ tới **IP của PC**. Nếu API chạy **trên Pi**, các máy khác mở dashboard bằng **IP của Pi**.

## API

| Phương thức | Đường dẫn | Mô tả |
|-------------|-----------|--------|
| `GET` | `/` | Trang dashboard tĩnh |
| `GET` | `/health` | Trạng thái service |
| `POST` | `/metrics` | Nhận một bản ghi metric (JSON) |
| `GET` | `/latest` | Metric mới nhất |
| `GET` | `/timeseries?minutes=10` | Chuỗi thời gian trong N phút gần đây |
| `GET` | `/history?page=1&page_size=20` | Lịch sử có phân trang |

### Body `POST /metrics` (JSON)

| Trường | Kiểu | Bắt buộc | Ghi chú |
|--------|------|----------|---------|
| `ts` | string (ISO datetime) | Không | Nếu bỏ qua, server dùng thời điểm nhận |
| `fps` | number | Có | |
| `latency_ms` | number | Có | |
| `cpu_temp_c` | number | Có | °C |
| `cpu_percent` | number | Có | |
| `ram_percent` | number | Có | |
| `detect_count` | integer | Không | Mặc định `0` |
| `camera_status` | string | Không | Mặc định `"ok"` |

Ví dụ tối thiểu:

```json
{
  "fps": 20.5,
  "latency_ms": 45.2,
  "cpu_temp_c": 55.0,
  "cpu_percent": 12.3,
  "ram_percent": 40.0,
  "detect_count": 0,
  "camera_status": "ok"
}
```

## Cấu trúc thư mục

```
app/
  main.py       # FastAPI, route
  db.py         # SQLite, model Metric
  schemas.py    # Pydantic MetricIn
  static/       # index.html, app.js, style.css
fake_metrics.py # Client giả lập gửi metric
requirements.txt
```

## Mạng và bảo mật

- Mở cổng **8000** trên tường lửa máy chủ nếu cần truy cập từ LAN.
- Repo này **không** có xác thực API; chỉ dùng trong mạng tin cậy hoặc bạn tự bổ sung (API key, VPN, v.v.) nếu expose ra Internet.

## Chạy nền trên Linux / Pi (gợi ý)

Dùng **systemd** với `Restart=always` để script gửi metric hoặc `uvicorn` tự khởi động lại sau reboot.
