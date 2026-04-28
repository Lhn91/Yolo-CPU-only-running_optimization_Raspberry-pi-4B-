from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock
from typing import Optional

from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, Response
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from sqlalchemy.orm import Session

from .db import SessionLocal, Metric, init_db
from .schemas import MetricIn, BoxesIn, StreamConfigIn

app = FastAPI(title="CoralVisionRT Remote Monitor")


# ─── In-memory live state (no DB) ─────────────────────────────────────────────

_state_lock = Lock()
_latest_boxes: dict = {"txt": "", "frame_w": 0, "frame_h": 0, "ts": None}
_stream_url: Optional[str] = None
_latest_snapshot: bytes = b""
_latest_snapshot_ts: Optional[datetime] = None
_servo_direction: int = 0  # -1 (CCW), 0 (Stop), 1 (CW)

class ServoIn(BaseModel):
    direction: int


def metric_to_dict(row: Metric):
    return {
        "id": row.id,
        "ts": row.ts.isoformat() if row.ts else None,
        "fps": row.fps,
        "latency_ms": row.latency_ms,
        "cpu_temp_c": row.cpu_temp_c,
        "cpu_percent": row.cpu_percent,
        "ram_percent": row.ram_percent,
        "detect_count": row.detect_count,
        "camera_status": row.camera_status,
    }


def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


@app.on_event("startup")
def on_startup():
    init_db()


# ─── Health ───────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    return {"status": "ok", "time": datetime.utcnow().isoformat()}


# ─── Metrics (DB-backed) ──────────────────────────────────────────────────────

@app.post("/metrics")
def push_metric(payload: MetricIn, db: Session = Depends(get_db)):
    row = Metric(
        ts=payload.ts or datetime.utcnow(),
        fps=payload.fps,
        latency_ms=payload.latency_ms,
        cpu_temp_c=payload.cpu_temp_c,
        cpu_percent=payload.cpu_percent,
        ram_percent=payload.ram_percent,
        detect_count=payload.detect_count,
        camera_status=payload.camera_status,
    )
    db.add(row)
    db.commit()
    db.refresh(row)
    return {"ok": True, "id": row.id}


@app.get("/latest")
def latest(db: Session = Depends(get_db)):
    row = db.query(Metric).order_by(Metric.id.desc()).first()
    if not row:
        return {"ok": False, "message": "No data"}
    return {"ok": True, "data": metric_to_dict(row)}


@app.get("/timeseries")
def timeseries(minutes: int = 10, db: Session = Depends(get_db)):
    since = datetime.utcnow() - timedelta(minutes=minutes)
    rows = (
        db.query(Metric)
        .filter(Metric.ts >= since)
        .order_by(Metric.ts.asc())
        .all()
    )
    data = [
        {
            "ts": r.ts.isoformat() if r.ts else None,
            "fps": r.fps,
            "latency_ms": r.latency_ms,
            "cpu_temp_c": r.cpu_temp_c,
            "detect_count": r.detect_count,
        }
        for r in rows
    ]
    return {"ok": True, "minutes": minutes, "count": len(data), "data": data}


@app.get("/history")
def history(page: int = 1, page_size: int = 20, db: Session = Depends(get_db)):
    page = max(page, 1)
    page_size = max(1, min(page_size, 200))

    base_query = db.query(Metric)
    total = base_query.count()
    rows = (
        base_query
        .order_by(Metric.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )

    return {
        "ok": True,
        "page": page,
        "page_size": page_size,
        "total": total,
        "total_pages": (total + page_size - 1) // page_size,
        "data": [metric_to_dict(row) for row in rows],
    }


# ─── Bounding boxes (in-memory, latest only) ──────────────────────────────────

@app.post("/boxes")
def push_boxes(payload: BoxesIn):
    global _latest_boxes
    with _state_lock:
        _latest_boxes = {
            "txt": payload.txt,
            "frame_w": payload.frame_w,
            "frame_h": payload.frame_h,
            "ts": payload.ts or datetime.utcnow().isoformat(),
        }
    return {"ok": True}


@app.get("/boxes")
def get_boxes():
    with _state_lock:
        return {"ok": True, "data": dict(_latest_boxes)}


# ─── Stream config (Pi tells dashboard its Tailscale MJPEG URL) ───────────────

@app.post("/stream-config")
def post_stream_config(payload: StreamConfigIn):
    global _stream_url
    with _state_lock:
        _stream_url = payload.stream_url
    return {"ok": True, "stream_url": payload.stream_url}


@app.get("/stream-config")
def get_stream_config():
    with _state_lock:
        return {"ok": True, "stream_url": _stream_url}


# ─── Snapshot (1 fps JPEG via Cloudflare for public viewers) ──────────────────

@app.post("/snapshot")
async def upload_snapshot(file: UploadFile = File(...)):
    """Pi POSTs latest annotated JPEG ~1 Hz so public viewers (no Tailscale)
    still see something. Stored in RAM only."""
    global _latest_snapshot, _latest_snapshot_ts
    data = await file.read()
    if not data:
        raise HTTPException(status_code=400, detail="empty payload")
    with _state_lock:
        _latest_snapshot = data
        _latest_snapshot_ts = datetime.utcnow()
    return {"ok": True, "size": len(data)}


@app.get("/snapshot.jpg")
def get_snapshot():
    with _state_lock:
        data = _latest_snapshot
    if not data:
        raise HTTPException(status_code=404, detail="no snapshot yet")
    return Response(
        content=data,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "no-store, no-cache, must-revalidate",
            "Pragma": "no-cache",
        },
    )


# ─── Servo Control ────────────────────────────────────────────────────────────

@app.post("/servo")
def post_servo(payload: ServoIn):
    global _servo_direction
    with _state_lock:
        _servo_direction = max(-1, min(1, payload.direction))
    return {"ok": True, "direction": _servo_direction}


@app.get("/servo")
def get_servo():
    with _state_lock:
        return {"ok": True, "direction": _servo_direction}


# ─── Static frontend ──────────────────────────────────────────────────────────

static_dir = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=static_dir), name="static")


@app.get("/")
def index():
    return FileResponse(static_dir / "index.html")
