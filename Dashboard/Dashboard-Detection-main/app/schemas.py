from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class MetricIn(BaseModel):
    ts: Optional[datetime] = None
    fps: float
    latency_ms: float
    cpu_temp_c: float
    cpu_percent: float
    ram_percent: float
    detect_count: int = 0
    camera_status: str = "ok"


class MetricOut(MetricIn):
    id: int

    class Config:
        from_attributes = True


class BoxesIn(BaseModel):
    """YOLO-style TXT bbox payload from Pi inference loop."""
    txt: str = ""
    frame_w: int
    frame_h: int
    ts: Optional[str] = None


class StreamConfigIn(BaseModel):
    """MJPEG stream URL registered by Pi (Tailscale IP preferred)."""
    stream_url: str
