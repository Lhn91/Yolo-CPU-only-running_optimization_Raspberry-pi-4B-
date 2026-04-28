import os
from pathlib import Path

from sqlalchemy import create_engine, Column, Integer, Float, String, DateTime
from sqlalchemy.orm import declarative_base, sessionmaker
from sqlalchemy.sql import func

# SQLite mặc định: file metrics.db nằm cạnh thư mục app/ (cùng cấp với requirements.txt)
DEFAULT_DB_PATH = Path(__file__).resolve().parent.parent / "metrics.db"
DB_URL = os.getenv("DB_URL", f"sqlite:///{DEFAULT_DB_PATH.as_posix()}")

# SQLite cần `check_same_thread=False` để FastAPI multi-thread dùng được
_engine_kwargs = {"pool_pre_ping": True}
if DB_URL.startswith("sqlite"):
    _engine_kwargs["connect_args"] = {"check_same_thread": False}

engine = create_engine(DB_URL, **_engine_kwargs)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


class Metric(Base):
    __tablename__ = "metrics"

    id = Column(Integer, primary_key=True, index=True)
    ts = Column(DateTime(timezone=True), server_default=func.now(), index=True)
    fps = Column(Float, nullable=False)
    latency_ms = Column(Float, nullable=False)
    cpu_temp_c = Column(Float, nullable=False)
    cpu_percent = Column(Float, nullable=False)
    ram_percent = Column(Float, nullable=False)
    detect_count = Column(Integer, nullable=False, default=0)
    camera_status = Column(String(32), nullable=False, default="ok")


def init_db():
    Base.metadata.create_all(bind=engine)
