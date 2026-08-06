"""FastAPI Web UI / API（:8000）。可选依赖：pip install -e .[web]。"""
from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(title="Pi ANC Demo", version="0.1.0")


class MeasureRequest(BaseModel):
    duration_s: float = 30.0
    fs: int = 16000
    synthetic: bool = True


class ANCRequest(BaseModel):
    mode: str = "harmonic"
    fs: int = 16000


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.get("/api/status")
def status() -> dict:
    from app.anc.pipeline import ANCController

    return {"demo": "pi-anc", "controller": ANCController().summary()}


@app.post("/api/measure")
def measure(req: MeasureRequest) -> dict:
    """触发一次测量（默认合成数据，避免阻塞开发机）。"""
    import numpy as np

    from app.analyze import analyze, to_dict
    from app.synth import printer_noise

    samples, sources = printer_noise(fs=req.fs, duration=req.duration_s)
    report = analyze(samples, req.fs)
    return {"sources": sources, "analysis": to_dict(report)}


@app.post("/api/anc/start")
def anc_start(req: ANCRequest) -> dict:
    """离线跑一段 ANC 演示并返回降噪报告。"""
    import numpy as np

    from app.anc.fxlms import simulate as fx_sim
    from app.anc.harmonic import simulate as harm_sim
    from app.synth import printer_noise

    samples, _ = printer_noise(fs=req.fs, duration=5.0)
    if req.mode == "harmonic":
        res = harm_sim(samples, req.fs)
    else:
        res = fx_sim(samples, req.fs, secondary_path=np.array([0.4, 0.6, 1.0, 0.6, 0.3]))
    return {
        "algorithm": req.mode,
        "reduction_db": round(res["reduction_db"], 2),
        "f0": res.get("f0"),
    }


@app.get("/api/report")
def report() -> dict:
    """读取最近一次测量报告（若存在）。"""
    import json
    from pathlib import Path

    reports = sorted(Path("data/reports").glob("*/report.json"))
    if not reports:
        return {"found": False}
    return {"found": True, "path": str(reports[-1]),
            "data": json.loads(reports[-1].read_text(encoding="utf-8"))}
