"""FastAPI Web UI / API（:8000）。可选依赖：pip install -e .[web]。"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.anc.live import LiveANCEngine
from app.grid import GridWorker
from app.monitor import Monitor
from app.quiet_zone import check_feasibility

WEB_DIR = Path(__file__).resolve().parent / "web"

monitor = Monitor()
grid_worker = GridWorker(monitor=monitor)
anc_live: LiveANCEngine | None = None


@asynccontextmanager
async def lifespan(_: FastAPI):
    monitor.start()
    yield
    monitor.stop()


app = FastAPI(title="Pi ANC Demo", version="0.1.0", lifespan=lifespan)


class MeasureRequest(BaseModel):
    duration_s: float = 30.0
    fs: int = 16000
    synthetic: bool = True


class GridRequest(BaseModel):
    origin_x: float = 0.0
    origin_y: float = 0.0
    size_x: float = 1.0
    size_y: float = 1.0
    step: float = 0.5
    per_point_s: float = 10.0
    fs: int = 48000
    synthetic: bool = False
    height_m: float = 0.5  # 测量高度：误差麦克风/噪声源离地高度（z 轴）


class ANCRequest(BaseModel):
    mode: str = "harmonic"
    fs: int = 16000


class ANCLiveRequest(BaseModel):
    fs: int = 48000
    in_device: str | None = None   # 误差麦克风设备
    out_device: str | None = None  # 扬声器输出设备
    f0: float | None = None
    gain: float = 0.4
    baseline_s: float = 5.0
    duration_s: float = 60.0
    synthetic: bool = False
    echo_gain: float = 0.15
    mic_delay_ms: float = 5.0  # 扬声器→误差麦延迟补偿（毫秒）


class QuietZoneRequest(BaseModel):
    x: float
    y: float
    z: float | None = None  # 点选高度，缺省取最近网格测量高度
    source_x: float | None = None
    source_y: float | None = None
    source_z: float | None = None


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "monitor_running": monitor.state.running}


@app.get("/api/status")
def status() -> dict:
    from app.anc.pipeline import ANCController

    return {"demo": "pi-anc", "controller": ANCController().summary()}


@app.get("/api/live")
def live() -> dict:
    """实时噪声监控快照 + Pi 状态。"""
    return monitor.snapshot()


@app.post("/api/grid/measure")
def grid_measure(req: GridRequest) -> dict:
    """启动后台网格测量任务。"""
    return grid_worker.start(
        origin_x=req.origin_x, origin_y=req.origin_y,
        size_x=req.size_x, size_y=req.size_y,
        step=req.step, per_point_s=req.per_point_s, fs=req.fs,
        synthetic=req.synthetic, height_m=req.height_m,
    )


@app.get("/api/grid/status")
def grid_status() -> dict:
    """网格任务进度与结果。"""
    return grid_worker.status()


@app.post("/api/quiet-zone")
def quiet_zone(req: QuietZoneRequest) -> dict:
    """点选静音区：返回该点相对噪声源的 3D ANC 可行性。"""
    source_xy = (req.source_x, req.source_y) if req.source_x is not None and req.source_y is not None \
        else None
    if source_xy is None:
        return {"error": "需要 source_x / source_y（未指定噪声源）"}
    # z 轴：点选高度缺省取最近网格测量的源高度；源 z 缺省同理
    last = grid_worker.result.quiet_zone
    last_z = float(last["source_pos_m"][2]) if last and last.get("source_pos_m") else 0.5
    source_z = req.source_z if req.source_z is not None else last_z
    sel_z = req.z if req.z is not None else source_z
    # 主频优先取最近一次网格测量的来源频率，否则默认步进电机谐波
    freq_hz = 120.0
    if last and last.get("dominant_freq_hz"):
        freq_hz = float(last["dominant_freq_hz"])
    return {
        "selected_m": [req.x, req.y, sel_z],
        "source_m": [req.source_x, req.source_y, source_z],
        "feasibility": check_feasibility(
            printer_pos_m=(req.source_x, req.source_y, source_z),
            quiet_pos_m=(req.x, req.y, sel_z),
            dominant_freq_hz=freq_hz,
        ),
    }


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


@app.get("/api/audio/devices")
def audio_devices() -> dict:
    """列出音频设备（输入/输出），供 ANC 现场配置。"""
    try:
        import sounddevice as sd
    except ImportError:
        return {"devices": [], "default": None, "error": "未安装 sounddevice"}
    devs = []
    for i, dev in enumerate(sd.query_devices()):
        devs.append({
            "index": i,
            "name": dev["name"],
            "in_channels": int(dev["max_input_channels"]),
            "out_channels": int(dev["max_output_channels"]),
            "default_samplerate": int(dev["default_samplerate"]),
        })
    return {"devices": devs, "default": {"in": sd.default.device[0], "out": sd.default.device[1]}}


@app.post("/api/anc/live/start")
def anc_live_start(req: ANCLiveRequest) -> dict:
    """启动实时 ANC 环路（自参考谐波消除）。"""
    global anc_live
    if anc_live is not None and anc_live._thread is not None and anc_live._thread.is_alive():
        return {"started": False, "message": "ANC 已在运行，请先停止"}
    anc_live = LiveANCEngine(
        fs=req.fs, in_device=req.in_device or None, out_device=req.out_device or None,
        f0=req.f0, output_gain=req.gain, baseline_s=req.baseline_s,
        max_duration_s=req.duration_s, synthetic=req.synthetic,
        echo_gain=req.echo_gain,
        speaker_mic_delay_s=req.mic_delay_ms / 1000.0)
    return anc_live.start()


@app.post("/api/anc/live/stop")
def anc_live_stop() -> dict:
    """停止实时 ANC，返回最终状态。"""
    global anc_live
    if anc_live is None:
        return {"state": "idle", "phase": "idle"}
    st = anc_live.stop()
    return st


@app.get("/api/anc/live/status")
def anc_live_status() -> dict:
    """实时 ANC 状态快照（仪表盘轮询）。"""
    global anc_live
    if anc_live is None:
        return {"state": "idle", "phase": "idle", "synthetic": False}
    return anc_live.status()


@app.get("/api/anc/live/report")
def anc_live_report() -> dict:
    """实时 ANC 完成后的 A/B 降噪报告。"""
    global anc_live
    if anc_live is None:
        return {"found": False}
    d, e = anc_live.get_signals()
    if d is None or e is None or len(d) == 0 or len(e) == 0:
        return {"found": False}
    from app.evaluate import evaluate_before_after

    fs = anc_live.fs
    st = anc_live.status()
    rep = evaluate_before_after(d, e, fs=fs)
    return {
        "found": True,
        "f0_hz": st["f0"],
        "baseline_spl_db": st["baseline_spl_db"],
        "cancelling_spl_db": st["cancelling_spl_db"],
        "broadband_reduction_db": rep["broadband_reduction_db"],
        "a_weighted_reduction_db": rep["a_weighted_reduction_db"],
        "peak_reductions": rep["peak_reductions"][:5],
    }


@app.get("/api/report")
def report() -> dict:
    """读取最近一次测量报告（若存在）。"""
    import json

    reports = sorted(Path("data/reports").glob("*/report.json"))
    if not reports:
        return {"found": False}
    return {"found": True, "path": str(reports[-1]),
            "data": json.loads(reports[-1].read_text(encoding="utf-8"))}


if (WEB_DIR / "index.html").exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
