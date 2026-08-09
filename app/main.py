"""FastAPI Web UI / API（:8000）。可选依赖：pip install -e .[web]。"""
from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import numpy as np

from app.anc.live import LiveANCEngine
from app.agent import NoiseAgentWorker
from app.grid import GridWorker
from app.monitor import Monitor
from app.quiet_zone import check_feasibility

WEB_DIR = Path(__file__).resolve().parent / "web"

monitor = Monitor()
grid_worker = GridWorker(monitor=monitor)
anc_live: LiveANCEngine | None = None


def _anc_status_provider() -> dict:
    """供噪声检测 Agent 查询实时 ANC 引擎状态（判断啸叫是否来自 ANC 输出）。

    只透传原始状态；running 的判定统一收敛到 agent.py 的 _anc_snapshot
    （state==running 或 phase∈{baseline,cancelling}），避免两处口径不一致。
    """
    global anc_live
    if anc_live is None:
        return {"state": "idle", "phase": "idle"}
    return anc_live.status()


def _anc_residual_provider() -> np.ndarray | None:
    """供噪声检测 Agent 取实时 ANC 残差尾部（误差麦真实信号），分析啸叫。"""
    global anc_live
    if anc_live is None:
        return None
    return anc_live.residual_tail(int(anc_live.fs * 1.0))


def _anc_control(action: str, params: dict) -> dict:
    """供噪声检测 Agent 的 adjust_anc 工具调节实时 ANC 引擎参数。"""
    global anc_live
    if anc_live is None:
        return {"ok": False, "error": "ANC 引擎未运行"}
    return anc_live.control(action, params)


noise_agent = NoiseAgentWorker(monitor=monitor, anc_status_provider=_anc_status_provider,
                               anc_control=_anc_control,
                               anc_residual_provider=_anc_residual_provider)


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
    gain: float = 0.5
    baseline_s: float = 5.0
    duration_s: float = 60.0
    synthetic: bool = False
    echo_gain: float = 0.15
    mic_delay_ms: float = 5.0  # 扬声器→误差麦延迟补偿（毫秒）
    feedback_cancel_gain: float = 0.3  # 反馈中和：扬声器→麦克风耦合增益估计
    auto_scan_delay: bool = True  # cancelling 开始时自动扫描最优延迟补偿
    watchdog_enabled: bool = True  # 规则式 watchdog：啸叫自动降增益 / 降噪不足自动增增益
    # 风噪门控：低频能量占比超阈值 → 静音反相输出（阻止对不相关的风喷噪声）
    wind_gate_enabled: bool = False
    wind_gate_cutoff_hz: float = 100.0
    wind_gate_ratio_thresh: float = 0.8
    wind_gate_window_s: float = 0.25
    input_highpass_hz: float = 0.0  # 输入高通（去低频风底，防 NLMS 追风）
    dual_mic: bool = False  # 双麦差分风噪检测（第二路 capsule 当参考）
    wind_gate_diff_thresh: float = 0.6  # 双麦差分风噪门控阈值


class QuietZoneRequest(BaseModel):
    x: float
    y: float
    z: float | None = None  # 点选高度，缺省取最近网格测量高度
    source_x: float | None = None
    source_y: float | None = None
    source_z: float | None = None


class AgentRequest(BaseModel):
    fresh_sample: bool = True   # 先重新采样再交给 Agent
    duration_s: float = 3.0     # 重新采样时长（秒）
    synthetic: bool | None = None  # None=按系统配置（ANC_SYNTHETIC）
    question: str | None = None


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
    from app.analyze import analyze, to_dict
    from app.synth import printer_noise

    samples, sources = printer_noise(fs=req.fs, duration=req.duration_s)
    report = analyze(samples, req.fs)
    return {"sources": sources, "analysis": to_dict(report)}


@app.post("/api/anc/start")
def anc_start(req: ANCRequest) -> dict:
    """离线跑一段 ANC 演示并返回降噪报告。"""
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


class CalibRequest(BaseModel):
    known_spl: float
    duration_s: float = 10.0
    fs: int = 16000


@app.get("/api/calibration")
def calibration_status() -> dict:
    """当前麦克风标定状态（offset_db 等）。"""
    from app.calibration import get_offset_db, info

    return {
        "configured": bool(get_offset_db()),
        "offset_db": get_offset_db(),
        "info": info(),
    }


@app.post("/api/calibration")
def calibration_start(req: CalibRequest) -> dict:
    """现场标定：用手机声级计读到的真实 SPL，计算 dBFS→dB SPL 偏移并立即生效。"""
    from app import calibration
    from app.monitor import SYNTHETIC

    if SYNTHETIC:
        return {"ok": False, "error": "合成模式下无法标定（合成噪声没有真实声压），请用真机模式"}
    # 暂停监控线程释放麦克风，避免设备被占用导致录音失败
    monitor.set_paused(True)
    _wait_monitor_idle(3.5)
    try:
        payload = calibration.calibrate(known_spl=req.known_spl,
                                        duration_s=req.duration_s, fs=req.fs)
        return {"ok": True, **payload}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}
    finally:
        monitor.set_paused(False)


@app.post("/api/anc/live/start")
def anc_live_start(req: ANCLiveRequest) -> dict:
    """启动实时 ANC 环路（自参考谐波消除）。"""
    global anc_live
    if anc_live is not None and anc_live._thread is not None and anc_live._thread.is_alive():
        return {"started": False, "message": "ANC 已在运行，请先停止"}
    # 暂停监控线程，释放麦克风/音频设备，避免设备被占用导致流打不开
    monitor.set_paused(True)
    _wait_monitor_idle(3.5)
    anc_live = LiveANCEngine(
        fs=req.fs, in_device=req.in_device or None, out_device=req.out_device or None,
        f0=req.f0, output_gain=req.gain, baseline_s=req.baseline_s,
        max_duration_s=req.duration_s, synthetic=req.synthetic,
        echo_gain=req.echo_gain,
        speaker_mic_delay_s=req.mic_delay_ms / 1000.0,
        feedback_cancel_gain=req.feedback_cancel_gain,
        auto_scan_delay=req.auto_scan_delay,
        watchdog_enabled=req.watchdog_enabled,
        wind_gate_enabled=req.wind_gate_enabled,
        wind_gate_cutoff_hz=req.wind_gate_cutoff_hz,
        wind_gate_ratio_thresh=req.wind_gate_ratio_thresh,
        wind_gate_window_s=req.wind_gate_window_s,
        input_highpass_hz=req.input_highpass_hz,
        dual_mic=req.dual_mic,
        wind_gate_diff_thresh=req.wind_gate_diff_thresh)
    res = anc_live.start()
    if not res.get("started"):
        monitor.set_paused(False)
    else:
        # 守护线程：ANC 自然结束（到时长/出错）后恢复监控线程
        import threading

        threading.Thread(target=_resume_monitor_when_anc_done,
                         name="anc-monitor-resume", daemon=True).start()
    return res


def _resume_monitor_when_anc_done() -> None:
    """轮询 ANC 引擎状态，结束后恢复监控线程（释放音频设备）。"""
    import time

    while True:
        st = anc_live.status()
        if st["state"] in ("stopped", "error"):
            monitor.set_paused(False)
            return
        time.sleep(1.0)


def _wait_monitor_idle(timeout_s: float) -> None:
    """等监控线程真正停下手里的常驻录音流（释放音频设备），最多等 timeout 秒。"""
    monitor.wait_paused(timeout_s)


@app.post("/api/anc/live/stop")
def anc_live_stop() -> dict:
    """停止实时 ANC，返回最终状态。"""
    global anc_live
    if anc_live is None:
        monitor.set_paused(False)
        return {"state": "idle", "phase": "idle"}
    st = anc_live.stop()
    monitor.set_paused(False)
    return st


@app.get("/api/anc/live/status")
def anc_live_status() -> dict:
    """实时 ANC 状态快照（仪表盘轮询）。"""
    global anc_live
    if anc_live is None:
        return {"state": "idle", "phase": "idle", "synthetic": False}
    return anc_live.status()


class ANCControlRequest(BaseModel):
    action: str  # set_gain | increase_gain | decrease_gain | set_mic_delay_ms | stop
    value: float | None = None
    delta: float | None = None
    mic_delay_ms: float | None = None
    reason: str = ""


@app.post("/api/anc/live/control")
def anc_live_control(req: ANCControlRequest) -> dict:
    """手动/Agent 调节实时 ANC 参数：增益、延迟补偿或停止。"""
    global anc_live
    if anc_live is None:
        return {"ok": False, "error": "ANC 引擎未运行"}
    params: dict = {}
    if req.value is not None:
        params["value"] = req.value
    if req.delta is not None:
        params["delta"] = req.delta
    if req.mic_delay_ms is not None:
        params["mic_delay_ms"] = req.mic_delay_ms
    params["reason"] = req.reason or ""
    return anc_live.control(req.action, params)


class ANCGainRequest(BaseModel):
    gain: float = 0.5  # 目标输出增益；会被钳制到安全范围 [min, max]，不会超限


@app.post("/api/anc/live/gain")
def anc_live_gain(req: ANCGainRequest) -> dict:
    """外部噪声检测 Agent / 脚本设定 ANC 目标输出增益（轻量接口）。

    契约：
    - POST /api/anc/live/gain，body: {"gain": <float>}（0.02–1.0，超出自动截断）
    - 运行时生效，不影响正在运行的 ANC；返回 {"ok": true, ...status()}
    - 引擎另有硬性安全兜底：compute_safe_output 的 ±clip_level 硬限幅与
      RMS 比例门控，外部 Agent 无法把输出推到危险水平。
    - 也可以改用 POST /api/anc/live/control（action=set_gain）做更细控制。
    """
    global anc_live
    if anc_live is None:
        return {"ok": False, "error": "ANC 引擎未运行"}
    return anc_live.set_output_gain(req.gain, reason="external_agent")


@app.get("/api/anc/source")
def anc_source() -> dict:
    """ANC 板块噪声源识别：基于最近一次监控分析，返回来源类型、建议 f0 与 ANC 可行性。

    树莓派不区分"输入/输出设备"：噪声由手机/电脑在旁边播放，误差麦采集，
    树莓派通过默认输出播放反相波。此端点只回答"当前是什么噪声源、ANC 是否值得做"。
    """
    report = monitor.last_report()
    if report is None:
        return {"found": False, "error": "暂无分析数据（等待监控采样）"}

    from app.source_id import load_profiles, match_sources, recommend_anc

    # 人声检测基于原始采样（稳定段剔除会洗掉语音帧，不适合判定）
    voice = None
    raw = monitor.last_samples()
    if raw is not None and len(raw) > 0:
        from app.voice import detect_voice
        voice = detect_voice(raw, monitor.fs)

    hits = match_sources(report)
    rec = recommend_anc(report, voice=voice)
    best = hits[0] if hits else None
    profiles = {p.get("id"): p for p in load_profiles()}
    info = profiles.get(best.source) if best else None

    # 谐波家族基频优先作为 f0 建议（谐波消除的关键参数）；人声不作为 ANC 目标
    is_voice = bool(voice and voice.get("is_voice"))
    fund = None if is_voice else (report.harmonic_family[0] if report.harmonic_family else report.dominant_freq)
    return {
        "found": True,
        "source_id": best.source if best else None,
        "source_name": (info or {}).get("name") if best else None,
        "confidence": best.confidence if best else None,
        "freqs_hz": best.freqs_hz if best else [],
        "recommended_f0": round(float(fund), 1) if fund else None,
        "dominant_freq": report.dominant_freq,
        "is_voice": is_voice,
        "anc_worthwhile": rec["anc_worthwhile"],
        "reasons": rec["reasons"],
        "feasibility": (info or {}).get("anc_feasibility"),
        "notes": (info or {}).get("notes"),
        "reference_mic": (info or {}).get("reference_mic"),
    }


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
        "tone_reduction_db": rep["tone_reduction_db"],
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


@app.get("/api/agent/status")
def agent_status() -> dict:
    """Kimi 噪声检测 Agent 状态（配置 / 运行 / 最近结论）。"""
    return noise_agent.status_dict()


@app.post("/api/agent/analyze")
def agent_analyze(req: AgentRequest) -> dict:
    """启动 Kimi 噪声自动检测 Agent（后台线程）。"""
    return noise_agent.start(
        fresh_sample=req.fresh_sample,
        duration_s=req.duration_s,
        synthetic=req.synthetic,
        question=req.question,
    )


if (WEB_DIR / "index.html").exists():
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
