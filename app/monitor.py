"""实时噪声监控线程：周期采样 → 分析 → 更新共享状态。

供 Web 仪表盘轮询（/api/live）。网格测量运行时通过 `set_paused(True)`
暂停采样，避免与录音任务抢占音频设备。
"""
from __future__ import annotations

import os
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from app import capture
from app.analyze import AnalysisReport, analyze, fast_spl_db, stable_segment
from app.calibration import get_offset_db
from app.synth import printer_noise
from app.voice import detect_voice

SYNTHETIC = os.environ.get("ANC_SYNTHETIC", "0") == "1"

DEFAULT_SAMPLE_S = 2.0
DEFAULT_INTERVAL_S = 3.0
DEFAULT_FS = 48000


def _uptime_s() -> float | None:
    try:
        return float(Path("/proc/uptime").read_text().split()[0])
    except Exception:
        return None


def _cpu_temp_c() -> float | None:
    try:
        raw = Path("/sys/class/thermal/thermal_zone0/temp").read_text().strip()
        return round(float(raw) / 1000.0, 1)
    except Exception:
        return None


@dataclass
class MonitorState:
    running: bool = False
    paused: bool = False
    spl_db: float | None = None
    spl_db_a: float | None = None
    rms_db: float | None = None
    dominant_freq: float | None = None
    source_guess: str | None = None
    source_confidence: float | None = None
    band_spl_db: dict[str, float] = field(default_factory=dict)
    # 人声检测：实时监控判断当前信号是否更像人声（非 ANC 目标）
    is_voice: bool | None = None
    voice_score: float | None = None
    voice_reasons: list[str] = field(default_factory=list)
    spectrum_freqs: list[float] = field(default_factory=list)
    spectrum_db: list[float] = field(default_factory=list)
    last_update_ts: float | None = None
    error: str | None = None
    mic_ok: bool | None = None  # None=未知, True=有有效输入, False=无麦克风/信号无效
    # Pi 状态
    uptime_s: float | None = None
    cpu_temp_c: float | None = None


class Monitor:
    """后台监控线程。"""

    def __init__(self, sample_s: float = DEFAULT_SAMPLE_S,
                 interval_s: float = DEFAULT_INTERVAL_S,
                 fs: int = DEFAULT_FS) -> None:
        self.sample_s = sample_s
        self.interval_s = interval_s
        self.fs = fs
        self.state = MonitorState()
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._paused = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_report: AnalysisReport | None = None
        self._last_raw: np.ndarray | None = None
        self._recorder: capture.ContinuousRecorder | None = None

    # ---- 线程控制 ----

    def start(self) -> None:
        if self._thread and self._thread.is_alive():
            return
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name="anc-monitor", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=2.0)
        self._close_recorder()

    def set_paused(self, paused: bool) -> None:
        if paused:
            self._paused.set()
        else:
            self._paused.clear()

    def wait_paused(self, timeout_s: float = 4.0) -> bool:
        """等待监控线程真正停下常驻录音流（释放音频设备）。

        供 ANC/网格/标定在 `set_paused(True)` 后调用，确保麦克风已释放、
        可以被独占打开，避免设备被占用导致录音/流打开失败。
        """
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            with self._lock:
                if self.state.paused:
                    return True
            time.sleep(0.05)
        return False

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "running": self.state.running,
                "paused": self.state.paused,
                "spl_db": round(self.state.spl_db, 1) if self.state.spl_db is not None else None,
                "spl_db_a": round(self.state.spl_db_a, 1) if self.state.spl_db_a is not None else None,
                "rms_db": round(self.state.rms_db, 1) if self.state.rms_db is not None else None,
                "dominant_freq": self.state.dominant_freq,
                "source_guess": self.state.source_guess,
                "source_confidence": self.state.source_confidence,
                "band_spl_db": {k: round(v, 1) for k, v in self.state.band_spl_db.items()},
                "is_voice": self.state.is_voice,
                "voice_score": self.state.voice_score,
                "voice_reasons": list(self.state.voice_reasons),
                "spectrum_freqs": self.state.spectrum_freqs,
                "spectrum_db": self.state.spectrum_db,
                "last_update_ts": self.state.last_update_ts,
                "last_update_age_s": (
                    round(time.time() - self.state.last_update_ts, 1)
                    if self.state.last_update_ts else None
                ),
                "mic_ok": self.state.mic_ok,
                "uptime_s": self.state.uptime_s,
                "cpu_temp_c": self.state.cpu_temp_c,
                "error": self.state.error,
            }

    def last_report(self) -> AnalysisReport | None:
        """返回最近一次分析报告（供 ANC 板块做来源识别与参数建议）。"""
        with self._lock:
            return self._last_report

    def last_samples(self) -> np.ndarray | None:
        """返回最近一次采样的原始信号（未做稳定段剔除，供人声检测等使用）。"""
        with self._lock:
            return self._last_raw

    # ---- 内部 ----

    def _sample(self) -> tuple[np.ndarray, AnalysisReport, np.ndarray]:
        """返回 (稳定段样本, 分析报告, 原始信号)，样本供频谱绘制。

        SPL 读数走"原始信号短窗峰值"路径（不经 stable_segment），保证
        敲击/突发噪声立刻反映在分贝值上；频谱与声源识别仍用稳定段，
        避免瞬态污染频谱。原始信号另用于人声检测（稳定段会剔除语音帧，
        不适合做人声判定）。
        """
        if SYNTHETIC:
            samples, _ = printer_noise(fs=self.fs, duration=self.sample_s, seed=int(time.time()))
            raw = samples
        else:
            if self._recorder is None or not self._recorder.is_open:
                raise RuntimeError("常驻录音流未打开")
            raw = self._recorder.read(self.sample_s)
            capture.ensure_valid_signal(raw)
            samples = stable_segment(raw, self.fs)
            if len(samples) == 0:
                raise RuntimeError("无有效信号（瞬态过多）")
        report = analyze(samples, self.fs, calibration_offset_db=get_offset_db())
        # 灵敏 SPL：原始信号短窗峰值（0.1s 帧峰值保持），敲击立即可见。
        # analyze() 的 spl_db 基于稳定段，这里用原始段重新算并覆盖。
        db_fast, db_fast_a = fast_spl_db(raw, self.fs)
        offset = get_offset_db()
        report.rms_db = db_fast
        report.spl_db = db_fast + offset if offset else None
        report.spl_db_a = db_fast_a + offset if offset else None
        return samples, report, raw

    def _open_recorder(self) -> None:
        if SYNTHETIC:
            return
        if self._recorder is None:
            self._recorder = capture.ContinuousRecorder(fs=self.fs, channels=2)
        if not self._recorder.is_open:
            self._recorder.open()

    def _close_recorder(self) -> None:
        if self._recorder is not None:
            self._recorder.close()

    def _run(self) -> None:
        with self._lock:
            self.state.running = True
            self.state.uptime_s = _uptime_s()
            self.state.cpu_temp_c = _cpu_temp_c()
        try:
            while not self._stop.is_set():
                if self._paused.is_set():
                    # 释放麦克风，供 ANC/网格/标定等任务独占使用
                    self._close_recorder()
                    with self._lock:
                        self.state.paused = True
                    time.sleep(0.5)
                    continue
                try:
                    self._open_recorder()
                    samples, report, raw = self._sample()
                    freqs, psd_db = self._spectrum(samples)
                    voice = detect_voice(raw, self.fs)
                    with self._lock:
                        self._last_report = report
                        self._last_raw = raw
                        self.state.paused = False
                        self.state.mic_ok = True
                        self.state.spl_db = report.spl_db or report.rms_db
                        self.state.spl_db_a = report.spl_db_a
                        self.state.rms_db = report.rms_db
                        self.state.dominant_freq = report.dominant_freq
                        self.state.band_spl_db = report.band_spl_db
                        self.state.is_voice = voice["is_voice"]
                        self.state.voice_score = voice["score"]
                        self.state.voice_reasons = voice["reasons"]
                        self.state.last_update_ts = time.time()
                        self.state.error = None
                        # 降采样到约 256 点，前端 canvas 足够
                        step = max(1, len(freqs) // 256)
                        self.state.spectrum_freqs = [round(float(f), 1) for f in freqs[::step]]
                        self.state.spectrum_db = [round(float(v), 1) for v in psd_db[::step]]
                        self.state.source_guess, self.state.source_confidence = self._guess_source(report)
                except RuntimeError as exc:
                    # 无输入设备 / 信号无效：明确标记，绝不报误导性分贝
                    with self._lock:
                        self._last_report = None
                        self.state.mic_ok = False
                        self.state.error = str(exc)
                        self.state.spl_db = None
                        self.state.spl_db_a = None
                        self.state.rms_db = None
                        self.state.dominant_freq = None
                        self.state.source_guess = None
                        self.state.source_confidence = None
                        self.state.band_spl_db = {}
                        self.state.is_voice = None
                        self.state.voice_score = None
                        self.state.voice_reasons = []
                        self.state.spectrum_freqs = []
                        self.state.spectrum_db = []
                        self.state.last_update_ts = time.time()
                except Exception as exc:  # 麦克风被占用等：记录但不退出
                    with self._lock:
                        self.state.error = str(exc)
                        self.state.last_update_ts = time.time()
                # 用事件等待代替固定 sleep：pause 请求一到立即返回，及时释放音频设备
                self._paused.wait(self.interval_s)
        finally:
            with self._lock:
                self.state.running = False

    def _spectrum(self, samples: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        from app.analyze import spectrum_db

        return spectrum_db(samples, self.fs)

    def _guess_source(self, report: AnalysisReport) -> tuple[str | None, float | None]:
        from app.source_id import load_profiles, match_sources

        hits = match_sources(report)
        if not hits:
            return None, None
        best = hits[0]
        return best.source, best.confidence
