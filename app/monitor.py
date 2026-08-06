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
from app.analyze import AnalysisReport, analyze, stable_segment
from app.synth import printer_noise

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
    spectrum_freqs: list[float] = field(default_factory=list)
    spectrum_db: list[float] = field(default_factory=list)
    last_update_ts: float | None = None
    error: str | None = None
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

    def set_paused(self, paused: bool) -> None:
        if paused:
            self._paused.set()
        else:
            self._paused.clear()

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
                "spectrum_freqs": self.state.spectrum_freqs,
                "spectrum_db": self.state.spectrum_db,
                "last_update_ts": self.state.last_update_ts,
                "last_update_age_s": (
                    round(time.time() - self.state.last_update_ts, 1)
                    if self.state.last_update_ts else None
                ),
                "uptime_s": self.state.uptime_s,
                "cpu_temp_c": self.state.cpu_temp_c,
                "error": self.state.error,
            }

    # ---- 内部 ----

    def _sample(self) -> tuple[np.ndarray, AnalysisReport]:
        """返回 (原始样本, 分析报告)，样本供频谱绘制。"""
        if SYNTHETIC:
            samples, _ = printer_noise(fs=self.fs, duration=self.sample_s, seed=int(time.time()))
        else:
            raw = capture.record_buffer(self.sample_s, fs=self.fs)
            samples = stable_segment(raw, self.fs)
            if len(samples) == 0:
                raise RuntimeError("无有效信号（瞬态过多）")
        return samples, analyze(samples, self.fs)

    def _run(self) -> None:
        with self._lock:
            self.state.running = True
            self.state.uptime_s = _uptime_s()
            self.state.cpu_temp_c = _cpu_temp_c()
        try:
            while not self._stop.is_set():
                if self._paused.is_set():
                    with self._lock:
                        self.state.paused = True
                    time.sleep(0.5)
                    continue
                try:
                    samples, report = self._sample()
                    freqs, psd_db = self._spectrum(samples)
                    with self._lock:
                        self.state.paused = False
                        self.state.spl_db = report.spl_db or report.rms_db
                        self.state.spl_db_a = report.spl_db_a
                        self.state.rms_db = report.rms_db
                        self.state.dominant_freq = report.dominant_freq
                        self.state.band_spl_db = report.band_spl_db
                        self.state.last_update_ts = time.time()
                        self.state.error = None
                        # 降采样到约 256 点，前端 canvas 足够
                        step = max(1, len(freqs) // 256)
                        self.state.spectrum_freqs = [round(float(f), 1) for f in freqs[::step]]
                        self.state.spectrum_db = [round(float(v), 1) for v in psd_db[::step]]
                        self.state.source_guess, self.state.source_confidence = self._guess_source(report)
                except Exception as exc:  # 麦克风被占用等：记录但不退出
                    with self._lock:
                        self.state.error = str(exc)
                        self.state.last_update_ts = time.time()
                time.sleep(self.interval_s)
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
