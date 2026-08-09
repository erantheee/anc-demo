"""啸叫（声反馈）检测的离线测试。"""
from __future__ import annotations

import numpy as np

from app.feedback import detect_feedback
from app.synth import printer_noise


def _growing_sine(fs: float, duration: float, freq: float = 1000.0,
                  growth_db_s: float = 20.0, base_level: float = 0.005) -> np.ndarray:
    """模拟啸叫：频率固定、电平随时间指数增长的强音调。"""
    n = int(fs * duration)
    t = np.arange(n) / fs
    amp = base_level * 10.0 ** (growth_db_s / 20.0 * t)
    amp = np.clip(amp, 0.0, 0.5)
    return amp * np.sin(2.0 * np.pi * freq * t)


def _constant_sine(fs: float, duration: float, freq: float = 1000.0,
                   level: float = 0.1) -> np.ndarray:
    n = int(fs * duration)
    t = np.arange(n) / fs
    return level * np.sin(2.0 * np.pi * freq * t)


def test_steady_printer_noise_is_not_howling():
    x, _ = printer_noise(fs=16000, duration=4.0, seed=3)
    r = detect_feedback(x, 16000)
    assert r["is_howling"] is False
    assert r["signal_class"] == "environment_noise"
    assert r["insufficient_data"] is False


def test_growing_sine_is_howling():
    x = _growing_sine(fs=16000, duration=3.0, freq=1500.0)
    r = detect_feedback(x, 16000)
    assert r["is_howling"] is True, r
    assert r["signal_class"] == "acoustic_feedback"
    assert r["growth_db_per_s"] is not None and r["growth_db_per_s"] > 4.0
    assert r["howling_score"] > 0.3
    # 候选频率应接近 1500 Hz
    assert r["candidate_freq_hz"] is not None and abs(r["candidate_freq_hz"] - 1500.0) < 100.0


def test_constant_tone_is_not_howling():
    """恒定幅度纯音 = 稳态环境噪声（不是啸叫，因为无增长）。"""
    x = _constant_sine(fs=16000, duration=3.0, freq=1500.0, level=0.1)
    r = detect_feedback(x, 16000)
    assert r["is_howling"] is False
    assert r["growth_db_per_s"] is not None and r["growth_db_per_s"] < 4.0


def test_insufficient_data():
    x = _growing_sine(fs=16000, duration=0.2, freq=1500.0)
    r = detect_feedback(x, 16000)
    assert r["insufficient_data"] is True
    assert r["is_howling"] is False


def test_frequency_wandering_wideband_is_not_howling():
    """宽带噪声（白噪声）主峰频率漂移大，不应判为啸叫。"""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(int(16000 * 3.0)) * 0.1
    r = detect_feedback(x, 16000)
    assert r["is_howling"] is False
