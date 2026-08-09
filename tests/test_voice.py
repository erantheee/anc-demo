"""人声检测：能区分稳态周期噪声（ANC 目标）与人声（非 ANC 目标）。"""
import numpy as np

from app.synth import printer_noise, speech_like
from app.voice import detect_voice


def _steady_tone(fs=16000, duration=4.0, f0=140.6):
    t = np.arange(int(duration * fs)) / fs
    return (0.3 * np.sin(2 * np.pi * f0 * t)
            + 0.15 * np.sin(2 * np.pi * 2 * f0 * t)
            + 0.05 * np.sin(2 * np.pi * 3 * f0 * t))


def test_steady_tone_not_voice():
    x = _steady_tone()
    r = detect_voice(x, 16000)
    assert r["is_voice"] is False, r


def test_printer_noise_not_voice():
    x, _ = printer_noise(fs=16000, duration=4.0, seed=3)
    r = detect_voice(x, 16000)
    assert r["is_voice"] is False, r


def test_speech_like_signal_is_voice():
    x = speech_like(fs=16000, duration=5.0)
    r = detect_voice(x, 16000)
    assert r["is_voice"] is True, r


def test_short_sample_not_voice():
    x = _steady_tone(fs=16000, duration=0.3)
    r = detect_voice(x, 16000)
    assert r["is_voice"] is False
    assert "样本过短" in r["reasons"][0]


def test_wideband_noise_not_voice():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(16000 * 4) * 0.05
    r = detect_voice(x, 16000)
    assert r["is_voice"] is False, r
