import numpy as np

from app.analyze import (a_weight_db, analyze, find_harmonic_family, find_peaks,
                         rms_db, spectrum_db, stable_segment, tonality_ratio)
from app.synth import printer_noise


def test_a_weight_1000hz_is_zero():
    a = a_weight_db(np.array([1000.0]))
    assert abs(a[0]) < 0.5


def test_tone_peak_detection():
    fs = 16000
    t = np.arange(fs) / fs
    x = 0.5 * np.sin(2 * np.pi * 200.0 * t)
    report = analyze(x, fs)
    peaks = [p for p in report.peaks if p.freq > 40]
    assert len(peaks) >= 1
    assert abs(peaks[0].freq - 200.0) < 10.0
    assert report.tonality_ratio > 0.5


def test_printer_noise_dominant_family():
    x, _ = printer_noise(fs=16000, duration=4.0, seed=2)
    report = analyze(x, 16000)
    assert report.harmonic_family, "应检测到谐波家族"
    fund = report.harmonic_family[0]
    assert 80 <= fund <= 200, f"基频应在 120 Hz 附近，实际 {fund:.1f}"


def test_harmonic_family_grouping():
    from app.analyze import TonalPeak

    peaks = [
        TonalPeak(freq=100.0, level_db=-30.0, prominence_db=10.0),
        TonalPeak(freq=200.0, level_db=-40.0, prominence_db=8.0),
        TonalPeak(freq=300.0, level_db=-50.0, prominence_db=6.0),
        TonalPeak(freq=450.0, level_db=-35.0, prominence_db=5.0),
    ]
    fund, matched = find_harmonic_family(peaks)
    assert fund is not None
    assert abs(fund - 100.0) < 1.0
    orders = sorted(o for _, o in matched)
    assert orders == [1, 2, 3]


def test_rms_db_pure_sine():
    t = np.arange(16000) / 16000
    x = np.sin(2 * np.pi * 100 * t)
    assert abs(rms_db(x) - (-3.01)) < 0.1


def test_tonality_ratio_bounds():
    rng = np.random.default_rng(0)
    x = rng.standard_normal(16000)
    report = analyze(x, 16000)
    assert 0.0 <= report.tonality_ratio <= 1.0


def test_stable_segment_removes_transient():
    fs = 16000
    x = 0.05 * np.sin(2 * np.pi * 200 * np.arange(fs) / fs)
    x[fs // 2: fs // 2 + fs // 4] += 5.0  # 瞬态突发（高 40 dB）
    kept = stable_segment(x, fs)
    rms_kept = np.sqrt(np.mean(kept ** 2))
    assert rms_kept < 0.15, f"瞬态应被剔除，RMS={rms_kept:.3f}"


def test_stable_segment_keeps_clean_signal():
    fs = 16000
    x = 0.05 * np.sin(2 * np.pi * 200 * np.arange(fs) / fs)
    kept = stable_segment(x, fs)
    assert len(kept) > len(x) * 0.7
