import numpy as np

from app.analyze import (a_weight_db, analyze, fast_spl_db, find_harmonic_family,
                         find_peaks, rms_db, spectrum_db, stable_segment,
                         tonality_ratio)
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


# ---- 灵敏实时 SPL（短窗峰值，敲击立即可见） ----

def test_fast_spl_db_reflects_short_burst():
    """短促响脉冲（模拟敲击外壳）应显著抬高快速 SPL。

    旧路径（2s 整段 RMS + stable_segment 剔除瞬态）对敲击几乎无反应；
    fast_spl_db 用 0.1s 短窗峰值保持，脉冲所在帧能量不被长窗稀释，
    灵敏度应明显高于整段 RMS。
    """
    fs = 16000
    rng = np.random.default_rng(0)
    quiet = 0.01 * rng.standard_normal(int(2.0 * fs))  # 安静背景 ~ -40 dB
    burst = quiet.copy()
    start = int(1.0 * fs)
    burst[start:start + int(0.03 * fs)] += 0.5  # 30ms 响脉冲

    fast_quiet, _ = fast_spl_db(quiet, fs)
    fast_burst, _ = fast_spl_db(burst, fs)
    jump_fast = fast_burst - fast_quiet
    jump_whole = rms_db(burst) - rms_db(quiet)
    # 短窗峰值路径的灵敏度应明显高于整段 RMS（至少多 8 dB），
    # 且自身跳升显著（> 15 dB）
    assert jump_fast > 15.0, \
        f"敲击应显著抬高快速 SPL: {fast_quiet:.1f} -> {fast_burst:.1f}"
    assert jump_fast - jump_whole > 8.0, \
        f"快速路径灵敏度应远超整段 RMS: fast={jump_fast:.1f}dB vs whole={jump_whole:.1f}dB"


def test_fast_spl_db_stable_when_quiet():
    """安静背景下快速 SPL 读数稳定（不因无信号而乱跳）。"""
    fs = 16000
    rng = np.random.default_rng(1)
    x1 = 0.01 * rng.standard_normal(int(2.0 * fs))
    x2 = 0.01 * rng.standard_normal(int(2.0 * fs))
    a, _ = fast_spl_db(x1, fs)
    b, _ = fast_spl_db(x2, fs)
    assert abs(a - b) < 3.0, f"安静背景下读数应稳定: {a:.1f} vs {b:.1f}"


def test_fast_spl_db_single_frame_input():
    """输入短于一个窗（如 <0.1s）时退化为整段 RMS，不崩溃。"""
    fs = 16000
    rng = np.random.default_rng(3)
    x = 0.05 * rng.standard_normal(int(0.05 * fs))  # 50ms < 0.1s 窗
    db, db_a = fast_spl_db(x, fs)
    assert np.isfinite(db) and np.isfinite(db_a)
    assert abs(db - rms_db(x)) < 0.5


def test_fast_spl_db_pure_sine_level():
    """纯正弦输入：短窗峰值 RMS ≈ 正弦 RMS（-3 dBFS）。"""
    fs = 16000
    t = np.arange(int(1.0 * fs)) / fs
    x = np.sin(2 * np.pi * 440 * t)
    db, _ = fast_spl_db(x, fs)
    assert abs(db - (-3.0)) < 0.5
