import numpy as np

from app.synth import printer_noise, two_sources, wind_noise


def test_printer_noise_normalized():
    x, _ = printer_noise(fs=16000, duration=2.0, seed=0)
    assert x.dtype == np.float32
    assert np.max(np.abs(x)) <= 0.5 + 1e-6


def test_printer_noise_has_stepper_family():
    x, _ = printer_noise(fs=16000, duration=2.0, seed=1)
    n = len(x)
    fft = np.abs(np.fft.rfft(x - x.mean()))
    freqs = np.fft.rfftfreq(n, 1 / 16000.0)
    peak_idx = int(np.argmax(fft[(freqs > 40) & (freqs < 2000)]))
    peak_freq = freqs[(freqs > 40) & (freqs < 2000)][peak_idx]
    assert 100 <= peak_freq <= 140, f"stepper 基频应为 ~120 Hz，实际 {peak_freq:.1f}"


def test_two_sources_shape():
    x, meta = two_sources(fs=16000, duration=1.0, seed=0)
    assert len(x) == 16000
    assert meta["f1"] == 120.0


def test_wind_noise_normalized():
    x = wind_noise(fs=16000, duration=2.0, seed=3)
    assert x.dtype == np.float32
    assert len(x) == 32000
    assert np.max(np.abs(x)) <= 0.6 + 1e-6
    assert np.std(x) > 0.01, "风噪不能是静音"


def test_wind_noise_lf_dominated():
    """风噪物理特征：能量集中在低频（<200Hz 占绝对主导），与周期/宽带噪声区分。"""
    x = wind_noise(fs=48000, duration=4.0, seed=7, cutoff_hz=600.0)
    X = np.abs(np.fft.rfft(x - x.mean())) ** 2
    f = np.fft.rfftfreq(len(x), 1 / 48000.0)
    total = float(np.sum(X))
    lf_ratio = float(np.sum(X[f < 200]) / total)
    assert lf_ratio > 0.6, f"风噪低频占比应很高，实际 {lf_ratio:.2f}"
    # 与打印机噪声对比：打印机的中频宽带（800Hz 起）把低频占比压下去
    p, _ = printer_noise(fs=48000, duration=2.0, seed=42)
    p = p.astype(np.float64)
    Xp = np.abs(np.fft.rfft(p - np.mean(p))) ** 2
    fp = np.fft.rfftfreq(len(p), 1 / 48000.0)
    lf_ratio_p = float(np.sum(Xp[fp < 200]) / float(np.sum(Xp)))
    assert lf_ratio_p < lf_ratio - 0.3, \
        f"风噪低频占比应显著高于打印机噪声：{lf_ratio:.2f} vs {lf_ratio_p:.2f}"
