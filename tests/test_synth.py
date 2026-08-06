import numpy as np

from app.synth import printer_noise, two_sources


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
