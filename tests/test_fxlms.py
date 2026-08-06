import numpy as np

from app.anc.fxlms import FXLMS, simulate


def _tonal_reference(fs=16000, duration=3.0, freq=150.0):
    t = np.arange(int(fs * duration)) / fs
    x = 0.3 * np.sin(2 * np.pi * freq * t) + 0.15 * np.sin(2 * np.pi * 2 * freq * t)
    return x


def test_fxlms_converges_for_tone():
    fs = 16000
    ns = _tonal_reference(fs)
    secondary_path = np.array([0.4, 0.6, 1.0, 0.6, 0.3])
    res = simulate(ns, fs, secondary_path=secondary_path, plant_delay_samples=16,
                   num_taps=256, mu=1e-4)
    assert res["reduction_db"] < -10.0, f"音调降噪应 ≥ 10 dB，实际 {res['reduction_db']:.1f}"


def test_fxlms_step_returns_finite():
    anc = FXLMS(num_taps=32, mu=1e-3)
    y, e = anc.step(0.1, 0.1)
    assert np.isfinite(y) and np.isfinite(e)


def test_fxlms_zero_reference_no_nan():
    anc = FXLMS(num_taps=16, mu=1e-2)
    y, e = anc.step(0.0, 0.0)
    assert np.isfinite(y) and np.isfinite(e)
