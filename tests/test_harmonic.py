import numpy as np

from app.anc.harmonic import (HarmonicCanceller, estimate_fundamental, simulate)
from app.synth import printer_noise


def test_estimate_fundamental_printer():
    x, _ = printer_noise(fs=16000, duration=2.0, seed=5)
    f0 = estimate_fundamental(x, 16000, low=60, high=300)
    assert f0 is not None
    assert 100 <= f0 <= 140, f"应估计 ~120 Hz，实际 {f0:.1f}"


def test_harmonic_canceller_reduces_printer_tonal():
    fs = 16000
    x, _ = printer_noise(fs=fs, duration=4.0, seed=6)
    f0 = estimate_fundamental(x, fs, low=60, high=300)
    res = simulate(x, fs, f0=f0, max_harmonics=10, mu=1e-3)
    assert res["f0"] is not None
    assert res["reduction_db"] < -3.0, f"谐波消除应显著降噪，实际 {res['reduction_db']:.1f}"


def test_canceller_step_shape():
    c = HarmonicCanceller(fs=16000, f0=120.0)
    y, e = c.step(0.1)
    assert np.isfinite(y) and np.isfinite(e)
