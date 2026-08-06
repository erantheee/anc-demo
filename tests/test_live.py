import time

import numpy as np

from app.anc.harmonic import estimate_fundamental
from app.anc.live import BlockHarmonicCanceller, LiveANCEngine
from app.synth import printer_noise


def _reduction(d: np.ndarray, e: np.ndarray) -> float:
    return 20.0 * np.log10(max(np.sqrt(np.mean(e ** 2)), 1e-12)
                           / max(np.sqrt(np.mean(d ** 2)), 1e-12))


def _steady_reduction(sig: np.ndarray, e: np.ndarray, fs: float,
                      skip_s: float = 0.6) -> float:
    n = min(len(sig), len(e))
    skip = int(skip_s * fs)
    d = sig[skip:n]
    r = e[skip:n]
    return 20.0 * np.log10(max(np.sqrt(np.mean(r ** 2)), 1e-12)
                           / max(np.sqrt(np.mean(d ** 2)), 1e-12))


def test_block_canceller_reduces_pure_harmonics():
    fs = 48000
    t = np.arange(3 * fs) / fs
    sig = (0.7 * np.sin(2 * np.pi * 120 * t)
           + 0.42 * np.sin(2 * np.pi * 240 * t))
    c = BlockHarmonicCanceller(fs=fs, f0=120.0, block=512, mu=0.02, max_harmonics=5)
    es = []
    for i in range(0, len(sig) - c.block, c.block):
        _, e = c.process_block(sig[i:i + c.block])
        es.append(e)
    e = np.concatenate(es)
    red = _steady_reduction(sig, e, fs)
    assert red < -12.0, f"纯谐波应被大幅消除，实际 {red:.1f} dB"


def test_block_canceller_reduces_printer_noise():
    fs = 48000
    noise, _ = printer_noise(fs=fs, duration=3.0, seed=42)
    noise = noise.astype(np.float64)
    f0 = estimate_fundamental(noise, fs)
    assert f0 is not None
    c = BlockHarmonicCanceller(fs=fs, f0=f0, block=512, mu=0.02, max_harmonics=10)
    es = []
    for i in range(0, len(noise) - c.block, c.block):
        _, e = c.process_block(noise[i:i + c.block])
        es.append(e)
    e = np.concatenate(es)
    red = _steady_reduction(noise, e, fs)
    assert red < -4.0, f"打印机谐波应显著降低，实际 {red:.1f} dB"


def test_basis_phase_continuous_across_blocks():
    """回归：_build_basis 曾把秒值再除以 fs，导致每个 block 相位从 0 重影。"""
    c = BlockHarmonicCanceller(fs=48000, f0=120.0, block=512, max_harmonics=2)
    cos_vals = []
    for _ in range(3):
        cos_vals.append(c._basis[0, 0])
        c._basis = c._build_basis()
    # 相位必须连续推进，不应每个 block 都是 cos(0)=1
    assert cos_vals[0] == 1.0
    assert cos_vals[1] < 0.0 and cos_vals[2] < 0.0, f"基频相位未连续推进: {cos_vals}"


def test_live_engine_synthetic_full_loop():
    eng = LiveANCEngine(fs=48000, synthetic=True, baseline_s=1.0,
                        max_duration_s=4.0, echo_gain=0.15)
    res = eng.start()
    assert res["started"] is True
    deadline = time.time() + 15
    while time.time() < deadline:
        st = eng.status()
        if st["state"] in ("stopped", "error"):
            break
        time.sleep(0.2)
    st = eng.status()
    assert st["state"] == "stopped", st.get("error")
    assert st["phase"] == "done"
    assert st["f0"] is not None and 100 <= st["f0"] <= 140, st["f0"]
    assert st["baseline_spl_db"] is not None and st["cancelling_spl_db"] is not None
    assert st["reduction_db"] is not None and st["reduction_db"] < -4.0, st
