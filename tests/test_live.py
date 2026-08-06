import time

import numpy as np

from app.anc.harmonic import estimate_fundamental
from app.anc.live import (BlockHarmonicCanceller, LiveANCEngine,
                          compute_safe_output)
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


def test_weight_norm_stays_bounded_under_echo_feedback():
    """回归：声学正反馈（输出被误差麦再次采到）不能使权重发散。

    用高回声增益（0.9）把扬声器输出送回输入，模拟自激条件；权重范数必须
    被限制在 max_weight_norm 内，且输出 RMS 保持很小。
    """
    fs = 48000
    c = BlockHarmonicCanceller(fs=fs, f0=120.0, block=512, mu=0.05,
                               max_harmonics=10, output_gain=0.2,
                               max_weight_norm=5.0)
    rng = np.random.default_rng(0)
    noise, _ = printer_noise(fs=fs, duration=2.0, seed=1)
    noise = noise.astype(np.float64)
    y_hist = np.zeros(c.block)
    idx = 0
    max_out = 0.0
    max_norm = 0.0
    while idx + c.block <= len(noise):
        # 误差麦采到：参考噪声 + 0.9×扬声器回声（正反馈）
        blk = noise[idx:idx + c.block] + 0.9 * y_hist
        y, _ = c.process_block(blk)
        in_rms = float(np.sqrt(np.mean(blk ** 2)))
        out = compute_safe_output(y, in_rms, c.output_gain)
        max_out = max(max_out, float(np.max(np.abs(out))))
        max_norm = max(max_norm, float(np.linalg.norm(c.w)))
        y_hist = np.roll(y_hist, c.block)
        y_hist[:c.block] = out
        idx += c.block
    assert max_norm <= 5.0 + 1e-6, f"权重范数应被限制在 5.0 内，实际 {max_norm:.2f}"
    assert max_out <= 0.12 + 1e-9, f"输出不得超过硬限幅 0.12，实际 {max_out:.3f}"


def test_output_hard_clipped_even_with_large_weights():
    """回归：即使权重巨大（模拟发散），输出也不能超过限幅电平。"""
    c = BlockHarmonicCanceller(fs=48000, f0=120.0, block=512, max_harmonics=10)
    c.w = np.full(2 * c.max_harmonics, 10.0)  # 巨大权重
    blk = np.ones(c.block) * 0.5
    y, _ = c.process_block(blk)
    in_rms = float(np.sqrt(np.mean(blk ** 2)))
    out = compute_safe_output(y, in_rms, c.output_gain)
    assert float(np.max(np.abs(out))) <= 0.12 + 1e-9
    # 权重范数上限也应生效：10.0 的初始权重会被缩回
    assert float(np.linalg.norm(c.w)) <= 5.0 + 1e-6


def test_safe_output_quiet_silence():
    """安静环境（输入能量极低）时输出静音，不喷出自身噪声。"""
    y = np.ones(512) * 5.0  # 即使 y 很大，输入安静也要静音
    out = compute_safe_output(y, in_rms=1e-6, gain=0.2)
    assert np.all(out == 0.0)
