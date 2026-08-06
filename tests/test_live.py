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


# ---- 扬声器→误差麦延迟补偿（次级路径对齐） ----

def _fan_like_noise(fs: float, duration: float, f0: float = 140.6) -> np.ndarray:
    """仿风扇：基频 140.6Hz + 前三次谐波。"""
    t = np.arange(int(duration * fs)) / fs
    return (0.08 * np.sin(2 * np.pi * f0 * t)
            + 0.05 * np.sin(2 * np.pi * 2 * f0 * t)
            + 0.03 * np.sin(2 * np.pi * 3 * f0 * t))


def _simulate_delay_loop(noise: np.ndarray, fs: float, f0: float, delay_s: float,
                         comp_s: float, echo_gain: float, output_gain: float = 1.0,
                         block: int = 512, mu: float = 0.02,
                         max_harmonics: int = 10) -> np.ndarray:
    """模拟物理环路：误差麦听到 noise(t) + echo_gain·out(t-δ)（扬声器输出经 δ
    到达误差麦），消除器对该信号实时输出反相波。返回误差麦处的物理残差
    r(t) = noise(t) + echo_gain·out(t-δ)，即用户实际听到的降噪效果。
    """
    delay_samples = int(round(delay_s * fs))
    c = BlockHarmonicCanceller(fs=fs, f0=f0, block=block, mu=mu,
                               max_harmonics=max_harmonics,
                               output_gain=output_gain,
                               speaker_mic_delay_s=comp_s)
    n = len(noise)
    out_all = np.zeros(n)
    idx = 0
    while idx + block <= n:
        d = noise[idx:idx + block].astype(np.float64).copy()
        if delay_samples > 0 and idx >= delay_samples:
            d += echo_gain * out_all[idx - delay_samples: idx - delay_samples + block]
        y, e = c.process_block(d)
        in_rms = float(np.sqrt(np.mean(d ** 2)))
        out_all[idx:idx + block] = compute_safe_output(y, in_rms, c.output_gain)
        idx += block
    r_mic = noise.astype(np.float64).copy()
    if delay_samples > 0:
        r_mic[delay_samples:] += echo_gain * out_all[:n - delay_samples]
    return r_mic


def _rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def test_output_predicts_noise_forward_by_delay():
    """机制：输出重建基必须"向前预测" δ，即 y_out(t) ≈ noise(t + δ)。

    扬声器在时刻 t 播放的声音经 δ 秒才到达误差麦（时刻 t+δ），所以反相输出
    必须对准噪声在 t+δ 的相位。该测试无反馈，直接验证预测方向/符号。
    """
    fs = 48000
    f0 = 140.6
    delay_s = 0.005
    ds = int(round(delay_s * fs))
    noise = _fan_like_noise(fs, duration=4.0, f0=f0)
    c = BlockHarmonicCanceller(fs=fs, f0=f0, block=512, mu=0.02,
                               max_harmonics=10, output_gain=0.08,
                               speaker_mic_delay_s=delay_s)
    ys = []
    idx = 0
    while idx + c.block <= len(noise):
        y, _ = c.process_block(noise[idx:idx + c.block])
        ys.append(y)
        idx += c.block
    y_out = np.concatenate(ys)

    skip = int(1.0 * fs)
    seg_y = y_out[skip: -ds]
    seg_fwd = noise[skip + ds: skip + ds + len(seg_y)]
    seg_now = noise[skip: skip + len(seg_y)]
    base = _rms(noise[skip: skip + len(seg_y)])
    # y_out 应对准"未来"的噪声（同相预测），而不是当前噪声或取反
    assert _rms(seg_y - seg_fwd) / base < 1e-1, "y_out 未向前预测 delay"
    assert _rms(seg_y - seg_now) / base > 0.3, "y_out 不应停留在当前相位（未补偿）"
    assert _rms(-seg_y - seg_fwd) / base > 0.3, "y_out 符号方向错误"


def test_delay_compensation_reduces_mic_residual():
    """闭环：已知扬声器→麦延迟 δ 时，开启 δ 补偿后误差麦残差显著低于不补偿。

    模拟真实物理环路（扬声器输出经 δ 延迟被误差麦回采），比较误差麦处残差：
    不补偿时反相波到达误差麦时已与噪声错位 → 残差不降反升；补偿 δ 后对准 →
    残差显著下降（至少 3-6 dB，实测 δ=2ms、环路增益 0.6 时差 ~9 dB）。
    """
    fs = 48000
    f0 = 140.6
    delay_s = 0.002  # 2ms，麦克风靠近扬声器的量级
    duration = 8.0
    noise = _fan_like_noise(fs, duration=duration, f0=f0)
    echo_gain, output_gain = 0.6, 1.0
    skip = int(2.0 * fs)
    base = _rms(noise[skip:])

    r_yes = _simulate_delay_loop(noise, fs, f0, delay_s, delay_s,
                                 echo_gain, output_gain)
    r_no = _simulate_delay_loop(noise, fs, f0, delay_s, 0.0,
                                echo_gain, output_gain)
    r_wrong = _simulate_delay_loop(noise, fs, f0, delay_s, -delay_s,
                                   echo_gain, output_gain)

    red_yes = 20 * np.log10(_rms(r_yes[skip:]) / base)
    red_no = 20 * np.log10(_rms(r_no[skip:]) / base)
    red_wrong = 20 * np.log10(_rms(r_wrong[skip:]) / base)
    assert red_yes < -3.0, f"补偿后应真正降噪，实际 {red_yes:.1f} dB"
    assert red_yes < red_no - 3.0, \
        f"延迟补偿应比不补偿至少好 3 dB：comp={red_yes:.1f} vs none={red_no:.1f}"
    assert red_yes < red_wrong, \
        f"延迟方向错误（向后预测）应更差：comp={red_yes:.1f} vs -δ={red_wrong:.1f}"


def test_delay_compensation_optimum_at_true_delay():
    """方向扫查：残差在补偿值≈真实延迟 δ 处最小（过大/过小/取反都更差）。"""
    fs = 48000
    f0 = 140.6
    delay_s = 0.001  # 1ms
    noise = _fan_like_noise(fs, duration=6.0, f0=f0)
    echo_gain, output_gain = 1.0, 1.0
    skip = int(2.0 * fs)
    base = _rms(noise[skip:])

    comps = [-delay_s, 0.0, 0.5 * delay_s, delay_s, 2 * delay_s]
    reds = {}
    for comp_s in comps:
        r = _simulate_delay_loop(noise, fs, f0, delay_s, comp_s,
                                 echo_gain, output_gain)
        reds[comp_s] = 20 * np.log10(_rms(r[skip:]) / base)
    best = min(reds, key=lambda k: reds[k])
    assert best == delay_s, f"最优补偿应等于真实延迟 δ，实际最优在 {best*1000:.1f}ms: {reds}"
    assert reds[delay_s] < reds[0.0] - 3.0
    assert reds[delay_s] < reds[-delay_s] - 3.0
