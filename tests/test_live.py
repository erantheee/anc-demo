import threading
import time

import numpy as np

from app.anc.harmonic import estimate_fundamental
from app.anc.live import (BlockHarmonicCanceller, LiveANCEngine,
                          compute_safe_output, differential_wind_ratio,
                          lf_energy_ratio)
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
    assert max_out <= 0.6 + 1e-9, f"输出不得超过硬限幅 0.6，实际 {max_out:.3f}"


def test_output_hard_clipped_even_with_large_weights():
    """回归：即使权重巨大（模拟发散），输出也不能超过限幅电平。"""
    c = BlockHarmonicCanceller(fs=48000, f0=120.0, block=512, max_harmonics=10)
    c.w = np.full(2 * c.max_harmonics, 10.0)  # 巨大权重
    blk = np.ones(c.block) * 0.5
    y, _ = c.process_block(blk)
    in_rms = float(np.sqrt(np.mean(blk ** 2)))
    out = compute_safe_output(y, in_rms, c.output_gain)
    assert float(np.max(np.abs(out))) <= 0.6 + 1e-9
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


# ---- 规则式 watchdog（防啸叫 / 降噪不足自动调节） ----

def _wd_engine(fs: int = 16000, block: int = 512, gain: float = 0.12,
               baseline_db: float = -40.0, elapsed_s: float = 20.0) -> LiveANCEngine:
    """构造一个停在 cancelling 相位、带消除器的引擎，供 watchdog 单测喂 SPL。"""
    eng = LiveANCEngine(fs=fs, block=block, output_gain=gain, baseline_s=2.0,
                        watchdog_enabled=True)
    eng.state.phase = "cancelling"
    eng.state.baseline_spl_db = baseline_db
    eng.state.elapsed_s = elapsed_s
    eng.state.gain = gain
    eng._canceller = BlockHarmonicCanceller(fs=fs, f0=120.0, block=block,
                                            output_gain=gain)
    eng._wd_blocks_since_check = 0
    return eng


def test_watchdog_reduces_gain_on_rising_spl():
    """啸叫特征：误差麦 SPL 快速上升 → watchdog 自动降增益并记日志。"""
    eng = _wd_engine()
    # 每 5 个值涨 1 dB，窗口内整体斜率 → 每秒斜率远高于 3 dB/s
    vals = [-42.0 + 1.0 * (i // 2) for i in range(60)]
    for v in vals:
        eng._watchdog_feed(v)
    st = eng.status()
    assert st["gain"] < 0.12, f"增益应被调小，实际 {st['gain']}"
    assert st["watchdog"]["reduce_count"] >= 1, st["watchdog"]
    assert any(e["action"] == "reduce_gain" for e in st["watchdog"]["log"])


def test_watchdog_increases_gain_when_no_reduction():
    """降噪不足：cancelling 一段时间后 SPL 仍接近基线 → watchdog 自动增增益。"""
    eng = _wd_engine(baseline_db=-40.0)
    # 稳态：SPL 恒定在基线-1 dB 内（无降噪），且不增长 → 应增增益
    for v in [-39.5] * 60:
        eng._watchdog_feed(v)
    st = eng.status()
    assert st["gain"] > 0.12, f"增益应被调大，实际 {st['gain']}"
    assert st["watchdog"]["increase_count"] >= 1, st["watchdog"]
    assert any(e["action"] == "increase_gain" for e in st["watchdog"]["log"])


def test_watchdog_does_not_increase_when_reduction_achieved():
    """降噪有效：SPL 明显低于基线（已降 >1.5dB）→ 不盲目增增益。"""
    eng = _wd_engine(baseline_db=-40.0)
    for v in [-46.0] * 60:  # 已降 6 dB
        eng._watchdog_feed(v)
    st = eng.status()
    assert st["watchdog"]["increase_count"] == 0, st["watchdog"]
    assert st["gain"] == 0.12


def test_watchdog_reduces_gain_when_anc_amplifies_noise():
    """ANC 放大噪声：当前 SPL 比基线高很多（反相波不匹配 → 能量叠加）→
    watchdog 应降增益，而不是误判为"降噪不足"继续加增益。

    回归保护：基线 -56.6dB、SPL 稳定在 -7dB（高 49dB）时，旧逻辑会一路
    加增益放大尖锐声；新逻辑必须降增益。降增益后若 SPL 随之下降，说明
    确实是 ANC 在放大 → 保持低增益。
    """
    eng = _wd_engine(baseline_db=-56.6, gain=0.40)
    # 先喂高位稳定 SPL（触发 2a 降增益），随后 SPL 随增益下降（有效）
    vals = ([-7.0] * 40) + ([-12.0] * 80)
    for v in vals:
        eng._watchdog_feed(v)
    st = eng.status()
    assert st["gain"] < 0.40, f"放大噪声时应降增益，实际 {st['gain']}"
    assert st["watchdog"]["reduce_count"] >= 1, st["watchdog"]
    assert any(e["action"] == "reduce_gain" for e in st["watchdog"]["log"])
    assert st["watchdog"]["increase_count"] == 0, st["watchdog"]


def test_watchdog_restores_gain_when_reduction_ineffective():
    """降增益后 SPL 不降（外部噪声源 / 声耦合主导，与增益无关）→ 不应把增益
    一路砍到无法降噪，而应恢复增益并冻结 2a/2b。

    回归保护：真机测试中出现基线 -53dB（安静）、用户中途开大打印机声源、
    误差麦 SPL 稳定在 -8dB 时，旧逻辑连降 4 级增益到 0.02 仍无济于事。
    """
    eng = _wd_engine(baseline_db=-53.5, gain=0.20)
    # 稳定高位且不随增益变化（模拟外部声源），喂足够多次覆盖 3 次验证窗口
    for v in [-8.0] * 200:
        eng._watchdog_feed(v)
    st = eng.status()
    # 增益应被恢复/保持在原值，而不是被砍到 min
    assert st["gain"] >= 0.20, f"无效降增益应恢复，实际 {st['gain']}"
    assert any(e["action"] == "restore_gain" for e in st["watchdog"]["log"]), st["watchdog"]["log"]


def test_watchdog_gain_clamped_to_max():
    """增增益不超过 watchdog_max_gain（安全上限）。"""
    eng = _wd_engine(gain=0.30)
    for v in [-39.5] * 200:
        eng._watchdog_feed(v)
    st = eng.status()
    assert st["gain"] <= eng.watchdog_max_gain, st


# ---- 可写控制（Agent / Web 调节） ----

def test_control_set_gain_clamped():
    eng = _wd_engine()
    r = eng.control("set_gain", {"value": 10.0})
    assert r["ok"] is True
    assert r["gain"] <= eng.watchdog_max_gain


def test_control_increase_decrease_gain():
    eng = _wd_engine(gain=0.12)
    r = eng.control("increase_gain", {"delta": 0.05})
    assert r["ok"] is True and r["gain"] > 0.12
    r2 = eng.control("decrease_gain", {"delta": 0.05})
    assert r2["ok"] is True and r2["gain"] < r["gain"]


def test_control_set_mic_delay_ms():
    eng = _wd_engine()
    r = eng.control("set_mic_delay_ms", {"mic_delay_ms": 25.0})
    assert r["ok"] is True
    assert abs(r["mic_delay_ms"] - 25.0) < 1e-6
    assert eng._canceller.predict_ahead_samples == int(round(25.0 * eng.fs / 1000.0))


def test_control_unknown_action():
    eng = _wd_engine()
    r = eng.control("bogus", {})
    assert r["ok"] is False


def test_control_missing_param_returns_error():
    eng = _wd_engine()
    r = eng.control("set_gain", {})
    assert r["ok"] is False
    r2 = eng.control("set_mic_delay_ms", {})
    assert r2["ok"] is False


def test_step_output_gain_atomic_under_concurrency(monkeypatch):
    """回归：并发 step_output_gain 不丢失增量。

    旧实现是「锁内读 cur → 放锁 → set_output_gain(cur+delta)」（两次加锁），
    两线程会基于同一个旧值计算，其中一次增量被覆盖。修复后读-算-钳-写全在
    单次锁内完成，step_output_gain 不再经过 set_output_gain。

    黑盒并发压测（Barrier + 高步数）在本机 CPython 上无法稳定触发 TOCTOU
    窗口，因此这里直接给 set_output_gain 注入栅栏：旧实现每次 step 都会经过
    它，两线程必然在同一旧基值上汇合，确定性丢失增量；新实现不调用它，
    栅栏保持惰性，正常串行累加。
    """
    eng = _wd_engine(gain=0.05)
    n_steps = 20
    delta = 0.002

    barrier = threading.Barrier(2)
    orig = eng.set_output_gain

    def sync_write(value, reason=""):
        barrier.wait(timeout=5.0)
        return orig(value, reason)

    eng.set_output_gain = sync_write

    start = threading.Barrier(2)
    errors = []

    def worker():
        try:
            start.wait()
            for _ in range(n_steps):
                eng.step_output_gain(delta)
        except Exception as exc:  # pragma: no cover —— 失败路径
            errors.append(exc)

    t1 = threading.Thread(target=worker)
    t2 = threading.Thread(target=worker)
    t1.start()
    t2.start()
    t1.join()
    t2.join()
    assert not errors
    expected = 0.05 + 2 * n_steps * delta
    final = eng.status()["gain"]
    assert abs(final - expected) < 1e-6, f"并发增量丢失：期望 {expected:.4f}，实际 {final:.4f}"
    assert final <= eng.watchdog_max_gain


def test_residual_tail_empty_when_no_engine_data():
    eng = _wd_engine()
    assert eng.residual_tail(1000) is None
    assert eng.residual_tail(0) is None


def test_residual_tail_returns_last_n_samples():
    eng = _wd_engine()
    eng.e_buf = [np.ones(64), np.ones(64) * 2.0, np.ones(64) * 3.0]
    tail = eng.residual_tail(64)
    assert tail is not None
    assert np.all(tail == 3.0), "应取最后一个 block 的尾部"
    # 跨 block 拼接：取 100 个样本应包含最后 36 个 3.0 与 64 个 2.0
    tail2 = eng.residual_tail(100)
    assert len(tail2) == 100
    assert np.all(tail2[-64:] == 3.0)
    assert np.all(tail2[:36] == 2.0)
    # 返回副本，后续修改不影响引擎缓冲
    tail2[0] = -999.0
    assert np.all(eng.e_buf[-2][:1] == 2.0)


# ---- 人声闸门（基线像人声时不进入 cancelling） ----

def test_begin_cancel_refuses_voice_baseline():
    """人声基线：自动估计基频时不应进入 cancelling（避免反相波"吃掉"说话声）。"""
    from app.synth import speech_like

    eng = LiveANCEngine(fs=16000, synthetic=True, baseline_s=1.0)
    eng.state.state = "running"
    eng.state.phase = "baseline"
    eng.d_buf = [speech_like(fs=16000, duration=4.0)]
    eng._begin_cancel()
    st = eng.status()
    assert st["state"] == "error"
    assert "人声" in st["error"], st["error"]
    assert st["phase"] == "baseline"


def test_begin_cancel_proceeds_on_steady_noise():
    """稳态周期噪声基线：自动估计基频应进入 cancelling。"""
    fs = 16000
    t = np.arange(2 * fs) / fs
    sig = (0.3 * np.sin(2 * np.pi * 140.6 * t)
           + 0.15 * np.sin(2 * np.pi * 2 * 140.6 * t)).astype(np.float32)
    eng = LiveANCEngine(fs=fs, synthetic=True, baseline_s=1.0)
    eng.state.state = "running"
    eng.state.phase = "baseline"
    eng.d_buf = [sig]
    eng._begin_cancel()
    st = eng.status()
    assert st["state"] == "running", st
    assert st["phase"] == "cancelling", st
    assert st["f0"] is not None and abs(st["f0"] - 140.6) < 3.0, st["f0"]


def test_begin_cancel_manual_f0_bypasses_voice_gate():
    """手动指定 f0 时跳过人声闸门（用户明确要消除该频率）。"""
    fs = 16000
    t = np.arange(2 * fs) / fs
    sig = (0.3 * np.sin(2 * np.pi * 140.6 * t)).astype(np.float32)
    eng = LiveANCEngine(fs=fs, synthetic=True, baseline_s=1.0, f0=140.6)
    eng.state.state = "running"
    eng.state.phase = "baseline"
    eng.d_buf = [sig]
    eng._begin_cancel()
    st = eng.status()
    assert st["phase"] == "cancelling", st


# ---- 实时人声门控（ANC 运行中检测到说话声 → 静音反相输出） ----

def _voice_gate_engine(fs: int = 16000, block: int = 512) -> LiveANCEngine:
    eng = LiveANCEngine(fs=fs, block=block, synthetic=True, baseline_s=1.0)
    eng.state.phase = "cancelling"
    return eng


def _feed_voice_hist(eng: LiveANCEngine, sig: np.ndarray) -> None:
    blk = eng.block
    eng._voice_hist = [
        sig[i:i + blk].astype(np.float64).copy()
        for i in range(0, len(sig) - blk + 1, blk)
    ]


def test_voice_gate_mutes_on_speech_during_cancelling():
    """ANC 运行中误差麦出现人声 → 门控静音反相输出并记日志。"""
    from app.synth import speech_like

    eng = _voice_gate_engine()
    _feed_voice_hist(eng, speech_like(fs=16000, duration=3.0))
    eng._voice_last_check = 0.0
    eng._voice_gate_tick()
    st = eng.status()
    assert st["voice_gate"]["mute"] is True, st["voice_gate"]
    assert st["voice_gate"]["reasons"], st["voice_gate"]
    assert any(e["action"] == "voice_mute" for e in st["watchdog"]["log"])


def test_voice_gate_stays_unmuted_on_steady_noise():
    """稳态周期噪声（无说话声）→ 门控不误静音。"""
    fs = 16000
    t = np.arange(2 * fs) / fs
    sig = (0.3 * np.sin(2 * np.pi * 140.6 * t)
           + 0.15 * np.sin(2 * np.pi * 2 * 140.6 * t)).astype(np.float64)
    eng = _voice_gate_engine()
    _feed_voice_hist(eng, sig)
    eng._voice_last_check = 0.0
    eng._voice_gate_tick()
    st = eng.status()
    assert st["voice_gate"]["mute"] is False, st["voice_gate"]


def test_voice_gate_unmutes_after_speech_stops():
    """人声停止 ~1.5s 后门控自动恢复 ANC 输出。"""
    from app.synth import speech_like

    eng = _voice_gate_engine()
    _feed_voice_hist(eng, speech_like(fs=16000, duration=3.0))
    eng._voice_last_check = 0.0
    eng._voice_gate_tick()
    assert eng.status()["voice_gate"]["mute"] is True

    fs = eng.fs
    t = np.arange(2 * fs) / fs
    steady = (0.3 * np.sin(2 * np.pi * 140.6 * t)).astype(np.float64)
    _feed_voice_hist(eng, steady)
    for _ in range(4):
        eng._voice_last_check = 0.0
        eng._voice_gate_tick()
    st = eng.status()
    assert st["voice_gate"]["mute"] is False, st["voice_gate"]
    assert any(e["action"] == "voice_unmute" for e in st["watchdog"]["log"])


def test_voice_gate_skips_when_hist_short():
    """误差麦缓冲不足（<3 blocks）时不跑语音检测，也不误静音。"""
    eng = _voice_gate_engine()
    eng._voice_hist = [np.zeros(eng.block)]
    eng._voice_last_check = 0.0
    eng._voice_gate_tick()
    assert eng.status()["voice_gate"]["mute"] is False


# ---- 反馈中和环形缓冲（写指针方案） ----

def test_read_echo_returns_sample_written_delay_ago():
    """回归（Bug #2 根因）：_read_echo 必须精确返回 delay_n 个采样点前的输出。

    旧实现（np.roll + 头写 + [len-delay_n] 读）把 40ms 延迟读成了 ~130ms，
    中和信号错位 → NLMS 追错残差 → 输出与噪声同相叠加（Pi 现场 +1.8 dB）。
    写指针方案读 (写指针 - delay_n) % len，等价于理想延迟线。
    """
    fs = 48000
    eng = _wd_engine(fs=fs, block=64)
    n = len(eng._out_hist)
    assert n % eng.block == 0, "环形缓冲长度必须是 block 的整数倍"

    delay_n = 5 * eng.block  # 5 个 block 前（160 samples @48k = 3.3ms）
    eng._out_pos = 0
    # 按时间顺序写入带标记的 block：第 k 个 block 全为 k
    for k in range(10):
        eng._out_hist[eng._out_pos:eng._out_pos + eng.block] = float(k)
        eng._out_pos = (eng._out_pos + eng.block) % n

    # 写指针现指向 10 个 block 的位置；delay_n=5 前的 block 应为 10-5=5
    echo = eng._read_echo(delay_n)
    assert len(echo) == eng.block
    assert np.all(echo == 5.0), "应精确读到 5 个 block 前写入的 block"


def test_read_echo_handles_wraparound():
    """读位置跨环形缓冲尾部绕回时，拼接结果仍正确。"""
    fs = 48000
    eng = _wd_engine(fs=fs, block=64)
    n = len(eng._out_hist)
    nblocks = n // eng.block
    assert nblocks >= 20

    eng._out_pos = 0
    for k in range(nblocks):
        eng._out_hist[eng._out_pos:eng._out_pos + eng.block] = float(k)
        eng._out_pos = (eng._out_pos + eng.block) % n

    # 写满一周：_out_pos 回到 0（下一写位置），最近写入的是 block nblocks-1。
    # 当前 block 编号 = nblocks，echo(delay=2 blocks) → block nblocks-2。
    echo = eng._read_echo(2 * eng.block)
    assert np.all(echo == float(nblocks - 2)), "绕回后应读到正确的历史 block"


def test_read_echo_silent_during_warmup():
    """预热期（写入历史不足 delay_n）返回静音，等价于尚无回声。"""
    fs = 48000
    eng = _wd_engine(fs=fs, block=64)
    eng._out_pos = 0
    for k in range(2):
        eng._out_hist[eng._out_pos:eng._out_pos + eng.block] = float(k)
        eng._out_pos = (eng._out_pos + eng.block) % len(eng._out_hist)
    # 只写了 2 个 block，读 5 个 block 前的延迟 → 缓冲区中对应的历史从未写入
    echo = eng._read_echo(5 * eng.block)
    assert np.all(echo == 0.0)


# ---- 风噪门控 / 输入高通 / f0 风噪拒绝 ----

def test_wind_gate_mutes_on_wind_during_cancelling():
    """风噪门控：低频能量占比超阈值 → 静音反相输出并记日志。"""
    from app.synth import wind_noise
    from app.anc.live import lf_energy_ratio

    fs = 48000
    eng = LiveANCEngine(fs=fs, block=512, synthetic=True, baseline_s=1.0,
                        wind_gate_enabled=True, wind_gate_cutoff_hz=100.0,
                        wind_gate_ratio_thresh=0.8)
    eng.state.phase = "cancelling"
    eng._canceller = BlockHarmonicCanceller(fs=fs, f0=120.0, block=eng.block,
                                            output_gain=0.5)
    w = wind_noise(fs=fs, duration=1.5, seed=7, cutoff_hz=600.0).astype(np.float64)
    eng._wind_hist = [w[:eng.block], w[eng.block:2 * eng.block], w[2 * eng.block:3 * eng.block]]
    assert lf_energy_ratio(np.concatenate(eng._wind_hist), fs, cutoff=100.0) > 0.8
    eng._wind_gate_tick()
    st = eng.status()
    assert st["wind_gate"]["enabled"] is True
    assert st["wind_gate"]["mute"] is True, st["wind_gate"]
    assert any(e["action"] == "wind_mute" for e in st["watchdog"]["log"])


def test_wind_gate_stays_unmuted_on_steady_noise():
    """稳态周期噪声（无风）→ 风噪门控不误静音。"""
    fs = 48000
    t = np.arange(2 * fs) / fs
    sig = (0.3 * np.sin(2 * np.pi * 140.6 * t)
           + 0.15 * np.sin(2 * np.pi * 2 * 140.6 * t)).astype(np.float64)
    eng = LiveANCEngine(fs=fs, block=512, synthetic=True, baseline_s=1.0,
                        wind_gate_enabled=True, wind_gate_cutoff_hz=100.0,
                        wind_gate_ratio_thresh=0.8)
    eng.state.phase = "cancelling"
    eng._wind_hist = [sig[:eng.block], sig[eng.block:2 * eng.block], sig[2 * eng.block:3 * eng.block]]
    eng._wind_gate_tick()
    st = eng.status()
    assert st["wind_gate"]["mute"] is False, st["wind_gate"]


def test_wind_gate_unmutes_after_wind_stops():
    """风噪消退 → 门控自动恢复 ANC 输出并记日志。"""
    from app.synth import wind_noise

    fs = 48000
    eng = LiveANCEngine(fs=fs, block=512, synthetic=True, baseline_s=1.0,
                        wind_gate_enabled=True, wind_gate_cutoff_hz=100.0,
                        wind_gate_ratio_thresh=0.8)
    eng.state.phase = "cancelling"
    eng._canceller = BlockHarmonicCanceller(fs=fs, f0=120.0, block=eng.block,
                                            output_gain=0.5)
    w = wind_noise(fs=fs, duration=1.5, seed=7, cutoff_hz=600.0).astype(np.float64)
    eng._wind_hist = [w[:eng.block], w[eng.block:2 * eng.block], w[2 * eng.block:3 * eng.block]]
    eng._wind_gate_tick()
    assert eng.status()["wind_gate"]["mute"] is True

    t = np.arange(2 * fs) / fs
    steady = (0.3 * np.sin(2 * np.pi * 140.6 * t)).astype(np.float64)
    eng._wind_hist = [steady[:eng.block], steady[eng.block:2 * eng.block], steady[2 * eng.block:3 * eng.block]]
    eng._wind_gate_tick()
    st = eng.status()
    assert st["wind_gate"]["mute"] is False, st["wind_gate"]
    assert any(e["action"] == "wind_unmute" for e in st["watchdog"]["log"])


def test_input_highpass_removes_lf_wind_floor():
    """输入高通：纯风噪经高通后低频（<200Hz）能量显著下降。"""
    from app.synth import wind_noise
    from app.anc.live import highpass_filter

    fs = 48000
    eng = LiveANCEngine(fs=fs, block=512, input_highpass_hz=100.0)
    w = wind_noise(fs=fs, duration=2.0, seed=7, cutoff_hz=600.0).astype(np.float64)
    out = highpass_filter(w, fs, 100.0)

    def band_db(x):
        X = np.abs(np.fft.rfft(x - x.mean())) ** 2
        f = np.fft.rfftfreq(len(x), 1 / fs)
        m = (f > 0) & (f < 200)
        return 10.0 * np.log10(np.mean(X[m]) + 1e-12)

    before_db, after_db = band_db(w), band_db(out)
    assert before_db - after_db > 6.0, \
        f"高通应显著压低 <200Hz 能量：{before_db:.1f} → {after_db:.1f} dB"


def test_begin_cancel_refuses_wind_baseline():
    """风噪基线：低频占比超阈值时拒绝进入 cancelling（不输出伪 f0 反相波）。"""
    from app.synth import wind_noise

    fs = 48000
    eng = LiveANCEngine(fs=fs, synthetic=True, baseline_s=1.0,
                        wind_gate_enabled=True, wind_gate_cutoff_hz=100.0,
                        wind_gate_ratio_thresh=0.8)
    eng.state.state = "running"
    eng.state.phase = "baseline"
    eng.d_buf = [wind_noise(fs=fs, duration=2.0, seed=7, cutoff_hz=600.0)]
    eng._begin_cancel()
    st = eng.status()
    assert st["state"] == "error"
    assert "风噪" in st["error"], st["error"]


def test_begin_cancel_allows_noisy_baseline():
    """稳态周期噪声基线（低频占比低）→ 正常进入 cancelling。"""
    fs = 48000
    t = np.arange(2 * fs) / fs
    sig = (0.3 * np.sin(2 * np.pi * 140.6 * t)
           + 0.15 * np.sin(2 * np.pi * 2 * 140.6 * t)).astype(np.float32)
    eng = LiveANCEngine(fs=fs, synthetic=True, baseline_s=1.0,
                        wind_gate_enabled=True, wind_gate_cutoff_hz=100.0,
                        wind_gate_ratio_thresh=0.8)
    eng.state.state = "running"
    eng.state.phase = "baseline"
    eng.d_buf = [sig]
    eng._begin_cancel()
    st = eng.status()
    assert st["state"] == "running", st
    assert st["phase"] == "cancelling", st


def test_begin_cancel_lowfreq_hum_not_refused_when_gate_off():
    """风噪门控关闭时，低频周期源（50Hz 哼声，LF 占比 > 0.8）不被误拒。"""
    fs = 48000
    t = np.arange(2 * fs) / fs
    hum = (0.4 * np.sin(2 * np.pi * 50.0 * t)).astype(np.float32)
    eng = LiveANCEngine(fs=fs, synthetic=True, baseline_s=1.0,
                        wind_gate_enabled=False)  # 门控关闭：不启用风噪 f0 拒绝
    eng.state.state = "running"
    eng.state.phase = "baseline"
    eng.d_buf = [hum]
    eng._begin_cancel()
    st = eng.status()
    assert st["state"] == "running", st
    assert st["phase"] == "cancelling", st


# ---- 双麦差分风噪检测 ----

def test_differential_wind_ratio_low_for_coherent():
    """相干周期声（两路同相）→ 差分指标低（不应判为风）。"""
    fs = 48000
    t = np.arange(4 * fs) / fs
    sig = (0.3 * np.sin(2 * np.pi * 140.6 * t)
           + 0.15 * np.sin(2 * np.pi * 2 * 140.6 * t))
    ratio = differential_wind_ratio(sig, sig.copy(), fs)
    assert ratio < 0.3, f"相干声差分指标应低，实际 {ratio:.2f}"


def test_differential_wind_ratio_high_for_decorrelated():
    """两路不相关（模拟风噪湍流）→ 差分指标高。"""
    fs = 48000
    n = 4 * fs
    rng = np.random.default_rng(7)
    a = rng.standard_normal(n)
    b = rng.standard_normal(n)
    ratio = differential_wind_ratio(a, b, fs)
    assert ratio > 0.5, f"不相关信号差分指标应高，实际 {ratio:.2f}"


def test_differential_wind_ratio_ignores_50hz_hum():
    """50Hz 工频哼声（含谐波）在两路相干 → 差分指标低，不误判为风。

    对比单麦 lf_energy_ratio：纯低频占比检测会把 50Hz 哼声误判为风
    （低频占比≈1），而双麦差分看的是两路相干性，相干工频哼声差分≈0。
    注意：真实工频哼声必带 150/250/350Hz 谐波，这些谐波落在差分检测的
    60–2000Hz 带内且相干；若只有纯 50Hz 正弦（带内无相干能量），差分指标
    会退回本底——那是检测器频带下限的正常行为。
    """
    fs = 48000
    t = np.arange(4 * fs) / fs
    hum = (0.4 * np.sin(2 * np.pi * 50.0 * t)
           + 0.2 * np.sin(2 * np.pi * 150.0 * t)
           + 0.1 * np.sin(2 * np.pi * 250.0 * t)
           + 0.05 * np.sin(2 * np.pi * 350.0 * t))
    # 两路几乎同相（只加一点点独立噪声模拟 capsule 失配）
    rng = np.random.default_rng(1)
    a = hum + 0.001 * rng.standard_normal(len(hum))
    b = hum + 0.001 * rng.standard_normal(len(hum))
    diff_ratio = differential_wind_ratio(a, b, fs)
    lf_ratio = lf_energy_ratio(a, fs, cutoff=100.0)
    # 关键对比：单麦低频占比显著高于双麦差分指标——单麦会倾向误判为风，
    # 双麦看相干性则干净利落地不判风。
    assert lf_ratio > 0.5, f"50Hz 哼声低频占比应偏高，实际 {lf_ratio:.2f}"
    assert diff_ratio < 0.3, f"双麦差分对相干哼声应低，实际 {diff_ratio:.2f}"
    assert diff_ratio < lf_ratio - 0.3, \
        f"差分指标应显著低于单麦低频占比：diff={diff_ratio:.2f} vs lf={lf_ratio:.2f}"


def test_wind_gate_uses_diff_metric_when_dual_mic():
    """双麦模式：两路不相关 → 差分指标驱动门控静音。"""
    fs = 48000
    rng = np.random.default_rng(11)
    n = 512 * 3
    l = rng.standard_normal(n).astype(np.float64)
    r = rng.standard_normal(n).astype(np.float64)
    eng = LiveANCEngine(fs=fs, block=512, synthetic=True, baseline_s=1.0,
                        wind_gate_enabled=True, dual_mic=True,
                        wind_gate_diff_thresh=0.5)
    eng.state.phase = "cancelling"
    eng._wind_hist = [(l[:512], r[:512]), (l[512:1024], r[512:1024]),
                      (l[1024:1536], r[1024:1536])]
    eng._wind_gate_tick()
    st = eng.status()
    assert st["wind_gate"]["dual_mic"] is True
    assert st["wind_gate"]["mute"] is True, st["wind_gate"]
    assert st["wind_gate"]["diff_ratio"] is not None
    assert st["wind_gate"]["diff_ratio"] > 0.5
    assert any(e["action"] == "wind_mute" for e in st["watchdog"]["log"])


def test_wind_gate_diff_metric_not_muted_on_coherent():
    """双麦模式：两路相干周期声 → 差分指标低，门控不静音。"""
    fs = 48000
    t = np.arange(512 * 3) / fs
    sig = (0.3 * np.sin(2 * np.pi * 140.6 * t)
           + 0.15 * np.sin(2 * np.pi * 2 * 140.6 * t)).astype(np.float64)
    eng = LiveANCEngine(fs=fs, block=512, synthetic=True, baseline_s=1.0,
                        wind_gate_enabled=True, dual_mic=True,
                        wind_gate_diff_thresh=0.5)
    eng.state.phase = "cancelling"
    eng._wind_hist = [(sig[:512], sig[:512]), (sig[512:1024], sig[512:1024]),
                      (sig[1024:1536], sig[1024:1536])]
    eng._wind_gate_tick()
    st = eng.status()
    assert st["wind_gate"]["mute"] is False, st["wind_gate"]
    assert st["wind_gate"]["diff_ratio"] < 0.5, st["wind_gate"]


def test_wind_gate_diff_metric_ignores_quiet_floor():
    """双麦模式：安静本底（两路各自的电子噪声）差分指标会被本底污染，
    但能量门槛应阻止门控误触发。"""
    fs = 48000
    rng = np.random.default_rng(21)
    # 极弱信号：两路独立小噪声，RMS ≈ -60dBFS
    n = 512 * 3
    l = 1e-3 * rng.standard_normal(n).astype(np.float64)
    r = 1e-3 * rng.standard_normal(n).astype(np.float64)
    eng = LiveANCEngine(fs=fs, block=512, synthetic=True, baseline_s=1.0,
                        wind_gate_enabled=True, dual_mic=True,
                        wind_gate_diff_thresh=0.5)
    eng.state.phase = "cancelling"
    eng._wind_hist = [(l[:512], r[:512]), (l[512:1024], r[512:1024]),
                      (l[1024:1536], r[1024:1536])]
    eng._wind_gate_tick()
    st = eng.status()
    assert st["wind_gate"]["mute"] is False, st["wind_gate"]
    assert st["wind_gate"]["diff_ratio"] == 0.0, st["wind_gate"]

