"""风噪离线仿真验证：量化各方案对风噪的效果。

镜像 live.py 信号通路（误差麦 = 噪声 + 回声、反馈中和、谐波消除、防啸叫输出），
在"纯风噪"和"打印机噪声 + 风噪混合"两种场景下对比：

  A. ANC 关闭（基线）
  B. 现有谐波 ANC（无风噪处理）
  C. B + 输入高通（~100Hz 去低频风底噪）
  D. B + 风噪门控（低频能量占比超阈值 → 静音输出，防"对着风喷不相关噪声"）
  E. C + D 组合

另外验证两件事：
  - estimate_fundamental 在风噪下会不会骗出"伪 f0"（卡门涡街的机理）
  - 低频能量占比检测器能否把风噪与打印机噪声/人声分开

指标：宽带降噪 dB、A 加权降噪 dB、低频带(<200Hz)降噪 dB。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy import signal

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.anc.harmonic import estimate_fundamental  # noqa: E402
from app.anc.live import BlockHarmonicCanceller, compute_safe_output  # noqa: E402
from app.synth import printer_noise, wind_noise  # noqa: E402


def rms(x: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.asarray(x, dtype=np.float64) ** 2)))


def _band_psd_db(x: np.ndarray, fs: float, lo: float, hi: float) -> float:
    x = np.asarray(x, dtype=np.float64)
    X = np.abs(np.fft.rfft(x - x.mean())) ** 2
    f = np.fft.rfftfreq(len(x), 1 / fs)
    m = (f >= lo) & (f <= hi)
    if not np.any(m):
        return 0.0
    return 10.0 * np.log10(np.mean(X[m]) + 1e-12)


def metrics(before: np.ndarray, after: np.ndarray, fs: float) -> dict:
    skip = min(len(before), len(after))
    d = before[skip // 3:]
    e = after[skip // 3:]
    broadband = 20.0 * np.log10(max(rms(e), 1e-12) / max(rms(d), 1e-12))
    lf = _band_psd_db(e, fs, 0, 200) - _band_psd_db(d, fs, 0, 200)
    # 中频带（300–2000Hz）：风噪被伪 f0 谐波"梳状切除"的效应在这里显现，
    # 宽带数会被巨大的低频风能淹没，看不到"残差变尖锐"的谱形状变化。
    mid = _band_psd_db(e, fs, 300, 2000) - _band_psd_db(d, fs, 300, 2000)
    return {"broadband_db": broadband, "lf_200hz_db": lf, "midband_db": mid}


def lf_energy_ratio(x: np.ndarray, fs: float, cutoff: float = 200.0) -> float:
    """低频能量占比：<cutoff 功率 / 总功率。风噪 >> 打印机噪声 >> 人声/周期噪声。"""
    x = np.asarray(x, dtype=np.float64)
    X = np.abs(np.fft.rfft(x - x.mean())) ** 2
    f = np.fft.rfftfreq(len(x), 1 / fs)
    total = float(np.sum(X))
    if total < 1e-12:
        return 0.0
    return float(np.sum(X[f < cutoff]) / total)


def highpass(x: np.ndarray, fs: float, cutoff: float, order: int = 4) -> np.ndarray:
    sos = signal.butter(order, cutoff, btype="high", fs=fs, output="sos")
    return signal.sosfilt(sos, np.asarray(x, dtype=np.float64))


def run_wind_loop(noise: np.ndarray, fs: int = 48000, block: int = 512,
                  mu: float = 2e-2, n_harm: int = 10, f0: float | None = None,
                  gain: float = 0.5, clip: float = 0.6, ratio: float = 2.5,
                  coupling: float = 0.6, delay_s: float = 0.040,
                  fbc: float = 0.3, hp_cutoff: float | None = None,
                  wind_gate: bool = False, gate_ratio_thresh: float = 0.6,
                  gate_cutoff: float = 100.0, gate_win_s: float = 0.25,
                  label: str = "", baseline_s: float = 1.5) -> dict:
    """返回 (残差, 门控时间占比)。残差 = 误差麦真实信号（噪声 + 扬声器回声）。

    coupling>0：扬声器反相输出经 delay_n 延迟回到误差麦（真实声学环路），
    残差才反映用户实际听到的降噪效果；coupling=0 时反相波不进麦，残差=输入。
    """
    n = len(noise)
    delay_n = int(round(delay_s * fs))
    buf_len = max(8192, int(0.2 * fs), block)
    buf_len = (buf_len // block) * block
    out_hist = np.zeros(buf_len)
    out_pos = 0

    def _est_input() -> np.ndarray:
        seg = np.asarray(noise[:int(baseline_s * fs)], dtype=np.float64)
        if hp_cutoff is not None:
            seg = highpass(seg, fs, hp_cutoff)
        return seg

    c = None
    if f0 is not None:
        c = BlockHarmonicCanceller(fs=fs, f0=float(f0), max_harmonics=n_harm, mu=mu,
                                   block=block, output_gain=gain,
                                   speaker_mic_delay_s=delay_s)
    else:
        # 自参考：用基线尾部估 f0（有风噪时可能骗出伪 f0——这正是要量化的）
        est = estimate_fundamental(_est_input(), fs)
        if est is not None:
            c = BlockHarmonicCanceller(fs=fs, f0=float(est), max_harmonics=n_harm, mu=mu,
                                       block=block, output_gain=gain,
                                       speaker_mic_delay_s=delay_s)
    e_real: list[np.ndarray] = []
    gate_fraction = 0.0
    idx = 0
    n_gate = 0
    n_total = 0
    def _read_echo(delay_n: int) -> np.ndarray:
        """读 delay_n 样本前的输出（写指针环形缓冲，绕回拼接，同 live.py）。"""
        if delay_n <= 0 or delay_n >= buf_len or delay_n + block > buf_len:
            return np.zeros(block)
        start = (out_pos - delay_n) % buf_len
        end = start + block
        if end <= buf_len:
            return out_hist[start:end]
        return np.concatenate([out_hist[start:], out_hist[:end - buf_len]])

    while idx + block <= n:
        blk = np.asarray(noise[idx: idx + block], dtype=np.float64).copy()
        d = blk
        if hp_cutoff is not None:
            d = highpass(d, fs, hp_cutoff)
        if c is None:
            # 无有效 f0（纯风噪时常见）：系统不输出反相波，残差 = 输入
            e_real.append(d.copy())
            idx += block
            continue
        if wind_gate:
            win = np.asarray(noise[max(0, idx - int(gate_win_s * fs) + block): idx + block],
                             dtype=np.float64)
            if len(win) < block:
                win = d
            if lf_energy_ratio(win, fs, cutoff=gate_cutoff) > gate_ratio_thresh:
                n_gate += 1
                # 风噪状态：静音输出（不喷不相关反相波），NLMS 冻结
                c.skip_block()
                e_real.append(blk.copy())  # 误差麦听到的只有原始噪声
                idx += block
                n_total += 1
                continue
        # 声学回声（扬声器输出延迟 delay_n 到达误差麦）
        if delay_n > 0 and coupling > 0:
            src = _read_echo(delay_n)
            d_full = blk + coupling * src
            d_alg = d_full - fbc * src
        else:
            d_full = blk
            d_alg = d
        y, _ = c.process_block(d_alg)
        in_rms = rms(d_full)
        out = compute_safe_output(y, in_rms, gain, clip_level=clip, ratio=ratio)
        out_hist[out_pos:out_pos + block] = out
        out_pos = (out_pos + block) % buf_len
        e_real.append(d_full.copy())
        n_total += 1
        idx += block
    gate_fraction = n_gate / max(n_total, 1)
    resid = np.concatenate(e_real) if e_real else np.zeros(n)
    return {"residual": resid, "gate_fraction": gate_fraction, "f0": getattr(c, "f0", None)}


def report(name: str, before: np.ndarray, after: np.ndarray, fs: int,
           extra: str = "") -> None:
    m = metrics(before, after, fs)
    print(f"{name:44s} 宽带 {m['broadband_db']:+6.2f} dB"
          f"  LF<200Hz {m['lf_200hz_db']:+6.2f} dB"
          f"  中频300-2k {m['midband_db']:+6.2f} dB  {extra}")


def main() -> None:
    fs = 48000
    dur = 8.0

    print("========== 场景 1：纯风噪（无周期噪声） ==========")
    wind = np.asarray(wind_noise(fs=fs, duration=dur, seed=7, strength=0.6,
                                 cutoff_hz=600.0), dtype=np.float64)
    est = estimate_fundamental(wind[:int(1.5 * fs)], fs)
    print(f"风噪上 estimate_fundamental → f0={est}（None=正确拒绝，数值=被卡门涡街骗出伪 f0）")

    r_base = wind.copy()  # ANC off
    report("A. ANC 关闭（基线）", wind, r_base, fs)

    r_anc = run_wind_loop(wind, fs, f0=None)["residual"]
    report("B. 现有谐波 ANC（无处理）", wind, r_anc, fs)

    r_hp = run_wind_loop(wind, fs, f0=None, hp_cutoff=100.0)["residual"]
    report("C. B + 输入高通 100Hz", wind, r_hp, fs)

    g = run_wind_loop(wind, fs, f0=None, wind_gate=True, gate_ratio_thresh=0.8)
    report("D. B + 风噪门控（阈值0.8）", wind, g["residual"], fs,
           f"门控占比 {g['gate_fraction'] * 100:.0f}%")

    g2 = run_wind_loop(wind, fs, f0=None, hp_cutoff=100.0, wind_gate=True,
                       gate_ratio_thresh=0.8)
    report("E. C + D 组合", wind, g2["residual"], fs,
           f"门控占比 {g2['gate_fraction'] * 100:.0f}%")

    print("\n-- 伪 f0 路径：把自动估出的 f0=495 强制作为消除目标 --")
    r_f0 = run_wind_loop(wind, fs, f0=495.0)["residual"]
    report("B' 强制 f0=495（风噪宽带被梳状切除）", wind, r_f0, fs)
    r_f0g = run_wind_loop(wind, fs, f0=495.0, wind_gate=True,
                          gate_ratio_thresh=0.8)
    report("D' 强制 f0=495 + 风噪门控", wind, r_f0g["residual"], fs,
           f"门控占比 {r_f0g['gate_fraction'] * 100:.0f}%")

    print("\n========== 场景 2：打印机噪声(f0≈120) + 风噪混合 ==========")
    printer, _ = printer_noise(fs=fs, duration=dur, seed=42)
    mix = np.asarray(printer, dtype=np.float64) + 0.5 * wind
    mix = mix / np.max(np.abs(mix)) * 0.7
    est2 = estimate_fundamental(mix[:int(1.5 * fs)], fs)
    print(f"混合信号 estimate_fundamental → f0={est2}")

    r_anc2 = run_wind_loop(mix, fs, f0=est2)["residual"]
    report("B. 现有谐波 ANC", mix, r_anc2, fs)

    r_hp2 = run_wind_loop(mix, fs, f0=est2, hp_cutoff=80.0)["residual"]
    report("C. B + 输入高通 80Hz（保留 120Hz 基频）", mix, r_hp2, fs)

    g3 = run_wind_loop(mix, fs, f0=est2, wind_gate=True, gate_ratio_thresh=0.8)
    report("D. B + 风噪门控（阈值0.8）", mix, g3["residual"], fs,
           f"门控占比 {g3['gate_fraction'] * 100:.0f}%")

    g4 = run_wind_loop(mix, fs, f0=est2, hp_cutoff=80.0, wind_gate=True,
                       gate_ratio_thresh=0.8)
    report("E. C + D 组合", mix, g4["residual"], fs,
           f"门控占比 {g4['gate_fraction'] * 100:.0f}%")

    print("\n========== 风噪检测器分离度（低频能量占比，cutoff=100Hz） ==========")
    for cut in (100.0, 200.0):
        wind_ratio = lf_energy_ratio(wind, fs, cutoff=cut)
        printer_ratio = lf_energy_ratio(np.asarray(printer, dtype=np.float64), fs, cutoff=cut)
        mix_ratio = lf_energy_ratio(mix, fs, cutoff=cut)
        from app.synth import speech_like
        voice_ratio = lf_energy_ratio(np.asarray(speech_like(fs=fs, duration=3.0), dtype=np.float64), fs, cutoff=cut)
        print(f"cutoff={cut:6.1f}Hz  纯风噪 {wind_ratio:.2f} | 打印机 {printer_ratio:.2f}"
              f" | 混合 {mix_ratio:.2f} | 人声 {voice_ratio:.2f}")


if __name__ == "__main__":
    main()
