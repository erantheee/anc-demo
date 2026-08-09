"""离线声学环路仿真：验证「降噪没有效果」的根因并量化修复效果。

镜像 live.py _run_audio 的真实数据通路：
  d(误差麦) = 噪声 + coupling × 输出(延迟 τ)
  反馈中和 → NLMS 谐波消除 → compute_safe_output → 扬声器 → 回声回到误差麦
残差取误差麦真实信号 d（含扬声器反相波回采），而不是数字域理想误差 e。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.anc.live import BlockHarmonicCanceller, compute_safe_output  # noqa: E402
from app.anc.harmonic import estimate_fundamental  # noqa: E402
from app.synth import printer_noise  # noqa: E402


def rms_db(x: np.ndarray) -> float:
    return 20.0 * np.log10(max(float(np.sqrt(np.mean(np.asarray(x) ** 2))), 1e-12))


def read_echo_modulo(out_hist: np.ndarray, out_pos: int, delay_n: int,
                     block: int) -> np.ndarray:
    """与 live.py `_read_echo` 等价：写指针环形缓冲，读 delay_n 样本前的输出。"""
    n = len(out_hist)
    if delay_n <= 0 or delay_n >= n or delay_n + block > n:
        return np.zeros(block)
    start = (out_pos - delay_n) % n
    end = start + block
    if end <= n:
        return out_hist[start:end]
    return np.concatenate([out_hist[start:], out_hist[:end - n]])


def run_loop(fs: int = 48000, f0: float | None = None, max_harmonics: int = 10,
             mu: float = 2e-2, block: int = 512, output_gain: float = 0.5,
             clip_level: float = 0.6, ratio: float = 2.5,
             coupling: float = 0.6, delay_s: float = 0.040,
             feedback_cancel_gain: float = 0.3,
             baseline_s: float = 3.0, total_s: float = 12.0,
             seed: int = 42, label: str = "", quiet_rms: float = 1e-3) -> dict:
    n_total = int(total_s * fs)
    n_baseline = int(baseline_s * fs)
    noise, _ = printer_noise(fs=fs, duration=total_s, seed=seed)
    noise = noise.astype(np.float64)
    delay_n = int(round(delay_s * fs))

    d_buf: list[np.ndarray] = []
    e_real: list[np.ndarray] = []   # 误差麦真实信号（含回声）
    buf_len = max(8192, int(0.2 * fs), block)
    buf_len = (buf_len // block) * block  # 与 live.py 一致：block 对齐，避免写回绕截断
    out_hist = np.zeros(buf_len, dtype=np.float64)
    out_pos = 0  # 写指针

    canceller: BlockHarmonicCanceller | None = None
    f0_used: float | None = None
    idx = 0
    while idx < n_total:
        blk = noise[idx: idx + block]
        if len(blk) < block:
            break
        if canceller is None:
            d_buf.append(blk.copy())
            if idx + block >= n_baseline:
                d_all = np.concatenate(d_buf)
                tail = d_all[-fs:]
                f0_used = f0 or estimate_fundamental(tail, fs)
                if f0_used is None:
                    break
                canceller = BlockHarmonicCanceller(
                    fs=fs, f0=float(f0_used), max_harmonics=max_harmonics,
                    mu=mu, block=block, output_gain=output_gain,
                    speaker_mic_delay_s=delay_s)
        else:
            # 扬声器回声：此刻输出会延迟 delay_n 后到达误差麦（写指针读 delay 前）
            src = read_echo_modulo(out_hist, out_pos, delay_n, block)
            echo = coupling * src
            d = blk + echo
            # 反馈中和：从麦克风信号中减去"扬声器回声估计"
            d_alg = d - feedback_cancel_gain * src
            y, _ = canceller.process_block(d_alg)
            in_rms = float(np.sqrt(np.mean(d ** 2)))
            out = compute_safe_output(y, in_rms, canceller.output_gain,
                                      clip_level=clip_level, ratio=ratio,
                                      quiet_rms=quiet_rms)
            out_hist[out_pos:out_pos + block] = out
            out_pos = (out_pos + block) % buf_len
            e_real.append(d.copy())
        idx += block

    d_all = np.concatenate(d_buf)
    e_all = np.concatenate(e_real) if e_real else np.zeros(1)
    skip = min(int(0.3 * len(e_all)), fs)
    rms_d = np.sqrt(np.mean(d_all ** 2))
    rms_e = np.sqrt(np.mean(e_all[skip:] ** 2)) if len(e_all) > skip else np.sqrt(np.mean(e_all ** 2))
    reduction_db = 20.0 * np.log10(max(rms_e, 1e-12) / max(rms_d, 1e-12))
    print(f"{label:52s} f0={f0_used:8.1f}Hz gain={output_gain:5.2f} "
          f"clip={clip_level:5.2f} ratio={ratio:4.1f} coupling={coupling:4.2f} "
          f"fbc={feedback_cancel_gain:4.2f} τ={delay_s*1000:5.1f}ms → 降噪 {reduction_db:+6.2f} dB")
    return {"reduction_db": reduction_db, "f0": f0_used}


if __name__ == "__main__":
    print("=== 1. 数字域（coupling=0，无回声）：算法本身能达到多少 ===")
    run_loop(coupling=0.0, output_gain=1.0, clip_level=0.5, ratio=2.0,
             label="纯数字：coupling=0")
    run_loop(coupling=0.0, output_gain=1.0, clip_level=0.5, ratio=2.0,
             f0=120.0, label="纯数字：coupling=0 + 手动f0=120")

    print("\n=== 2. 回声耦合：反馈中和匹配 vs 不匹配 ===")
    run_loop(coupling=0.6, output_gain=1.0, clip_level=0.5, ratio=2.0,
             feedback_cancel_gain=0.6, label="coupling=0.6, fbc=0.6(完美匹配)")
    run_loop(coupling=0.6, output_gain=1.0, clip_level=0.5, ratio=2.0,
             feedback_cancel_gain=0.3, label="coupling=0.6, fbc=0.3(低估,现状)")
    run_loop(coupling=0.6, output_gain=1.0, clip_level=0.5, ratio=2.0,
             feedback_cancel_gain=0.0, label="coupling=0.6, fbc=0(无中和)")
    run_loop(coupling=0.3, output_gain=1.0, clip_level=0.5, ratio=2.0,
             feedback_cancel_gain=0.3, label="coupling=0.3, fbc=0.3(完美匹配)")

    print("\n=== 3. 输出限幅（clip/ratio）对降噪的影响（coupling=0.6, fbc=0.3 现状）===")
    run_loop(output_gain=0.5, clip_level=0.6, ratio=2.5,
             label="新默认：gain=0.5, clip=0.6, ratio=2.5")
    run_loop(output_gain=0.12, clip_level=0.12, ratio=1.0,
             label="旧默认：gain=0.12, clip=0.12, ratio=1")
    run_loop(output_gain=1.0, clip_level=0.6, ratio=2.5,
             label="更高增益：gain=1.0, clip=0.6, ratio=2.5")

    print("\n=== 4. 真机复现：f0=960 手动错误基频 ===")
    run_loop(f0=960.0, output_gain=1.0, clip_level=0.5, ratio=2.0,
             label="f0=960(错误) + 宽松限幅")
    run_loop(f0=960.0, output_gain=0.42, clip_level=0.12, ratio=1.0,
             label="f0=960(错误) + gain=0.42(真机当时)")
