"""合成噪声源，用于无硬件时验证分析管线（M1）与 ANC 算法（M2）。"""
from __future__ import annotations

import numpy as np
from scipy import signal


def tone(samples: np.ndarray, fs: float, freq: float, amplitude: float = 1.0,
         phase: float = 0.0) -> np.ndarray:
    t = np.arange(len(samples)) / fs
    return amplitude * np.sin(2 * np.pi * freq * t + phase)


def harmonics(samples: np.ndarray, fs: float, fundamental: float,
              amplitudes: list[float], phase: float = 0.0) -> np.ndarray:
    """以基频 fundamental 生成 amplitudes 对应幅度的谐波叠加。"""
    t = np.arange(len(samples)) / fs
    out = np.zeros(len(samples), dtype=float)
    for k, a in enumerate(amplitudes, start=1):
        out += a * np.sin(2 * np.pi * fundamental * k * t + phase * k)
    return out


def filtered_noise(samples: np.ndarray, fs: float, low: float, high: float,
                   rng: np.random.Generator, order: int = 6) -> np.ndarray:
    noise = rng.standard_normal(len(samples))
    low = max(float(low), 10.0)
    high = min(float(high), 0.45 * fs)  # 必须 < fs/2
    if high <= low:
        return np.zeros(len(samples), dtype=float)
    sos = signal.butter(order, [low, high], btype="band", fs=fs, output="sos")
    return signal.sosfilt(sos, noise)


def printer_noise(fs: float = 16000.0, duration: float = 10.0, seed: int = 0,
                  stepper_fundamental: float = 120.0, blade_freq: float = 2500.0) -> tuple[np.ndarray, dict]:
    """合成 3D 打印机噪声。

    成分：步进电机谐波音调、冷却风扇宽带 + 叶片频率音调、结构共振窄带、房间底噪。
    返回 (samples, sources)，samples 已归一化到 [-0.5, 0.5]。
    """
    rng = np.random.default_rng(seed)
    n = int(fs * duration)
    x = np.zeros(n)
    sources: dict[str, dict] = {}

    stepper_amp = harmonics(x, fs, stepper_fundamental,
                            amplitudes=[1.0, 0.6, 0.35, 0.15, 0.08])
    sources["stepper"] = {"fundamental": stepper_fundamental, "n_harmonics": 5}
    x += 0.7 * stepper_amp

    fan_broad = filtered_noise(x, fs, 800.0, 4000.0, rng)
    blade = tone(x, fs, blade_freq, 0.25) + 0.08 * tone(x, fs, 2 * blade_freq, 0.10)
    sources["fan_broadband"] = {"low": 800.0, "high": 4000.0}
    sources["fan_blade"] = {"freq": blade_freq, "harmonic": 2 * blade_freq}
    x += 0.5 * fan_broad + blade

    x += 0.4 * tone(x, fs, 180.0, 0.4)  # 结构共振窄带
    sources["resonance"] = {"freq": 180.0}

    x += 0.05 * filtered_noise(x, fs, 50.0, 8000.0, rng)  # 房间底噪

    peak = np.max(np.abs(x))
    if peak > 0.5:
        x = x / peak * 0.5
    return x.astype(np.float32), sources


def wind_noise(fs: float = 16000.0, duration: float = 10.0, seed: int = 3,
               strength: float = 0.6, cutoff_hz: float = 200.0,
               gust_hz: float = 1.2, aeolian_hz: float | None = 45.0) -> np.ndarray:
    """合成风噪：湍流低频宽带 + 阵风包络 + 可选卡门涡街准周期音。

    风噪物理特征（与打印机/风扇谐波噪声的关键区别）：
    - 湍流压力脉动集中在 <200–500 Hz，且随频率快速下降（≈1/f，低频堆积）；
    - 非平稳：阵风包络在 ~0.5–2 Hz 尺度起伏（一阵一阵），NLMS 无法跟踪；
    - 卡门涡街（风哨/涡脱落）产生 ~20–60 Hz 缓慢漂移的准周期音，
      可能把基频估计骗到一个"伪 f0"上——这是风噪让谐波 ANC 出怪音的机理。

    返回归一化到 ±strength 的 float32 信号。
    """
    rng = np.random.default_rng(seed)
    n = int(fs * duration)
    t = np.arange(n) / fs

    # 1) 湍流：白噪声低通 → 低频堆积；再积分塑形（额外 -6dB/oct 低频强调）
    sos_lp = signal.butter(6, min(cutoff_hz, 0.45 * fs), btype="low",
                           fs=fs, output="sos")
    turb = signal.sosfilt(sos_lp, rng.standard_normal(n))
    turb = np.cumsum(turb - np.mean(turb)) / fs
    # 去积分的极低频漂移（保留 1/f 特征，但不产生不可控 DC 游走）
    sos_dc = signal.butter(2, 0.3, btype="high", fs=fs, output="sos")
    turb = signal.sosfilt(sos_dc, turb)
    turb -= np.mean(turb)

    # 2) 阵风包络：0.3–2·gust_hz 慢变带通噪声整流，归一化均值≈1
    gust_sos = signal.butter(2, [max(0.2, 0.3), 2.0 * gust_hz], btype="band",
                             fs=fs, output="sos")
    env = signal.sosfilt(gust_sos, rng.standard_normal(n))
    env = np.abs(env)
    env /= (np.mean(env) + 1e-12)
    x = turb * env

    # 3) 可选卡门涡街准周期音：瞬时相位慢漂移 → 基频会"游走"，测 f0 估计鲁棒性
    if aeolian_hz is not None:
        drift = 2.0 * np.cumsum(rng.standard_normal(n)) / fs
        x += 0.35 * strength * np.sin(2.0 * np.pi * aeolian_hz * t + drift)

    peak = np.max(np.abs(x))
    if peak > 1e-9:
        x = x / peak * strength
    return x.astype(np.float32)


def two_sources(fs: float = 16000.0, duration: float = 10.0, seed: int = 0,
                f1: float = 120.0, f2: float = 350.0) -> tuple[np.ndarray, dict]:
    """两个音调源 + 宽带，用于测试谐波家族区分与来源归属。"""
    rng = np.random.default_rng(seed)
    n = int(fs * duration)
    x = np.zeros(n)
    x += 0.6 * harmonics(x, fs, f1, [1.0, 0.5, 0.25, 0.12])
    x += 0.4 * harmonics(x, fs, f2, [1.0, 0.4, 0.2])
    x += 0.3 * filtered_noise(x, fs, 600.0, 5000.0, rng)
    peak = np.max(np.abs(x))
    if peak > 0.5:
        x = x / peak * 0.5
    return x.astype(np.float32), {"f1": f1, "f2": f2}


def speech_like(fs: float = 16000.0, duration: float = 5.0, seed: int = 7) -> np.ndarray:
    """粗仿人声：变调浊音段 + 音节间停顿 + 轻微清音噪声。

    用于人声检测测试与演示：浊音段基频 90–160Hz 随语调起伏、音节间有静默
    间隙、含无周期帧，与稳态机械噪声（恒定基频 + 平稳能量）形成对照。
    """
    rng = np.random.default_rng(seed)
    n = int(fs * duration)
    x = np.zeros(n)
    syllables = [
        (110, 0.30), (135, 0.22), (95, 0.28), (150, 0.20),
        (120, 0.26), (160, 0.18), (100, 0.28), (140, 0.22),
        (130, 0.26), (90, 0.20), (125, 0.28), (155, 0.18),
    ]
    i = 0
    for k, (f0, dur) in enumerate(syllables):
        m = int(dur * fs)
        if i + m > n:
            break
        t = np.arange(m) / fs
        env = np.hanning(m)  # 音节内起音/收尾包络
        x[i:i + m] = env * (0.45 * np.sin(2 * np.pi * f0 * t)
                            + 0.20 * np.sin(2 * np.pi * 2 * f0 * t)
                            + 0.10 * np.sin(2 * np.pi * 3 * f0 * t))
        x[i:i + m] += 0.02 * rng.standard_normal(m)
        i += m
        gap = int((0.10 + 0.15 * rng.random()) * fs)  # 音节间停顿
        if k % 3 == 2:
            gap += int(0.25 * fs)  # 每三个音节一个稍长停顿
        if i + gap > n:
            break
        i += gap
    return x
