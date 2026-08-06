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
