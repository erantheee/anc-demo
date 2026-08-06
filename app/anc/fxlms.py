"""Filtered-x LMS 前馈主动降噪。

参考麦克风 x → 自适应滤波器 W → 扬声器 y → 次级路径 S → 误差麦克风 e。
权重更新使用滤波参考 x' = S * x（必须经次级路径估计，否则环路不稳定）。
"""
from __future__ import annotations

import numpy as np


class FXLMS:
    def __init__(self, num_taps: int = 256, mu: float = 1e-4, leak: float = 0.0,
                 secondary_path: np.ndarray | None = None):
        self.num_taps = num_taps
        self.mu = mu
        self.leak = leak
        self.w = np.zeros(num_taps)
        if secondary_path is None:
            secondary_path = np.array([1.0])
        self.s = np.asarray(secondary_path, dtype=np.float64)
        self.x_buf = np.zeros(num_taps)
        self.xf_buf = np.zeros(num_taps)
        self.y_hist = np.zeros(len(self.s))

    def step(self, reference: float, desired: float) -> tuple[float, float]:
        """处理一个采样点。

        reference: 参考麦克风当前样本（噪声源）。
        desired: 误差麦克风处当前样本（含未抵消噪声）。
        返回 (y, e)：扬声器输出与误差。
        """
        self.x_buf = np.roll(self.x_buf, 1)
        self.x_buf[0] = reference
        self.xf_buf = np.roll(self.xf_buf, 1)
        self.xf_buf[0] = float(np.dot(self.s, self.x_buf[: len(self.s)]))

        y = float(self.w @ self.x_buf)
        self.y_hist = np.roll(self.y_hist, 1)
        self.y_hist[0] = y
        e = desired - float(self.s @ self.y_hist)

        self.w = self.w + self.mu * e * self.xf_buf - self.leak * self.w
        return y, e


def simulate(ns: np.ndarray, fs: float, secondary_path: np.ndarray,
             plant_delay_samples: int = 16, num_taps: int = 256, mu: float = 1e-4,
             skip_head: int | None = None) -> dict:
    """离线模拟前馈 ANC。

    ns: 参考信号（噪声源）。
    plant: 噪声源 → 误差麦克风的声学路径，用纯延迟近似。
    secondary_path: 扬声器 → 误差麦克风路径（FIR）。
    返回 d/e（降噪前/后误差处信号）与统计。
    """
    h_plant = np.zeros(plant_delay_samples + 1)
    h_plant[plant_delay_samples] = 1.0
    d = np.convolve(ns, h_plant)[: len(ns)]

    anc = FXLMS(num_taps=num_taps, mu=mu, secondary_path=secondary_path)
    n = len(ns)
    e = np.zeros(n)
    y = np.zeros(n)
    for i in range(n):
        y[i], e[i] = anc.step(float(ns[i]), float(d[i]))

    skip = skip_head if skip_head is not None else int(0.3 * fs)
    rms_d = np.sqrt(np.mean(d[skip:] ** 2))
    rms_e = np.sqrt(np.mean(e[skip:] ** 2))
    reduction_db = 20.0 * np.log10(max(rms_e, 1e-12) / max(rms_d, 1e-12)) if rms_d > 0 else 0.0
    return {
        "d": d,
        "e": e,
        "y": y,
        "reduction_db": float(reduction_db),
        "rms_d": float(rms_d),
        "rms_e": float(rms_e),
    }


def recommend_mu(secondary_path: np.ndarray, num_taps: int, reference_power: float) -> float:
    """根据参考功率给一个稳定的 μ 初值（经验式）。"""
    return min(1e-3, 0.1 / (num_taps * max(reference_power, 1e-8)))
