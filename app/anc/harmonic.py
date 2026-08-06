"""周期噪声谐波消除（自适应陷波滤波器组）。

步进电机、风扇叶片频率等噪声近似周期信号：估计基频 f0 后，对每个谐波 k·f0
用正交基（cos/sin）做 LMS 自适应幅度/相位跟踪，叠加反相输出。
对周期噪声比纯宽带 FXLMS 更稳、对延迟更不敏感。
"""
from __future__ import annotations

import numpy as np


def _parabolic_f0(seg: np.ndarray, peak_idx: int, lo: int, fs: float) -> float:
    """抛物线插值精化滞后，得到亚采样精度的基频。peak_idx 为 seg 内索引。"""
    i = peak_idx
    lag = float(i + lo)
    if 1 <= i < len(seg) - 1:
        denom = seg[i - 1] - 2.0 * seg[i] + seg[i + 1]
        if denom != 0.0:
            delta = 0.5 * (seg[i - 1] - seg[i + 1]) / denom
            lag = i + lo + float(np.clip(delta, -0.5, 0.5))
    return fs / lag


def _refine_fundamental(x: np.ndarray, fs: float, f_coarse: float,
                        search: float = 6.0, harmonics: int = 8,
                        step: float = 0.05) -> float:
    """频谱梳状精化：在粗估附近细扫，使各谐波频点能量和最大。

    对基频精度敏感的场景（如谐波消除），比自相关更准。
    """
    n = len(x)
    X = np.fft.rfft(x - x.mean())
    bin_hz = fs / n
    cands = np.arange(max(f_coarse - search, 20.0), f_coarse + search, step)
    best_f, best_score = f_coarse, -1.0
    for f in cands:
        score = 0.0
        for k in range(1, harmonics + 1):
            i = int(round(k * f / bin_hz))
            if 1 <= i < len(X):
                score += abs(X[i]) ** 2
        if score > best_score:
            best_score, best_f = score, f
    return float(best_f)


def estimate_fundamental(x: np.ndarray, fs: float, low: float = 40.0,
                         high: float = 500.0) -> float | None:
    """自相关法估计基频，返回 Hz；含倍频校正与抛物线精化。无显著周期返回 None。"""
    from scipy.signal import find_peaks

    x = np.asarray(x, dtype=np.float64)
    if len(x) < fs // 8:
        return None
    x = x - x.mean()
    r = np.correlate(x, x, mode="full")[len(x) - 1:]
    r = r / (r[0] + 1e-12)

    lo = max(int(fs / high), 2)
    hi = min(int(fs / low), len(r) - 1)
    if hi <= lo:
        return None
    seg = r[lo:hi]
    idx, _ = find_peaks(seg, prominence=0.05)
    if len(idx) == 0:
        idx = np.array([int(np.argmax(seg))])
    candidates = [int(i) for i in idx]

    best_idx = candidates[int(np.argmax(seg[candidates]))]
    best_lag = best_idx + lo  # 实际滞后
    best_val = float(seg[best_idx])

    # 倍频校正：若半周期（实际滞后域）附近有强峰值，取其中相关值最高者
    half_lag = best_lag // 2
    near_half = [idx + lo for idx in candidates if abs(idx + lo - half_lag) <= 3]
    if near_half:
        chosen_actual = max(near_half, key=lambda lag: seg[lag - lo])
        if seg[chosen_actual - lo] >= 0.8 * best_val:
            coarse = _parabolic_f0(seg, chosen_actual - lo, lo, fs)
        else:
            coarse = _parabolic_f0(seg, best_idx, lo, fs)
    else:
        coarse = _parabolic_f0(seg, best_idx, lo, fs)
    return _refine_fundamental(x, fs, coarse)


class HarmonicCanceller:
    """自适应谐波消除器。每个谐波一组 cos/sin 权重，用 LMS 跟踪。"""

    def __init__(self, fs: float, max_harmonics: int = 10, mu: float = 1e-3,
                 f0: float | None = None):
        self.fs = float(fs)
        self.max_harmonics = max_harmonics
        self.mu = mu
        self.f0 = f0
        self.reset()

    def reset(self) -> None:
        self.a = np.zeros(self.max_harmonics)  # cos 权重
        self.b = np.zeros(self.max_harmonics)  # sin 权重
        self.n = 0

    def step(self, desired: float) -> tuple[float, float]:
        """desired: 误差处信号。返回 (y, e)。基频需先 estimate/设置。"""
        if self.f0 is None:
            return 0.0, desired
        t = self.n / self.fs
        y = 0.0
        for k in range(1, self.max_harmonics + 1):
            w = 2.0 * np.pi * self.f0 * k * t
            c = np.cos(w)
            s = np.sin(w)
            y += self.a[k - 1] * c + self.b[k - 1] * s
        e = desired - y
        for k in range(1, self.max_harmonics + 1):
            w = 2.0 * np.pi * self.f0 * k * t
            c = np.cos(w)
            s = np.sin(w)
            self.a[k - 1] += self.mu * e * c
            self.b[k - 1] += self.mu * e * s
        self.n += 1
        return y, e


def simulate(ns: np.ndarray, fs: float, f0: float | None = None,
             max_harmonics: int = 10, mu: float = 1e-3,
             skip_head: int | None = None) -> dict:
    """离线模拟谐波消除。ns 即误差处信号（假设参考与误差同源）。"""
    if f0 is None:
        f0 = estimate_fundamental(ns, fs)
    if f0 is None:
        return {"d": ns, "e": ns, "y": np.zeros_like(ns), "reduction_db": 0.0,
                "rms_d": 0.0, "rms_e": 0.0, "f0": None}
    canceller = HarmonicCanceller(fs=fs, max_harmonics=max_harmonics, mu=mu, f0=f0)
    n = len(ns)
    e = np.zeros(n)
    y = np.zeros(n)
    for i in range(n):
        y[i], e[i] = canceller.step(float(ns[i]))

    skip = skip_head if skip_head is not None else int(0.3 * fs)
    rms_d = np.sqrt(np.mean(ns[skip:] ** 2))
    rms_e = np.sqrt(np.mean(e[skip:] ** 2))
    reduction_db = 20.0 * np.log10(max(rms_e, 1e-12) / max(rms_d, 1e-12)) if rms_d > 0 else 0.0
    return {
        "d": np.asarray(ns, dtype=np.float64),
        "e": e,
        "y": y,
        "reduction_db": float(reduction_db),
        "rms_d": float(rms_d),
        "rms_e": float(rms_e),
        "f0": float(f0),
    }
