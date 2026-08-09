"""啸叫（声反馈）检测。

声反馈 / 啸叫（howling / squealing）是"麦克风 → 功放 → 扬声器 → 麦克风"闭环
环路增益 ≥ 1 时产生的自激振荡。特征：
- 频谱：一个频率相对稳定、电平**随时间增长**的窄带强音调；
- 增长到饱和后，振幅在高低之间摆动（限幅/非线性）；
- 音调占比（tonality）极高。

与稳态环境噪声（风机叶片音、步进电机音、压缩机音）的关键区别在**时域趋势**：
- 啸叫：主峰电平随帧推进持续上升（正斜率、拟合优度高）；
- 稳态噪声：主峰电平基本恒定（斜率 ≈ 0）。

因此本检测以"跟踪主峰电平的时域增长"为核心判据，音调占比与 ANC 引擎状态
（由调用方在工具层补充）作辅助。
"""
from __future__ import annotations

import numpy as np
from scipy import stats

from app.analyze import find_peaks, spectrum_db, tonality_ratio

FEEDBACK_BAND_HZ = (100.0, 6000.0)
DEFAULT_FRAME_S = 0.1
DEFAULT_MIN_FRAMES = 5
DEFAULT_GROWTH_THRESH_DB_S = 4.0  # 主峰电平每秒增长超过此值判为"增长"
DEFAULT_FREQ_TOL = 0.06  # 主峰频率相对中值漂移容忍度（6%）


def _empty(insufficient: bool = True) -> dict:
    return {
        "is_howling": False,
        "howling_score": 0.0,
        "signal_class": "uncertain",
        "candidate_freq_hz": None,
        "growth_db_per_s": None,
        "growth_fit_r2": None,
        "tonality_ratio": None,
        "frame_freqs": [],
        "frame_levels": [],
        "insufficient_data": insufficient,
    }


def _track_dominant_peak(x: np.ndarray, fs: float, frame_s: float) -> tuple[list, list, int]:
    frame_len = max(int(fs * frame_s), 256)
    n = len(x) // frame_len
    freqs: list[float | None] = []
    levels: list[float | None] = []
    for i in range(n):
        seg = x[i * frame_len:(i + 1) * frame_len]
        f, psd = spectrum_db(seg, fs)
        peaks = find_peaks(f, psd, min_freq=FEEDBACK_BAND_HZ[0], max_freq=FEEDBACK_BAND_HZ[1],
                           min_prominence=1.5, min_level=-90.0)
        if peaks:
            top = max(peaks, key=lambda p: p.level_db)
            freqs.append(float(top.freq))
            levels.append(float(top.level_db))
        else:
            freqs.append(None)
            levels.append(None)
    return freqs, levels, frame_len


def detect_feedback(samples: np.ndarray, fs: float, frame_s: float = DEFAULT_FRAME_S,
                    min_frames: int = DEFAULT_MIN_FRAMES,
                    growth_thresh_db_s: float = DEFAULT_GROWTH_THRESH_DB_S,
                    freq_tol: float = DEFAULT_FREQ_TOL) -> dict:
    """检测样本中是否存在声反馈（啸叫）。

    # 判定决策树
    #
    #   样本长度 < 0.5s 或帧数不足 ──────────────► insufficient_data=True
    #   主峰频率漂移 > freq_tol (6%) ───────────► environment_noise（宽带/时变）
    #   稳定频率段做 linregress（主峰电平随时间）：
    #     增长斜率 > 4 dB/s 且 r² > 0.5 且音调占比 > 0.4 ─► acoustic_feedback（啸叫）
    #     否则 ──────────────────────────────────────────► environment_noise
    #   （suspected_feedback 由工具层叠加「ANC 运行 + 高音调稳态」补充）
    #
    # 与稳态环境噪声的区别在时域趋势：啸叫主峰电平持续上升，稳态噪声斜率≈0。

    返回 dict 关键字段：
    - is_howling: bool，是否判定为啸叫
    - howling_score: 0..1
    - signal_class: environment_noise | acoustic_feedback | suspected_feedback | uncertain
      （suspected_feedback 由工具层根据 ANC 状态补充；本函数只产出前三种）
    - candidate_freq_hz / growth_db_per_s / growth_fit_r2 / tonality_ratio
    - frame_freqs / frame_levels: 每帧主峰轨迹（供解释与调试）
    - insufficient_data: bool
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if len(x) < max(int(fs * 0.5), 512):
        return _empty(insufficient=True)
    frame_len = max(int(fs * frame_s), 256)
    n = len(x) // frame_len
    if n < min_frames:
        return _empty(insufficient=True)

    freqs, levels, _ = _track_dominant_peak(x, fs, frame_s)
    valid = [i for i in range(n) if freqs[i] is not None]
    if len(valid) < min_frames:
        return _empty(insufficient=False)

    med = float(np.median([freqs[i] for i in valid]))
    if med <= 0:
        return _empty(insufficient=False)
    stable = [i for i in valid if abs(freqs[i] - med) / med < freq_tol]
    if len(stable) < min_frames:
        # 主峰频率漂移大：更像宽带/时变噪声，而非啸叫
        out = _empty(insufficient=False)
        out["candidate_freq_hz"] = round(med, 1)
        out["frame_freqs"] = [round(f, 1) if f is not None else None for f in freqs]
        out["frame_levels"] = [round(v, 1) if v is not None else None for v in levels]
        return out

    t = np.arange(len(stable), dtype=float)
    lv = np.array([levels[i] for i in stable], dtype=float)
    slope, _intercept, r, _p, _se = stats.linregress(t, lv)
    growth_db_per_s = float(slope) * (fs / frame_len)
    r2 = float(r * r)

    f_all, psd_all = spectrum_db(x, fs)
    peaks_all = find_peaks(f_all, psd_all, min_freq=FEEDBACK_BAND_HZ[0], max_freq=FEEDBACK_BAND_HZ[1])
    tr = float(tonality_ratio(f_all, psd_all, peaks_all))

    is_howling = growth_db_per_s > growth_thresh_db_s and r2 > 0.5 and tr > 0.4
    growth_score = float(np.clip((growth_db_per_s - growth_thresh_db_s) / 12.0, 0.0, 1.0))
    score = float(np.clip(0.45 * growth_score + 0.3 * r2 + 0.25 * tr, 0.0, 1.0)) if is_howling else 0.0

    return {
        "is_howling": is_howling,
        "howling_score": round(score, 2),
        "signal_class": "acoustic_feedback" if is_howling else "environment_noise",
        "candidate_freq_hz": round(med, 1),
        "growth_db_per_s": round(growth_db_per_s, 1),
        "growth_fit_r2": round(r2, 3),
        "tonality_ratio": round(tr, 3),
        "frame_freqs": [round(f, 1) if f is not None else None for f in freqs],
        "frame_levels": [round(v, 1) if v is not None else None for v in levels],
        "insufficient_data": False,
    }
