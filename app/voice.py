"""人声/语音检测：区分「稳态周期机械噪声」（ANC 的消除目标）与人声。

谐波消除（BlockHarmonicCanceller）只对稳态周期噪声有效。人声虽然浊音段
有周期性基频，但基频随语调起伏、音节/静默之间能量强调制、且夹杂大量无
基频的清音/气声帧——与风扇/电机/打印机的「恒定基频 + 平稳能量」明显不同。
若把说话声当成周期源启动 ANC，反相波会去"消除"说话声，听感上就像把人的
声音也吃掉了。

本模块在 ANC 决策点（基线 f0 估计、ANC 可行性建议）加一道闸：信号更像
人声时，不进入降噪。判据基于逐帧（默认 150ms）分析：
- 基频稳定度：机械噪声帧间基频几乎不变（CV 低）；人声基频随语调抖动（CV 高）。
- 能量调制：机械噪声能量平稳；人声在音节/静默间强调制（帧 RMS CV 高）。
- 浊音占比：机械噪声几乎每帧都有稳定基频；人声含大量清音/无声帧。

信号几乎没有稳定周期成分（宽带噪声、静音）不算人声——这类信号本就走
「无法估计基频 → ANC 不可用」的分支，不需要语音闸门。

基频估计用 FFT 加速的自相关（Wiener–Khinchin），比时域自相关 O(n²) 快得多，
可在实时监控线程里逐采样周期调用。
"""
from __future__ import annotations

import numpy as np

SPEECH_F0_HZ = (80.0, 400.0)  # 人声基频粗范围（浊音）
DEFAULT_FRAME_S = 0.15
MIN_FRAMES = 6            # 帧数不足无法判定
MIN_VOICED_FRAMES = 3     # 至少这么多帧有稳定周期才进入人声判定
MIN_CORR = 0.25           # 帧内归一化自相关峰值低于此值视为无周期

# 判据阈值与权重（保守：宁可漏检语音，不要误杀真正的周期噪声）
F0_CV_THRESH = 0.10
F0_CV_WEIGHT = 0.45
RMS_CV_THRESH = 0.30
RMS_CV_WEIGHT = 0.35
VOICED_RATIO_MAX = 0.80
VOICED_RATIO_WEIGHT = 0.20
VOICE_SCORE_MIN = 0.45    # 配合 len(reasons) >= 2 使用


def _frame_features(x: np.ndarray, fs: float, frame: int) -> tuple[np.ndarray, np.ndarray]:
    """逐帧 RMS 与基频估计。返回 (rms, f0)，f0 在无周期处为 NaN。"""
    n = len(x) // frame
    rms = np.empty(n, dtype=np.float64)
    f0 = np.full(n, np.nan, dtype=np.float64)
    lo = max(int(fs / SPEECH_F0_HZ[1]), 2)
    hi = min(int(fs / SPEECH_F0_HZ[0]), frame - 1)
    for i in range(n):
        seg = x[i * frame:(i + 1) * frame].astype(np.float64)
        rms[i] = float(np.sqrt(np.mean(seg ** 2)))
        if hi <= lo:
            continue
        seg = seg - seg.mean()
        X = np.fft.rfft(seg)
        acf = np.fft.irfft(np.abs(X) ** 2, len(seg))
        acf = acf / (acf[0] + 1e-12)
        window = acf[lo:hi]
        if window.size == 0:
            continue
        peak = float(np.max(window))
        if peak < MIN_CORR:
            continue
        lag = lo + int(np.argmax(window))
        f0[i] = fs / lag
    return rms, f0


def detect_voice(samples: np.ndarray, fs: float,
                 frame_s: float = DEFAULT_FRAME_S) -> dict:
    """判断信号是否更像人声/语音而非稳态周期机械噪声。

    返回 {"is_voice": bool, "score": 0..1, "reasons": [命中的判据]}。
    无稳定周期成分或样本过短时 is_voice=False（此类信号走「无基频 → ANC
    不可用」的分支，不需要语音闸门）。
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    frame = max(int(fs * frame_s), 512)
    n = len(x) // frame
    if n < MIN_FRAMES:
        return {"is_voice": False, "score": 0.0,
                "reasons": [f"样本过短（{n} 帧 < {MIN_FRAMES}），无法判定"]}

    rms, f0 = _frame_features(x, fs, frame)
    voiced = ~np.isnan(f0)
    n_voiced = int(np.sum(voiced))
    if n_voiced < MIN_VOICED_FRAMES:
        return {"is_voice": False, "score": 0.0,
                "reasons": ["无稳定周期成分（宽带噪声或静音，非人声特征）"]}

    med_rms = float(np.median(rms))
    if med_rms <= 0:
        return {"is_voice": False, "score": 0.0, "reasons": ["无有效信号"]}

    f0_v = f0[voiced]
    f0_cv = float(np.std(f0_v) / (np.mean(f0_v) + 1e-9))
    rms_cv = float(np.std(rms) / (np.mean(rms) + 1e-9))
    voiced_ratio = n_voiced / n

    reasons: list[str] = []
    score = 0.0
    if f0_cv >= F0_CV_THRESH:
        reasons.append(f"基频不稳定（帧间 CV={f0_cv:.2f}，语音语调起伏）")
        score += F0_CV_WEIGHT
    if rms_cv >= RMS_CV_THRESH:
        reasons.append(f"帧能量强调制（CV={rms_cv:.2f}，音节/静默交替）")
        score += RMS_CV_WEIGHT
    if voiced_ratio <= VOICED_RATIO_MAX:
        reasons.append(f"浊音占比偏低（{voiced_ratio:.0%}，含清音/无声帧）")
        score += VOICED_RATIO_WEIGHT

    # 保守：至少命中两项判据才判为人声，避免误杀真周期噪声
    is_voice = len(reasons) >= 2 and score >= VOICE_SCORE_MIN
    return {
        "is_voice": is_voice,
        "score": round(min(1.0, score), 2),
        "reasons": reasons if reasons else ["未检出明显人声特征"],
    }
