"""信号分析：SPL、频谱、音调峰值、谐波家族、A 加权、音调占比。

无标定时输出 dBFS（相对值），标定后输出 dB SPL（绝对值）。
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import signal
from scipy.io import wavfile

P_REF = 20e-6  # Pa，人耳听阈参考声压
LOW_BAND = (40.0, 500.0)
MID_BAND = (500.0, 2000.0)
HIGH_BAND = (2000.0, 8000.0)


def a_weight_db(freqs: np.ndarray) -> np.ndarray:
    """IEC 61672 A 加权曲线，输入 Hz，输出 dB 修正量。"""
    f = np.asarray(freqs, dtype=float)
    ra = (12194.0 ** 2 * f ** 4) / (
        (f ** 2 + 20.6 ** 2)
        * np.sqrt((f ** 2 + 107.7 ** 2) * (f ** 2 + 737.9 ** 2))
        * (f ** 2 + 12194.0 ** 2)
    )
    with np.errstate(divide="ignore"):
        a = 20.0 * np.log10(np.maximum(ra, 1e-30)) + 2.0
    return a


def rms_db(samples: np.ndarray) -> float:
    x = np.asarray(samples, dtype=np.float64)
    if x.size == 0:
        return -120.0
    rms = np.sqrt(np.mean(x ** 2))
    if rms <= 0:
        return -120.0
    return float(20.0 * np.log10(rms))


@dataclass
class TonalPeak:
    freq: float
    level_db: float
    prominence_db: float
    harmonic_order: int = 1


@dataclass
class AnalysisReport:
    fs: float
    duration_s: float
    rms_db: float
    spl_db: float | None
    spl_db_a: float | None
    peaks: list[TonalPeak]
    tonality_ratio: float
    dominant_freq: float | None
    harmonic_family: list[float]
    band_spl_db: dict[str, float]
    calibration_offset_db: float = field(default=0.0)


def spectrum_db(samples: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """返回 (freqs, psd_db)。Welch 平均，nperseg 自适应窗口。"""
    x = np.asarray(samples, dtype=np.float64)
    nperseg = min(len(x), 4096)
    nperseg = max(256, int(2 ** np.floor(np.log2(nperseg))))
    f, psd = signal.welch(x, fs=fs, nperseg=nperseg)
    psd_db = 10.0 * np.log10(np.maximum(psd, 1e-20))
    return f, psd_db


def find_peaks(freqs: np.ndarray, psd_db: np.ndarray, min_freq: float = 40.0,
               max_freq: float = 8000.0, min_prominence: float = 3.0,
               min_level: float = -70.0) -> list[TonalPeak]:
    mask = (freqs >= min_freq) & (freqs <= max_freq)
    f = freqs[mask]
    p = psd_db[mask]
    idx, props = signal.find_peaks(p, prominence=min_prominence, height=min_level)
    peaks: list[TonalPeak] = []
    for k, i in enumerate(idx):
        prom = float(props["prominences"][k])
        peaks.append(TonalPeak(freq=float(f[i]), level_db=float(p[i]), prominence_db=prom))
    return peaks


def find_harmonic_family(peaks: list[TonalPeak], tolerance: float = 0.04) -> tuple[float | None, list[tuple[TonalPeak, int]]]:
    """在峰值中找最强谐波家族。返回 (基频, [(peak, 谐波阶数)])。"""
    best_fund: float | None = None
    best_matched: list[tuple[TonalPeak, int]] = []
    for base in peaks:
        fund = base.freq
        matched: list[tuple[TonalPeak, int]] = []
        for pk in peaks:
            order = round(pk.freq / fund)
            if order >= 1 and abs(pk.freq - order * fund) <= tolerance * fund:
                matched.append((pk, order))
        if len(matched) >= 2 and len(matched) > len(best_matched):
            best_fund = fund
            best_matched = matched
    return best_fund, best_matched


def tonality_ratio(freqs: np.ndarray, psd_db: np.ndarray, peaks: list[TonalPeak]) -> float:
    """音调能量占比：峰值频点能量 / 总能量。"""
    power = 10.0 ** (psd_db / 10.0)
    total = float(np.sum(power))
    if total <= 0:
        return 0.0
    peak_power = 0.0
    for pk in peaks:
        i = int(np.argmin(np.abs(freqs - pk.freq)))
        peak_power += float(power[i])
    return float(np.clip(peak_power / total, 0.0, 1.0))


def band_energy_db(samples: np.ndarray, fs: float) -> dict[str, float]:
    f, psd_db = spectrum_db(samples, fs)
    df = float(f[1] - f[0])
    out: dict[str, float] = {}
    for name, (lo, hi) in {"low": LOW_BAND, "mid": MID_BAND, "high": HIGH_BAND}.items():
        m = (f >= lo) & (f <= hi)
        power = float(np.sum(10.0 ** (psd_db[m] / 10.0)) * df)
        out[name] = 10.0 * np.log10(max(power, 1e-20))
    return out


def stable_segment(samples: np.ndarray, fs: float, frame_s: float = 0.25,
                   keep_ratio: float = 0.8, max_jump_db: float = 20.0) -> np.ndarray:
    """剔除瞬态/突发噪声，返回稳定段。

    把信号分帧算 RMS，剔除显著高于中位数的帧（瞬态污染，如开关门、
    说话、碰触），保留约 keep_ratio 的稳定部分。用于真实测量时避免
    突发噪声抬高 SPL 或污染频谱。
    """
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    if len(x) == 0:
        return x
    frame = max(int(fs * frame_s), 1)
    n_frames = len(x) // frame
    if n_frames < 2:
        return x
    fr = x[: n_frames * frame].reshape(n_frames, frame)
    rms = np.sqrt(np.mean(fr ** 2, axis=1))
    med = np.median(rms)
    thresh = med * (10.0 ** (max_jump_db / 20.0))
    keep = rms <= thresh
    n_keep = int(np.sum(keep))
    if n_keep == 0:
        return x  # 全异常时放弃剔除
    target = min(n_keep, max(int(n_frames * keep_ratio), 1))
    keep_idx = np.argsort(rms)[:target]  # 保留最低 RMS 的 target 帧
    keep_mask = np.zeros(n_frames, dtype=bool)
    keep_mask[keep_idx] = True
    kept = fr[keep_mask].ravel()
    return kept.astype(np.float64)


def analyze(samples: np.ndarray, fs: float, calibration_offset_db: float = 0.0) -> AnalysisReport:
    x = np.asarray(samples, dtype=np.float64)
    if x.ndim > 1:
        x = x.mean(axis=1)
    f, psd_db = spectrum_db(x, fs)
    peaks = find_peaks(f, psd_db)
    fund, matched = find_harmonic_family(peaks)
    for pk, order in matched:
        pk.harmonic_order = order

    rms = rms_db(x)
    tr = tonality_ratio(f, psd_db, peaks)

    df = float(f[1] - f[0])
    w_a = 10.0 ** (a_weight_db(f) / 10.0)
    power_a = float(np.sum(psd_db_to_linear(psd_db) * w_a) * df)
    db_a_rel = 20.0 * np.log10(max(power_a, 1e-20)) if power_a > 0 else -120.0

    dominant = None
    if peaks:
        dominant = max(peaks, key=lambda p: p.level_db).freq

    return AnalysisReport(
        fs=float(fs),
        duration_s=len(x) / fs,
        rms_db=rms,
        spl_db=rms + calibration_offset_db if calibration_offset_db else None,
        spl_db_a=db_a_rel + calibration_offset_db if calibration_offset_db else None,
        peaks=sorted(peaks, key=lambda p: p.level_db, reverse=True),
        tonality_ratio=tr,
        dominant_freq=float(dominant) if dominant else None,
        harmonic_family=[fund] + [pk.freq for pk, _ in matched[1:]] if fund else [],
        band_spl_db=band_energy_db(x, fs),
        calibration_offset_db=calibration_offset_db,
    )


def psd_db_to_linear(psd_db: np.ndarray) -> np.ndarray:
    return 10.0 ** (psd_db / 10.0)


def analyze_file(path: str, calibration_offset_db: float = 0.0) -> AnalysisReport:
    fs, data = wavfile.read(path)
    if data.dtype == np.int16:
        samples = data.astype(np.float64) / 32768.0
    elif data.dtype == np.float32 or data.dtype == np.float64:
        samples = data.astype(np.float64)
    else:
        samples = data.astype(np.float64) / np.iinfo(data.dtype).max
    return analyze(samples, fs, calibration_offset_db)


def to_dict(report: AnalysisReport) -> dict:
    return {
        "fs": report.fs,
        "duration_s": round(report.duration_s, 2),
        "rms_db": round(report.rms_db, 1),
        "spl_db": round(report.spl_db, 1) if report.spl_db is not None else None,
        "spl_db_a": round(report.spl_db_a, 1) if report.spl_db_a is not None else None,
        "dominant_freq": report.dominant_freq,
        "tonality_ratio": round(report.tonality_ratio, 3),
        "harmonic_family": [round(f, 1) for f in report.harmonic_family],
        "band_spl_db": {k: round(v, 1) for k, v in report.band_spl_db.items()},
        "peaks": [
            {"freq": round(p.freq, 1), "level_db": round(p.level_db, 1),
             "prominence_db": round(p.prominence_db, 1), "harmonic_order": p.harmonic_order}
            for p in report.peaks[:20]
        ],
    }
