"""ANC 前后 A/B 评估：宽带/音调/A 加权降噪量。"""
from __future__ import annotations

import numpy as np

from app.analyze import analyze, spectrum_db, to_dict


def _level_at(freqs: np.ndarray, psd_db: np.ndarray, freq: float) -> float:
    i = int(np.argmin(np.abs(freqs - freq)))
    return float(psd_db[i])


def evaluate_before_after(before: np.ndarray, after: np.ndarray, fs: float,
                          calibration_offset_db: float = 0.0) -> dict:
    ra = analyze(before, fs, calibration_offset_db)
    rb = analyze(after, fs, calibration_offset_db)

    _, psd_after = spectrum_db(np.asarray(after, dtype=np.float64), fs)
    f, _ = spectrum_db(np.asarray(before, dtype=np.float64), fs)

    peak_reductions = []
    for p in ra.peaks[:8]:
        after_level = _level_at(f, psd_after, p.freq)
        peak_reductions.append({
            "freq": round(p.freq, 1),
            "before_db": round(p.level_db, 1),
            "after_db": round(after_level, 1),
            "reduction_db": round(p.level_db - after_level, 1),
        })

    a_before = ra.spl_db_a if ra.spl_db_a is not None else ra.rms_db
    a_after = rb.spl_db_a if rb.spl_db_a is not None else rb.rms_db

    return {
        "before": to_dict(ra),
        "after": to_dict(rb),
        "broadband_reduction_db": round(ra.rms_db - rb.rms_db, 2),
        "a_weighted_reduction_db": round(a_before - a_after, 2),
        "peak_reductions": peak_reductions,
    }
