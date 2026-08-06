"""噪声源归属：把分析报告的频谱特征匹配到 data/profiles/*.json 的噪声源 Profile。"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.analyze import AnalysisReport

PROFILES_DIR = Path(__file__).resolve().parent.parent / "data" / "profiles"


@dataclass
class SourceHit:
    source: str
    confidence: float
    freqs_hz: list[float]
    matched_signatures: list[str]


def load_profiles(profiles_dir: Path = PROFILES_DIR) -> list[dict]:
    profiles = []
    for p in sorted(profiles_dir.glob("*.json")):
        with open(p, encoding="utf-8") as fh:
            profiles.append(json.load(fh))
    return profiles


def _match_signature(sig: dict, report: AnalysisReport) -> bool:
    kind = sig.get("kind")
    if kind == "harmonic_family":
        fund = report.harmonic_family[0] if report.harmonic_family else None
        if fund is None:
            return False
        lo, hi = sig.get("low_hz", 40.0), sig.get("high_hz", 8000.0)
        return lo <= fund <= hi and len(report.harmonic_family) >= sig.get("n_harmonics_min", 3)
    if kind == "band_peak":
        lo, hi = sig.get("low_hz", 0.0), sig.get("high_hz", 20000.0)
        return any(lo <= p.freq <= hi for p in report.peaks)
    if kind == "band_energy_dominant":
        band = sig.get("band", "mid")
        bands = {"low": "low", "mid": "mid", "high": "high"}
        band = bands.get(band, band)
        if band not in report.band_spl_db:
            return False
        others = {k: v for k, v in report.band_spl_db.items() if k != band}
        return report.band_spl_db[band] >= max(others.values(), default=-120.0)
    return False


def match_sources(report: AnalysisReport, profiles: list[dict] | None = None) -> list[SourceHit]:
    if profiles is None:
        profiles = load_profiles()
    hits: list[SourceHit] = []
    for prof in profiles:
        sigs = prof.get("signatures", [])
        if not sigs:
            continue
        matched = [s for s in sigs if _match_signature(s, report)]
        conflict = [s for s in prof.get("negative_signatures", []) if _match_signature(s, report)]
        if matched:
            freqs = [round(p.freq, 1) for p in report.peaks[:8]]
            # 匹配比例为基分；出现负向特征（不该有的频谱特征却出现）按 0.5 折损；
            # 匹配到的特征数作为排序加成，用于区分重叠特征源。
            score = (len(matched) / len(sigs)) * (0.5 ** len(conflict)) + 0.05 * len(matched)
            hits.append(SourceHit(
                source=prof.get("id", "unknown"),
                confidence=round(min(1.0, score), 2),
                freqs_hz=freqs,
                matched_signatures=[s.get("kind", "") for s in matched],
            ))
    return sorted(hits, key=lambda h: (h.confidence, len(h.matched_signatures)), reverse=True)


def recommend_anc(report: AnalysisReport) -> dict:
    """根据分析报告给出"是否需要降噪"的量化建议。"""
    fund = report.harmonic_family[0] if report.harmonic_family else (report.dominant_freq or 0.0)
    reasons: list[str] = []
    worthwhile = False

    if report.tonality_ratio >= 0.3:
        reasons.append("tonal_ratio_high")
        worthwhile = True
    if fund and fund < 500:
        reasons.append("dominant_freq_low_mid")
        worthwhile = True
    if fund and 500 <= fund <= 2000:
        reasons.append("dominant_freq_mid_partial")
    if fund and fund > 2000:
        reasons.append("dominant_freq_high_limited_anc")

    high_band = report.band_spl_db.get("high", -120.0)
    low_band = report.band_spl_db.get("low", -120.0)
    if high_band - low_band > 6:
        reasons.append("wideband_high_freq_use_passive")

    return {
        "anc_worthwhile": worthwhile,
        "reasons": reasons,
        "dominant_freq": fund,
        "tonality_ratio": round(report.tonality_ratio, 3),
    }
