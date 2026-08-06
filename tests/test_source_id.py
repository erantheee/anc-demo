from app.analyze import analyze
from app.source_id import match_sources, recommend_anc
from app.synth import printer_noise


def test_printer_profile_matched():
    x, _ = printer_noise(fs=16000, duration=4.0, seed=3)
    report = analyze(x, 16000)
    hits = match_sources(report)
    ids = [h.source for h in hits]
    assert "3d_printer" in ids, f"应命中 3d_printer，实际 {ids}"


def test_recommend_anc_worthwhile_for_printer():
    x, _ = printer_noise(fs=16000, duration=4.0, seed=4)
    report = analyze(x, 16000)
    rec = recommend_anc(report)
    assert rec["anc_worthwhile"] is True
    assert "tonal_ratio_high" in rec["reasons"] or "dominant_freq_low_mid" in rec["reasons"]
