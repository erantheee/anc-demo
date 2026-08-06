"""M1 测量 CLI：网格录音 → 分析 → 来源归属 → 报告。

用法：
  # 合成数据（无硬件自测）
  python scripts/measure.py --synthetic --out data/reports/synthetic-demo

  # 真实测量（默认 sounddevice，失败可用 --driver arecord）
  python scripts/measure.py --grid "0,0 1,0 0,1 1,1" --duration 30 \
      --calibration data/reports/calibration.json --out data/reports/room-001
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# 允许 `python scripts/measure.py` 直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app import capture
from app.analyze import AnalysisReport, analyze, analyze_file, stable_segment
from app.noise_map import GridPoint, build_noise_map, write_report
from app.source_id import load_profiles, match_sources, recommend_anc
from app.synth import printer_noise


def _read_wav(path: str) -> tuple[np.ndarray, int]:
    """读取 WAV，返回 (samples_float64_mono, fs)。兼容 int16 与 float 格式。"""
    from scipy.io import wavfile

    fs, data = wavfile.read(path)
    if data.dtype == np.int16:
        samples = data.astype(np.float64) / 32768.0
    elif data.dtype in (np.float32, np.float64):
        samples = data.astype(np.float64)
    else:
        samples = data.astype(np.float64) / np.iinfo(data.dtype).max
    if samples.ndim > 1:
        samples = samples.mean(axis=1)
    return samples, int(fs)


def parse_grid(s: str) -> list[tuple[float, float]]:
    pts = []
    for tok in s.split():
        x, y = tok.split(",")
        pts.append((float(x), float(y)))
    return pts or [(0.0, 0.0)]


def load_calibration(path: str | None) -> float:
    if not path:
        return 0.0
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return float(data.get("offset_db", 0.0))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--grid", default="0,0", help='网格点，如 "0,0 1,0 0,1 1,1"')
    ap.add_argument("--duration", type=float, default=30.0)
    ap.add_argument("--fs", type=int, default=16000)
    ap.add_argument("--out", default="data/reports/measure", help="报告输出目录")
    ap.add_argument("--calibration", default=None, help="calibration.json 路径")
    ap.add_argument("--synthetic", action="store_true", help="用合成噪声，不录音")
    ap.add_argument("--driver", choices=["auto", "sounddevice", "arecord"], default="auto")
    args = ap.parse_args()

    grid = parse_grid(args.grid)
    offset = load_calibration(args.calibration)
    profiles = load_profiles()

    points: list[GridPoint] = []
    reports: list[AnalysisReport] = []
    for i, (x, y) in enumerate(grid):
        print(f"[{i+1}/{len(grid)}] 测量点 ({x}, {y}) ...", flush=True)
        if args.synthetic:
            dist = np.hypot(x, y)
            samples, _ = printer_noise(fs=args.fs, duration=args.duration, seed=i)
            samples = samples * float(np.clip(1.0 - 0.08 * dist, 0.3, 1.0))
            report = analyze(samples, args.fs, offset)
        else:
            if args.driver == "arecord":
                wav = capture.record_with_arecord(
                    args.duration, fs=args.fs, out_path=Path("data/recordings") / f"p{i}.wav")
            else:
                wav = capture.record(
                    args.duration, fs=args.fs, out_path=Path("data/recordings") / f"p{i}.wav")
            samples, fs_read = _read_wav(str(wav))
            # 真实录音可能含瞬态突发（开关门/说话/碰触），只保留稳定段
            samples = stable_segment(samples, fs_read)
            if len(samples) == 0:
                print(f"  [警告] 点 ({x},{y}) 无有效信号，跳过")
                continue
            report = analyze(samples, fs_read, offset)
        hits = match_sources(report, profiles)
        spl = report.spl_db if report.spl_db is not None else report.rms_db
        points.append(GridPoint(x=x, y=y, spl_db=float(spl),
                                source_hits=[h.__dict__ for h in hits]))
        reports.append(report)
        time.sleep(1.0)

    agg: dict[str, dict] = {}
    for p in points:
        for h in (p.source_hits or []):
            cur = agg.get(h["source"])
            if cur is None or h["confidence"] > cur["confidence"]:
                agg[h["source"]] = h
    dominant = sorted(agg.values(), key=lambda h: h["confidence"], reverse=True)[:5]

    merged_report = {"grid": [p.__dict__ for p in points],
                     "dominant_sources": dominant,
                     "calibration_offset_db": offset}

    if reports:
        ref_report = max(reports, key=lambda r: r.rms_db)
        merged_report["recommendation"] = recommend_anc(ref_report)

    try:
        xi, yi, zi = build_noise_map(points)
        map_path = Path(args.out) / "noise-map.png"
        from app.noise_map import save_png

        save_png(xi, yi, zi, map_path)
        merged_report["noise_map"] = {
            "image": str(map_path),
            "max_db": round(float(np.nanmax(zi)), 1),
            "min_db": round(float(np.nanmin(zi)), 1),
        }
    except Exception as exc:
        print(f"噪声地图渲染跳过（需要 matplotlib）：{exc}")

    write_report(points, merged_report, args.out)
    print(f"报告已写入 {args.out}/report.json")


if __name__ == "__main__":
    main()
