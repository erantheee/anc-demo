"""麦克风灵敏度标定。

方法：用手机声级计 APP（或 94 dB 校准器）在麦克风处读出真实 SPL，
程序算出 offset_db，供 analyze 输出绝对 dB SPL。

用法：
  python scripts/calibrate_mic.py --known-spl 75 --duration 10 --out data/reports/calibration.json
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# 允许 `python scripts/calibrate_mic.py` 直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.analyze import rms_db
from app.capture import record


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--known-spl", type=float, required=True, help="手机声级计在麦克风处读到的 dB SPL")
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--fs", type=int, default=16000)
    ap.add_argument("--out", default="data/reports/calibration.json")
    ap.add_argument("--driver", choices=["auto", "arecord"], default="auto")
    args = ap.parse_args()

    wav = record(args.duration, fs=args.fs, out_path=Path("data/recordings/_calibration.wav"))
    from scipy.io import wavfile

    fs, data = wavfile.read(str(wav))
    samples = data.astype("float64") / 32768.0 if data.dtype == "int16" else data.astype("float64")
    rel = rms_db(samples)
    offset = round(args.known_spl - rel, 2)

    payload = {
        "offset_db": offset,
        "known_spl_db": args.known_spl,
        "measured_rms_db": round(rel, 2),
        "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"offset_db = {offset}（校准文件：{out}）")


if __name__ == "__main__":
    main()
