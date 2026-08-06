"""ANC 运行 CLI。

离线演示（默认，无需硬件）：
  python scripts/run_anc.py --mode harmonic
  python scripts/run_anc.py --mode fxlms

真实实时 ANC（M2，需要 WM8960 I2S 编解码器）：
  python scripts/run_anc.py --mode harmonic --realtime

realtime 模式当前为接口占位：请按 docs/ENGINEERING_PLAN.md 的 ALSA 低延迟
配置接入参考/误差麦克风与扬声器。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

# 允许 `python scripts/run_anc.py` 直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.anc.harmonic import estimate_fundamental
from app.anc.pipeline import ANCController
from app.synth import printer_noise


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["harmonic", "fxlms"], default="harmonic")
    ap.add_argument("--fs", type=int, default=16000)
    ap.add_argument("--duration", type=float, default=10.0)
    ap.add_argument("--realtime", action="store_true", help="实时模式（占位）")
    args = ap.parse_args()

    controller = ANCController(mode=args.mode, fs=args.fs)

    if args.realtime:
        raise NotImplementedError(
            "实时 ANC 环路尚未接入 ALSA。完成 M0/M2 硬件验证后，"
            "在 app/anc/pipeline.py 的 ANCController 里实现音频循环。"
        )

    samples, sources = printer_noise(fs=args.fs, duration=args.duration)
    reference = samples
    error = samples  # 离线模拟：参考与误差同源，谐波消除直接作用于误差

    if args.mode == "harmonic":
        f0 = estimate_fundamental(samples, args.fs)
        print(f"估计基频: {f0:.1f} Hz" if f0 else "未检测到显著基频")
        res = controller.run_offline(reference, error)
    else:
        res = controller.run_offline(reference, error)

    summary = controller.summary()
    print(json_dumps(summary))
    print(f"降噪量（稳态 RMS）：{res['reduction_db']:+.1f} dB")


def json_dumps(d: dict) -> str:
    import json

    return json.dumps(d, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
