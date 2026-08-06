"""现场实时 ANC demo（M2）。

用法：
  python scripts/run_anc_live.py --list                          # 列出音频设备
  python scripts/run_anc_live.py --synthetic --duration 15       # 无硬件自测
  python scripts/run_anc_live.py --in-device "USB Mic" --out-device "3.5mm" \
      --fs 48000 --baseline 5 --duration 60                      # 现场真机

流程：先采 baseline（ANC off）估计基频并记录噪声水平；随后实时输出反相谐波，
在误差麦克风处形成安静区；结束输出 A/B 降噪报告。

原理：自参考谐波消除（无需参考麦克风）。对步进电机 / 风扇叶片 / 压缩机等
稳态周期噪声有效。若基频自动估计失败，用 --f0 手动指定（如 120.0）。
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

# 允许 `python scripts/run_anc_live.py` 直接运行
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.anc.live import LiveANCEngine
from app.evaluate import evaluate_before_after


def list_devices() -> None:
    import sounddevice as sd

    print("音频设备（* 为默认）：")
    default_in = sd.default.device[0]
    default_out = sd.default.device[1]
    for i, dev in enumerate(sd.query_devices()):
        mark_in = "*" if i == default_in else " "
        mark_out = "*" if i == default_out else " "
        ch_in, ch_out = dev["max_input_channels"], dev["max_output_channels"]
        print(f"[{i}] {dev['name']}")
        print(f"     输入{mark_in} {ch_in}ch / 输出{mark_out} {ch_out}ch  默认采样率 {dev['default_samplerate']}Hz")


def wait_for_finish(engine: LiveANCEngine, poll_s: float = 0.5) -> None:
    last = None
    while True:
        st = engine.status()
        line = (f"\r阶段={st['phase']:<10} f0={st['f0']} "
                f"实时SPL={st['spl_now_db']} dB  基线={st['baseline_spl_db']} dB "
                f"时间={st['elapsed_s']:.0f}s  状态={st['state']}")
        if line != last:
            sys.stdout.write(line)
            sys.stdout.flush()
            last = line
        if st["state"] in ("stopped", "error"):
            sys.stdout.write("\n")
            sys.stdout.flush()
            if st["error"]:
                print(f"[错误] {st['error']}")
            return
        time.sleep(poll_s)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="列出音频设备后退出")
    ap.add_argument("--synthetic", action="store_true", help="合成模式（无硬件自测）")
    ap.add_argument("--in-device", type=str, default=None, help="误差麦克风设备（名称或编号）")
    ap.add_argument("--out-device", type=str, default=None, help="扬声器输出设备（名称或编号）")
    ap.add_argument("--fs", type=int, default=48000, help="采样率，默认 48000")
    ap.add_argument("--baseline", type=float, default=5.0, help="基线采集时长（秒），默认 5")
    ap.add_argument("--duration", type=float, default=60.0, help="总演示时长（秒），默认 60")
    ap.add_argument("--f0", type=float, default=None, help="手动指定基频（Hz），跳过自动估计")
    ap.add_argument("--harmonics", type=int, default=10, help="谐波数量，默认 10")
    ap.add_argument("--gain", type=float, default=0.4, help="反相输出增益（防啸叫，默认 0.4）")
    ap.add_argument("--mu", type=float, default=0.02,
                    help="NLMS 步长，默认 0.02（block 归一化后）")
    ap.add_argument("--echo-gain", type=float, default=0.0,
                    help="合成模式：反相输出被误差麦回采的增益（模拟声学环路）")
    ap.add_argument("--block", type=int, default=512, help="音频 block 大小，默认 512")
    args = ap.parse_args()

    if args.list:
        list_devices()
        return

    if args.duration <= args.baseline:
        ap.error("--duration 需大于 --baseline")

    engine = LiveANCEngine(
        fs=args.fs, in_device=args.in_device, out_device=args.out_device,
        block=args.block, f0=args.f0, max_harmonics=args.harmonics, mu=args.mu,
        output_gain=args.gain, baseline_s=args.baseline,
        max_duration_s=args.duration, synthetic=args.synthetic,
        echo_gain=args.echo_gain)

    mode = "合成" if args.synthetic else "真机"
    print(f"[M2] {mode}模式：基线 {args.baseline:.0f}s（ANC off）→ 实时消除（ANC on）"
          f"，总长 {args.duration:.0f}s")
    if not args.synthetic and args.in_device is None:
        print("[提示] 未指定 --in-device，将使用系统默认输入设备。"
              "可先跑 --list 查看设备。")

    engine.start()
    wait_for_finish(engine)

    st = engine.status()
    if st["error"]:
        print(f"\n实时 ANC 失败：{st['error']}")
        sys.exit(1)

    d, e = engine.get_signals()
    if d is None or e is None or len(d) == 0 or len(e) == 0:
        print("未采集到有效信号。")
        sys.exit(1)

    report = evaluate_before_after(d, e, fs=args.fs)
    summary = {
        "f0_hz": st["f0"],
        "baseline_spl_db": st["baseline_spl_db"],
        "cancelling_spl_db": st["cancelling_spl_db"],
        "broadband_reduction_db": report["broadband_reduction_db"],
        "a_weighted_reduction_db": report["a_weighted_reduction_db"],
        "peak_reductions": report["peak_reductions"][:5],
    }
    print("\n==== A/B 降噪报告 ====")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
