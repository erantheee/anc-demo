"""端到端 ANC 演示：播放合成打印机噪声 → 麦克风采集 → 降噪 → 量化效果。

流程：
1. 低音量播放合成 3D 打印机噪声（办公室安全音量）
2. USB 麦克风同步采集
3. 分析：频谱、基频、音调占比 → 确认噪声来源
4. 降噪：谐波消除 与 FXLMS 各跑一遍
5. 输出降噪前后 dB 差 + 峰值衰减

注意：处理为离线批量（采集完成后统一处理），因此不受 USB 采集延迟
限制。这是"先测量确认、再评估降噪"闭环的演示。

用法（在树莓派上）：
  .venv/bin/python scripts/anc_demo_live.py --volume 0.02 --duration 6
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np

from app.analyze import analyze, stable_segment
from app.anc.fxlms import simulate as fx_sim
from app.anc.harmonic import estimate_fundamental, simulate as harm_sim
from app.synth import printer_noise


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fs", type=int, default=48000, help="声卡采样率（USB 通常 48k）")
    ap.add_argument("--duration", type=float, default=6.0)
    ap.add_argument("--volume", type=float, default=0.02, help="播放音量（办公室安全，建议 ≤0.05）")
    ap.add_argument("--out", default="data/reports/anc-demo", help="输出目录")
    args = ap.parse_args()

    import sounddevice as sd
    import soundfile as sf

    # 1. 生成低音量打印机噪声
    noise, sources = printer_noise(fs=args.fs, duration=args.duration, seed=42)
    noise = noise * args.volume
    stereo = np.column_stack([noise, noise])
    sd.default.samplerate = args.fs

    print(f"[1/4] 播放合成打印机噪声 {args.duration}s（音量 {args.volume:.2f}）...")
    rec = sd.playrec(stereo, samplerate=args.fs, channels=2, dtype="float32")
    sd.wait()
    recorded = rec[:, 0].astype(np.float64)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_dir / "recorded_noise.wav"), rec, args.fs)

    # 2. 分析采集到的噪声（含瞬态剔除）
    kept = stable_segment(recorded, args.fs)
    report = analyze(kept, args.fs)
    print(f"[2/4] 采集 {len(kept)/args.fs:.1f}s 有效信号，RMS {report.rms_db:.1f} dBFS")
    if report.rms_db < -40.0:
        print("      [警告] 采集信号很弱——扬声器未接、音量过低或采集被占用。"
              "请检查扬声器连接后重试。")
    # 基频从参考信号（播放的噪声）估计：信噪比高，不受房间底噪影响，
    # 对应真实系统中"参考麦克风贴噪声源"的做法
    f0 = estimate_fundamental(noise, args.fs, low=40, high=300)
    dom = report.dominant_freq
    print(f"      参考信号基频 {f0:.1f} Hz，采集音调占比 {report.tonality_ratio:.2f}，"
          f"主频 {dom:.0f} Hz" if dom else
          f"      参考信号基频 {f0:.1f} Hz，采集音调占比 {report.tonality_ratio:.2f}")

    # 3. 降噪处理
    print("[3/4] 运行谐波消除与 FXLMS ...")
    harm_res = harm_sim(kept, args.fs, f0=f0, max_harmonics=10, mu=2e-3)
    fx_res = fx_sim(kept, args.fs,
                    secondary_path=np.array([0.4, 0.6, 1.0, 0.6, 0.3]),
                    plant_delay_samples=16, num_taps=256, mu=1e-4)

    # 4. 量化效果
    print("[4/4] 评估降噪量 ...")
    harm_report = analyze(harm_res["e"], args.fs)
    fx_report = analyze(fx_res["e"], args.fs)

    # 谐波峰值衰减：评估谐波消除在音调成分上的效果（即使整体 RMS 降幅被宽带掩盖）
    def tonal_peak_db(report) -> float:
        f0_pk = next((p.level_db for p in report.peaks if p.harmonic_order == 1), None)
        return f0_pk if f0_pk is not None else -120.0

    before_f0 = tonal_peak_db(report)
    after_f0_harm = tonal_peak_db(harm_report)
    after_f0_fx = tonal_peak_db(fx_report)

    result = {
        "setup": {"fs": args.fs, "duration": args.duration, "volume": args.volume,
                  "recorded_dbfs": round(report.rms_db, 1)},
        "noise_analysis": {
            "f0_hz": round(f0, 1) if f0 else None,
            "tonality_ratio": round(report.tonality_ratio, 3),
            "dominant_freq_hz": report.dominant_freq,
            "peaks": [round(p.freq, 1) for p in report.peaks[:6]],
        },
        "harmonic": {
            "reduction_db": round(harm_res["reduction_db"], 2),
            "residual_tonality": round(harm_report.tonality_ratio, 3),
            "f0_peak_reduction_db": round(before_f0 - after_f0_harm, 1),
        },
        "fxlms": {
            "reduction_db": round(fx_res["reduction_db"], 2),
            "residual_tonality": round(fx_report.tonality_ratio, 3),
            "f0_peak_reduction_db": round(before_f0 - after_f0_fx, 1),
        },
    }

    import json
    (out_dir / "anc-demo-result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== 结果 ===")
    print(f"采集噪声 RMS:          {report.rms_db:+.1f} dBFS（音调占比 {report.tonality_ratio:.2f}）")
    print(f"谐波消除: 整体 {harm_res['reduction_db']:+.2f} dB，"
          f"基频峰值 {before_f0 - after_f0_harm:+.1f} dB")
    print(f"FXLMS:    整体 {fx_res['reduction_db']:+.2f} dB，"
          f"基频峰值 {before_f0 - after_f0_fx:+.1f} dB")
    print(f"结果保存: {out_dir / 'anc-demo-result.json'}")
    print("\n注：若谐波消除整体降幅小但基频峰值大幅下降，说明扬声器对低频")
    print("输出弱、采集端音调能量低——算法在工作，硬件低频响应是瓶颈。")


if __name__ == "__main__":
    main()
