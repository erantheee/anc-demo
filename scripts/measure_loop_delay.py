"""实测扬声器→误差麦克风环路延迟（ANC 相位对齐的关键参数）。

实时 ANC（自参考谐波消除）要求 `--mic-delay-ms` 大致等于"扬声器播放 →
声波到达误差麦"的总延迟，包括设备缓冲延迟（USB 声卡通常 10–60ms）与
声学传播延迟。默认 5ms 只覆盖声学部分，USB 设备必须用本脚本实测。

用法：
  python scripts/measure_loop_delay.py --list                 # 列出设备
  python scripts/measure_loop_delay.py --out-device 0 --in-device 0
  python scripts/measure_loop_delay.py --gain 0.05 --duration 2

原理：通过扬声器播放低幅度噪声脉冲，同时录音，用互相关找出输出信号在
输入中的位置 → 环路延迟。结果中的 `recommended_mic_delay_ms` 可直接用于
run_anc_live.py --mic-delay-ms 或 Web 仪表盘"麦克风延迟"。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.capture import input_devices  # noqa: E402


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


PROBE_FREQ_HZ = 2000.0  # 探测音调频率（选环境频谱较干净的频段）
PROBE_START_S = 0.15    # 探测突发在录音中的起始时刻


def _envelope(sig: np.ndarray, fs: int, f0: float, lp_ms: float = 8.0) -> np.ndarray:
    """正交解调包络：sig * e^{-j2π f0 t} 后低通，幅度即包络。"""
    t = np.arange(len(sig)) / fs
    iq = sig * np.exp(-1j * 2.0 * np.pi * f0 * t)
    k = max(1, int(lp_ms / 1000.0 * fs))
    kernel = np.ones(k) / k
    lp = np.convolve(iq, kernel, mode="same")
    return np.abs(lp)


def measure(device: int | str | None, fs: int, duration_s: float, gain: float,
            burst_s: float = 0.25) -> dict:
    import sounddevice as sd

    if device is None:
        inp = input_devices()
        if not inp:
            sys.exit("未检测到任何输入设备")
        device = inp[0]["index"]
    elif isinstance(device, str):
        # 名称 → 索引（sounddevice 的 playrec 对部分 ALSA 名称匹配有差异）
        for i, d in enumerate(sd.query_devices()):
            if device in d["name"] and d["max_input_channels"] > 0 and d["max_output_channels"] > 0:
                device = i
                break
        if isinstance(device, str):
            sys.exit(f"找不到设备 {device}")

    n_total = int(duration_s * fs)
    n_start = int(PROBE_START_S * fs)
    n_burst = int(burst_s * fs)
    t = np.arange(n_burst) / fs
    tone = gain * np.sin(2.0 * np.pi * PROBE_FREQ_HZ * t).astype(np.float32)
    # 音调突发 + 前后留白（探测信号在 n_start 处开始播放）
    probe = np.zeros((n_total, 1), dtype=np.float32)
    probe[n_start: n_start + n_burst, 0] = tone

    # sounddevice 0.5.x：playrec 的 channels 是输入通道数，输出通道取 data.shape[1]
    rec = sd.playrec(probe, samplerate=fs, channels=1,
                     device=(device, device), dtype="float32")
    sd.wait()
    inp = np.asarray(rec[:, 0], dtype=np.float64)

    clipped = bool(np.abs(inp).max() >= 0.99)
    env = _envelope(inp, fs, PROBE_FREQ_HZ)
    # 用突发开始前的区间估计本底，找首个明显超过本底的包络点 → 环路延迟
    noise_floor = float(np.median(env[:n_start])) + 1e-9
    thr = max(3.0 * noise_floor, gain * 0.05)
    lag = None
    for i in range(n_start, n_total):
        if env[i] > thr:
            lag = i - n_start
            break
    if lag is None:
        peak_abs = float(np.abs(inp).max())
        return {
            "device_index": device,
            "fs": fs,
            "found": False,
            "loop_delay_samples": None,
            "loop_delay_ms": None,
            "noise_floor": round(noise_floor, 4),
            "clipped": clipped,
            "recording_peak": round(peak_abs, 3),
            "error": "未检测到扬声器回波。请确认扬声器音量足够、麦克风可采集到扬声器声音，"
                     "或增大 --gain。",
        }
    delay_ms = lag / fs * 1000.0
    return {
        "device_index": device,
        "fs": fs,
        "found": True,
        "loop_delay_samples": lag,
        "loop_delay_ms": round(delay_ms, 1),
        "noise_floor": round(noise_floor, 4),
        "clipped": clipped,
        "recording_peak": round(float(np.abs(inp).max()), 3),
        "recommended_mic_delay_ms": max(0.0, round(delay_ms, 1)),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="列出音频设备后退出")
    ap.add_argument("--device", default=None,
                    help="音频设备（编号、名称或 ALSA PCM，如 default；缺省取默认输入）")
    ap.add_argument("--fs", type=int, default=48000, help="采样率，默认 48000")
    ap.add_argument("--duration", type=float, default=2.0, help="总时长（秒），默认 2")
    ap.add_argument("--gain", type=float, default=0.15,
                    help="探测音调幅度（默认 0.15）")
    args = ap.parse_args()

    if args.list:
        list_devices()
        return

    import json

    res = measure(args.device, args.fs, args.duration, args.gain)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if not res.get("found"):
        print(f"\n[提示] {res.get('error', '未检测到回波')}")
    else:
        if res.get("clipped"):
            print("\n[警告] 录音已削波（峰值接近满量程）：建议调低麦克风增益或 --gain。")
        print(f"\n[结论] 建议 --mic-delay-ms {res['recommended_mic_delay_ms']}"
              f"（环路延迟 {res['loop_delay_ms']}ms / {res['loop_delay_samples']} samples）")


if __name__ == "__main__":
    main()
