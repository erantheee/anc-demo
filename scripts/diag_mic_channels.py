"""诊断：USB 麦克风的 2 声道是真双 capsule，还是单声道复制？

双麦空间差分防风的前提：两路输入来自两支独立 capsule 且间距几厘米
（远端周期声场在两路相干、风噪在两路不相关）。很多"立体声 USB 麦克风"
内部就是两支 capsule（等同现成双麦）；而部分声卡/单麦只是把同一路信号
复制到两个声道。

判定（录一段双声道，依次做：稳态纯音 → 对麦克风吹气/晃手制造气流）：
- 真双 capsule：纯音下左右声道在该频段相干（coherence≈1），吹气时中高频
  coherence 显著掉下去（两路听到的是不同的湍流）；
- 单声道复制：左右声道逐样本几乎相同（diff_rms≈0、corr≈1.0），全频带
  coherence≈1，吹气时也不分家。

在树莓派上跑（USB 麦已接）：
    .venv/bin/python scripts/diag_mic_channels.py --seconds 3
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.capture import _effective_input_channels


def _band_coherence(x: np.ndarray, y: np.ndarray, fs: float,
                    bands: dict[str, tuple[float, float]]) -> dict[str, float]:
    """各频段的平均幅度平方相干（Welch 估计）。"""
    from scipy import signal

    f, C = signal.coherence(x, y, fs=fs, nperseg=min(4096, len(x)))
    out: dict[str, float] = {}
    for name, (lo, hi) in bands.items():
        m = (f >= lo) & (f < hi)
        out[name] = float(np.mean(C[m])) if m.any() else 0.0
    return out


def _record_arecord(device: str, fs: int, seconds: float) -> tuple[np.ndarray, np.ndarray, int, int]:
    """用 arecord（ALSA 直连）录双声道，返回 (L, R, fs, channels)。

    绕开 PortAudio 枚举：Pi 上 sounddevice 常把 USB 声卡输入报成 0 声道
    （设备被占用时尤其如此），但 ALSA 层 arecord -l 能看到采集能力。
    """
    import subprocess
    import tempfile

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        wav_path = tf.name
    cmd = ["arecord", "-D", device, "-f", "S16_LE", "-r", str(fs),
           "-c", "2", "-d", str(int(max(seconds, 1))), wav_path]
    print("运行:", " ".join(cmd))
    print("在这段时间里：0–1.5s 播放纯音/风扇声，1.5s 后对麦克风缓慢吹气。")
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        raise RuntimeError(f"arecord 失败: {proc.stderr.strip() or proc.stdout.strip()}")
    import wave

    with wave.open(wav_path, "rb") as w:
        fs = w.getframerate()
        n_ch = w.getnchannels()
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    raw = raw.astype(np.float64) / 32768.0
    x = raw[0::n_ch]
    y = raw[1::n_ch]
    return x, y, fs, n_ch


def _record_selftone(alsa_device: str | None, fs: int, seconds: float,
                     tone_freq: float, tone_gain: float) -> tuple[np.ndarray, np.ndarray, int]:
    """纯音自测：Pi 从扬声器播放纯音，同时用 arecord 双声道采集。

    判据（真双 capsule vs 单声道复制）：
    - 单声道复制：左右声道逐样本相同 → 纯音频段差分能量 ~-90dB（只剩量化噪声），
      两路在该频段相干性≈1；
    - 真双 capsule：两 capsule 间距几 cm，纯音到达两路有微小相位差/幅度差 →
      差分能量落在 -20~-40dB 量级，且两路在纯音频段相干≈1（听到的是同一个声场），
      中高频（>2kHz）环境本底相干低。
    """
    import subprocess
    import tempfile
    import wave

    duration = max(float(seconds), 2.0)
    t = np.arange(int(duration * fs)) / fs
    tone = (tone_gain * np.sin(2.0 * np.pi * tone_freq * t)).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        wav_out = tf.name
    with wave.open(wav_out, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(int(fs))
        w.writeframes((tone * 32767.0).astype(np.int16).tobytes())

    rec_wav = wav_out.replace(".wav", "-rec.wav")
    dev = alsa_device or "hw:2,0"
    print(f"纯音自测：扬声器播放 {tone_freq:.0f}Hz（幅 {tone_gain}），"
          f"同步 {alsa_device or 'hw:2,0'} 双声道采集 {duration:.0f}s …")
    aplay = subprocess.Popen(["aplay", "-D", dev, wav_out],
                             stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT)
    proc = subprocess.run(["arecord", "-D", dev, "-f", "S16_LE", "-r", str(int(fs)),
                           "-c", "2", "-d", str(int(duration)), rec_wav],
                          capture_output=True, text=True)
    aplay.wait(timeout=30)
    if proc.returncode != 0:
        raise RuntimeError(f"arecord 失败: {proc.stderr.strip() or proc.stdout.strip()}")
    with wave.open(rec_wav, "rb") as w:
        fs = w.getframerate()
        n_ch = w.getnchannels()
        raw = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
    raw = raw.astype(np.float64) / 32768.0
    x = raw[0::n_ch]
    y = raw[1::n_ch]
    # 报告纯音频段的差分能量（判定主指标）
    seg = slice(int(0.2 * fs), int(1.2 * fs))  # 跳过起播瞬态
    xs, ys = x[seg], y[seg]
    d = xs - ys
    rms_l = float(np.sqrt(np.mean(xs ** 2)))
    rms_r = float(np.sqrt(np.mean(ys ** 2)))
    diff_db = 20.0 * np.log10(max(float(np.sqrt(np.mean(d ** 2))), 1e-12)
                              / max(max(rms_l, rms_r), 1e-12))
    print(f"纯音段两路 RMS L={20*np.log10(max(rms_l,1e-12)):.1f}dBFS "
          f"R={20*np.log10(max(rms_r,1e-12)):.1f}dBFS，差分能量 {diff_db:+.1f} dB")
    print("判定提示：差分 ~-90dB = 单声道复制；-20~-40dB = 真双 capsule。")
    return x, y, fs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--seconds", type=float, default=3.0, help="录音时长，默认 3s")
    ap.add_argument("--fs", type=int, default=48000, help="采样率，默认 48kHz")
    ap.add_argument("--device", type=str, default=None, help="输入设备名/索引，默认系统默认")
    ap.add_argument("--alsa-device", type=str, default=None,
                    help="ALSA 直连设备名（如 hw:2,0 / plughw:2,0），绕过 PortAudio 枚举。"
                         "Pi 上 sounddevice 可能查不到 USB 声卡输入，用这个最可靠")
    ap.add_argument("--out", type=str, default=None, help="把原始双声道 WAV 存到该路径")
    ap.add_argument("--selftone", action="store_true",
                    help="纯音自测：Pi 自己从扬声器播放 300Hz 纯音并同步双声道采集，"
                         "无需人工操作。用纯音下两路差分能量判定真双 capsule 还是单声道复制。")
    ap.add_argument("--tone-freq", type=float, default=300.0, help="纯音频率（--selftone，默认 300Hz）")
    ap.add_argument("--tone-gain", type=float, default=0.08, help="纯音幅度（--selftone，默认 0.08）")
    ap.add_argument("--delay", type=float, default=0.0,
                    help="开始录音前等待秒数（给你时间放好声源、准备吹风）")
    args = ap.parse_args()

    if args.delay > 0:
        import time as _time
        print(f"将在 {args.delay:.0f}s 后开始录音，请现在准备声源/吹风姿势……")
        _time.sleep(args.delay)

    if args.selftone:
        x, y, fs = _record_selftone(args.alsa_device, args.fs, args.seconds,
                                    args.tone_freq, args.tone_gain)
    elif args.alsa_device is not None:
        x, y, fs, ch = _record_arecord(args.alsa_device, args.fs, args.seconds)
    else:
        import sounddevice as sd

        ch = _effective_input_channels(args.device, 2)
        print(f"输入设备可用声道数: {ch}（2 = 立体声，可能是双 capsule）")
        if ch < 2:
            print("单声道输入：不可能做双麦差分。需要换 USB 立体声麦 / 麦克风阵列，")
            print("或按 README 蓝图加 WM8960 I2S HAT + 两支 MEMS 麦。")
            return
        print(f"录音 {args.seconds}s…… 请在这段时间里依次：")
        print("  1) 播放一个稳态纯音/风扇声（0–1.5s）")
        print("  2) 对麦克风缓慢吹气/晃手制造气流（1.5–3s）")
        rec = sd.rec(int(args.seconds * args.fs), samplerate=args.fs, channels=ch,
                     device=args.device, dtype="float32")
        sd.wait()
        fs = args.fs
        x = rec[:, 0].astype(np.float64)
        y = rec[:, 1].astype(np.float64)

    if args.out:
        import soundfile as sf

        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(out_path), np.stack([x, y], axis=1).astype(np.float32), int(fs))
        print(f"已保存双声道原始音频: {out_path}")

    x, y = x - x.mean(), y - y.mean()
    rms_x = float(np.sqrt(np.mean(x ** 2)))
    rms_y = float(np.sqrt(np.mean(y ** 2)))
    diff = x - y
    diff_rms = float(np.sqrt(np.mean(diff ** 2)))
    diff_db = 20.0 * np.log10(max(diff_rms, 1e-12) / max(max(rms_x, rms_y), 1e-12))
    corr = float(np.corrcoef(x, y)[0, 1])

    bands = {"<100Hz": (0, 100), "100-500Hz": (100, 500),
             "500-2kHz": (500, 2000), ">2kHz": (2000, args.fs / 2)}
    coh = _band_coherence(x, y, args.fs, bands)

    print("\n========== 左右声道分析 ==========")
    print(f"L RMS {rms_x:.4f} | R RMS {rms_y:.4f} | 差分 RMS {diff_rms:.4f}"
          f"（{diff_db:+.1f} dB）")
    print(f"全局相关系数: {corr:.4f}")
    print("频段相干性（幅度平方相干，1=完全相干，0=不相关）:")
    for name, v in coh.items():
        print(f"  {name:<12} {v:.3f}")

    print("\n========== 判定 ==========")
    if corr > 0.999 and diff_db < -30:
        print("很可能是『单声道复制』：左右声道几乎逐样本相同。"
              "想双麦差分必须换真双 capsule 设备（USB 立体声麦/阵列）或加 WM8960 HAT。")
    elif args.selftone:
        # 纯音自测：差分能量已在 _record_selftone 报告，这里补相干性判读
        if diff_db > -60:
            print(f"纯音下差分能量 {diff_db:+.1f} dB（远高于量化噪声）→ 真双 capsule："
                  "两路是独立 capsule，可做差分风噪检测。")
        else:
            print(f"纯音下差分能量 {diff_db:+.1f} dB（接近量化噪声）→ 疑似单声道复制。")
        print("纯音频段相干性：", "  ".join(f"{k}={v:.2f}" for k, v in coh.items()))
    elif coh[">2kHz"] < 0.6:
        print("疑似『真双 capsule』且中高频不相关：吹气时中高频相干性低，"
              "说明两路听到的是不同湍流——这就是现成的紧邻双麦，可做差分风噪检测。")
        print("（如果纯音段相干高、吹气段相干低，判定更稳。）")
    else:
        print("声道数=2 但无法判定（可能双 capsule 间距太近或信号太弱）。"
              "重跑一次，确保纯音足够响、吹气足够明显。")
    if not args.selftone:
        print("\n提示：真双 capsule 的『分家』现象是——吹气时 >500Hz 相干性明显低于纯音段。")


if __name__ == "__main__":
    main()
