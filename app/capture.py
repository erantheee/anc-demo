"""音频采集。优先 sounddevice（USB 麦克风），Pi 上可用 arecord 兜底。"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np


def list_devices() -> list[dict]:
    try:
        import sounddevice as sd
        return [{"index": i, "name": d["name"], "channels": d["max_input_channels"]}
                for i, d in enumerate(sd.query_devices())]
    except Exception:
        return []


def record(duration: float, fs: int = 48000, channels: int = 2,
           device: str | int | None = None,
           out_path: str | Path | None = None) -> Path:
    """用 sounddevice 录音到 WAV。返回输出文件路径。

    默认 48kHz 双声道（常见 USB 麦克风/声卡）。多声道录音会写多声道 WAV，
    分析阶段统一取均值降为单声道。
    """
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        raise RuntimeError(
            "需要 sounddevice + soundfile：pip install -e .[audio]；"
            "或使用 record_with_arecord() 走 ALSA 兜底"
        )

    out_path = Path(out_path) if out_path else Path(f"data/recordings/rec-{int(__import__('time').time())}.wav")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rec = sd.rec(int(duration * fs), samplerate=fs, channels=channels, dtype="float32", device=device)
    sd.wait()
    sf.write(str(out_path), rec, fs)
    return out_path


def record_buffer(duration: float, fs: int = 48000, channels: int = 2,
                  device: str | int | None = None) -> np.ndarray:
    """内存录音，返回 (n, channels) float32 ndarray。用于高频采样，不写磁盘。"""
    try:
        import sounddevice as sd
    except ImportError:
        raise RuntimeError("需要 sounddevice：pip install -e .[audio]")
    try:
        rec = sd.rec(int(duration * fs), samplerate=fs, channels=channels,
                     dtype="float32", device=device)
        sd.wait()
    except Exception as exc:
        raise RuntimeError(
            f"录音设备不可用（{exc}）。请检查麦克风连接，"
            "或用勾选/设置 ANC_SYNTHETIC=1 走合成模式"
        ) from exc
    if rec.ndim > 1:
        rec = rec.mean(axis=1)
    return np.asarray(rec, dtype=np.float64)


def record_with_arecord(duration: float, fs: int = 16000, channels: int = 1,
                        out_path: str | Path | None = None) -> Path:
    """Pi 兜底：arecord → WAV（需 `sudo apt install alsa-utils`）。"""
    out_path = Path(out_path) if out_path else Path(f"data/recordings/rec-{int(__import__('time').time())}.wav")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "arecord", "-f", "S16_LE", "-r", str(fs), "-c", str(channels),
        "-d", str(int(duration)), str(out_path),
    ]
    subprocess.run(cmd, check=True)
    return out_path
