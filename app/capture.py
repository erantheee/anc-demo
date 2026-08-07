"""音频采集。优先 sounddevice（USB 麦克风），Pi 上可用 arecord 兜底。"""
from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np

# 信号有效性阈值：低于此 RMS(dBFS) 视为"无有效信号"（麦克风悬空/未接）
MIN_VALID_RMS_DB = -60.0


def list_devices() -> list[dict]:
    try:
        import sounddevice as sd
        return [{"index": i, "name": d["name"], "channels": d["max_input_channels"]}
                for i, d in enumerate(sd.query_devices())]
    except Exception:
        return []


def input_devices() -> list[dict]:
    """列出所有有输入通道的音频设备。"""
    try:
        import sounddevice as sd
        return [{"index": i, "name": d["name"], "channels": int(d["max_input_channels"])}
                for i, d in enumerate(sd.query_devices())
                if d["max_input_channels"] > 0]
    except Exception:
        return []


def default_input_device() -> dict | None:
    """返回默认输入设备；无任何可用输入设备返回 None。"""
    devs = input_devices()
    if not devs:
        return None
    try:
        import sounddevice as sd
        idx = sd.default.device[0]
        for d in devs:
            if d["index"] == idx:
                return d
    except Exception:
        pass
    return devs[0]


def record(duration: float, fs: int = 48000, channels: int = 2,
           device: str | int | None = None,
           out_path: str | Path | None = None) -> Path:
    """用 sounddevice 录音到 WAV。返回输出文件路径。

    默认 48kHz 双声道（常见 USB 麦克风/声卡）。多声道录音会写多声道 WAV，
    分析阶段统一取均值降为单声道。找不到输入设备时抛 RuntimeError。
    """
    try:
        import sounddevice as sd
        import soundfile as sf
    except ImportError:
        raise RuntimeError(
            "需要 sounddevice + soundfile：pip install -e .[audio]；"
            "或使用 record_with_arecord() 走 ALSA 兜底"
        )

    if device is None:
        inp = default_input_device()
        if inp is None:
            raise RuntimeError(
                "未检测到任何麦克风/输入设备。请接入 USB 麦克风（或 I2S 编解码器）后重试"
            )
        device = inp["name"]

    out_path = Path(out_path) if out_path else Path(f"data/recordings/rec-{int(__import__('time').time())}.wav")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rec = sd.rec(int(duration * fs), samplerate=fs, channels=channels, dtype="float32", device=device)
    sd.wait()
    sf.write(str(out_path), rec, fs)
    return out_path


def record_buffer(duration: float, fs: int = 48000, channels: int = 2,
                  device: str | int | None = None,
                  require_valid_signal: bool = True) -> np.ndarray:
    """内存录音，返回单声道 float64 ndarray。

    require_valid_signal=True（默认）时，若找不到可用输入设备、或录到的信号
    低于 MIN_VALID_RMS_DB（悬空/未接麦克风），抛 RuntimeError——避免把
    "无信号"误报成环境分贝。
    """
    try:
        import sounddevice as sd
    except ImportError:
        raise RuntimeError("需要 sounddevice：pip install -e .[audio]")

    if device is None:
        inp = default_input_device()
        if inp is None:
            raise RuntimeError(
                "未检测到任何麦克风/输入设备。请接入 USB 麦克风（或 I2S 编解码器），"
                "或用勾选/设置 ANC_SYNTHETIC=1 走合成模式"
            )
        device = inp["name"]

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
    samples = np.asarray(rec, dtype=np.float64)

    if require_valid_signal:
        rms = np.sqrt(np.mean(samples ** 2))
        if rms <= 0 or 20.0 * np.log10(max(rms, 1e-30)) < MIN_VALID_RMS_DB:
            raise RuntimeError(
                f"输入信号无效（RMS {20.0 * np.log10(max(rms, 1e-30)):.0f} dBFS，"
                "低于可用阈值）。麦克风可能未连接或线缆悬空，请检查后重试"
            )
    return samples


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
