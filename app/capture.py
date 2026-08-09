"""音频采集。优先 sounddevice（USB 麦克风），Pi 上可用 arecord 兜底。"""
from __future__ import annotations

import subprocess
import threading
from pathlib import Path

import numpy as np

# 信号有效性阈值：低于此 RMS(dBFS) 视为"无有效信号"（麦克风悬空/未接）
MIN_VALID_RMS_DB = -60.0


def ensure_valid_signal(samples: np.ndarray) -> np.ndarray:
    """信号有效性校验：RMS 过低或无能量时抛 RuntimeError（与 record_buffer 同口径）。

    返回原样本，供调用方链式使用。
    """
    rms = np.sqrt(np.mean(samples ** 2))
    if rms <= 0 or 20.0 * np.log10(max(rms, 1e-30)) < MIN_VALID_RMS_DB:
        raise RuntimeError(
            f"输入信号无效（RMS {20.0 * np.log10(max(rms, 1e-30)):.0f} dBFS，"
            "低于可用阈值）。麦克风可能未连接或线缆悬空，请检查后重试"
        )
    return samples


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


def _effective_input_channels(device: str | int | None, channels: int) -> int:
    """把请求的录音通道数收敛到设备实际支持的最大输入通道数。

    Mac 内置麦克风是单声道，请求 2 声道会让 PortAudio/CoreAudio 报
    "Invalid number of channels"；Pi 的立体声 USB 麦克风（2 声道）不受影响。
    """
    try:
        import sounddevice as sd
        max_in = int(sd.query_devices(device)["max_input_channels"])
        if max_in > 0:
            return min(int(channels), max_in)
    except Exception:
        pass
    return int(channels)


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

    rec = sd.rec(int(duration * fs), samplerate=fs, channels=_effective_input_channels(device, channels),
                 dtype="float32", device=device)
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
        rec = sd.rec(int(duration * fs), samplerate=fs, channels=_effective_input_channels(device, channels),
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
        ensure_valid_signal(samples)
    return samples


class ContinuousRecorder:
    """常驻输入流：打开一次、持续采集，按需取最近 N 秒音频。

    相比每次采样开关一次录音流（record/record_buffer），常驻流只在启动/停止时
    开关设备。廉价 USB 声卡在流开关瞬间可能在模拟输出上产生爆音，常驻流
    消除周期性开关；采样节奏（每隔 interval 秒取 sample_s 秒分析）不变。

    - 线程安全：PortAudio 回调线程写入环形缓冲，read() 可从任意线程取快照。
    - 使用前须 open()，结束后 close()（暂停/停止监控时释放设备，供 ANC、网格、
      标定等任务独占使用）。
    - read() 在缓冲不足时抛 RuntimeError（如刚打开、或长时间无人取数被挤掉）。
    """

    def __init__(self, fs: int = 48000, channels: int = 2,
                 device: str | int | None = None,
                 buffer_s: float = 90.0, block_s: float = 0.25) -> None:
        self.fs = int(fs)
        self.channels = int(channels)
        self.device = device
        self._block_s = float(block_s)
        self._ring_len = int(buffer_s * self.fs)
        self._ring = np.zeros(self._ring_len, dtype=np.float32)
        self._pos = 0
        self._filled = 0
        self._stream = None
        self._lock = threading.Lock()

    @property
    def is_open(self) -> bool:
        return self._stream is not None

    def open(self) -> None:
        import sounddevice as sd

        if self._stream is not None:
            return
        if self.device is None:
            inp = default_input_device()
            if inp is None:
                raise RuntimeError(
                    "未检测到任何麦克风/输入设备。请接入 USB 麦克风（或 I2S 编解码器），"
                    "或用勾选/设置 ANC_SYNTHETIC=1 走合成模式"
                )
            self.device = inp["name"]
        ch = _effective_input_channels(self.device, self.channels)
        self._stream = sd.InputStream(
            samplerate=self.fs, channels=ch, dtype="float32",
            device=self.device, blocksize=int(self._block_s * self.fs),
            callback=self._on_audio)
        self._stream.start()

    def _on_audio(self, indata, frames, time_info, status) -> None:
        """回调线程：写环形缓冲。多声道取均值降为单声道。"""
        data = np.asarray(indata, dtype=np.float32)
        if data.ndim > 1:
            data = data.mean(axis=1)
        n = len(data)
        with self._lock:
            if n >= self._ring_len:
                self._ring[:] = data[-self._ring_len:]
                self._filled = self._ring_len
                self._pos = 0
                return
            end = self._pos + n
            if end <= self._ring_len:
                self._ring[self._pos:end] = data
            else:
                head = self._ring_len - self._pos
                self._ring[self._pos:] = data[:head]
                self._ring[:end - self._ring_len] = data[head:]
            self._pos = end % self._ring_len
            self._filled = min(self._ring_len, self._filled + n)

    def read(self, duration: float) -> np.ndarray:
        """返回最近 duration 秒的单声道 float64 样本。缓冲不足时抛 RuntimeError。"""
        n = int(duration * self.fs)
        with self._lock:
            filled = self._filled
            ring = self._ring.copy()
            pos = self._pos
        if filled < n:
            raise RuntimeError(
                f"常驻录音缓冲不足（已有 {filled / self.fs:.1f}s，需要 {duration:.1f}s）。"
                "麦克风可能刚打开，请稍后重试"
            )
        start = (pos - n) % self._ring_len
        if start + n <= self._ring_len:
            out = ring[start:start + n]
        else:
            out = np.concatenate([ring[start:], ring[:(start + n) % self._ring_len]])
        return np.asarray(out, dtype=np.float64)

    def close(self) -> None:
        stream, self._stream = self._stream, None
        if stream is None:
            return
        try:
            stream.stop()
        except Exception:
            pass
        try:
            stream.close()
        except Exception:
            pass


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
