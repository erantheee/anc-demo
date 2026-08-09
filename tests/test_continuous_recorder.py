"""ContinuousRecorder（常驻录音流）环形缓冲正确性测试：不依赖真实硬件。"""
from __future__ import annotations

import sys

import numpy as np

from app import capture


class _FakeStream:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.callback = kwargs["callback"]
        self.started = False
        self.closed = False

    def start(self):
        self.started = True

    def stop(self):
        pass

    def close(self):
        self.closed = True


class _FakeSD:
    def __init__(self):
        self.streams = []

    def InputStream(self, **kwargs):
        stream = _FakeStream(**kwargs)
        self.streams.append(stream)
        return stream


def _make_recorder(monkeypatch, buffer_s=2.0, fs=48000, channels=2, block_s=0.25):
    import app.capture as cap

    fake = _FakeSD()
    monkeypatch.setattr(cap, "default_input_device",
                        lambda: {"index": 0, "name": "fake", "channels": 2})
    monkeypatch.setattr(cap, "_effective_input_channels", lambda *a, **k: channels)
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    rec = cap.ContinuousRecorder(fs=fs, channels=channels,
                                 buffer_s=buffer_s, block_s=block_s)
    rec.open()
    return rec, fake.streams[0]


def _feed(stream, mono_f32):
    block = stream.kwargs["blocksize"]
    for i in range(0, len(mono_f32), block):
        chunk = mono_f32[i:i + block]
        if len(chunk) < block:
            chunk = np.pad(chunk, (0, block - len(chunk)))
        stream.callback(chunk.reshape(-1, 1), len(chunk), None, None)


def test_recorder_reads_last_window(monkeypatch):
    """read(1.0) 返回最近 1 秒，且为单声道 float64。"""
    rec, stream = _make_recorder(monkeypatch, buffer_s=2.0)
    n_total = 2 * rec.fs
    data = np.arange(n_total, dtype=np.float32)
    _feed(stream, data)

    out = rec.read(1.0)
    assert out.shape == (rec.fs,)
    assert out.dtype == np.float64
    assert np.allclose(out, data[rec.fs:])  # 最近 1 秒


def test_recorder_ring_wraps(monkeypatch):
    """缓冲写满后回绕：只保留最近 buffer_s 秒。"""
    rec, stream = _make_recorder(monkeypatch, buffer_s=2.0)
    n_total = 4 * rec.fs  # 超过缓冲长度
    data = np.arange(n_total, dtype=np.float32)
    _feed(stream, data)

    out = rec.read(2.0)
    assert np.allclose(out, data[2 * rec.fs:])  # 最近 2 秒


def test_recorder_insufficient_buffer_raises(monkeypatch):
    """缓冲还没攒够就 read 应抛 RuntimeError。"""
    rec, stream = _make_recorder(monkeypatch, buffer_s=2.0)
    data = np.arange(rec.fs, dtype=np.float32)  # 只喂了 1 秒
    _feed(stream, data)

    import pytest

    with pytest.raises(RuntimeError, match="缓冲不足"):
        rec.read(2.0)


def test_recorder_open_close_lifecycle(monkeypatch):
    """open 打开流、close 关闭流并复位 is_open。"""
    rec, stream = _make_recorder(monkeypatch)
    assert rec.is_open
    assert stream.started

    rec.close()
    assert not rec.is_open
    assert stream.closed
