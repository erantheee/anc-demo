"""输入设备/信号有效性校验测试：无麦克风时必须明确报错，绝不误报分贝。"""
from __future__ import annotations

import numpy as np
import pytest

from app import capture


def test_default_input_device_none_when_no_audio_lib(monkeypatch):
    """没有可输入设备时返回 None（而非凭空一个设备）。"""
    import app.capture as cap

    monkeypatch.setattr(cap, "input_devices", lambda: [])
    assert cap.default_input_device() is None
    assert cap.input_devices() == []


def test_record_buffer_raises_without_input_device(monkeypatch):
    """默认输入设备不存在时 record_buffer 必须抛"未检测到麦克风"。"""
    import app.capture as cap

    monkeypatch.setattr(cap, "default_input_device", lambda: None)
    with pytest.raises(RuntimeError, match="未检测到任何麦克风"):
        cap.record_buffer(0.5)


def test_record_buffer_raises_on_quiet_signal(monkeypatch):
    """信号极弱（悬空/未接麦）必须抛"输入信号无效"，不能返回低电平样本。"""
    import sys
    import app.capture as cap

    quiet = np.zeros(24000, dtype=np.float32).reshape(-1, 1)  # 悬空麦克风的本底

    class _FakeSD:
        @staticmethod
        def rec(*a, **k):
            return quiet

        @staticmethod
        def wait():
            return None

    monkeypatch.setattr(cap, "default_input_device", lambda: {"index": 0, "name": "fake", "channels": 2})
    monkeypatch.setitem(sys.modules, "sounddevice", _FakeSD())

    with pytest.raises(RuntimeError, match="输入信号无效"):
        cap.record_buffer(0.5)


def test_record_buffer_accepts_real_signal(monkeypatch):
    """有真实声音时正常返回单声道样本。"""
    import sys
    import app.capture as cap

    rng = np.random.default_rng(0)
    signal = (0.1 * rng.standard_normal(24000)).astype(np.float32).reshape(-1, 1)

    class _FakeSD:
        @staticmethod
        def rec(*a, **k):
            return signal

        @staticmethod
        def wait():
            return None

    monkeypatch.setattr(cap, "default_input_device", lambda: {"index": 0, "name": "fake", "channels": 2})
    monkeypatch.setitem(sys.modules, "sounddevice", _FakeSD())

    out = cap.record_buffer(0.5)
    assert isinstance(out, np.ndarray)
    assert out.ndim == 1
    assert out.shape[0] == 24000
    assert float(np.sqrt(np.mean(out ** 2))) > 0.05


def test_quiet_gate_threshold_constant():
    """MIN_VALID_RMS_DB 阈值合理性：-60 dBFS 附近为可用边界。"""
    # -60 dBFS ≈ 0.001 线性
    assert abs(capture.MIN_VALID_RMS_DB - (-60.0)) < 1e-6
    near = 0.0012  # ~ -58.4 dBFS，应通过校验
    assert 20.0 * np.log10(near) > capture.MIN_VALID_RMS_DB
