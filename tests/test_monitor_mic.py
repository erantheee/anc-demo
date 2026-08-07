"""Monitor 在无麦克风/信号无效时，必须报 mic_ok=False 且清空分贝，绝不误报。"""
from __future__ import annotations

import time

import pytest


@pytest.fixture
def no_mic_monitor(monkeypatch):
    from app import monitor as mon
    import app.capture as cap

    monkeypatch.setattr(cap, "record_buffer", lambda *a, **k: (_ for _ in ()).throw(
        RuntimeError("未检测到任何麦克风/输入设备。请接入 USB 麦克风（或 I2S 编解码器），"
                     "或用勾选/设置 ANC_SYNTHETIC=1 走合成模式")))
    m = mon.Monitor(sample_s=0.5, interval_s=0.01)
    return m


def test_monitor_reports_no_mic(no_mic_monitor):
    m = no_mic_monitor
    m.start()
    try:
        deadline = time.time() + 5
        snap = None
        while time.time() < deadline:
            snap = m.snapshot()
            if snap["mic_ok"] is False:
                break
            time.sleep(0.05)
        assert snap is not None
        assert snap["mic_ok"] is False
        assert snap["spl_db"] is None
        assert snap["rms_db"] is None
        assert snap["dominant_freq"] is None
        assert snap["error"] and "麦克风" in snap["error"]
    finally:
        m.stop()


def test_monitor_good_signal_sets_mic_ok(monkeypatch):
    from app import monitor as mon
    import numpy as np
    import app.capture as cap

    rng = np.random.default_rng(1)
    good = (0.05 * rng.standard_normal(24000)).astype(np.float32)
    monkeypatch.setattr(cap, "record_buffer", lambda *a, **k: good)
    m = mon.Monitor(sample_s=0.5, interval_s=0.01)
    m.start()
    try:
        deadline = time.time() + 5
        snap = None
        while time.time() < deadline:
            snap = m.snapshot()
            if snap["mic_ok"] is True:
                break
            time.sleep(0.05)
        assert snap is not None
        assert snap["mic_ok"] is True
        assert snap["spl_db"] is not None
        assert snap["error"] is None
    finally:
        m.stop()
