"""麦克风标定模块的离线测试（不触发真实录音）。"""
from __future__ import annotations

import json

import pytest

from app import calibration


def test_default_uncalibrated():
    # 默认状态（无标定文件）：偏移 0，未配置
    assert calibration.get_offset_db() == 0.0
    assert calibration.info() == {}


def test_calibrate_writes_and_activates(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration, "CALIB_PATH", tmp_path / "calibration.json")

    def fake_record_buffer(duration_s, fs):
        # 模拟 0.1 RMS（≈ -20 dBFS）的输入信号
        return __import__("numpy").full(int(duration_s * fs), 0.1)

    monkeypatch.setattr(calibration.capture, "record_buffer", fake_record_buffer)
    payload = calibration.calibrate(known_spl=75.0, duration_s=1.0, fs=16000)
    # 75 - (-20) = 95 dB 偏移
    assert payload["offset_db"] == pytest.approx(95.0, abs=1.0)
    assert calibration.get_offset_db() == pytest.approx(payload["offset_db"])
    assert calibration.info()["known_spl_db"] == 75.0
    assert tmp_path.joinpath("calibration.json").exists()


def test_load_from_disk_on_import(tmp_path, monkeypatch):
    monkeypatch.setattr(calibration, "CALIB_PATH", tmp_path / "calibration.json")
    tmp_path.joinpath("calibration.json").write_text(
        json.dumps({"offset_db": 30.0, "known_spl_db": 60.0}), encoding="utf-8")
    calibration._load_from_disk()
    assert calibration.get_offset_db() == 30.0
    assert calibration.info()["known_spl_db"] == 60.0
