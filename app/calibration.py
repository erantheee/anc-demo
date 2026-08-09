"""麦克风灵敏度标定：记录 dBFS → 真实 dB SPL 的偏移，并在进程内缓存生效。

标定方法（与 scripts/calibrate_mic.py 一致）：用手机声级计 APP 在麦克风处读出
真实 SPL，程序算出 offset_db = known_spl - measured_rms_db，之后
`analyze(..., calibration_offset_db=offset)` 就输出绝对 dB SPL。

- `calibrate()` 现场采集并计算偏移，写入 `data/reports/calibration.json`，
  同时更新进程内缓存（Monitor / Grid / Agent 下次采样即生效）。
- `get_offset_db()` / `info()` 供各模块读取；模块导入时自动从磁盘加载。
- 合成模式（ANC_SYNTHETIC=1）下无法标定，因为合成噪声没有真实声压。
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path

from app import capture
from app.analyze import rms_db

CALIB_PATH = Path(__file__).resolve().parent.parent / "data" / "reports" / "calibration.json"

_lock = threading.Lock()
_current_offset_db = 0.0
_info: dict = {}


def _load_from_disk() -> None:
    global _current_offset_db, _info
    try:
        data = json.loads(CALIB_PATH.read_text(encoding="utf-8"))
        _info = data
        _current_offset_db = float(data.get("offset_db", 0.0))
    except Exception:
        _current_offset_db = 0.0
        _info = {}


_load_from_disk()


def get_offset_db() -> float:
    """当前生效的标定偏移（dB）。未标定返回 0.0。"""
    with _lock:
        return _current_offset_db


def info() -> dict:
    """最近一次标定信息：offset_db / known_spl_db / measured_rms_db / calibrated_at。"""
    with _lock:
        return dict(_info)


def calibrate(known_spl: float, duration_s: float = 10.0, fs: int = 16000) -> dict:
    """现场采集一段真实噪声，计算 offset_db 并立即生效。

    无可用麦克风 / 信号无效时抛 RuntimeError（由调用方转为友好错误）。
    """
    samples = capture.record_buffer(duration_s, fs=fs)
    rel = rms_db(samples)
    offset = round(known_spl - rel, 2)
    payload = {
        "offset_db": offset,
        "known_spl_db": known_spl,
        "measured_rms_db": round(rel, 2),
        "calibrated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    CALIB_PATH.parent.mkdir(parents=True, exist_ok=True)
    CALIB_PATH.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    with _lock:
        global _current_offset_db, _info
        _current_offset_db = offset
        _info = payload
    return payload
