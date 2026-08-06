"""静音区选择与可行性检查。"""
from __future__ import annotations

import numpy as np

SPEED_OF_SOUND = 343.0  # m/s


def zone_of_quiet_diameter(freq_hz: float) -> float:
    """安静区直径 ≈ λ/10。"""
    return SPEED_OF_SOUND / max(freq_hz, 1.0) / 10.0


def check_feasibility(printer_pos_m: tuple[float, float, float],
                      quiet_pos_m: tuple[float, float, float],
                      dominant_freq_hz: float) -> dict:
    """检查静音点的 ANC 可行性：距离、传播延迟、安静区尺寸。"""
    d = float(np.linalg.norm(np.asarray(printer_pos_m) - np.asarray(quiet_pos_m)))
    diameter = zone_of_quiet_diameter(dominant_freq_hz)
    delay_us = d / SPEED_OF_SOUND * 1e6
    return {
        "distance_m": round(d, 2),
        "propagation_delay_us": round(delay_us, 1),
        "zone_of_quiet_diameter_m": round(diameter, 2),
        "verdict": (
            "good" if d <= 3.0 and diameter >= 0.05
            else "marginal" if d <= 5.0
            else "poor"
        ),
    }


def recommend_quiet_zone(sources: list, origin_m: tuple[float, float, float] = (0, 0, 0),
                         radius_m: float = 2.0) -> tuple[float, float, float] | None:
    """在房间模型里建议静音点：取源的位置向房间原点方向外推。

    简化启发式：Demo 阶段主要用于给用户一个可在地图上点选/调整的起点。
    """
    if not sources:
        return None
    center = np.mean([np.asarray(s.position_m, dtype=float)
                      for s in sources if s.position_m is not None], axis=0)
    origin = np.asarray(origin_m, dtype=float)
    direction = origin - center
    norm = np.linalg.norm(direction)
    if norm < 1e-6:
        direction = np.array([1.0, 0.0, 0.0])
    else:
        direction = direction / norm
    return tuple(float(v) for v in center + direction * radius_m)
