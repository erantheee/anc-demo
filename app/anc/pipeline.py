"""ANC 控制器：状态机 + 模式选择 + 评估报告。

离线模式用整段缓冲跑算法；实时模式预留 ALSA（WM8960 I2S）接口。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum, auto

import numpy as np

from app.anc import fxlms, harmonic


class ANCState(Enum):
    IDLE = auto()
    MEASURING = auto()
    ACTIVE = auto()
    STOPPED = auto()


@dataclass
class PositionContext:
    """M3 空间建模后的几何上下文，用于调整 ANC 参数。"""
    printer_pos_m: tuple[float, float, float] | None = None
    quiet_zone_m: tuple[float, float, float] | None = None
    distance_m: float | None = None
    print_active: bool = False

    @classmethod
    def from_positions(cls, printer: tuple[float, float, float],
                       quiet: tuple[float, float, float]) -> "PositionContext":
        d = float(np.linalg.norm(np.asarray(printer) - np.asarray(quiet)))
        return cls(printer_pos_m=printer, quiet_zone_m=quiet, distance_m=d)


class ANCController:
    """演示级 ANC 控制器。mode: "harmonic" 或 "fxlms"。"""

    def __init__(self, mode: str = "harmonic", fs: float = 16000.0,
                 secondary_path: np.ndarray | None = None, **kwargs):
        if mode not in {"harmonic", "fxlms"}:
            raise ValueError(f"未知模式 {mode!r}")
        self.mode = mode
        self.fs = float(fs)
        self.kwargs = kwargs
        self.state = ANCState.IDLE
        self.position: PositionContext | None = None
        self._secondary_path = (secondary_path if secondary_path is not None
                                else np.array([0.4, 0.6, 1.0, 0.6, 0.3]))
        self.last_report: dict = {}

    def configure_position(self, ctx: PositionContext) -> None:
        """接入 M3 空间建模结果：距离越近/安静区越大越值得降噪。"""
        self.position = ctx

    def anc_feasibility_score(self) -> float:
        """0–1，由几何与距离粗略估算（M3 使用）。"""
        if self.position is None or self.position.distance_m is None:
            return 0.5
        d = self.position.distance_m
        # 距离越近，参考相干性越好，越容易降
        return float(np.clip(1.2 - 0.15 * d, 0.1, 1.0))

    def run_offline(self, reference: np.ndarray, error: np.ndarray) -> dict:
        """离线整段处理。reference: 参考麦信号；error: 误差麦信号（含噪声）。"""
        if self.state is not ANCState.IDLE:
            raise RuntimeError(f"状态应为 IDLE，当前 {self.state.name}")
        self.state = ANCState.ACTIVE
        try:
            if self.mode == "harmonic":
                res = harmonic.simulate(np.asarray(error, dtype=np.float64), self.fs,
                                        **self.kwargs)
                res["algorithm"] = "harmonic"
            else:
                kwargs = dict(self.kwargs)
                kwargs.pop("secondary_path", None)
                res = fxlms.simulate(np.asarray(reference, dtype=np.float64), self.fs,
                                     secondary_path=self._secondary_path, **kwargs)
                res["algorithm"] = "fxlms"
            self.last_report = res
            return res
        finally:
            self.state = ANCState.IDLE

    def summary(self) -> dict:
        res = self.last_report
        return {
            "state": self.state.name,
            "algorithm": res.get("algorithm"),
            "reduction_db": round(res.get("reduction_db", 0.0), 2),
            "feasibility_score": round(self.anc_feasibility_score(), 2),
        }
