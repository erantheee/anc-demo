"""后台网格测量任务：网格录音 → 分析 → 来源归属 → 插值 surface → 建议静音区。

供 Web 仪表盘触发（/api/grid/measure）。逻辑抽取自 scripts/measure.py，
任务在后台线程运行，进度与结果通过线程安全的状态字典对外暴露。
"""
from __future__ import annotations

import os
import threading
from dataclasses import dataclass, asdict

import numpy as np

from app import capture
from app.analyze import analyze, stable_segment
from app.calibration import get_offset_db
from app.noise_map import GridPoint, build_noise_map
from app.quiet_zone import zone_of_quiet_diameter
from app.source_id import load_profiles, match_sources
from app.synth import printer_noise

SYNTHETIC = os.environ.get("ANC_SYNTHETIC", "0") == "1"

SURFACE_MAX = 40  # 插值表面每维最大点数


@dataclass
class GridResult:
    state: str = "idle"  # idle | running | done | error
    progress: float = 0.0  # 0..1
    current_point: str = ""
    message: str = ""
    points: list[dict] = None  # [{x, y, spl_db, source_hits}]
    surface: dict = None  # {x: [], y: [], z: []}
    dominant_sources: list[dict] = None
    recommendation: dict = None
    quiet_zone: dict = None  # 建议静音区
    error: str = None


class GridWorker:
    """后台网格测量。同一时刻只允许一个任务。"""

    def __init__(self, monitor=None) -> None:
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self.result = GridResult()
        self.monitor = monitor  # 可选：测量期间暂停实时监控，避免抢音频设备

    def is_running(self) -> bool:
        return bool(self._thread and self._thread.is_alive())

    def status(self) -> dict:
        with self._lock:
            return asdict(self.result)

    def start(self, origin_x: float = 0.0, origin_y: float = 0.0,
              size_x: float = 1.0, size_y: float = 1.0, step: float = 0.5,
              per_point_s: float = 10.0, fs: int = 48000,
              synthetic: bool | None = None, height_m: float = 0.5) -> dict:
        if self.is_running():
            return {"started": False, "message": "已有任务在运行"}
        with self._lock:
            self.result = GridResult(state="running", message="任务启动")
        self._thread = threading.Thread(
            target=self._run, name="anc-grid", daemon=True,
            args=(origin_x, origin_y, size_x, size_y, step, per_point_s, fs,
                  synthetic, height_m))
        self._thread.start()
        return {"started": True}

    # ---- 内部 ----

    def _run(self, origin_x: float, origin_y: float, size_x: float, size_y: float,
             step: float, per_point_s: float, fs: int,
             synthetic: bool | None = None, height_m: float = 0.5) -> None:
        use_synth = SYNTHETIC if synthetic is None else synthetic
        xs = np.arange(origin_x, origin_x + size_x + step / 2, step)
        ys = np.arange(origin_y, origin_y + size_y + step / 2, step)
        grid = [(float(x), float(y)) for y in ys for x in xs]
        n = len(grid)

        profiles = load_profiles()
        points: list[GridPoint] = []
        reports = []

        try:
            if self.monitor is not None:
                self.monitor.set_paused(True)
                # 等监控释放常驻录音流后再逐点录音，避免设备被占用
                self.monitor.wait_paused()
            for i, (x, y) in enumerate(grid):
                with self._lock:
                    self.result.progress = i / n
                    self.result.current_point = f"({x:.2f}, {y:.2f})"
                if use_synth:
                    dist = np.hypot(x - origin_x, y - origin_y)
                    samples, _ = printer_noise(fs=fs, duration=per_point_s, seed=i)
                    samples = samples * float(np.clip(1.0 - 0.08 * dist, 0.3, 1.0))
                    report = analyze(samples, fs, calibration_offset_db=get_offset_db())
                else:
                    raw = capture.record_buffer(per_point_s, fs=fs)
                    samples = stable_segment(raw, fs)
                    if len(samples) == 0:
                        with self._lock:
                            self.result.message = f"点 ({x:.2f},{y:.2f}) 无有效信号，跳过"
                        continue
                    report = analyze(samples, fs, calibration_offset_db=get_offset_db())
                hits = match_sources(report, profiles)
                spl = report.spl_db if report.spl_db is not None else report.rms_db
                points.append(GridPoint(x=x, y=y, spl_db=float(spl),
                                        source_hits=[h.__dict__ for h in hits]))
                reports.append(report)

            # 合并来源归属
            agg: dict[str, dict] = {}
            for p in points:
                for h in (p.source_hits or []):
                    cur = agg.get(h["source"])
                    if cur is None or h["confidence"] > cur["confidence"]:
                        agg[h["source"]] = h
            dominant = sorted(agg.values(), key=lambda h: h["confidence"], reverse=True)[:5]

            # 插值 surface
            surface = self._build_surface(points)
            rec = {}
            if reports:
                from app.source_id import recommend_anc
                ref_report = max(reports, key=lambda r: r.rms_db)
                rec = recommend_anc(ref_report)

            # 建议静音区
            qz = self._recommend_quiet_zone(points, dominant, height_m)

            with self._lock:
                self.result.state = "done"
                self.result.progress = 1.0
                self.result.points = [asdict(p) for p in points]
                self.result.surface = surface
                self.result.dominant_sources = dominant
                self.result.recommendation = rec
                self.result.quiet_zone = qz
                self.result.message = "测量完成"
        except Exception as exc:
            with self._lock:
                self.result.state = "error"
                self.result.error = str(exc)
        finally:
            if self.monitor is not None:
                self.monitor.set_paused(False)

    def _build_surface(self, points: list[GridPoint]) -> dict:
        if len(points) < 3:
            return None
        try:
            xi, yi, zi = build_noise_map(points)
        except Exception:
            return None
        # 下采样到 SURFACE_MAX 内
        sx, sy = len(xi), len(yi)
        if sx > SURFACE_MAX:
            step_x = max(1, sx // SURFACE_MAX)
            xi, zi = xi[::step_x], zi[:, ::step_x]
        if sy > SURFACE_MAX:
            step_y = max(1, sy // SURFACE_MAX)
            yi, zi = yi[::step_y], zi[::step_y]
        # 掩掉 NaN（测量网格外无数据）
        nan_mask = np.isnan(zi)
        z_filled = zi.copy()
        z_filled[nan_mask] = 0.0
        return {
            "x": [round(float(v), 3) for v in xi],
            "y": [round(float(v), 3) for v in yi],
            "z": [[None if m else round(float(v), 1) for m, v in zip(row_nan, row)]
                  for row_nan, row in zip(nan_mask, z_filled)],
        }

    def _recommend_quiet_zone(self, points: list[GridPoint], dominant: list[dict],
                              height_m: float = 0.5) -> dict:
        """建议静音区：离噪声源约 0.5m 处（安静区直径 ~λ/10 内）。

        source_pos_m / quiet_pos_m 均为房间系 3D 坐标 [x, y, z]：
        z 取网格测量高度 height_m（即误差麦克风/打印机噪声源离地高度）。
        """
        if not points:
            return None
        # 噪声源 = SPL 最高的点（若用户未手动标记）
        loudest = max(points, key=lambda p: p.spl_db)
        src = (loudest.x, loudest.y, height_m)
        dist = 0.5  # 建议离源 0.5m
        cx, cy, cz = src
        qz_pos = (cx + dist, cy, cz)
        # 静音区直径基于主频（取最强来源的峰值频率，默认 120Hz 步进电机）
        fund = 120.0
        for d in (dominant or []):
            freqs = d.get("freqs_hz") or []
            if freqs:
                fund = float(freqs[0])
                break
        diameter = zone_of_quiet_diameter(fund)
        return {
            "source_pos_m": list(src),
            "quiet_pos_m": list(qz_pos),
            "distance_m": dist,
            "zone_of_quiet_diameter_m": round(diameter, 2),
            "dominant_freq_hz": round(fund, 1),
        }
