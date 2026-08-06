"""空间建模：房间坐标系 + 噪声源 + 静音区。

M3 起用摄像头（平面检测 + ArUco）建模；未来可接入 RPLIDAR 2D 地图
（lidar_map 字段）与麦克风阵列 DOA。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class NoiseSource:
    id: str
    name: str
    position_m: tuple[float, float, float] | None = None
    profile_id: str | None = None
    active: bool = True

    def __post_init__(self) -> None:
        if self.position_m is not None:
            self.position_m = tuple(float(v) for v in self.position_m)


@dataclass
class QuietZone:
    id: str
    position_m: tuple[float, float, float]
    target_freq_hz: float | None = None

    def __post_init__(self) -> None:
        self.position_m = tuple(float(v) for v in self.position_m)


@dataclass
class RoomModel:
    name: str
    sources: list[NoiseSource] = field(default_factory=list)
    quiet_zones: list[QuietZone] = field(default_factory=list)
    lidar_map: list[dict] | None = None
    camera_floor_plane: list[float] | None = None  # [a,b,c,d]：a·x+b·y+c·z+d=0
    camera_pose_m: tuple[float, float, float] = (0.0, 0.0, 0.0)
    camera_yaw_deg: float = 0.0

    def add_source(self, source: NoiseSource) -> None:
        self.sources.append(source)

    def add_quiet_zone(self, zone: QuietZone) -> None:
        self.quiet_zones.append(zone)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    @classmethod
    def load(cls, path: str | Path) -> "RoomModel":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(
            name=data["name"],
            sources=[NoiseSource(**s) for s in data.get("sources", [])],
            quiet_zones=[QuietZone(**z) for z in data.get("quiet_zones", [])],
            lidar_map=data.get("lidar_map"),
            camera_floor_plane=data.get("camera_floor_plane"),
            camera_pose_m=tuple(data.get("camera_pose_m", (0, 0, 0))),
            camera_yaw_deg=data.get("camera_yaw_deg", 0.0),
        )

    def estimate_print_active(self, image_motion_score: float, threshold: float = 0.05) -> bool:
        """用帧间运动估计打印机是否在打印（M3 摄像头逻辑的占位接口）。"""
        return image_motion_score > threshold
