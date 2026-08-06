"""网格测量 → 2D 噪声地图（插值 + PNG + JSON 导出）。"""
from __future__ import annotations

import json
from dataclasses import dataclass, asdict
from pathlib import Path

import numpy as np
from scipy.interpolate import griddata


@dataclass
class GridPoint:
    x: float
    y: float
    spl_db: float
    source_hits: list[dict] | None = None


def build_noise_map(points: list[GridPoint], resolution: float = 0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xs = np.array([p.x for p in points], dtype=float)
    ys = np.array([p.y for p in points], dtype=float)
    zs = np.array([p.spl_db for p in points], dtype=float)
    xi = np.arange(xs.min(), xs.max() + resolution, resolution)
    yi = np.arange(ys.min(), ys.max() + resolution, resolution)
    zi = griddata((xs, ys), zs, (xi[None, :], yi[:, None]), method="linear")
    return xi, yi, zi


def save_png(xi: np.ndarray, yi: np.ndarray, zi: np.ndarray, path: str | Path) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    path = Path(path)
    fig, ax = plt.subplots(figsize=(6, 5))
    cf = ax.contourf(xi, yi, zi, levels=24, cmap="viridis")
    ax.set_xlabel("x (m)")
    ax.set_ylabel("y (m)")
    ax.set_title("Room noise map (A-weighted SPL, dB)")
    ax.scatter(xi[0], yi[0], marker="+", s=0)
    cbar = fig.colorbar(cf)
    cbar.set_label("dB")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path


def write_report(points: list[GridPoint], report: dict, out_dir: str | Path) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "grid": [asdict(p) for p in points],
        **report,
    }
    (out_dir / "report.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return payload
