"""摄像头定位：ArUco 标记 → 打印机 6DoF 位姿。

需要 opencv-python（pip install -e .[vision]）。未标定相机时给出近似位姿，
精度足够 Demo（<30 cm 目标）；精确标定流程见 scripts/calibrate-camera.py。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def _default_camera_matrix(image_width: int, image_height: int) -> np.ndarray:
    """未标定时用近似内参（f ≈ 1.2×宽）。"""
    f = 1.2 * image_width
    return np.array([[f, 0, image_width / 2],
                     [0, f, image_height / 2],
                     [0, 0, 1]], dtype=np.float64)


def detect_marker_pose(image: np.ndarray, marker_size_m: float = 0.10,
                       camera_matrix: np.ndarray | None = None,
                       dist_coeffs: np.ndarray | None = None,
                       dictionary: str = "DICT_6X6_250") -> dict | None:
    """检测第一个 ArUco 标记，返回 {tvec_m, rvec, yaw_deg, marker_id} 或 None。"""
    import cv2

    dicts = {"DICT_4X4_50": cv2.aruco.DICT_4X4_50,
             "DICT_6X6_250": cv2.aruco.DICT_6X6_250}
    aruco_dict = cv2.aruco.getPredefinedDictionary(dicts.get(dictionary, cv2.aruco.DICT_6X6_250))
    detector = cv2.aruco.ArucoDetector(aruco_dict, cv2.aruco.DetectorParameters())
    corners, ids, _ = detector.detectMarkers(image)

    if ids is None or len(ids) == 0:
        return None

    h, w = image.shape[:2]
    if camera_matrix is None:
        camera_matrix = _default_camera_matrix(w, h)
    if dist_coeffs is None:
        dist_coeffs = np.zeros((5, 1))

    rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
        corners, marker_size_m, camera_matrix, dist_coeffs)
    i = 0
    tvec = np.asarray(tvecs[i][0], dtype=float)
    rvec = np.asarray(rvecs[i][0], dtype=float)
    rmat, _ = cv2.Rodrigues(rvec)
    yaw_deg = float(np.degrees(np.arctan2(rmat[1, 0], rmat[0, 0])))
    return {
        "marker_id": int(ids[i][0]),
        "tvec_m": tvec.tolist(),
        "rvec": rvec.tolist(),
        "yaw_deg": yaw_deg,
        "distance_m": float(np.linalg.norm(tvec)),
    }


def camera_to_room(pose: dict, camera_pose_m: tuple[float, float, float] = (0, 0, 0),
                   camera_yaw_deg: float = 0.0) -> tuple[float, float, float]:
    """把相机系位姿转到房间系（相机位置 + 朝向为外参）。"""
    t = np.asarray(pose["tvec_m"], dtype=float)
    yaw = np.radians(camera_yaw_deg)
    R = np.array([[np.cos(yaw), -np.sin(yaw), 0],
                  [np.sin(yaw), np.cos(yaw), 0],
                  [0, 0, 1]], dtype=float)
    room = R @ t + np.asarray(camera_pose_m, dtype=float)
    return tuple(float(v) for v in room)
