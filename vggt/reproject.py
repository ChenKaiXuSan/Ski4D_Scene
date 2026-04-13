#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/vggt/reproject.py
Project: /workspace/code/vggt
Created Date: Saturday November 22nd 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Saturday November 22nd 2025 5:47:43 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date       By  Comments
---------- --- ---------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple

import cv2
import numpy as np
import logging

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------- #
# Utils
# --------------------------------------------------------------------- #
def _as_float32(x) -> np.ndarray:
    return np.asarray(x, dtype=np.float32)


def _ensure_bgr_uint8(img: np.ndarray) -> np.ndarray:
    """HxW or HxWx1/3, any dtype -> HxWx3 BGR uint8"""
    im = np.asarray(img)
    if im.ndim == 2:
        im = cv2.cvtColor(im, cv2.COLOR_GRAY2BGR)
    elif im.ndim == 3 and im.shape[2] == 1:
        im = np.repeat(im, 3, axis=2)
    elif im.ndim == 3 and im.shape[2] == 3:
        pass
    else:
        raise ValueError(f"Unsupported image shape: {im.shape}")

    if im.dtype != np.uint8:
        im = np.clip(im, 0, 255)
        if im.max() <= 1.0:
            im = (im * 255.0).astype(np.uint8)
        else:
            im = im.astype(np.uint8)
    return im


def _clip_xy(xy: np.ndarray, w: int, h: int) -> np.ndarray:
    """clip (N,2) to [0,w-1]/[0,h-1]"""
    out = xy.copy()
    out[:, 0] = np.clip(out[:, 0], 0, w - 1)
    out[:, 1] = np.clip(out[:, 1], 0, h - 1)
    return out


# --------------------------------------------------------------------- #
# Reprojection core
# --------------------------------------------------------------------- #
def reproject_points(
    X3: np.ndarray,      # (J,3) 3D joints in WORLD coordinates
    K1: np.ndarray,
    dist1: Optional[np.ndarray],
    K2: np.ndarray,
    dist2: Optional[np.ndarray],
    R: np.ndarray,
    T: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    使用相机外参进行重投影。

    支持两种模式：
      1) VGGT 风格（推荐）:
         R, T 形状为 (2,3,3)、(2,3)，表示两个相机的世界→相机外参:
             X_cam = R * X_world + T

      2) 传统 stereo 风格:
         R, T 形状为 (3,3)、(3,)，表示 Cam1→Cam2 的相对外参:

    Cam1 始终作为参考相机 (rvec=0, tvec=0)，X3 被认为在“世界坐标”
    下已经与 Cam1 的外参对齐。
    """
    X3 = _as_float32(X3).reshape(-1, 1, 3)

    R = np.asarray(R, np.float32)
    T = np.asarray(T, np.float32)

    # ---- 解析相机外参 ----
    if R.ndim == 3:
        # VGGT 模式: R,T 为 (2,3,3)、(2,3)，分别对应 cam1, cam2 的世界→相机外参
        if R.shape[0] != 2 or T.shape[0] != 2:
            raise ValueError(f"Expect R,T shape (2,3,3),(2,3), got {R.shape}, {T.shape}")

        R1 = R[0]          # world -> cam1
        t1 = T[0].reshape(3, 1)

        R2 = R[1]          # world -> cam2
        t2 = T[1].reshape(3, 1)

        # cam1 -> cam2 的相对外参
        # X_cam1 = R1 X_world + t1
        # X_cam2 = R2 X_world + t2
        # => X_cam2 = R2 R1^T X_cam1 + (t2 - R2 R1^T t1)
        R_rel = R2 @ R1.T
        t_rel = t2 - R_rel @ t1

    elif R.ndim == 2:
        # 传统 stereo 模式: R,T 已经是 cam1->cam2 的外参
        if R.shape != (3, 3) or T.shape not in [(3,), (3, 1)]:
            raise ValueError(f"Expect R(3,3), T(3,) for relative extrinsic, got {R.shape}, {T.shape}")
        R_rel = R
        t_rel = T.reshape(3, 1)
    else:
        raise ValueError(f"Unsupported R shape: {R.shape}")

    # ---- Cam1: 视作世界对齐相机，rvec,tvec = 0 ----
    rvec1 = np.zeros((3, 1), np.float32)
    tvec1 = np.zeros((3, 1), np.float32)

    # ---- Cam2: 使用相对外参 ----
    rvec2, _ = cv2.Rodrigues(R_rel)
    tvec2 = t_rel.astype(np.float32)

    K1 = _as_float32(K1).reshape(3, 3)
    K2 = _as_float32(K2).reshape(3, 3)
    d1 = None if dist1 is None else _as_float32(dist1).reshape(-1, 1)
    d2 = None if dist2 is None else _as_float32(dist2).reshape(-1, 1)

    proj1, _ = cv2.projectPoints(X3, rvec1, tvec1, K1, d1)
    proj2, _ = cv2.projectPoints(X3, rvec2, tvec2, K2, d2)

    return {
        "proj_L": proj1.reshape(-1, 2),
        "proj_R": proj2.reshape(-1, 2),
    }


# --------------------------------------------------------------------- #
# Visualization
# --------------------------------------------------------------------- #
def render_reprojection_panel(
    img1: np.ndarray,
    img2: np.ndarray,
    kptL: np.ndarray,      # (J,2) observed pixels in left/cam1
    kptR: np.ndarray,      # (J,2) observed pixels in right/cam2
    proj_L: np.ndarray,    # (J,2) reprojected pixels
    proj_R: np.ndarray,    # (J,2)
    joint_names: Optional[Sequence[str]] = None,
    circle_r: int = 5,
    thickness: int = 2,
    align_height: bool = True,
    title_left: str = "Left/Cam1 (Green=Observed, Red=Reprojected)",
    title_right: str = "Right/Cam2",
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    在左右图上绘制观测点(绿)、重投影点(红)和误差向量(青)，并拼接为对比图。
    返回:
      vis_left, vis_right, panel
    """
    vis1 = _ensure_bgr_uint8(img1)
    vis2 = _ensure_bgr_uint8(img2)

    if align_height and vis1.shape[0] != vis2.shape[0]:
        # 对齐高度，宽度按比例缩放
        target_h = max(vis1.shape[0], vis2.shape[0])
        if vis1.shape[0] != target_h:
            scale = target_h / vis1.shape[0]
            vis1 = cv2.resize(
                vis1,
                (int(vis1.shape[1] * scale), target_h),
                interpolation=cv2.INTER_LINEAR,
            )
        if vis2.shape[0] != target_h:
            scale = target_h / vis2.shape[0]
            vis2 = cv2.resize(
                vis2,
                (int(vis2.shape[1] * scale), target_h),
                interpolation=cv2.INTER_LINEAR,
            )

    # 绘制函数
    def _draw(vis: np.ndarray, obs: np.ndarray, rep: np.ndarray) -> np.ndarray:
        h, w = vis.shape[:2]
        obs = np.asarray(obs, dtype=float).reshape(-1, 2)
        rep = np.asarray(rep, dtype=float).reshape(-1, 2)
        rep = _clip_xy(rep, w, h)

        for j, (o, r) in enumerate(zip(obs, rep)):
            if not (np.all(np.isfinite(o)) and np.all(np.isfinite(r))):
                continue
            o_i = (int(round(o[0])), int(round(o[1])))
            r_i = (int(round(r[0])), int(round(r[1])))
            # 观测点-绿色
            cv2.circle(vis, o_i, circle_r, (0, 255, 0), thickness, cv2.LINE_AA)
            # 重投影-红色
            cv2.circle(vis, r_i, circle_r, (0, 0, 255), thickness, cv2.LINE_AA)
            # 误差向量-青色
            cv2.line(vis, o_i, r_i, (255, 255, 0), 1, cv2.LINE_AA)
            # 标注
            label = (
                str(joint_names[j])
                if (joint_names is not None and j < len(joint_names))
                else str(j)
            )
            cv2.putText(
                vis,
                label,
                (o_i[0] + 6, o_i[1] - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA,
            )
        return vis

    visL = _draw(vis1.copy(), kptL, proj_L)
    visR = _draw(vis2.copy(), kptR, proj_R)

    # 计算误差统计
    errL = np.linalg.norm(np.asarray(proj_L, float) - np.asarray(kptL, float), axis=1)
    errR = np.linalg.norm(np.asarray(proj_R, float) - np.asarray(kptR, float), axis=1)
    rmseL = float(np.sqrt(np.nanmean(errL**2)))
    rmseR = float(np.sqrt(np.nanmean(errR**2)))

    # 拼接面板
    h = max(visL.shape[0], visR.shape[0])
    w = visL.shape[1] + visR.shape[1]
    panel = np.zeros((h, w, 3), dtype=np.uint8)
    panel[: visL.shape[0], : visL.shape[1]] = visL
    panel[: visR.shape[0], visL.shape[1] : visL.shape[1] + visR.shape[1]] = visR

    # 标题与统计
    cv2.putText(
        panel,
        f"{title_left} | RMSE={rmseL:.2f}px  (mean={np.nanmean(errL):.2f}, "
        f"med={np.nanmedian(errL):.2f}, max={np.nanmax(errL):.2f})",
        (20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        f"{title_right} | RMSE={rmseR:.2f}px  (mean={np.nanmean(errR):.2f}, "
        f"med={np.nanmedian(errR):.2f}, max={np.nanmax(errR):.2f})",
        (visL.shape[1] + 20, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )

    return visL, visR, panel


# --------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------- #
def reproject_and_visualize(
    img1: np.ndarray,
    img2: np.ndarray,
    X3: np.ndarray,             # (J,3) 3D points in WORLD coords
    kptL: np.ndarray,           # (J,2) observed pixels in cam1
    kptR: np.ndarray,           # (J,2) observed pixels in cam2
    K1: np.ndarray,
    dist1: Optional[np.ndarray],
    K2: np.ndarray,
    dist2: Optional[np.ndarray],
    R: np.ndarray,
    T: np.ndarray,
    joint_names: Optional[Sequence[str]] = None,
    circle_r: int = 5,
    thickness: int = 2,
    out_path: Path = Path("reprojection_panel.jpg"),
) -> Dict[str, object]:
    """
    封装：重投影 + 可视化 + 保存。

    R,T 支持两种输入:
      - (2,3,3)/(2,3): 两个相机的世界→相机外参 (VGGT 风格)
      - (3,3)/(3,):   Cam1→Cam2 的相对外参 (stereo 风格)

    X3 为世界坐标系下的 3D 点，默认与 Cam1 对齐。
    返回:
      proj_L/proj_R/err_L/err_R/统计信息/out_path 等
    """
    # 1) 投影
    proj = reproject_points(X3, K1, dist1, K2, dist2, R, T)
    proj_L, proj_R = proj["proj_L"], proj["proj_R"]

    # 2) 面板渲染
    visL, visR, panel = render_reprojection_panel(
        img1,
        img2,
        kptL,
        kptR,
        proj_L,
        proj_R,
        joint_names=joint_names,
        circle_r=circle_r,
        thickness=thickness,
    )

    # 3) 误差统计
    errL = np.linalg.norm(proj_L - np.asarray(kptL, float), axis=1)
    errR = np.linalg.norm(proj_R - np.asarray(kptR, float), axis=1)

    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), panel)

    return {
        "proj_L": proj_L,
        "proj_R": proj_R,
        "err_L": errL,
        "err_R": errR,
        "rmse_L": float(np.sqrt(np.nanmean(errL**2))),
        "rmse_R": float(np.sqrt(np.nanmean(errR**2))),
        "mean_err_L": float(np.nanmean(errL)),
        "mean_err_R": float(np.nanmean(errR)),
        "median_err_L": float(np.nanmedian(errL)),
        "median_err_R": float(np.nanmedian(errR)),
        "max_err_L": float(np.nanmax(errL)),
        "max_err_R": float(np.nanmax(errR)),
        "out_path": str(out_path),
        "vis_left": visL,
        "vis_right": visR,
        "panel": panel,
    }
