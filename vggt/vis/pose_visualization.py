#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, List, Optional, Tuple, Union, Dict

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import numpy as np
import torch

logger = logging.getLogger(__name__)

# --------------------------- 常量 ---------------------------

# COCO-17 骨架（左/右臂、腿、躯干、头部）
COCO_SKELETON: List[Tuple[int, int]] = [
    (0, 1),
    (0, 2),
    (1, 3),
    (2, 4),
    (5, 7),
    (7, 9),
    (6, 8),
    (8, 10),
    (5, 6),
    (5, 11),
    (6, 12),
    (11, 12),
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]

__all__ = [
    "COCO_SKELETON",
    "draw_camera",
    "visualize_3d_joints",
    "compute_bone_lengths",
    "compute_bone_stats",
]

# COCO 17 关节点的典型索引
COCO_NOSE = 0
COCO_L_SHOULDER = 5
COCO_R_SHOULDER = 6
COCO_L_HIP = 11
COCO_R_HIP = 12
COCO_L_ANKLE = 15
COCO_R_ANKLE = 16


def _set_axes_by_body_shape(
    ax3d, pts_plot: np.ndarray, camera_centers: Optional[np.ndarray] = None
):
    """
    根据 COCO 3D 关键点自动设置坐标范围，但同时考虑摄像机位置，避免摄像机被裁掉。
    """
    pts = np.asarray(pts_plot, float)

    # === 把相机加入整体点云范围（不影响骨盆/头部检测） ===
    if camera_centers is not None:
        camera_centers = np.asarray(camera_centers, float)
        pts = np.vstack([pts, camera_centers])

    J = pts_plot.shape[0]

    def safe_get(idx):
        return pts_plot[idx] if idx < J and np.all(np.isfinite(pts_plot[idx])) else None

    nose = safe_get(COCO_NOSE)
    l_sh = safe_get(COCO_L_SHOULDER)
    r_sh = safe_get(COCO_R_SHOULDER)
    l_hip = safe_get(COCO_L_HIP)
    r_hip = safe_get(COCO_R_HIP)
    l_ank = safe_get(COCO_L_ANKLE)
    r_ank = safe_get(COCO_R_ANKLE)

    if l_hip is not None and r_hip is not None:
        pelvis = 0.5 * (l_hip + r_hip)
    else:
        pelvis = np.nanmean(pts_plot, axis=0)

    if l_ank is not None and r_ank is not None:
        foot = 0.5 * (l_ank + r_ank)
    else:
        foot = pts_plot[np.argmin(pts_plot[:, 2])]

    if nose is not None:
        head = nose
    else:
        head = pts_plot[np.argmax(pts_plot[:, 2])]

    body_height = np.linalg.norm(head - foot) + 1e-6
    shoulder_width = (
        np.linalg.norm(l_sh - r_sh)
        if l_sh is not None and r_sh is not None
        else pts_plot[:, 0].max() - pts_plot[:, 0].min()
    )
    depth_range_human = pts_plot[:, 1].max() - pts_plot[:, 1].min()

    # === 关键变化：加入相机的全局范围 ===
    depth_range = max(depth_range_human, pts[:, 1].max() - pts[:, 1].min())
    width_range = max(shoulder_width, pts[:, 0].max() - pts[:, 0].min())
    height_range = max(body_height, pts[:, 2].max() - pts[:, 2].min())

    half = 0.3 * max(height_range, width_range, depth_range)

    cx, cy, cz = pelvis
    ax3d.set_xlim(cx - half, cx + half)
    ax3d.set_ylim(cy - half, cy + half)
    ax3d.set_zlim(cz - half, cz + half)


def compute_bone_lengths(
    pts: np.ndarray,
    skeleton: Iterable[Tuple[int, int]],
    *,
    ignore_nan: bool = True,
) -> np.ndarray:
    """
    计算一帧 3D 关键点在给定骨架下的骨长。
    pts: (K,3)
    返回: (E,) 对应 skeleton 中每条边的长度；无效边为 np.nan
    """
    P = np.asarray(pts, dtype=float)
    L: List[float] = []
    for i, j in skeleton:
        if i >= len(P) or j >= len(P):
            L.append(np.nan)
            continue
        a, b = P[i], P[j]
        if ignore_nan and (not np.all(np.isfinite(a)) or not np.all(np.isfinite(b))):
            L.append(np.nan)
            continue
        L.append(float(np.linalg.norm(a - b)))
    return np.asarray(L, dtype=float)


def compute_bone_stats(lengths: np.ndarray) -> Dict[str, float]:
    """
    对骨长（含 nan）做统计，返回 mean/median/std/min/max/valid_count。
    """
    x = np.asarray(lengths, dtype=float)
    valid = np.isfinite(x)
    if not np.any(valid):
        return dict(
            mean=np.nan,
            median=np.nan,
            std=np.nan,
            min=np.nan,
            max=np.nan,
            valid_count=0,
        )
    xv = x[valid]
    return dict(
        mean=float(np.nanmean(xv)),
        median=float(np.nanmedian(xv)),
        std=float(np.nanstd(xv)),
        min=float(np.nanmin(xv)),
        max=float(np.nanmax(xv)),
        valid_count=int(valid.sum()),
    )


# --------------------------- 3D 可视化 ---------------------------


def draw_camera(
    ax: plt.Axes,
    R: np.ndarray,  # (3,3)
    T: np.ndarray,  # (3,) or (3,1)
    K: np.ndarray,  # (3,3)
    image_size: Tuple[int, int],  # (W, H) in px
    axis_len: float = 0.1,
    frustum_depth: float = 0.1,
    label: Optional[str] = None,
    ray_scale_mode: Literal[
        "depth", "focal"
    ] = "depth",  # "depth": 固定 z_cam；"focal": 以焦距近似尺度
    linewidths: Dict[str, float] = None,  # {"axis": 2.0, "frustum": 1.0}
    frustum_alpha: float = 1.0,
) -> np.ndarray:
    """
    在 Matplotlib 3D 轴上绘制 OpenCV 相机 (坐标轴 & 视锥)，并对齐到 Matplotlib 3D 坐标系：
      OpenCV:     x→右, y→下, z→前
      Matplotlib: x→右, y→前, z→上

    参数
    ----
    R, T : 相机外参
      - 如果 convention="world2cam"（默认），满足 Xc = R @ Xw + T
      - 如果 convention="cam2world"，满足 Xw = R @ Xc + T
    K : 内参矩阵 (3x3)
    image_size : (W, H) 像素
    axis_len : 相机坐标轴长度（世界单位）
    frustum_depth : 视锥体深度（世界单位），当 ray_scale_mode="depth" 时生效
    colors : 坐标轴颜色 (X, Y, Z)
    label : 相机中心标注
    convention : 外参定义方式
    ray_scale_mode :
        - "depth": 将像素反投影射线缩放到 z_cam=frustum_depth
        - "focal": 以焦距近似尺度（更像真实FOV），深度感基于 fx/fy
    linewidths : 线宽配置，默认为 {"axis": 2.0, "frustum": 1.0}
    frustum_alpha : 视锥边透明度

    返回
    ----
    C_plt : (3,) 相机中心（已映射到 Matplotlib 世界系）
    """
    # ------- 参数与形状 -------
    if linewidths is None:
        linewidths = {"axis": 2.0, "frustum": 1.0}

    R = np.asarray(R, dtype=float).reshape(3, 3)
    T = np.asarray(T, dtype=float).reshape(3, 1)
    K = np.asarray(K, dtype=float).reshape(3, 3)
    W, H = [float(x) for x in image_size]
    # W, H = image_size

    # ------- OpenCV(world) -> Matplotlib(world) 线性映射 -------
    # x 保持；z(前) -> y(前)；-y(上) -> z(上)
    M = np.array(
        [
            [1.0, 0.0, 0.0],  # x -> x
            [0.0, 0.0, 1.0],  # z -> y
            [0.0, -1.0, 0.0],  # -y -> z
        ],
        dtype=float,
    )

    # ------- 相机中心 (OpenCV 世界系) -------
    # Xc = R Xw + T  => 0 = R C + T  => C = -R^T T
    C_cv = -R.T @ T
    R_wc = R.T  # world <- cam
    t_wc = C_cv.reshape(3)  # same as camera center

    # ------- 相机中心映射到 Matplotlib 世界系 -------
    C_plt = (M @ C_cv).reshape(3)
    
    logger.info(f"[Draw Camera]{label} Center (Matplotlib world): {C_plt}")

    # ------- 画相机坐标轴（使用 R_wc 的行向量作为相机轴在世界系中的表示）-------
    # rows of R_wc: x_cam,y_cam,z_cam expressed in world(OpenCV) coords
    axes_cv = R_wc.copy()
    lw_axis = float(linewidths.get("axis", 2.0))
    for axis_cv, color in zip(axes_cv, ("r", "g", "b")):
        end_vec_cv = axis_cv * axis_len  # (3,)
        end_vec_plt = M @ end_vec_cv
        ax.plot(
            [C_plt[0], C_plt[0] + end_vec_plt[0]],
            [C_plt[1], C_plt[1] + end_vec_plt[1]],
            [C_plt[2], C_plt[2] + end_vec_plt[2]],
            c=color,
            lw=lw_axis,
        )

    # ------- 构造视锥：像素四角反投影 -> 相机系 -> 世界系(OpenCV) -> Matplotlib -------
    # 像素四角齐次坐标
    corners_px = np.array(
        [
            [0.0, 0.0, 1.0],
            [W - 1, 0.0, 1.0],
            [W - 1, H - 1, 1.0],
            [0.0, H - 1, 1.0],
        ],
        dtype=float,
    )

    Kinv = np.linalg.inv(K)
    # 射线方向（相机系）
    rays_cam = (Kinv @ corners_px.T).T  # (4,3)

    if ray_scale_mode == "depth":
        # 令 z_cam = frustum_depth
        scale = frustum_depth / np.clip(rays_cam[:, 2:3], 1e-12, None)
        rays_cam = rays_cam * scale
    else:
        # 按焦距尺度：以 fx/fy 的均值近似一个“单位深度”量级
        fx, fy = K[0, 0], K[1, 1]
        s = float((fx + fy) * 0.5)
        if s <= 1e-12:
            s = 1.0
        rays_cam = rays_cam / s * max(axis_len, frustum_depth)

    # cam -> world(OpenCV)
    corners_w_cv = (R_wc @ rays_cam.T).T + t_wc.reshape(1, 3)  # (4,3)
    # world(OpenCV) -> world(Matplotlib)
    corners_w_plt = (M @ corners_w_cv.T).T

    # ------- 画视锥边 -------
    lw_frustum = float(linewidths.get("frustum", 1.0))
    for p in corners_w_plt:
        ax.plot(
            [C_plt[0], p[0]],
            [C_plt[1], p[1]],
            [C_plt[2], p[2]],
            c="k",
            lw=lw_frustum,
            alpha=frustum_alpha,
        )
    loop = [0, 1, 2, 3, 0]
    ax.plot(
        corners_w_plt[loop, 0],
        corners_w_plt[loop, 1],
        corners_w_plt[loop, 2],
        c="k",
        lw=lw_frustum,
        alpha=frustum_alpha,
    )

    # ------- 标签 -------
    if label:
        ax.text(C_plt[0], C_plt[1], C_plt[2], label, fontsize=9, color="black")

    return C_plt


def _ensure_joints3d_arr(joints_3d: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    arr = torch.as_tensor(joints_3d).detach().cpu().numpy()
    if arr.ndim == 2 and arr.shape[1] == 3:
        return arr
    if arr.ndim == 2 and arr.shape[0] == 3:
        return arr.T
    raise ValueError(f"joints_3d must be (K,3) or (3,K), got {arr.shape}")


def visualize_3d_joints(
    joints_3d: Union[np.ndarray, torch.Tensor],
    R: List[np.ndarray],
    T: List[np.ndarray],
    K: List[np.ndarray],
    image_size: Tuple[int, int],
    save_path: Path,
    title: str = "Triangulated 3D Joints",
    show_stats: bool = True,  # 是否叠加骨长统计文本
) -> Optional[Dict[str, float]]:
    """
    保存 3D 关键点+相机示意图，并可：
      - 统一等比例 & 统一刻度（axis_limits / tick_step）
      - 计算并显示骨长统计（均值/中位数/方差等）
      - 可选在每条骨中点标注长度
    返回: 若 show_stats=True 则返回统计字典；否则 None
    """
    save_path.parent.mkdir(parents=True, exist_ok=True)

    pts = _ensure_joints3d_arr(joints_3d).astype(float)
    R = np.asarray(R, dtype=float).reshape(2, 3, 3)
    T = np.asarray(T, dtype=float).reshape(2, 3)

    # 初始化图形
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")

    # 相机
    center_L = draw_camera(
        ax=ax, R=R[0], T=T[0], K=K[0], image_size=image_size, label="Left Cam"
    )
    center_R = draw_camera(
        ax=ax, R=R[1], T=T[1], K=K[1], image_size=image_size, label="Right Cam"
    )

    camera_centers = np.stack([center_L, center_R])  # (2,3)

    # 可视化坐标置换（Y 朝上）
    # 把pts从cv2世界系映射到matplotlib世界系
    M = np.array(
        [
            [1.0, 0.0, 0.0],  # x -> x
            [0.0, 0.0, 1.0],  # z -> y
            [0.0, -1.0, 0.0],  # -y -> z
        ],
        dtype=float,
    )
    pts_plot = (M @ pts.T).T
    xlab, ylab, zlab = "X", "Z", "Y (up)"

    # 点与索引
    ax.scatter(pts_plot[:, 0], pts_plot[:, 1], pts_plot[:, 2], c="blue", s=30)
    for i, (x, y, z) in enumerate(pts_plot):
        ax.text(x, y, z, str(i), size=8)

    # 骨架与长度
    skeleton = COCO_SKELETON
    if skeleton is not None:
        lengths = compute_bone_lengths(
            pts, skeleton
        )  # 用原坐标算长度（与可视化变换无关）

        # 画线（在绘图坐标系）
        for i, j in skeleton:
            if i < len(pts_plot) and j < len(pts_plot):
                if not (
                    np.all(np.isfinite(pts_plot[i]))
                    and np.all(np.isfinite(pts_plot[j]))
                ):
                    continue
                ax.plot(
                    [pts_plot[i, 0], pts_plot[j, 0]],
                    [pts_plot[i, 1], pts_plot[j, 1]],
                    [pts_plot[i, 2], pts_plot[j, 2]],
                    c="red",
                    linewidth=2,
                )
        stats = compute_bone_stats(lengths)
    else:
        stats = None

    ax.set_title(title)
    ax.set_xlabel(xlab)
    ax.set_ylabel(ylab)
    ax.set_zlabel(zlab)

    # 叠加统计文本
    if show_stats and stats is not None:
        txt = (
            f"bones: {stats['valid_count']}\n"
            f"mean:  {stats['mean']:.3f}\n"
            f"median:{stats['median']:.3f}\n"
            f"std:   {stats['std']:.3f}\n"
            f"min:   {stats['min']:.3f}\n"
            f"max:   {stats['max']:.3f}"
        )
        ax.text2D(
            0.02,
            0.98,
            txt,
            transform=ax.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
        )

    _set_axes_by_body_shape(ax, pts_plot, camera_centers)

    plt.tight_layout()

    fig.savefig(str(save_path), dpi=500)
    plt.close(fig)
    logger.info(f"[Saved] {save_path}")

    return stats if show_stats else None


def save_stereo_pose_frame(
    img_left: np.ndarray,  # (H,W,3)
    img_right: np.ndarray,  # (H,W,3)
    kpt_left: np.ndarray,  # (J,2)
    kpt_right: np.ndarray,  # (J,2)
    pose_3d: np.ndarray,  # (J,3)
    output_path: Path,
    R: np.ndarray,  # (2,3,3)
    T: np.ndarray,  # (2,3)
    K: np.ndarray,  # (2,3,3)
    repoj_error: Dict[str, float] = None,
    frame_num: int = 0,
):
    """
    渲染一个 frame：左图+右图+3D pose，并保存成PNG。
    """
    skeleton = COCO_SKELETON
    J = pose_3d.shape[0]

    fig = plt.figure(figsize=(10, 8))
    fig.suptitle(f"Frame {frame_num}")
    gs = GridSpec(2, 2, figure=fig)

    # -------- 右视角 ---------- #
    axR = fig.add_subplot(gs[1, 0])
    axR.imshow(img_right)
    axR.axis("off")
    axR.set_title("Right view")

    for i, j in skeleton:
        axR.plot(
            [kpt_right[i, 0], kpt_right[j, 0]],
            [kpt_right[i, 1], kpt_right[j, 1]],
            color="blue",
        )
    axR.scatter(kpt_right[:, 0], kpt_right[:, 1], s=10, c="yellow", edgecolors="white")
    for i, (x, y) in enumerate(kpt_right):
        axR.text(x, y, str(i), size=8)

    # -------- 左视角 ---------- #
    axL = fig.add_subplot(gs[0, 0])
    axL.imshow(img_left)
    axL.axis("off")
    axL.set_title("Left view")

    for i, j in skeleton:
        if i < J and j < J:
            axL.plot(
                [kpt_left[i, 0], kpt_left[j, 0]],
                [kpt_left[i, 1], kpt_left[j, 1]],
                color="red",
            )
    axL.scatter(kpt_left[:, 0], kpt_left[:, 1], s=10, c="yellow", edgecolors="white")
    for i, (x, y) in enumerate(kpt_left):
        axL.text(x, y, str(i), size=8)

    # -------- 3D pose ---------- #
    ax3d = fig.add_subplot(gs[:, 1], projection="3d")
    ax3d.set_title("3D Pose")
    ax3d.axis("on")

    # --- 在原世界坐标系中计算人体尺度 ---
    pts = pose_3d.astype(float)

    # 骨盆宽度
    pelvis = np.linalg.norm(pts[11] - pts[12])
    target_pelvis = 0.38  # 期望肩宽/骨盆宽度（单位 m，可根据需要调整）

    # scale = target_pelvis / pelvis
    scale = 1.0  # 回退，不缩放

    # 应用缩放（只在可视化空间里用）
    pts_scaled = pts * scale

    # OpenCV 世界系 -> Matplotlib 世界系 (Y up)
    M = np.array(
        [
            [1.0, 0.0, 0.0],  # x -> x
            [0.0, 0.0, 1.0],  # z -> y
            [0.0, -1.0, 0.0],  # -y -> z
        ],
        dtype=float,
    )
    pts_plot = (M @ pts_scaled.T).T  # (J,3)，这下坐标系和相机绘制一致了

    # 相机（用同一套尺度绘制）
    center_l = draw_camera(
        ax=ax3d,
        R=R[0],
        T=T[0],
        K=K[0],
        image_size=(img_left.shape[1], img_left.shape[0]),
        label="Left Cam",
    )
    center_r = draw_camera(
        ax=ax3d,
        R=R[1],
        T=T[1],
        K=K[1],
        image_size=(img_right.shape[1], img_right.shape[0]),
        label="Right Cam",
    )

    camera_centers = np.stack([center_l, center_r])  # (2,3)

    # 画 3D 关节点
    ax3d.scatter(pts_plot[:, 0], pts_plot[:, 1], pts_plot[:, 2], c="blue", s=30)
    for i, (x, y, z) in enumerate(pts_plot):
        ax3d.text(x, y, z, str(i), size=8)

    for i, j in skeleton:
        if i < len(pts_plot) and j < len(pts_plot):
            if not (
                np.all(np.isfinite(pts_plot[i])) and np.all(np.isfinite(pts_plot[j]))
            ):
                continue
            ax3d.plot(
                [pts_plot[i, 0], pts_plot[j, 0]],
                [pts_plot[i, 1], pts_plot[j, 1]],
                [pts_plot[i, 2], pts_plot[j, 2]],
                c="red",
                linewidth=1,
            )

    # 按人体形状自动设置 xyz 取值范围（保持等比例）
    _set_axes_by_body_shape(ax3d, pts_plot, camera_centers)

    # Reproj error 文本
    if repoj_error is not None:
        txt = (
            f"Reproj Error:\n"
            f"L: {repoj_error['mean_err_L']:.2f} px\n"
            f"R: {repoj_error['mean_err_R']:.2f} px"
        )
        ax3d.text2D(
            0.02,
            0.98,
            txt,
            transform=ax3d.transAxes,
            va="top",
            fontsize=9,
            bbox=dict(facecolor="white", alpha=0.6, edgecolor="none"),
        )

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"[Saved] {output_path}")
