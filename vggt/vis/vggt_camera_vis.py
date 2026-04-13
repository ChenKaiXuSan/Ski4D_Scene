#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/vggt/camera_vis.py
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt


def plot_cameras_from_predictions(
    predictions: dict,
    out_path: str | Path,
    axis_len: float = 0.1,
    title: str = "Camera Poses",
    show_id: bool = True,
    include_points: bool = False,
    center_mode: str = "mean",  # "mean" | "first" | "none"
) -> None:
    """
    从 VGGT predictions 中读取相机外参，在一张 PNG 上画出所有相机。

    Args:
        predictions: 包含至少：
            - "extrinsic": (S,3,4) 或 (S,4,4)，世界->相机  [R|T]
          可选：
            - "world_points_from_depth" 或 "world_points": (S,H,W,3)，用来画点云
        out_path: 输出 PNG 路径
        axis_len: 每个相机局部坐标轴长度
        title: 图标题
        show_id: 是否在相机中心旁边标 id
        include_points: 是否画世界点云
        center_mode: 可视化时的平移方式：
            - "mean": 以所有相机中心均值为原点
            - "first": 以第 0 个相机为原点
            - "none": 不平移，保持原始世界坐标
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # -------- 1. 取出外参 R,T 并计算相机中心 --------
    E = np.asarray(predictions["extrinsic"])  # (S,3,4) or (S,4,4)
    if E.shape[-2:] == (3, 4):
        R = E[..., :3, :3]  # (S,3,3)
        T = E[..., :3, 3]  # (S,3)
    else:
        R = E[..., :3, :3]
        T = E[..., :3, 3]

    # 世界坐标系下的相机中心：
    # X_cam = R * X_world + T, 相机中心 X_cam=0 → C_world = - R^T * T
    C = -np.einsum("sij,sj->si", R.transpose(0, 2, 1), T)  # (S,3)

    # -------- 2. 可选点云 --------
    pts = None
    if include_points:
        if "world_points_from_depth" in predictions:
            pts = predictions["world_points_from_depth"]
        elif "world_points" in predictions:
            pts = predictions["world_points"]
        if pts is not None:
            pts = np.asarray(pts).reshape(-1, 3)
            pts = pts[np.isfinite(pts).all(axis=1)]

    # -------- 3. 为可视化做平移（不改真实外参） --------
    if center_mode == "mean":
        center = C.mean(axis=0, keepdims=True)  # (1,3)
    elif center_mode == "first":
        center = C[:1]  # 第 0 个相机
    elif center_mode == "none":
        center = np.zeros((1, 3), dtype=C.dtype)
    else:
        raise ValueError(f"Unknown center_mode: {center_mode}")

    C_plot = C - center
    if pts is not None:
        pts_plot = pts - center
    else:
        pts_plot = None

    # -------- 4. 设置多个视角（同一张图多子图） --------
    views = [
        dict(name="default", elev=30, azim=60),
        dict(name="front", elev=0, azim=90),
        dict(name="top", elev=90, azim=-90),
        dict(name="side", elev=0, azim=0),
    ]
    n_cols = 2
    n_rows = (len(views) + n_cols - 1) // n_cols
    fig = plt.figure(figsize=(4 * n_cols, 4 * n_rows))

    def _plot_one(ax, elev, azim, view_name):
        # 相机中心
        ax.scatter(C_plot[:, 0], C_plot[:, 1], C_plot[:, 2], s=25, label="Cameras")

        # 局部坐标轴方向（仍然用世界系方向，不再平移）
        Xw = np.einsum("sij,j->si", R.transpose(0, 2, 1), np.array([1, 0, 0]))
        Yw = np.einsum("sij,j->si", R.transpose(0, 2, 1), np.array([0, 1, 0]))
        Zw = np.einsum("sij,j->si", R.transpose(0, 2, 1), np.array([0, 0, 1]))

        for i in range(C_plot.shape[0]):
            o = C_plot[i]
            ax.plot(
                [o[0], o[0] + axis_len * Xw[i, 0]],
                [o[1], o[1] + axis_len * Xw[i, 1]],
                [o[2], o[2] + axis_len * Xw[i, 2]],
                "r",
                lw=1,
            )
            ax.plot(
                [o[0], o[0] + axis_len * Yw[i, 0]],
                [o[1], o[1] + axis_len * Yw[i, 1]],
                [o[2], o[2] + axis_len * Yw[i, 2]],
                "g",
                lw=1,
            )
            ax.plot(
                [o[0], o[0] + axis_len * Zw[i, 0]],
                [o[1], o[1] + axis_len * Zw[i, 1]],
                [o[2], o[2] + axis_len * Zw[i, 2]],
                "b",
                lw=1,
            )

            if show_id:
                ax.text(
                    o[0],
                    o[1],
                    o[2],
                    f"{i}",
                    fontsize=9,
                    ha="center",
                    va="bottom",
                    weight="bold",
                )

        if pts_plot is not None:
            ax.scatter(
                pts_plot[:, 0],
                pts_plot[:, 1],
                pts_plot[:, 2],
                s=1,
                c="gray",
                alpha=0.3,
                label="Points",
            )

        # 自适应坐标范围
        all_xyz = [C_plot]
        if pts_plot is not None and len(pts_plot) > 0:
            all_xyz.append(pts_plot)
        all_xyz = np.concatenate(all_xyz, axis=0)
        x_min, y_min, z_min = all_xyz.min(axis=0)
        x_max, y_max, z_max = all_xyz.max(axis=0)
        pad = 0.05 * max(x_max - x_min, y_max - y_min, z_max - z_min)
        ax.set_xlim(x_min - pad, x_max + pad)
        ax.set_ylim(y_min - pad, y_max + pad)
        ax.set_zlim(z_min - pad, z_max + pad)

        ax.set_title(view_name)
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")
        ax.set_box_aspect((1, 1, 1))
        ax.view_init(elev=elev, azim=azim)

    for i, v in enumerate(views):
        ax = fig.add_subplot(n_rows, n_cols, i + 1, projection="3d")
        _plot_one(ax, v["elev"], v["azim"], v["name"])

    fig.suptitle(f"{title} (center_mode={center_mode})")
    plt.tight_layout()
    fig.savefig(str(out_path), dpi=250)
    plt.close(fig)
    print(f"[Saved camera PNG] {out_path}")
