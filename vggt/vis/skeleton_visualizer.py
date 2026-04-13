#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/vggt/vis/skeleton_visualizer.py
Project: /workspace/code/vggt/vis
Created Date: Saturday December 6th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Saturday December 6th 2025 2:00:48 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from pathlib import Path
from typing import Dict, List, Literal, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np

K = np.array(
    [
        1116.9289548941917,
        0.0,
        955.77175993563799,
        0.0,
        1117.3341496962166,
        538.91061167202145,
        0.0,
        0.0,
        1.0,
    ]
).reshape(3, 3)

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


class SkeletonVisualizer:
    def __init__(
        self,
        bbox_color: Optional[Union[str, Tuple[int]]] = "green",
        kpt_color: Optional[Union[str, Tuple[Tuple[int]]]] = "red",
        link_color: Optional[Union[str, Tuple[Tuple[int]]]] = "blue",
        text_color: Optional[Union[str, Tuple[int]]] = (255, 255, 255),
        line_width: Union[int, float] = 1,
        radius: Union[int, float] = 3,
        alpha: float = 1.0,
        show_keypoint_weight: bool = False,
    ):
        self.bbox_color = bbox_color
        self.kpt_color = kpt_color
        self.link_color = link_color
        self.line_width = line_width
        self.text_color = text_color
        self.radius = radius
        self.alpha = alpha
        self.show_keypoint_weight = show_keypoint_weight

        # Pose specific meta info if available.
        self.skeleton = COCO_SKELETON

    def draw_camera(
        self,
        ax: plt.Axes,
        R: np.ndarray,  # (3,3)
        T: np.ndarray,  # (3,) or (3,1)
        K: np.ndarray = K,  # (3,3)
        image_size: Tuple[int, int] = (1920, 1080),  # (W, H) in px
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

    def draw_skeleton_2d(
        self,
        image: np.ndarray,
        keypoints: np.ndarray,
        kpt_thr: float = 0.3,
        show_kpt_idx: bool = False,
    ):
        """Draw keypoints and skeletons (optional) of GT or prediction.

        Args:
            image (np.ndarray): The image to draw.
            keypoints (np.ndarray): B x N x 3
            kpt_thr (float, optional): Minimum threshold of keypoints
                to be shown. Default: 0.3.
            show_kpt_idx (bool): Whether to show the index of keypoints.
                Defaults to ``False``

        Returns:
            np.ndarray: the drawn image which channel is RGB.
        """

        image = image.copy()
        img_h, img_w, _ = image.shape
        if len(keypoints.shape) == 2:
            keypoints = keypoints[None, :, :]

        # loop for each person
        for cur_keypoints in keypoints:
            kpts = cur_keypoints
            score = np.ones_like(kpts[:, 0])

            if self.kpt_color is None or isinstance(self.kpt_color, str):
                kpt_color = [self.kpt_color] * len(kpts)
            elif len(self.kpt_color) == len(kpts):
                kpt_color = self.kpt_color
            else:
                raise ValueError(
                    f"the length of kpt_color "
                    f"({len(self.kpt_color)}) does not matches "
                    f"that of keypoints ({len(kpts)})"
                )

            # draw links
            if self.skeleton is not None and self.link_color is not None:
                if self.link_color is None or isinstance(self.link_color, str):
                    link_color = [self.link_color] * len(self.skeleton)
                elif len(self.link_color) == len(self.skeleton):
                    link_color = self.link_color
                else:
                    raise ValueError(
                        f"the length of link_color "
                        f"({len(self.link_color)}) does not matches "
                        f"that of skeleton ({len(self.skeleton)})"
                    )

                for sk_id, sk in enumerate(self.skeleton):
                    pos1 = (int(kpts[sk[0], 0]), int(kpts[sk[0], 1]))
                    pos2 = (int(kpts[sk[1], 0]), int(kpts[sk[1], 1]))
                    if (
                        pos1[0] <= 0
                        or pos1[0] >= img_w
                        or pos1[1] <= 0
                        or pos1[1] >= img_h
                        or pos2[0] <= 0
                        or pos2[0] >= img_w
                        or pos2[1] <= 0
                        or pos2[1] >= img_h
                        or score[sk[0]] < kpt_thr
                        or score[sk[1]] < kpt_thr
                        or link_color[sk_id] is None
                    ):
                        # skip the link that should not be drawn
                        continue

                    color = link_color[sk_id]
                    if not isinstance(color, str):
                        color = tuple(int(c) for c in color)
                    transparency = self.alpha
                    if self.show_keypoint_weight:
                        transparency *= max(
                            0, min(1, 0.5 * (score[sk[0]] + score[sk[1]]))
                        )

                    image = cv2.line(
                        image,
                        pos1,
                        pos2,
                        2,  # FIXME: color was missing here
                        thickness=self.line_width,
                    )

            # draw each point on image
            for kid, kpt in enumerate(kpts):
                if score[kid] < kpt_thr or kpt_color[kid] is None:
                    # skip the point that should not be drawn
                    continue

                color = kpt_color[kid]
                if not isinstance(color, str):
                    color = tuple(int(c) for c in color)
                transparency = self.alpha
                if self.show_keypoint_weight:
                    transparency *= max(0, min(1, score[kid]))

                if transparency == 1.0:
                    image = cv2.circle(
                        image,
                        (int(kpt[0]), int(kpt[1])),
                        int(self.radius),
                        0,  # color,
                        -1,
                    )
                else:
                    temp = image = cv2.circle(
                        image.copy(),
                        (int(kpt[0]), int(kpt[1])),
                        int(self.radius),
                        0,  # color,
                        -1,
                    )
                    image = cv2.addWeighted(
                        image, 1 - transparency, temp, transparency, 0
                    )

                if show_kpt_idx:
                    cv2.putText(
                        image,
                        str(kid),
                        (int(kpt[0]), int(kpt[1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                        lineType=cv2.LINE_AA,
                    )

        return image

    def draw_skeleton_3d(
        self,
        ax: plt.Axes,
        points_3d: np.ndarray,
        window_title: str = "3D Skeleton Visualization",
    ):
        """
        使用 Open3D 绘制 3D 关键点和骨架连线。

        Args:
            points_3d (np.ndarray): N x 3 形状的 NumPy 数组，代表 3D 坐标 (x, y, z)。
            colors (Optional[np.ndarray]): N x 3 形状的 NumPy 数组，代表每个点的 RGB 颜色 (0.0 到 1.0)。
            link_colors (Optional[np.ndarray]): M x 3 形状的 NumPy 数组，代表每条连接线的 RGB 颜色 (0.0 到 1.0)。
            window_title (str): 可视化窗口的标题。
        """
        # 1. 初始化 3D 绘图

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
        pts_plot = (M @ points_3d.T).T

        ax.set_title(window_title)

        # 设置坐标轴标签
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Y (up)")

        # 2. 绘制 3D 关键点
        # 点与索引
        ax.scatter(
            pts_plot[:, 0],
            pts_plot[:, 1],
            pts_plot[:, 2],
            c=self.kpt_color,
            marker="o",
            s=self.radius * 10,  # 调整点的大小以便在 3D 中可见
            alpha=self.alpha,
        )
        for i, (x, y, z) in enumerate(pts_plot):
            ax.text(x, y, z, str(i), size=8)

        # 3. 绘制 3D 骨架连线
        if self.skeleton is not None:
            for i, (p1_idx, p2_idx) in enumerate(self.skeleton):
                # 获取连接线的颜色，确保在 0.0-1.0 范围
                color = self.link_color

                # 提取两个点的坐标
                p1 = pts_plot[p1_idx]
                p2 = pts_plot[p2_idx]

                # 绘制连接线
                ax.plot(
                    [p1[0], p2[0]],
                    [p1[1], p2[1]],
                    [p1[2], p2[2]],
                    color=color,
                    linewidth=self.line_width * 2,  # 调整线宽
                    alpha=self.alpha,
                )

        # 返回 Figure 对象，Notebook 会自动显示它
        return ax

    def draw_camera_with_skeleton(
        self,
        R: np.ndarray,
        T: np.ndarray,
        keypoints_3d: np.ndarray,
        K: np.ndarray = K,
        image_size: Tuple[int, int] = (1920, 1080),
        save_dir: Path = Path("skeleton_camera_viz.png"),
        window_title: str = "3D Camera and Skeleton Visualization",
    ) -> np.ndarray:
        """
        在 Matplotlib 3D 轴上绘制相机和 3D 骨架。

        参数
        ----
        R, T : 相机外参
        keypoints_3d : 3D 关键点 (N, 3)
        其他参数同 draw_camera 和 draw_skeleton_3d

        返回
        ----
        C_plt : (3,) 相机中心（已映射到 Matplotlib 世界系）
        """
        fig = plt.figure()
        ax = fig.add_subplot(111, projection="3d")

        C_plt = self.draw_camera(
            ax=ax,
            R=R[0],
            T=T[0],
            image_size=image_size,
            label="left cam",
        )

        C_plt = self.draw_camera(
            ax=ax,
            R=R[1],
            T=T[1],
            image_size=image_size,
            label="right cam",
        )

        self.draw_skeleton_3d(
            ax=ax,
            points_3d=keypoints_3d,
            window_title=window_title,
        )

        # save figure
        save_dir.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_dir)
        plt.close(fig)
