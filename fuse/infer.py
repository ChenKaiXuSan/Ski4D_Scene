#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
vggt_video_infer.py
从单个视频抽帧并执行 VGGT 推理，可作为函数调用。
"""

import gc
import logging
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List

import cv2
import numpy as np
import torch
import trimesh
from omegaconf import DictConfig
from tqdm import tqdm

from vggt.load import load_and_preprocess_images
from vggt.reproject import reproject_and_visualize
from vggt.save import save_camera_info

# 依赖 VGGT 官方模块
from vggt.vggt.models.vggt import VGGT
from vggt.vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.vis.pose_visualization import save_stereo_pose_frame, visualize_3d_joints
from vggt.vis.vggt_camera_vis import plot_cameras_from_predictions
from vggt.vis.visual_util import predictions_to_glb

from .load import (
    load_sam3d_body_results,
    load_vggt_results,
)
from .prepare_paths import PersonInfo

logger = logging.getLogger(__name__)


class FuseVGGTInfer:
    def __init__(self, cfg: DictConfig, out_dir: Path, inference_output_dir: Path):
        super().__init__()

        self.outdir = out_dir
        self.inference_output_dir = inference_output_dir

    @staticmethod
    def compute_kpts_in_point_cloud(
        kpts, 
        frame_size,
        point_cloud,
        conf_map=None,
        conf_thresh=None,
        remove_invalid=True,
    )
        pass

    @staticmethod
    def compute_bbox_in_point_cloud(
        bbox,
        frame_size,
        point_cloud,
        conf_map=None,
        conf_thresh=None,
        remove_invalid=True,
    ):
        """
        根据 2D bbox，在 VGGT point cloud 中取出对应区域的 3D 点。

        Args:
            bbox: tuple/list (x_min, y_min, x_max, y_max) in image pixel coordinates
            frame_size: tuple (width, height) of the original frame
            point_cloud: np.ndarray of shape (H_pc, W_pc, 3)
            conf_map: optional np.ndarray of shape (H_pc, W_pc), confidence map
            conf_thresh: optional float, only keep points with conf >= conf_thresh
            remove_invalid: bool, whether to remove invalid 3D points

        Returns:
            region_pts: np.ndarray of shape (N, 3), valid 3D points in bbox region
            center_pt: np.ndarray of shape (3,), center pixel 3D point
            grid_bbox: tuple (row_min, row_max, col_min, col_max)
        """
        x_min, y_min, x_max, y_max = bbox
        W_frame, H_frame = frame_size  # 注意：frame_size = (width, height)
        H_pc, W_pc, C = point_cloud.shape

        if C != 3:
            raise ValueError(
                f"point_cloud should have shape (H, W, 3), but got {point_cloud.shape}"
            )

        print(
            f"bbox: {bbox}, frame_size: {frame_size}, point_cloud_size: {point_cloud.shape}"
        )

        # 防止 bbox 越界 / 顺序错误
        x_min, x_max = sorted([x_min, x_max])
        y_min, y_max = sorted([y_min, y_max])

        x_min = np.clip(x_min, 0, W_frame - 1)
        x_max = np.clip(x_max, 0, W_frame - 1)
        y_min = np.clip(y_min, 0, H_frame - 1)
        y_max = np.clip(y_max, 0, H_frame - 1)

        # 原图 bbox -> point cloud 网格索引
        col_min = int(np.floor(x_min * W_pc / W_frame))
        col_max = int(np.floor(x_max * W_pc / W_frame))
        row_min = int(np.floor(y_min * H_pc / H_frame))
        row_max = int(np.floor(y_max * H_pc / H_frame))

        col_min = np.clip(col_min, 0, W_pc - 1)
        col_max = np.clip(col_max, 0, W_pc - 1)
        row_min = np.clip(row_min, 0, H_pc - 1)
        row_max = np.clip(row_max, 0, H_pc - 1)

        # 防止切片为空
        if col_max < col_min:
            col_min, col_max = col_max, col_min
        if row_max < row_min:
            row_min, row_max = row_max, row_min

        # 取 bbox 对应区域
        region = point_cloud[row_min : row_max + 1, col_min : col_max + 1]
        region_pts = region.reshape(-1, 3)

        # 可选：按 confidence 过滤
        if conf_map is not None:
            region_conf = conf_map[
                row_min : row_max + 1, col_min : col_max + 1
            ].reshape(-1)
            if conf_thresh is not None:
                keep = region_conf >= conf_thresh
                region_pts = region_pts[keep]

        # 去掉无效点
        if remove_invalid and len(region_pts) > 0:
            valid_mask = np.isfinite(region_pts).all(axis=1)
            valid_mask &= ~(np.abs(region_pts).sum(axis=1) == 0)
            region_pts = region_pts[valid_mask]

        # bbox 中心对应的 3D 点
        row_center = (row_min + row_max) // 2
        col_center = (col_min + col_max) // 2
        center_pt = point_cloud[row_center, col_center]

        grid_bbox = (row_min, row_max, col_min, col_max)

        # 返回四个角点的 3D 坐标、中心点的 3D 坐标，以及在 point cloud 网格中的 bbox 索引范围
        bbox_corners_3d = [
            point_cloud[row_min, col_min],  # top-left
            point_cloud[row_min, col_max],  # top-right
            point_cloud[row_max, col_max],  # bottom-right
            point_cloud[row_max, col_min],  # bottom-left
        ]

        return region_pts, center_pt, grid_bbox, bbox_corners_3d

    @staticmethod
    def extrinsic_to_RT(extrinsic):
        """
        extrinsic: (T,3,4) / (T,4,4) / (3,4) / (4,4)
        return: R (T,3,3), t (T,3), C (T,3)
        """
        E = np.asarray(extrinsic)

        # 扩展到 (T,3,4)
        if E.ndim == 2:  # 单帧
            E = E[None, ...]
        if E.shape[-2:] == (4, 4):
            E = E[:, :3, :]

        R = E[:, :3, :3]  # (T,3,3)
        t = E[:, :3, 3]  # (T,3)
        C = -np.einsum("tij,tj->ti", R.transpose(0, 2, 1), t)  # (T,3)

        return R, t, C

    def save(
        self,
        left_pts_flat,
        right_pts_flat,
        left_images,
        right_images,
        frame_idx,
    ) -> None:
        """
        保存点云和对应的图像颜色信息，方便后续分析和可视化。
        """
        # 把放在一起的点云保存起来成glb文件，方便在外部工具中查看

        # 处理左侧和右侧的图像颜色
        left_img_hw3 = left_images[0].transpose(1, 2, 0)  # (H, W, 3)
        left_colors_rgb = (
            (left_img_hw3.reshape(-1, 3) * 255.0).clip(0, 255).astype(np.uint8)
        )

        right_img_hw3 = right_images[0].transpose(1, 2, 0)  # (H, W, 3)
        right_colors_rgb = (
            (right_img_hw3.reshape(-1, 3) * 255.0).clip(0, 255).astype(np.uint8)
        )

        # 合并顶点和颜色
        vertices = np.vstack([left_pts_flat, right_pts_flat])
        colors = np.vstack([left_colors_rgb, right_colors_rgb])

        # 保存
        point_cloud = trimesh.PointCloud(vertices=vertices, colors=colors)
        point_cloud.export(
            self.inference_output_dir / f"{frame_idx}_merged_point_cloud.glb"
        )

    def process(self, *args, **kwargs) -> None:
        """
        处理单个人的推理流程。
        """
        pass

def process_one_person(
    p_info: PersonInfo,
    cfg: DictConfig,
) -> None:
    """处理单个人的推理流程。"""

    person_name = p_info.subject_name

    left_video_path = p_info.left_video_path
    right_video_path = p_info.right_video_path

    left_sam_3d_body_results_path = p_info.left_sam_3d_body_results_path
    right_sam_3d_body_results_path = p_info.right_sam_3d_body_results_path

    left_vggt_results_path = p_info.left_vggt_results_path
    right_vggt_results_path = p_info.right_vggt_results_path

    n_frames = len(left_sam_3d_body_results_path)  # 假设左右视频帧数相同

    inference_output_path = p_info.inference_output_path
    out_dir = p_info.output_dir

    inference_output_path.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    fused_infer = FuseVGGTInfer(
        cfg, out_dir=out_dir, inference_output_dir=inference_output_path
    )

    for idx in tqdm(range(n_frames), desc=f"Processing {person_name} dual-view frames"):

        left_sam_3d_body_results = load_sam3d_body_results(left_sam_3d_body_results_path[idx])
        right_sam_3d_body_results = load_sam3d_body_results(right_sam_3d_body_results_path[idx])

        left_vggt_preds = load_vggt_results(left_vggt_results_path[idx])
        right_vggt_preds = load_vggt_results(right_vggt_results_path[idx])

        
    # 清空大对象，释放内存
    gc.collect()
