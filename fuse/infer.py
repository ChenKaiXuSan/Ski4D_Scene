#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
vggt_video_infer.py
从单个视频抽帧并执行 VGGT 推理，可作为函数调用。
"""

import gc
import logging
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import trimesh
import numpy as np
import open3d as o3d

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

from .load import load_sam3d_body_results, load_vggt_results, rot_point_cloud
from .prepare_paths import PersonInfo

logger = logging.getLogger(__name__)


class FuseVGGTInfer:
    def __init__(self, cfg: DictConfig, out_dir: Path, inference_output_dir: Path):
        super().__init__()

        self.cfg = cfg
        self.outdir = out_dir
        self.inference_output_dir = inference_output_dir

        self.rigid_min_points = int(cfg.infer.get("rigid_min_points", 6))
        self.rigid_use_ransac = bool(cfg.infer.get("rigid_use_ransac", True))
        self.rigid_ransac_iters = int(cfg.infer.get("rigid_ransac_iters", 200))
        self.rigid_ransac_thresh = float(cfg.infer.get("rigid_ransac_thresh", 0.08))

        # 可选：额外导出带世界坐标轴和相机位置的 3D 场景（glb）
        self.save_scene_with_axes = bool(cfg.infer.get("save_scene_with_axes", False))
        self.scene_axis_length = float(cfg.infer.get("scene_axis_length", 0.5))
        self.scene_camera_size = float(cfg.infer.get("scene_camera_size", 0.03))

    @staticmethod
    def _ensure_points(x: Any) -> np.ndarray:
        """将输入标准化为 (N,3) float64。"""
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 3:
            arr = arr.reshape(-1, arr.shape[-1])
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"Expect points shape (N,3), got {arr.shape}")
        return arr

    @staticmethod
    def _kabsch(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """估计 src->dst 的刚体变换: dst ~= R @ src + t。"""
        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)

        src_centered = src - src_mean
        dst_centered = dst - dst_mean

        H = src_centered.T @ dst_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        if np.linalg.det(R) < 0:
            Vt[-1, :] *= -1.0
            R = Vt.T @ U.T

        t = dst_mean - R @ src_mean
        return R, t

    @staticmethod
    def _residuals(
        src: np.ndarray, dst: np.ndarray, R: np.ndarray, t: np.ndarray
    ) -> np.ndarray:
        pred = (R @ src.T).T + t[None, :]
        return np.linalg.norm(pred - dst, axis=1)

    def register_kpts(
        self,
        left_kpts_3d: np.ndarray,
        right_kpts_3d: np.ndarray,
    ) -> Dict[str, Any]:
        """
        用左右 3D 关键点估计刚体变换（右 -> 左），并返回配准结果。

        Returns:
            {
                "R": (3,3),
                "t": (3,),
                "valid_mask": (N,),
                "inlier_mask": (N_valid,),
                "num_points": int,
                "num_inliers": int,
                "rmse": float,
                "right_aligned": (N,3)
            }
        """
        left = self._ensure_points(left_kpts_3d)
        right = self._ensure_points(right_kpts_3d)

        if left.shape[0] != right.shape[0]:
            raise ValueError(
                f"Point count mismatch: left={left.shape[0]}, right={right.shape[0]}"
            )

        valid_mask = np.isfinite(left).all(axis=1) & np.isfinite(right).all(axis=1)
        src = right[valid_mask]  # 右 -> 左
        dst = left[valid_mask]

        if src.shape[0] < self.rigid_min_points:
            raise ValueError(
                f"Not enough valid correspondences: {src.shape[0]} < {self.rigid_min_points}"
            )

        if not self.rigid_use_ransac or src.shape[0] <= 3:
            R, t = self._kabsch(src, dst)
            inliers = np.ones((src.shape[0],), dtype=bool)
        else:
            rng = np.random.default_rng()
            best_inliers = None
            best_count = -1
            sample_size = min(4, src.shape[0])

            for _ in range(self.rigid_ransac_iters):
                idx = rng.choice(src.shape[0], size=sample_size, replace=False)
                try:
                    R_try, t_try = self._kabsch(src[idx], dst[idx])
                except np.linalg.LinAlgError:
                    continue

                err_try = self._residuals(src, dst, R_try, t_try)
                inliers_try = err_try < self.rigid_ransac_thresh
                count = int(inliers_try.sum())
                if count > best_count:
                    best_count = count
                    best_inliers = inliers_try

            if best_inliers is None or best_count < 3:
                R, t = self._kabsch(src, dst)
                inliers = np.ones((src.shape[0],), dtype=bool)
            else:
                R, t = self._kabsch(src[best_inliers], dst[best_inliers])
                inliers = best_inliers

        err = self._residuals(src, dst, R, t)
        rmse = (
            float(np.sqrt(np.mean(np.square(err[inliers]))))
            if np.any(inliers)
            else float(np.sqrt(np.mean(np.square(err))))
        )

        right_aligned = np.full_like(right, np.nan, dtype=np.float32)
        right_aligned_valid = (R @ src.T).T + t[None, :]
        right_aligned[valid_mask] = right_aligned_valid.astype(np.float32)

        return {
            "R": R.astype(np.float32),
            "t": t.astype(np.float32),
            "valid_mask": valid_mask,
            "inlier_mask": inliers,
            "num_points": int(src.shape[0]),
            "num_inliers": int(inliers.sum()),
            "rmse": rmse,
            "right_aligned": right_aligned,
        }

    @staticmethod
    # 根据2d kpt，寻找在点云中的对应点
    def compute_kpts_in_point_cloud(
        kpts,
        frame_size,
        point_cloud,
        conf_map=None,
        conf_thresh=None,
        remove_invalid=True,
    ):
        """
        根据 2D 关键点，在 VGGT point cloud 中取对应的 3D 点。

        Args:
            kpts: np.ndarray, shape (..., 2) 或 (..., >=2)，最后一维前两项为 (x, y)
            frame_size: tuple (width, height) of the original frame
            point_cloud: np.ndarray of shape (H_pc, W_pc, 3)
            conf_map: optional np.ndarray of shape (H_pc, W_pc), confidence map
            conf_thresh: optional float, only keep points with conf >= conf_thresh
            remove_invalid: bool, whether to mark invalid 3D points as NaN

        Returns:
            kpts_3d: np.ndarray, shape (..., 3)，与输入关键点结构一一对应
        """
        kpts_arr = np.asarray(kpts)

        if kpts_arr.ndim < 2 or kpts_arr.shape[-1] < 2:
            raise ValueError(
                f"kpts should have shape (..., 2) or (..., >=2), but got {kpts_arr.shape}"
            )

        W_frame, H_frame = frame_size  # 注意：frame_size = (width, height)
        if W_frame <= 0 or H_frame <= 0:
            raise ValueError(f"frame_size should be positive, but got {frame_size}")

        if point_cloud.ndim != 3 or point_cloud.shape[-1] != 3:
            raise ValueError(
                f"point_cloud should have shape (H, W, 3), but got {point_cloud.shape}"
            )

        H_pc, W_pc, _ = point_cloud.shape

        if conf_map is not None and conf_map.shape != (H_pc, W_pc):
            raise ValueError(
                f"conf_map should have shape {(H_pc, W_pc)}, but got {conf_map.shape}"
            )

        # 将任意前缀维度展平，统一处理像素映射
        kpts_xy = kpts_arr[..., :2].reshape(-1, 2).astype(np.float32)

        # 限制到图像有效范围，防止越界
        x = np.clip(kpts_xy[:, 0], 0, W_frame - 1)
        y = np.clip(kpts_xy[:, 1], 0, H_frame - 1)

        # 原图像素 -> point cloud 网格索引
        cols = np.floor(x * W_pc / W_frame).astype(np.int32)
        rows = np.floor(y * H_pc / H_frame).astype(np.int32)

        cols = np.clip(cols, 0, W_pc - 1)
        rows = np.clip(rows, 0, H_pc - 1)

        kpts_3d_flat = point_cloud[rows, cols].astype(np.float32, copy=True)

        if remove_invalid:
            valid_mask = np.isfinite(kpts_3d_flat).all(axis=1)
            valid_mask &= ~(np.abs(kpts_3d_flat).sum(axis=1) == 0)

            if conf_map is not None and conf_thresh is not None:
                kpts_conf = conf_map[rows, cols]
                valid_mask &= kpts_conf >= conf_thresh

            kpts_3d_flat[~valid_mask] = np.nan

        kpts_3d = kpts_3d_flat.reshape(*kpts_arr.shape[:-1], 3)
        return kpts_3d

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

    def _save_scene_with_axes_and_cameras(
        self,
        pts_flat: np.ndarray,
        colors_rgb: np.ndarray,
        extrinsics: np.ndarray,
        frame_idx: int,
        title: str,
    ) -> None:
        """将点云、坐标轴和相机中心一起导出为 glb。"""
        scene = trimesh.Scene()

        scene.add_geometry(
            trimesh.points.PointCloud(
                vertices=np.asarray(pts_flat, dtype=np.float32),
                colors=np.asarray(colors_rgb, dtype=np.uint8),
            ),
            geom_name="point_cloud",
        )

        world_axis = trimesh.creation.axis(axis_length=self.scene_axis_length)
        scene.add_geometry(world_axis, geom_name="world_axis")

        _, _, camera_centers = self.extrinsic_to_RT(extrinsics)
        for idx, center in enumerate(camera_centers):
            cam_marker = trimesh.creation.icosphere(
                subdivisions=2, radius=self.scene_camera_size
            )
            cam_marker.apply_translation(center)
            scene.add_geometry(cam_marker, geom_name=f"camera_{idx}")

        scene_path = self.inference_output_dir / f"{frame_idx}_scene_{title}.glb"
        scene.export(scene_path)

    def save_single(self, pts_flat, images, frame_idx, title, extrinsics=None):
        """
        保存点云和对应的图像颜色信息，方便后续分析和可视化。
        """

        pcd = o3d.geometry.PointCloud()

        # 处理图像颜色
        img_hw3 = images[0].transpose(1, 2, 0)  # (H, W, 3)
        colors_rgb = (img_hw3.reshape(-1, 3) * 255.0).clip(0, 255).astype(np.uint8)

        pcd.points = o3d.utility.Vector3dVector(pts_flat)  # (N,3)
        pcd.colors = o3d.utility.Vector3dVector(colors_rgb / 255.0)  # 需要 [0,1]

        # 保存
        o3d.io.write_point_cloud(
            self.inference_output_dir / f"{frame_idx}_point_cloud_{title}.ply", pcd
        )

        if self.save_scene_with_axes and extrinsics is not None:
            self._save_scene_with_axes_and_cameras(
                pts_flat=pts_flat,
                colors_rgb=colors_rgb,
                extrinsics=extrinsics,
                frame_idx=frame_idx,
                title=title,
            )

    def save_dual(
        self,
        left_pts_flat,
        right_pts_flat,
        left_images,
        right_images,
        frame_idx,
        title,
    ) -> None:
        """
        保存点云和对应的图像颜色信息，方便后续分析和可视化。
        """

        pcd = o3d.geometry.PointCloud()

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

        pcd.points = o3d.utility.Vector3dVector(vertices)  # (N,3)
        pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)  # 需要 [0,1]

        # 保存
        o3d.io.write_point_cloud(
            self.inference_output_dir / f"{frame_idx}_merged_point_cloud_{title}.ply",
            pcd,
        )


def process_one_person(
    p_info: PersonInfo,
    cfg: DictConfig,
) -> None:
    """处理单个人的推理流程。"""

    person_name = p_info.subject_name

    left_sam_3d_body_results_path = p_info.left_sam3d_body_results_path
    right_sam_3d_body_results_path = p_info.right_sam3d_body_results_path

    left_vggt_results_path = p_info.left_vggt_results_path
    right_vggt_results_path = p_info.right_vggt_results_path

    n_frames = min(
        len(left_sam_3d_body_results_path), len(right_sam_3d_body_results_path)
    )

    inference_output_path = p_info.inference_output_path
    out_dir = p_info.output_dir

    inference_output_path.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    fused_infer = FuseVGGTInfer(
        cfg, out_dir=out_dir, inference_output_dir=inference_output_path
    )

    for idx in tqdm(range(n_frames), desc=f"Processing {person_name} dual-view frames"):

        left_sam_3d_body_kpt_2d = load_sam3d_body_results(
            left_sam_3d_body_results_path[idx]
        )
        right_sam_3d_body_kpt_2d = load_sam3d_body_results(
            right_sam_3d_body_results_path[idx]
        )

        left_vggt_preds = load_vggt_results(left_vggt_results_path[idx])
        right_vggt_preds = load_vggt_results(right_vggt_results_path[idx])

        # 旋转点云，同一到一个坐标系下
        roted_left_world_pts, roted_left_extrinsics = rot_point_cloud(
            left_vggt_preds["world_points"], left_vggt_preds["extrinsic"], flag="left"
        )

        roted_right_world_pts, roted_right_extrinsics = rot_point_cloud(
            right_vggt_preds["world_points"],
            right_vggt_preds["extrinsic"],
            flag="right",
        )

        # 根据 2D 关键点，在 VGGT point cloud 中取对应的 3D 点
        left_kpt_3d_from_pt = fused_infer.compute_kpts_in_point_cloud(
            left_sam_3d_body_kpt_2d,
            frame_size=(
                left_vggt_preds["images"][0].shape[2],  # width
                left_vggt_preds["images"][0].shape[1],  # height
            ),
            point_cloud=roted_left_world_pts.squeeze(),  # shape (H_pc, W_pc, 3)
        )

        # center the point cloud for better numerical stability in registration
        left_feet_kpt = left_kpt_3d_from_pt[[16, 19]].mean(
            axis=0
        )  # 假设16,19是左右脚关键点索引

        moved_rot_left_world_pts = roted_left_world_pts - left_feet_kpt  # 移动点云

        moved_rot_left_extrinsics = roted_left_extrinsics.copy()
        for v in range(moved_rot_left_extrinsics.shape[0]):
            R = moved_rot_left_extrinsics[v, :3, :3]
            t = moved_rot_left_extrinsics[v, :3, 3]
            moved_rot_left_extrinsics[v, :3, 3] = t + R @ left_feet_kpt

        moved_rot_left_kpt_3d_from_pt = (
            left_kpt_3d_from_pt - left_feet_kpt
        )  # 移动关键点位置

        fused_infer.save_single(
            pts_flat=moved_rot_left_world_pts.reshape(-1, 3),
            images=left_vggt_preds["images"],
            frame_idx=idx,
            title="Left_Before_Alignment",
            extrinsics=moved_rot_left_extrinsics,
        )

        # center right
        right_kpt_3d_from_pt = fused_infer.compute_kpts_in_point_cloud(
            right_sam_3d_body_kpt_2d,
            frame_size=(
                right_vggt_preds["images"][0].shape[2],  # width
                right_vggt_preds["images"][0].shape[1],  # height
            ),
            point_cloud=roted_right_world_pts.squeeze(),  # shape (H_pc, W_pc, 3)
        )

        right_feet_kpt = right_kpt_3d_from_pt[[16, 19]].mean(
            axis=0
        )  # 假设16,19是左右脚关键点索引

        moved_rot_right_world_pts = roted_right_world_pts - right_feet_kpt  # 移动点云

        moved_rot_right_extrinsics = roted_right_extrinsics.copy()
        for v in range(moved_rot_right_extrinsics.shape[0]):
            R = moved_rot_right_extrinsics[v, :3, :3]
            t = moved_rot_right_extrinsics[v, :3, 3]
            moved_rot_right_extrinsics[v, :3, 3] = t + R @ right_feet_kpt

        moved_rot_right_kpt_3d_from_pt = (
            right_kpt_3d_from_pt - right_feet_kpt
        )  # 移动关键点位置

        fused_infer.save_single(
            pts_flat=moved_rot_right_world_pts.reshape(-1, 3),
            images=right_vggt_preds["images"],
            frame_idx=idx,
            title="Right_Before_Alignment",
            extrinsics=moved_rot_right_extrinsics,
        )

        fused_infer.save_dual(
            left_pts_flat=moved_rot_left_world_pts.reshape(-1, 3),
            right_pts_flat=moved_rot_right_world_pts.reshape(-1, 3),
            left_images=left_vggt_preds["images"],
            right_images=right_vggt_preds["images"],
            frame_idx=idx,
            title="Before_Alignment",
        )

        # 根据左右 3D 关键点估计刚体变换（右 -> 左），并返回配准结果
        reg_result = fused_infer.register_kpts(
            moved_rot_left_kpt_3d_from_pt, moved_rot_right_kpt_3d_from_pt
        )

        right_kpt_3d_aligned = reg_result["right_aligned"]

        fused_kpt_3d = np.nanmean(
            np.stack([moved_rot_left_kpt_3d_from_pt, right_kpt_3d_aligned], axis=0),
            axis=0,
        ).astype(np.float32)

        R = reg_result["R"]
        t = reg_result["t"]

        # 根据R，t来对齐右侧点云到左侧坐标系
        right_pts_aligned = (R @ moved_rot_right_world_pts.reshape(-1, 3).T).T + t[
            None, :
        ]
        right_pts_aligned = right_pts_aligned.reshape(moved_rot_right_world_pts.shape)

        # 保存点云和对应的图像颜色信息，方便后续分析和可视化
        fused_infer.save_dual(
            left_pts_flat=moved_rot_left_world_pts.reshape(-1, 3),
            right_pts_flat=right_pts_aligned.reshape(-1, 3),
            left_images=left_vggt_preds["images"],
            right_images=right_vggt_preds["images"],
            frame_idx=idx,
            title="After_Alignment",
        )

    # 清空大对象，释放内存
    gc.collect()
