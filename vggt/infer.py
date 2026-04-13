#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
vggt_video_infer.py
从单个视频抽帧并执行 VGGT 推理，可作为函数调用。
"""

import logging
from pathlib import Path
from typing import List, Optional, Union

import cv2
import numpy as np
import torch
from omegaconf import DictConfig
from torchvision.io import read_video
from tqdm import tqdm

from vggt.reproject import reproject_and_visualize
from vggt.save import save_camera_info, save_inference_results
from vggt.vggt.infer import CameraHead
from vggt.vis.pose_visualization import save_stereo_pose_frame, visualize_3d_joints

from vggt.rigid_transformation.infer import RigidTransformationInfer

from .prepare_paths import PersonInfo
from .vis.skeleton_visualizer import SkeletonVisualizer

from .load import load_sam3d_body_results, load_video_frames


logger = logging.getLogger(__name__)


def process_one_person(
    p_info: PersonInfo,
    cfg: DictConfig,
) -> Optional[Path]:
    """同时加载左右视频，并将同一时刻的双视角帧一起输入 VGGT。"""

    subject = p_info.subject_name

    left_video_path = p_info.left_video_path
    right_video_path = p_info.right_video_path

    left_sam3d_body_results_path = p_info.left_sam3d_body_results_path
    right_sam3d_body_results_path = p_info.right_sam3d_body_results_path

    inference_output_path = p_info.inference_output_path
    out_dir = p_info.output_dir

    inference_output_path.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[Run-DV] Processing dual-view frames -> {out_dir}")

    camera_head = CameraHead(cfg, out_dir=out_dir, inference_output_dir=inference_output_path)
    rigid_transformation_infer = RigidTransformationInfer(cfg)

    all_frame_camera_extrinsics = []
    all_frame_camera_intrinsics = []
    all_frame_R = []
    all_frame_t = []
    all_frame_C = []
    all_frame_relative_R = []
    all_frame_relative_t = []
    all_frame_relative_rmse = []
    all_frame_relative_inliers = []
    all_frame_pointcloud_centroid_error = []
    all_frame_fused_point_count = []
    all_frame_reproj_before_mean_err_l = []
    all_frame_reproj_before_mean_err_r = []
    all_frame_reproj_after_mean_err_l = []
    all_frame_reproj_after_mean_err_r = []

    def _as_path_list(x: Union[Path, List[Path]]) -> List[str]:
        if isinstance(x, Path):
            return [x.as_posix()]
        return [p.as_posix() for p in x]

    def _select_sam3d_result_path(
        sam_path: Union[Path, List[Path]],
        ref_video_path: Path,
    ) -> str:
        """从候选 SAM3D 结果中选出与视频最匹配的单个文件。"""
        candidates = [Path(p) for p in _as_path_list(sam_path)]
        if len(candidates) == 0:
            raise FileNotFoundError("No SAM3D result file found")
        if len(candidates) == 1:
            return candidates[0].as_posix()

        ref_name = ref_video_path.stem.lower()
        ref_tokens = [t for t in ref_name.replace("-", "_").split("_") if t]

        # 先做精确子串匹配
        exact_hits = [p for p in candidates if ref_name in p.stem.lower()]
        if len(exact_hits) == 1:
            return exact_hits[0].as_posix()
        if len(exact_hits) > 1:
            candidates = exact_hits

        # 再做 token 重叠匹配（例如 osmo_1 / osmo_2）
        best_path = candidates[0]
        best_score = -1
        for p in candidates:
            stem_tokens = set(p.stem.lower().replace("-", "_").split("_"))
            score = sum(1 for tok in ref_tokens if tok in stem_tokens)
            if score > best_score:
                best_score = score
                best_path = p

        if best_score <= 0:
            logger.warning(
                "Cannot confidently match SAM3D result for %s, fallback to %s",
                ref_video_path.name,
                best_path.name,
            )

        return best_path.as_posix()

    def _flatten_valid_points(points_map: np.ndarray) -> np.ndarray:
        """将 (H,W,3) 点云展开为 (N,3)，并过滤无效点。"""
        pts = np.asarray(points_map)
        if pts.ndim != 3 or pts.shape[-1] != 3:
            raise ValueError(f"Expect (H,W,3) point map, got {pts.shape}")

        valid_mask = np.isfinite(pts).all(axis=-1)
        flat = pts[valid_mask]

        # 深度为 0 的点通常是无效点，额外过滤
        if flat.size == 0:
            return flat.reshape(0, 3)
        non_zero_mask = np.linalg.norm(flat, axis=1) > 1e-8
        return flat[non_zero_mask]

    def _extract_first_person_keypoints(
        frame_result: object,
        key: str,
        expected_last_dim: int,
    ) -> Optional[np.ndarray]:
        """从单帧 SAM3D 结果中提取第一个人的关键点（2D/3D）。"""
        if frame_result is None:
            return None
        if isinstance(frame_result, np.ndarray) and frame_result.ndim == 0:
            frame_result = frame_result.item()
        if not isinstance(frame_result, dict):
            return None

        kpt = frame_result.get(key, None)
        if kpt is None:
            return None
        arr = np.asarray(kpt)

        if arr.ndim == 2 and arr.shape[1] == expected_last_dim:
            return arr.astype(np.float32)
        if arr.ndim == 3 and arr.shape[0] > 0 and arr.shape[2] == expected_last_dim:
            return arr[0].astype(np.float32)
        return None

    # 只加载与左右视频匹配的指定 SAM3D 结果文件
    # left_sam3d_file = _select_sam3d_result_path(
    #     left_sam3d_body_results_path,
    #     left_video_path,
    # )
    # right_sam3d_file = _select_sam3d_result_path(
    #     right_sam3d_body_results_path,
    #     right_video_path,
    # )

    # logger.info("Use left SAM3D result: %s", left_sam3d_file)
    # logger.info("Use right SAM3D result: %s", right_sam3d_file)

    # left_sam3d_results = load_sam3d_body_results([left_sam3d_file])
    # right_sam3d_results = load_sam3d_body_results([right_sam3d_file])

    left_video_frames, right_video_frames = load_video_frames(
        left_video_path, right_video_path
    )

    n_frames = min(len(left_video_frames), len(right_video_frames))

    # TODO：这里改成当前frame+-前后几帧的方式，增加鲁棒性

    for idx in tqdm(range(0, n_frames), desc="Processing dual-view frames"):
        img_dir = out_dir / "raw_frames" / f"frame_{idx:04d}"
        img_dir.mkdir(parents=True, exist_ok=True)

        fused_cloud_point_dir = out_dir / "fused_point_clouds"
        fused_cloud_point_dir.mkdir(parents=True, exist_ok=True)

        left_bgr = cv2.cvtColor(left_video_frames[idx].numpy(), cv2.COLOR_RGB2BGR)
        right_bgr = cv2.cvtColor(right_video_frames[idx].numpy(), cv2.COLOR_RGB2BGR)

        # 保存原始帧以供后续对齐检查
        cv2.imwrite((img_dir / f"left_{idx:04d}.png").as_posix(), left_bgr)
        cv2.imwrite((img_dir / f"right_{idx:04d}.png").as_posix(), right_bgr)

        (
            camera_extrinsics,
            camera_intrinsics_resized,
            R,
            t,
            C,
            world_points_from_depth,
            preds,
        ) = camera_head.reconstruct_from_frames(
            imgs=[left_video_frames[idx], right_video_frames[idx]],
            frame_id=idx,
        )

        all_frame_camera_extrinsics.append(camera_extrinsics)
        all_frame_camera_intrinsics.append(np.asarray(camera_intrinsics_resized))
        all_frame_R.append(R)
        all_frame_t.append(t)
        all_frame_C.append(C)

        # # 使用 SAM3D 的 3D 关键点估计左右视角相对位姿（左->右）
        # relative_pose = rigid_transformation_infer.infer_from_sam3d(
        #     left_sam3d_results.get(idx),
        #     right_sam3d_results.get(idx),
        # )
        # if relative_pose is not None:
        #     all_frame_relative_R.append(relative_pose["R"])
        #     all_frame_relative_t.append(relative_pose["t"])
        #     all_frame_relative_rmse.append(relative_pose["rmse"])
        #     all_frame_relative_inliers.append(relative_pose["num_inliers"])
        #     rel_R = relative_pose["R"]
        #     rel_t = relative_pose["t"]
        # else:
        #     all_frame_relative_R.append(np.eye(3, dtype=np.float32))
        #     all_frame_relative_t.append(np.zeros((3,), dtype=np.float32))
        #     all_frame_relative_rmse.append(np.nan)
        #     all_frame_relative_inliers.append(0)
        #     rel_R = np.eye(3, dtype=np.float32)
        #     rel_t = np.zeros((3,), dtype=np.float32)

        # right_to_left_R = rel_R.T
        # right_to_left_t = -right_to_left_R @ rel_t

        # # 使用估计出来的左右视角的相对位置，调整point cloud的姿态，使其在左右视角下都对齐更好
        # # relative_pose 是 left->right，点云对齐到左坐标系时要用其逆变换 (right->left)
        # if (
        #     isinstance(world_points_from_depth, np.ndarray)
        #     and world_points_from_depth.ndim == 4
        #     and world_points_from_depth.shape[0] >= 2
        # ):
        #     left_world_map = world_points_from_depth[0]
        #     right_world_map = world_points_from_depth[1]

        #     left_world_points = _flatten_valid_points(left_world_map)
        #     right_world_points = _flatten_valid_points(right_world_map)
        #     right_world_points_aligned = (
        #         right_to_left_R @ right_world_points.T
        #     ).T + right_to_left_t[None, :]

        #     if left_world_points.size == 0:
        #         fused_world_points = right_world_points_aligned
        #     elif right_world_points_aligned.size == 0:
        #         fused_world_points = left_world_points
        #     else:
        #         fused_world_points = np.concatenate(
        #             [left_world_points, right_world_points_aligned], axis=0
        #         )

        #     if left_world_points.size > 0 and right_world_points_aligned.size > 0:
        #         centroid_error = float(
        #             np.linalg.norm(
        #                 left_world_points.mean(axis=0)
        #                 - right_world_points_aligned.mean(axis=0)
        #             )
        #         )
        #     else:
        #         centroid_error = float("nan")

        #     preds["world_points_from_depth"] = fused_world_points.astype(np.float32)
        #     preds["depth_conf"] = np.ones(
        #         (fused_world_points.shape[0],), dtype=np.float32
        #     )

        #     # 使用经过 kpt 刚体矫正后的双目相机参数（世界坐标系=左相机坐标系）
        #     # left cam: X_l = I * X_world + 0
        #     # right cam: X_r = rel_R * X_world + rel_t
        #     corrected_extrinsic = np.zeros((2, 3, 4), dtype=np.float32)
        #     corrected_extrinsic[0, :3, :3] = np.eye(3, dtype=np.float32)
        #     corrected_extrinsic[0, :3, 3] = np.zeros((3,), dtype=np.float32)
        #     corrected_extrinsic[1, :3, :3] = rel_R.astype(np.float32)
        #     corrected_extrinsic[1, :3, 3] = rel_t.astype(np.float32)

        #     if len(camera_intrinsics_resized) >= 2:
        #         corrected_intrinsic = np.stack(
        #             [
        #                 np.asarray(camera_intrinsics_resized[0], dtype=np.float32),
        #                 np.asarray(camera_intrinsics_resized[1], dtype=np.float32),
        #             ],
        #             axis=0,
        #         )
        #     elif len(camera_intrinsics_resized) == 1:
        #         k0 = np.asarray(camera_intrinsics_resized[0], dtype=np.float32)
        #         corrected_intrinsic = np.stack([k0, k0], axis=0)
        #     else:
        #         corrected_intrinsic = np.stack(
        #             [np.eye(3, dtype=np.float32), np.eye(3, dtype=np.float32)],
        #             axis=0,
        #         )

        #     preds["extrinsic"] = corrected_extrinsic
        #     preds["intrinsic"] = corrected_intrinsic

        #     # 让可视化颜色数组与融合点数量一致，避免 reshape/mask 维度不匹配
        #     preds["images"] = np.ones(
        #         (fused_world_points.shape[0], 3), dtype=np.float32
        #     )
                                                                     
        #     # 可视化融合之后的点云(保存成glb)，检查对齐效果
        #     save_inference_results(
        #         preds=preds,
        #         outdir=fused_cloud_point_dir / f"frame_{idx:04d}",
        #         conf_thres=0.0,
        #         prediction_mode="FusedCloud",
        #         frame_id=idx,
        #     )

        #     all_frame_pointcloud_centroid_error.append(centroid_error)
        #     all_frame_fused_point_count.append(int(fused_world_points.shape[0]))

        #     np.savez_compressed(
        #         (fused_cloud_point_dir / f"aligned_point_cloud_{idx}.npz").as_posix(),
        #         left_world_points=left_world_points.astype(np.float32),
        #         right_world_points_aligned=right_world_points_aligned.astype(
        #             np.float32
        #         ),
        #         fused_world_points=fused_world_points.astype(np.float32),
        #         right_to_left_R=right_to_left_R.astype(np.float32),
        #         right_to_left_t=right_to_left_t.astype(np.float32),
        #         align_centroid_error=np.asarray(centroid_error, dtype=np.float32),
        #     )
        # else:
        #     all_frame_pointcloud_centroid_error.append(np.nan)
        #     all_frame_fused_point_count.append(0)

        # # 对齐前后重投影可视化（使用 SAM 2D 观测 + SAM 3D 关键点）
        # left_frame_result = left_sam3d_results.get(idx)
        # right_frame_result = right_sam3d_results.get(idx)

        # left_kpt2d = _extract_first_person_keypoints(
        #     left_frame_result, "pred_keypoints_2d", expected_last_dim=2
        # )
        # right_kpt2d = _extract_first_person_keypoints(
        #     right_frame_result, "pred_keypoints_2d", expected_last_dim=2
        # )
        # right_kpt3d = _extract_first_person_keypoints(
        #     right_frame_result, "pred_keypoints_3d", expected_last_dim=3
        # )

        # if (
        #     left_kpt2d is not None
        #     and right_kpt2d is not None
        #     and right_kpt3d is not None
        #     and left_kpt2d.shape[0] == right_kpt2d.shape[0]
        #     and left_kpt2d.shape[0] == right_kpt3d.shape[0]
        #     and len(camera_intrinsics_resized) >= 2
        # ):
        #     right_kpt3d_aligned = (right_to_left_R @ right_kpt3d.T).T + right_to_left_t[
        #         None, :
        #     ]

        #     reproj_before = reproject_and_visualize(
        #         img1=left_bgr,
        #         img2=right_bgr,
        #         X3=right_kpt3d,
        #         kptL=left_kpt2d,
        #         kptR=right_kpt2d,
        #         K1=np.asarray(camera_intrinsics_resized[0]),
        #         dist1=None,
        #         K2=np.asarray(camera_intrinsics_resized[1]),
        #         dist2=None,
        #         R=np.asarray(R),
        #         T=np.asarray(t),
        #         out_path=img_dir / "reprojection_before_align.jpg",
        #     )

        #     reproj_after = reproject_and_visualize(
        #         img1=left_bgr,
        #         img2=right_bgr,
        #         X3=right_kpt3d_aligned,
        #         kptL=left_kpt2d,
        #         kptR=right_kpt2d,
        #         K1=np.asarray(camera_intrinsics_resized[0]),
        #         dist1=None,
        #         K2=np.asarray(camera_intrinsics_resized[1]),
        #         dist2=None,
        #         R=np.asarray(R),
        #         T=np.asarray(t),
        #         out_path=img_dir / "reprojection_after_align.jpg",
        #     )

        #     all_frame_reproj_before_mean_err_l.append(reproj_before["mean_err_L"])
        #     all_frame_reproj_before_mean_err_r.append(reproj_before["mean_err_R"])
        #     all_frame_reproj_after_mean_err_l.append(reproj_after["mean_err_L"])
        #     all_frame_reproj_after_mean_err_r.append(reproj_after["mean_err_R"])
        # else:
        #     all_frame_reproj_before_mean_err_l.append(np.nan)
        #     all_frame_reproj_before_mean_err_r.append(np.nan)
        #     all_frame_reproj_after_mean_err_l.append(np.nan)
        #     all_frame_reproj_after_mean_err_r.append(np.nan)

    save_camera_info(
        out_pt_path=inference_output_path / f"{subject}_vggt_3d_info.npz",
        all_frame_camera_intrinsics=all_frame_camera_intrinsics,
        all_frame_R=all_frame_R,
        all_frame_t=all_frame_t,
        all_frame_C=all_frame_C,
    )

    return out_dir
