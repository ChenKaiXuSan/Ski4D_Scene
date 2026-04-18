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
from omegaconf import DictConfig
from tqdm import tqdm
import gc

from vggt.reproject import reproject_and_visualize
from vggt.save import save_camera_info
from vggt.vggt.infer import CameraHead
from vggt.vis.pose_visualization import save_stereo_pose_frame, visualize_3d_joints

from vggt.rigid_transformation.infer import RigidTransformationInfer

from .prepare_paths import PersonInfo

from .load import load_sam3d_body_results, load_video_frames


logger = logging.getLogger(__name__)


def process_one_person(
    p_info: PersonInfo,
    cfg: DictConfig,
) -> Optional[Path]:
    """同时加载左右视频，并将同一时刻的双视角帧一起输入 VGGT。"""

    person_name = p_info.subject_name

    left_video_path = p_info.left_video_path
    right_video_path = p_info.right_video_path

    left_sam3d_body_results_path = p_info.left_sam3d_body_results_path
    right_sam3d_body_results_path = p_info.right_sam3d_body_results_path

    inference_output_path = p_info.inference_output_path
    out_dir = p_info.output_dir

    inference_output_path.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[Run-DV] Processing dual-view frames -> {out_dir}")

    camera_head = CameraHead(
        cfg, out_dir=out_dir, inference_output_dir=inference_output_path
    )
    rigid_transformation_infer = RigidTransformationInfer(cfg)

    all_frame_camera_extrinsics = []
    all_frame_camera_intrinsics = []
    all_frame_R = []
    all_frame_t = []
    all_frame_C = []

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

    for idx in tqdm(
        range(0, n_frames), desc=f"Processing {person_name} dual-view frames"
    ):

        left_bgr = cv2.cvtColor(left_video_frames[idx].numpy(), cv2.COLOR_RGB2BGR)
        right_bgr = cv2.cvtColor(right_video_frames[idx].numpy(), cv2.COLOR_RGB2BGR)

        # 保存原始帧以供后续对齐检查
        img_dir = out_dir / f"frame_{idx:04d}"
        img_dir.mkdir(parents=True, exist_ok=True)
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

    save_camera_info(
        out_pt_path=inference_output_path / f"{person_name}_vggt_3d_info.npz",
        all_frame_camera_intrinsics=all_frame_camera_intrinsics,
        all_frame_R=all_frame_R,
        all_frame_t=all_frame_t,
        all_frame_C=all_frame_C,
    )

    # 清空大对象，释放内存
    del left_video_frames, right_video_frames
    del (
        all_frame_camera_extrinsics,
        all_frame_camera_intrinsics,
        all_frame_R,
        all_frame_t,
        all_frame_C,
    )

    del world_points_from_depth, preds

    gc.collect()

    return out_dir