#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
vggt_video_infer.py
从单个视频抽帧并执行 VGGT 推理，可作为函数调用。
"""

import logging
from concurrent.futures import ThreadPoolExecutor

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

from .load import (
    iter_dual_video_frames,
    iter_video_frames,
    load_sam3d_body_results,
)


logger = logging.getLogger(__name__)


def _process_single_view(
    flag: str,
    person_name: str,
    one_video_path,
    p_info: PersonInfo,
    cfg: DictConfig,
    progress_position: int,
) -> None:
    """处理单路视频并保存相机参数。"""

    out_dir = p_info.output_dir / flag
    out_dir.mkdir(parents=True, exist_ok=True)

    inference_output_path = p_info.inference_output_path / flag
    inference_output_path.mkdir(parents=True, exist_ok=True)

    camera_head = CameraHead(cfg, out_dir=out_dir, inference_output_dir=inference_output_path)

    all_frame_camera_extrinsics = []
    all_frame_camera_intrinsics = []
    all_frame_R = []
    all_frame_t = []
    all_frame_C = []

    for idx, frame in enumerate(
        tqdm(
            iter_video_frames(one_video_path),
            desc=f"Processing {person_name} {flag}-view frames",
            position=progress_position,
            leave=True,
        )
    ):
        bgr = cv2.cvtColor(frame.numpy(), cv2.COLOR_RGB2BGR)

        # 保存原始帧以供后续对齐检查
        img_dir = out_dir / f"frame_{idx:04d}"
        img_dir.mkdir(parents=True, exist_ok=True)
        cv2.imwrite((img_dir / f"{flag}_{idx:04d}.png").as_posix(), bgr)

        (
            camera_extrinsics,
            camera_intrinsics_resized,
            R,
            t,
            C,
            *_unused,
        ) = camera_head.reconstruct_from_frames(
            imgs=[frame],
            frame_id=idx,
        )

        all_frame_camera_extrinsics.append(camera_extrinsics)
        all_frame_camera_intrinsics.append(np.asarray(camera_intrinsics_resized))
        all_frame_R.append(R)
        all_frame_t.append(t)
        all_frame_C.append(C)

    save_camera_info(
        out_pt_path=inference_output_path / f"{person_name}_{flag}_vggt_3d_info.npz",
        all_frame_camera_intrinsics=all_frame_camera_intrinsics,
        all_frame_R=all_frame_R,
        all_frame_t=all_frame_t,
        all_frame_C=all_frame_C,
    )

    gc.collect()


def process_one_person(
    p_info: PersonInfo,
    cfg: DictConfig,
) -> None:
    """同时加载左右视频，并将同一时刻的双视角帧一起输入 VGGT。"""

    view_process = cfg.infer.view
    person_name = p_info.subject_name

    left_video_path = p_info.left_video_path
    right_video_path = p_info.right_video_path

    all_frame_camera_extrinsics = []
    all_frame_camera_intrinsics = []
    all_frame_R = []
    all_frame_t = []
    all_frame_C = []

    if view_process == "dual":

        inference_output_path = p_info.inference_output_path
        out_dir = p_info.output_dir

        inference_output_path.mkdir(parents=True, exist_ok=True)
        out_dir.mkdir(parents=True, exist_ok=True)

        camera_head = CameraHead(
            cfg, out_dir=out_dir, inference_output_dir=inference_output_path
        )

        for idx, (left_frame, right_frame) in enumerate(
            tqdm(
                iter_dual_video_frames(left_video_path, right_video_path),
                desc=f"Processing {person_name} dual-view frames",
            )
        ):
            left_bgr = cv2.cvtColor(left_frame.numpy(), cv2.COLOR_RGB2BGR)
            right_bgr = cv2.cvtColor(right_frame.numpy(), cv2.COLOR_RGB2BGR)

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
                *_unused,
            ) = camera_head.reconstruct_from_frames(
                imgs=[left_frame, right_frame],
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
        del (
            all_frame_camera_extrinsics,
            all_frame_camera_intrinsics,
            all_frame_R,
            all_frame_t,
            all_frame_C,
        )

        gc.collect()

    elif view_process == "single":
        with ThreadPoolExecutor(max_workers=2) as executor:
            left_future = executor.submit(
                _process_single_view,
                "left",
                person_name,
                left_video_path,
                p_info,
                cfg,
                0,
            )
            right_future = executor.submit(
                _process_single_view,
                "right",
                person_name,
                right_video_path,
                p_info,
                cfg,
                1,
            )

            # 触发子任务异常上抛，避免静默失败
            left_future.result()
            right_future.result()

    else:
        raise ValueError(f"Unsupported view type: {view_process}")
