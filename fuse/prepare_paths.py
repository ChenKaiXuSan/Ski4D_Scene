#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/Ski4D_Scene/vggt/prepare_paths.py
Project: /workspace/Ski4D_Scene/vggt
Created Date: Monday April 13th 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Monday April 13th 2026 2:31:16 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import dataclasses
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from omegaconf import DictConfig

logger = logging.getLogger(__name__)


@dataclasses.dataclass
class PersonInfo:
    subject_name: str
    left_video_path: Path
    right_video_path: Path
    left_sam3d_body_results_path: List[Path]
    right_sam3d_body_results_path: List[Path]
    left_vggt_results_path: List[Path]
    right_vggt_results_path: List[Path]
    output_dir: Path
    inference_output_path: Path


# --------------------------------------------------------------------------- #
# Utility
# --------------------------------------------------------------------------- #
def find_files(
    subject_dir: Path,
) -> List[Path]:
    """在 subject_dir 查找文件。
    如果是一个文件的话，就返回该文件的路径。
    如果是一个目录的话，就组装成为list。
    """
    files: List[Path] = []
    if subject_dir.is_file():
        files.append(subject_dir)
    elif subject_dir.is_dir():
        for f in subject_dir.iterdir():
            if f.is_file():
                files.append(f)
    else:
        logger.warning(f"Path is neither file nor directory: {subject_dir}")

    return sorted({f.resolve() for f in files})


def find_left_right(
    subject_dir: Path,
) -> Tuple[Optional[List[Path]], Optional[List[Path]]]:
    """在 subject_dir 查找 left/right 视频文件。
    返回 (left_video_path, right_video_path)，如果找不到则为 None。
    """
    left_video_path = None
    right_video_path = None

    for f in subject_dir.iterdir():
        if f.is_file():
            name_lower = f.name.lower()
            if "left" in name_lower or "osmo_2" in name_lower:
                left_video_path = f.resolve()
            elif "right" in name_lower or "osmo_1" in name_lower:
                right_video_path = f.resolve()

        if f.is_dir():
            if "left" in f.name.lower() or "osmo_2" in f.name.lower():
                left_video_path = find_files(f)
            elif "right" in f.name.lower() or "osmo_1" in f.name.lower():
                right_video_path = find_files(f)

    return left_video_path, right_video_path


def prepare_paths(cfg: DictConfig):
    """按照PersonInfo的格式准备路径并返回相关映射和目录。
    video path和sam3d body results path有可能是两个文件或者是一个文件
    """

    res_paths = {}

    # 读取路径
    video_root = Path(cfg.paths.video_path).resolve()
    sam_3d_body_results_root = Path(cfg.paths.sam3d_body_results_path).resolve()
    vggt_results_root = Path(cfg.paths.vggt_results_path).resolve()
    log_out_root = Path(cfg.log_path).resolve()
    inference_output_path = Path(cfg.paths.inference_output_path).resolve()

    if not video_root.exists():
        raise FileNotFoundError(f"video path not found: {video_root}")
    if not sam_3d_body_results_root.exists():
        raise FileNotFoundError(
            f"sam3d body results path not found: {sam_3d_body_results_root}"
        )
    if not vggt_results_root.exists():
        raise FileNotFoundError(f"vggt results path not found: {vggt_results_root}")

    # 1. 对比video root和sam3d_body_results_root的subject是否相同，准备subject
    subjects_video = sorted([p for p in video_root.iterdir() if p.is_dir()])
    sam_subjects = sorted([p for p in sam_3d_body_results_root.iterdir() if p.is_dir()])
    vggt_subjects = sorted([p for p in vggt_results_root.iterdir() if p.is_dir()])

    # 对比 video、sam、vggt 的 subject，取交集
    video_subject_names = {p.name for p in subjects_video}
    sam_subject_names = {p.name for p in sam_subjects}
    vggt_subject_names = {p.name for p in vggt_subjects}
    common_subjects = video_subject_names.intersection(sam_subject_names).intersection(
        vggt_subject_names
    )
    if not common_subjects:
        raise ValueError(
            f"No common subjects between video and SAM 3D body results:\n"
            f"  Video subjects: {sorted(video_subject_names)}\n"
            f"  SAM subjects: {sorted(sam_subject_names)}\n"
            f"  VGGT subjects: {sorted(vggt_subject_names)}"
        )
    logger.info(
        f"Found {len(common_subjects)} common subjects between video and SAM 3D body results"
    )

    for one_subject in common_subjects:

        video_subject_dir = video_root / one_subject
        sam_subject_dir = sam_3d_body_results_root / one_subject
        vggt_subject_dir = vggt_results_root / one_subject

        person_log_out_root = log_out_root / one_subject
        person_inference_output_path = inference_output_path / one_subject

        # 2. 构建 subject -> [video files] 的映射
        left_video_path, right_video_path = find_left_right(video_subject_dir)

        # 3. 构建 sam 3d_body_results_path 的目录结构输出
        left_sam3d_body_results_path, right_sam3d_body_results_path = find_left_right(
            sam_subject_dir
        )

        # 4. 构建 vggt results path 的目录结构输出
        left_vggt_results_path, right_vggt_results_path = find_left_right(
            vggt_subject_dir
        )

        res_paths[one_subject] = PersonInfo(
            subject_name=one_subject,
            left_video_path=left_video_path,
            right_video_path=right_video_path,
            left_sam3d_body_results_path=left_sam3d_body_results_path,
            right_sam3d_body_results_path=right_sam3d_body_results_path,
            left_vggt_results_path=left_vggt_results_path,
            right_vggt_results_path=right_vggt_results_path,
            output_dir=person_log_out_root,
            inference_output_path=person_inference_output_path,
        )

    return res_paths
