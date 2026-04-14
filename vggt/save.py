#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/vggt/save.py
Project: /workspace/code/vggt
Created Date: Monday November 24th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Monday November 24th 2025 4:37:41 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from pathlib import Path
import numpy as np
import logging

logger = logging.getLogger(__name__)


# update 3d information to pt file
def save_camera_info(
    out_pt_path: Path,
    all_frame_camera_intrinsics: list[np.ndarray],
    all_frame_R: list[np.ndarray],
    all_frame_t: list[np.ndarray],
    all_frame_C: list[np.ndarray],
):
    """
    更新 pt 文件，添加 3D 信息

    Args:
        out_pt_path: 输出 pt 文件路径
        reprojet_err: 重投影误差字典
    """

    data = {
        "camera_intrinsics": np.stack(
            all_frame_camera_intrinsics, axis=0
        ),  # (N, C, 3, 3)
        "R": np.stack(all_frame_R, axis=0),  # (N, C, 3, 3)
        "t": np.stack(all_frame_t, axis=0),  # (N, C, 3)
        "C": np.stack(all_frame_C, axis=0),  # (N, C, 3)
    }

    # 保存更新后的 pt 文件
    np.savez_compressed(out_pt_path.with_suffix(".npz"), **data)
    logger.info(f"Updated PT with 3D info → {out_pt_path}")
