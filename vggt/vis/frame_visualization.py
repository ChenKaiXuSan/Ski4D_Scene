#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/triangulation/vis/3d_visualization copy.py
Project: /workspace/code/triangulation/vis
Created Date: Wednesday September 24th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Wednesday September 24th 2025 10:45:33 am
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np
import torch


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

# --------------------------- 小工具 ---------------------------


def _to_numpy_img(frame: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
    """
    接受 HxW 或 HxWxC（RGB/灰度）:
      - torch.Tensor 会被搬到 CPU，并转换为 numpy
      - float 类型若范围在 [0,1]，会映射到 [0,255]
      - 输出为 HxWx3（BGR 或 RGB 由调用者决定）
    """
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()

    img = frame
    if img.ndim == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)  # 统一成 3 通道(RGB)
    elif img.ndim == 3 and img.shape[2] == 1:
        img = np.repeat(img, 3, axis=2)

    if img.dtype != np.uint8:
        vmax = float(np.nanmax(img)) if img.size else 1.0
        if vmax <= 1.0:
            img = (img * 255.0).clip(0, 255).astype(np.uint8)
        else:
            img = img.clip(0, 255).astype(np.uint8)
    return img


def _ensure_dir(path: Union[str, Path]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _valid_xy(pt: Sequence[float]) -> bool:
    return len(pt) >= 2 and np.isfinite(pt[0]) and np.isfinite(pt[1])


# --------------------------- 2D 关键点绘制 ---------------------------


def draw_and_save_keypoints_from_frame(
    frame: Union[np.ndarray, torch.Tensor],
    keypoints: Union[np.ndarray, torch.Tensor, Sequence[Sequence[float]]],
    save_path: Union[str, Path],
    *,
    color: Tuple[int, int, int] = (0, 255, 0),
    radius: int = 4,
    thickness: int = -1,
    with_index: bool = True,
    skeleton: Optional[Iterable[Tuple[int, int]]] = COCO_SKELETON,
    scores: Optional[Union[np.ndarray, torch.Tensor]] = None,
    score_thresh: Optional[float] = None,
    assume_input_is_rgb: bool = True,
) -> None:
    """
    在单帧上绘制 (K,2) 的关键点并保存。

    Parameters
    ----------
    frame : np.ndarray | torch.Tensor
        HxW 或 HxWxC，RGB/灰度均可；float 会自动缩放到 uint8。
    keypoints : array-like, shape=(K,2 or >=2)
        x,y(,?)；若包含 >=3 维也只取前两维。
    save_path : str | Path
        输出文件路径。
    color : BGR 颜色（cv2），默认绿色。
    thickness : -1 表示填充圆点。
    skeleton : None 或边索引列表
    scores : 可选 (K,) 或 (K,1)；若提供且有 score_thresh，会据此过滤绘制。
    score_thresh : 置信度阈值。
    assume_input_is_rgb : bool
        True 时保存前会把 RGB->BGR；False 则按当前通道顺序直接写。
    """
    save_path = Path(save_path)
    _ensure_dir(save_path)

    img_rgb = _to_numpy_img(frame)  # HxWx3, uint8, RGB
    img = (
        cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
        if assume_input_is_rgb
        else img_rgb.copy()
    )

    kpts = torch.as_tensor(keypoints).detach().cpu().numpy()
    if kpts.ndim != 2 or kpts.shape[1] < 2:
        raise ValueError(f"keypoints shape must be (K,>=2), got {kpts.shape}")
    kxy = kpts[:, :2]

    # 分数过滤（若提供）
    mask = np.ones(len(kxy), dtype=bool)
    if scores is not None and score_thresh is not None:
        sc = torch.as_tensor(scores).detach().cpu().numpy().reshape(-1)
        if sc.shape[0] != kxy.shape[0]:
            raise ValueError(f"scores len {sc.shape[0]} != keypoints K {kxy.shape[0]}")
        mask = sc >= float(score_thresh)
    kxy_draw = kxy[mask]

    # 画点
    for idx, (x, y) in enumerate(kxy_draw):
        if not _valid_xy((x, y)):
            continue
        c = (int(round(x)), int(round(y)))
        cv2.circle(img, c, radius, color, thickness, lineType=cv2.LINE_AA)
        if with_index:
            cv2.putText(
                img,
                str(idx),
                (c[0] + 4, c[1] - 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.4,
                (255, 255, 255),
                1,
                lineType=cv2.LINE_AA,
            )

    # 画骨架
    if skeleton is not None:
        for i, j in skeleton:
            if (
                i < len(kxy)
                and j < len(kxy)
                and _valid_xy(kxy[i])
                and _valid_xy(kxy[j])
            ):
                p1 = (int(round(kxy[i, 0])), int(round(kxy[i, 1])))
                p2 = (int(round(kxy[j, 0])), int(round(kxy[j, 1])))
                cv2.line(img, p1, p2, (0, 255, 255), 2, lineType=cv2.LINE_AA)

    cv2.imwrite(str(save_path), img)
    print(f"[INFO] Saved image with keypoints to: {save_path}")
