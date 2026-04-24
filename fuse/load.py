#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/vggt/load_fn.py
Project: /workspace/code/vggt
Created Date: Friday November 21st 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Friday November 21st 2025 2:42:14 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import torch
from PIL import Image
from torchvision import transforms as TF
from torchvision.io import VideoReader, read_video

logger = logging.getLogger(__name__)


def load_and_preprocess_images(image_path_list, mode="crop"):
    """
    A quick start function to load and preprocess images for model input.
    This assumes the images should have the same shape for easier batching, but our model can also work well with different shapes.

    Args:
        image_path_list (list): List of paths to image files
        mode (str, optional): Preprocessing mode, either "crop" or "pad".
                             - "crop" (default): Sets width to 518px and center crops height if needed.
                             - "pad": Preserves all pixels by making the largest dimension 518px
                               and padding the smaller dimension to reach a square shape.

    Returns:
        torch.Tensor: Batched tensor of preprocessed images with shape (N, 3, H, W)

    Raises:
        ValueError: If the input list is empty or if mode is invalid

    Notes:
        - Images with different dimensions will be padded with white (value=1.0)
        - A warning is printed when images have different shapes
        - When mode="crop": The function ensures width=518px while maintaining aspect ratio
          and height is center-cropped if larger than 518px
        - When mode="pad": The function ensures the largest dimension is 518px while maintaining aspect ratio
          and the smaller dimension is padded to reach a square shape (518x518)
        - Dimensions are adjusted to be divisible by 14 for compatibility with model requirements
    """
    # Check for empty list
    if len(image_path_list) == 0:
        raise ValueError("At least 1 image is required")

    # Validate mode
    if mode not in ["crop", "pad"]:
        raise ValueError("Mode must be either 'crop' or 'pad'")

    images = []
    shapes = set()
    to_tensor = TF.ToTensor()
    target_size = 518

    # First process all images and collect their shapes
    for image_path in image_path_list:
        # Open image
        # img = Image.open(image_path)
        img = image_path.numpy()
        img = Image.fromarray(img)

        # If there's an alpha channel, blend onto white background:
        if img.mode == "RGBA":
            # Create white background
            background = Image.new("RGBA", img.size, (255, 255, 255, 255))
            # Alpha composite onto the white background
            img = Image.alpha_composite(background, img)

        # Now convert to "RGB" (this step assigns white for transparent areas)
        img = img.convert("RGB")

        width, height = img.size

        if mode == "pad":
            # Make the largest dimension 518px while maintaining aspect ratio
            if width >= height:
                new_width = target_size
                new_height = (
                    round(height * (new_width / width) / 14) * 14
                )  # Make divisible by 14
            else:
                new_height = target_size
                new_width = (
                    round(width * (new_height / height) / 14) * 14
                )  # Make divisible by 14
        else:  # mode == "crop"
            # Original behavior: set width to 518px
            new_width = target_size
            # Calculate height maintaining aspect ratio, divisible by 14
            new_height = round(height * (new_width / width) / 14) * 14

        # Resize with new dimensions (width, height)
        img = img.resize((new_width, new_height), Image.Resampling.BICUBIC)
        img = to_tensor(img)  # Convert to tensor (0, 1)

        # Center crop height if it's larger than 518 (only in crop mode)
        if mode == "crop" and new_height > target_size:
            start_y = (new_height - target_size) // 2
            img = img[:, start_y : start_y + target_size, :]

        # For pad mode, pad to make a square of target_size x target_size
        if mode == "pad":
            h_padding = target_size - img.shape[1]
            w_padding = target_size - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                # Pad with white (value=1.0)
                img = torch.nn.functional.pad(
                    img,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=1.0,
                )

        shapes.add((img.shape[1], img.shape[2]))
        images.append(img)

    # Check if we have different shapes
    # In theory our model can also work well with different shapes
    if len(shapes) > 1:
        print(f"Warning: Found images with different shapes: {shapes}")
        # Find maximum dimensions
        max_height = max(shape[0] for shape in shapes)
        max_width = max(shape[1] for shape in shapes)

        # Pad images if necessary
        padded_images = []
        for img in images:
            h_padding = max_height - img.shape[1]
            w_padding = max_width - img.shape[2]

            if h_padding > 0 or w_padding > 0:
                pad_top = h_padding // 2
                pad_bottom = h_padding - pad_top
                pad_left = w_padding // 2
                pad_right = w_padding - pad_left

                img = torch.nn.functional.pad(
                    img,
                    (pad_left, pad_right, pad_top, pad_bottom),
                    mode="constant",
                    value=1.0,
                )
            padded_images.append(img)
        images = padded_images

    images = torch.stack(images)  # concatenate images

    # Ensure correct shape when single image
    if len(image_path_list) == 1:
        # Verify shape is (1, C, H, W)
        if images.dim() == 3:
            images = images.unsqueeze(0)

    return images


def load_sam3d_body_results(pt_file_path: Path) -> np.ndarray:
    """
    Load SAM3D Body inference results from NPZ file.
    """

    data = np.load(pt_file_path, allow_pickle=True)

    if "outputs" in data.files:
        data = data["outputs"].item()  # Assuming outputs is a single dictionary

    bbox = data["bbox"]  # shape (N_people, 4) in (x_min, y_min, x_max, y_max) format
    pred_keypoints_2d = data["pred_keypoints_2d"]  # shape (N_people, N_joints, 2)
    pred_keypoints_3d = data["pred_keypoints_3d"]  # shape (N_people, N_joints, 3)

    return pred_keypoints_2d


def load_vggt_results(pt_file_path: Path) -> Dict[int, Any]:
    """
    Load VGGT inference results from NPZ file.

    """

    data = np.load(pt_file_path, allow_pickle=True)

    # Handle different possible key names
    if "outputs" in data.files:
        outputs_data = data["outputs"]
    elif "output" in data.files:
        outputs_data = data["output"]
    else:
        outputs_data = data  # Assume the whole NPZ is the outputs dict

    return outputs_data.item() if isinstance(outputs_data, np.ndarray) else outputs_data


def rot_point_cloud(
    point_cloud: np.ndarray, extrinsics: np.ndarray, flag: str = "right"
) -> Tuple[np.ndarray, np.ndarray]:

    # 定义旋转矩阵
    if flag == "right":
        theta = -np.pi / 2  # 向右（顺时针）
    elif flag == "left":
        theta = np.pi / 2  # 向左（逆时针）
    else:
        raise ValueError("Flag must be either 'right' or 'left'")

    R_y = np.array(
        [
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)],
        ],
        dtype=np.float64,
    )

    rot_world_pts = point_cloud @ R_y.T

    # * 旋转 R, 暂时保持t不变
    rot_extrinsics = extrinsics.copy()
    rot_extrinsics[:, :3, :3] = extrinsics[:, :3, :3] @ R_y.T

    return rot_world_pts, rot_extrinsics


def iter_video_frames(video_path: Path) -> Iterator[torch.Tensor]:
    """按帧流式加载视频，返回 RGB 的 HWC uint8 张量。"""

    if not video_path.exists():
        raise FileNotFoundError(f"Failed to open video: {video_path}")

    reader = VideoReader(video_path.as_posix(), "video")
    for frame in reader:
        data = frame["data"]
        if data.ndim == 3 and data.shape[0] in (1, 3) and data.shape[-1] not in (1, 3):
            data = data.permute(1, 2, 0).contiguous()
        yield data


def iter_dual_video_frames(
    left_video_path: Path, right_video_path: Path
) -> Iterator[Tuple[torch.Tensor, torch.Tensor]]:
    """按帧同步流式加载双视角视频，遇到任一视频结束即停止。"""

    if not left_video_path.exists():
        raise FileNotFoundError(f"Failed to open left video: {left_video_path}")
    if not right_video_path.exists():
        raise FileNotFoundError(f"Failed to open right video: {right_video_path}")

    left_reader = VideoReader(left_video_path.as_posix(), "video")
    right_reader = VideoReader(right_video_path.as_posix(), "video")

    for left_frame, right_frame in zip(left_reader, right_reader):
        left_data = left_frame["data"]
        right_data = right_frame["data"]

        if (
            left_data.ndim == 3
            and left_data.shape[0] in (1, 3)
            and left_data.shape[-1] not in (1, 3)
        ):
            left_data = left_data.permute(1, 2, 0).contiguous()
        if (
            right_data.ndim == 3
            and right_data.shape[0] in (1, 3)
            and right_data.shape[-1] not in (1, 3)
        ):
            right_data = right_data.permute(1, 2, 0).contiguous()

        yield left_data, right_data
