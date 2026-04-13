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

import dataclasses
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torchvision import transforms as TF
from torchvision.io import read_video
# from .prepare_paths import PersonInfo

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


def load_sam3d_body_results(pt_file_path: List[str]) -> Dict[int, Any]:
    """
    Load SAM3D Body inference results from NPZ file.

    The NPZ file contains frame-by-frame inference results with 3D
    keypoints.

    Args:
        pt_file_path (str): Path to the SAM3D body results NPZ file.

    Returns:
        Dict[int, Any]: Dictionary mapping frame indices to predictions.
                       Each frame entry contains:
                       - pred_keypoints_3d: 3D keypoint coordinates
                         shape (N_people, N_joints, 3)
                       - pred_keypoints_2d: 2D keypoint coordinates
                         shape (N_people, N_joints, 2)
                       - pred_cam_t: Camera translation vector
                       - Other fields from SAM3D output

    Raises:
        FileNotFoundError: If the NPZ file does not exist.
        KeyError: If required data keys are missing in NPZ file.
    """

    res = {}

    for one_file in pt_file_path:
        data = np.load(one_file, allow_pickle=True)

        # Handle different possible key names
        if "outputs" in data.files:
            outputs_data = data["outputs"]
        elif "output" in data.files:
            outputs_data = data["output"]
        else:
            raise KeyError(
                f"Neither 'outputs' nor 'output' found in NPZ. "
                f"Available keys: {list(data.files)}"
            )

        # Convert to dictionary indexed by frame number
        results = {}
        if isinstance(outputs_data, np.ndarray):
            # Case 1: outputs is an array of dictionaries/results
            for idx, frame_result in enumerate(outputs_data):
                # Handle wrapped numpy arrays
                if isinstance(frame_result, np.ndarray) and frame_result.ndim == 0:
                    frame_result = frame_result.item()
                results[idx] = frame_result

        elif isinstance(outputs_data, dict):
            # Case 2: outputs is already a dictionary
            results = outputs_data
        else:
            logger.warning(
                f"Unexpected outputs type: {type(outputs_data)}, "
                "attempting to convert to dict"
            )
            results = dict(enumerate(outputs_data))

        res.update(results)

    return res


def load_video_frames(
    left_video_path: Path, right_video_path: Path
) -> Tuple[torch.Tensor, torch.Tensor]:
    """加载视频帧并返回一个形状为 (N, H, W, C) 的张量。"""
    left_video_frames, _, info = read_video(
        left_video_path.as_posix(), pts_unit="sec", output_format="THWC"
    )
    right_video_frames, _, info = read_video(
        right_video_path.as_posix(), pts_unit="sec", output_format="THWC"
    )
    return left_video_frames, right_video_frames
