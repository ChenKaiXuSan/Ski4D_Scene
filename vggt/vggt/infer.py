#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/code/vggt/vggt/infer.py
Project: /workspace/code/vggt/vggt
Created Date: Tuesday November 25th 2025
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Tuesday November 25th 2025 2:22:14 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2025 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from pathlib import Path
import logging
from typing import Any, Dict, List
from omegaconf import DictConfig

import numpy as np
import torch

# 依赖 VGGT 官方模块
from vggt.vggt.models.vggt import VGGT
from vggt.vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.vggt.utils.geometry import unproject_depth_map_to_point_map

from vggt.load import load_and_preprocess_images

from vggt.vis.visual_util import predictions_to_glb
from vggt.vis.vggt_camera_vis import plot_cameras_from_predictions


logger = logging.getLogger(__name__)


class CameraHead:
    def __init__(self, cfg: DictConfig, out_dir: Path, inference_output_dir: Path):
        super().__init__()
        self.device = f"cuda:{cfg.infer.gpu}" if torch.cuda.is_available() else "cpu"
        if "cuda" not in self.device:
            raise RuntimeError("VGGT 需要 GPU。")
        verbose = cfg.runtime.get("verbose", True)

        self.vggt = self.load_vggt_model(self.device, verbose=verbose)
        self.outdir = out_dir
        self.inference_output_dir = inference_output_dir

        self.conf_thres = cfg.get("conf_thres", 50.0)
        self.prediction_mode = cfg.get("prediction_mode", "All")

    @staticmethod
    def load_vggt_model(device="cuda", verbose=True):
        """加载预训练 VGGT 模型"""
        model = VGGT()
        url = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
        state = torch.hub.load_state_dict_from_url(
            url, map_location="cpu", progress=verbose
        )
        model.load_state_dict(state)
        model.eval().to(device)
        return model

    @torch.no_grad()
    def run_vggt(
        self,
        images: List[torch.Tensor],
    ) -> tuple[Dict[str, Any], int, int]:
        """对图像列表执行 VGGT 推理"""
        imgs = load_and_preprocess_images(images).to(self.device)
        dtype = (
            torch.bfloat16
            if torch.cuda.get_device_capability()[0] >= 8
            else torch.float16
        )
        with torch.cuda.amp.autocast(dtype=dtype):
            preds = self.vggt(imgs)

        H, W = imgs.shape[-2:]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(preds["pose_enc"], (H, W))
        preds["extrinsic"] = extrinsic
        preds["intrinsic"] = intrinsic

        # 转 numpy
        out = {
            k: (
                v.detach().cpu().numpy().squeeze(0)
                if isinstance(v, torch.Tensor)
                else v
            )
            for k, v in preds.items()
        }
        out["pose_enc_list"] = None
        depth = out["depth"]
        out["world_points_from_depth"] = unproject_depth_map_to_point_map(
            depth, out["extrinsic"], out["intrinsic"]
        )
        return out, H, W

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

    @staticmethod
    def scale_intrinsics(
        K: np.ndarray, orig_size: tuple[int, int], new_size: tuple[int, int]
    ) -> np.ndarray:
        """
        当图像从原始分辨率 resize 到新分辨率时，同步缩放相机内参 K。

        K: (3,3) 内参矩阵，像素坐标
        [[fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]]
        orig_size: (H, W) 原始图像分辨率
        new_size:  (H, W) 新图像分辨率

        返回缩放后的 K_new
        """
        H0, W0 = orig_size
        H1, W1 = new_size

        sx = W1 / W0  # 水平方向缩放比例
        sy = H1 / H0  # 垂直方向缩放比例

        K_new = K.copy().astype(float)
        K_new[0, 0] *= sx  # fx
        K_new[1, 1] *= sy  # fy
        K_new[0, 2] *= sx  # cx
        K_new[1, 2] *= sy  # cy

        return K_new

    def reconstruct_from_frames(
        self,
        frame_id: int,
        imgs: list[torch.Tensor],
    ) -> tuple[
        np.ndarray,
        list[np.ndarray],
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
        Dict[str, Any],
    ]:
        """
        从给定的帧图像列表中执行 VGGT 推理，保存结果，并返回相机参数和点云等信息。
        """

        log_dir = self.outdir / f"frame_{frame_id:04d}"
        log_dir.mkdir(parents=True, exist_ok=True)

        H, W = imgs[0].shape[:2]

        # 推理
        preds, orig_h, orig_w = self.run_vggt(imgs)

        plot_cameras_from_predictions(
            predictions=preds,
            out_path=log_dir / "camera_poses.png",
            axis_len=0.1,
            include_points=False,  # 想看点云就开
            center_mode="mean",  # 不以相机为原点，而是整体居中
        )

        # 保存结果
        self.save(
            preds=preds,
            frame_id=frame_id,
            conf_thres=self.conf_thres,
            prediction_mode=self.prediction_mode,
            log_out_dir=log_dir,
            inference_output_dir=self.inference_output_dir,
        )

        # 这里是相机在世界坐标系下的 R,t,C
        camera_extrinsics = preds["extrinsic"]
        camera_intrinsics = preds["intrinsic"]
        R, t, C = self.extrinsic_to_RT(camera_extrinsics)
        world_points_from_depth = preds["world_points_from_depth"]  # (N,3)

        camera_intrinsics_resized = []
        for i in range(len(camera_intrinsics)):
            camera_intrinsics_resized.append(
                self.scale_intrinsics(
                    camera_intrinsics[i],
                    orig_size=(orig_h, orig_w),
                    new_size=(H, W),
                )
            )

        return (
            camera_extrinsics,
            camera_intrinsics_resized,
            R,
            t,
            C,
            world_points_from_depth,
            preds,
        )

    def save(
        self,
        preds: dict,
        frame_id: int,
        conf_thres: float,
        prediction_mode: str,
        log_out_dir: Path,
        inference_output_dir: Path,
    ) -> None:
        """
        保存 VGGT 推理结果：
        1. 保存 npz
        2. 导出 glb

        Args:
            preds: VGGT 推理结果
            outdir: 输出目录
            imgs: 输入图像列表
            conf_thres: 导出 glb 时的置信度阈值
            prediction_mode: 导出 glb 时的预测模式
            frame_id: 当前帧号（可选）

        Returns:
            None
        """
        log_out_dir.mkdir(parents=True, exist_ok=True)
        inference_output_dir.mkdir(parents=True, exist_ok=True)

        # 保存 npz
        npz_path = inference_output_dir / f"frame_{frame_id:04d}_predictions.npz"
        np.savez(npz_path, **preds)

        # 导出 glb
        frame_tag = f"_frame{frame_id:04d}" if frame_id is not None else ""
        glb_path = log_out_dir / (
            "scene"
            f"{frame_tag}_conf{conf_thres}"
            f"_mode{prediction_mode.replace(' ', '_')}.glb"
        )
        glb = predictions_to_glb(
            preds,
            conf_thres=conf_thres,
            filter_by_frames="All",
            show_cam=True,
            mask_black_bg=False,
            mask_white_bg=False,
            mask_sky=False,
            target_dir=log_out_dir,
            prediction_mode=prediction_mode,
        )
        glb.export(file_obj=glb_path)
        logger.info(f"Saved GLB → {glb_path}")
