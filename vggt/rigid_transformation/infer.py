#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
File: /workspace/Ski4D_Scene/vggt/rigid_transformation/infer.py
Project: /workspace/Ski4D_Scene/vggt/rigid_transformation
Created Date: Monday April 13th 2026
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Monday April 13th 2026 4:49:43 pm
Modified By: the developer formerly known as Kaixu Chen at <chenkaixusan@gmail.com>
-----
Copyright (c) 2026 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
"""

from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig


logger = logging.getLogger(__name__)


class RigidTransformationInfer:
    def __init__(self, cfg: DictConfig):
        self.cfg = cfg
        self.device = f"cuda:{cfg.infer.gpu}" if torch.cuda.is_available() else "cpu"
        if "cuda" not in self.device:
            raise RuntimeError("VGGT 需要 GPU。")
        self.verbose = cfg.runtime.get("verbose", True)

        # 估计参数
        self.min_points = int(cfg.infer.get("rigid_min_points", 6))
        self.use_ransac = bool(cfg.infer.get("rigid_use_ransac", True))
        self.ransac_iters = int(cfg.infer.get("rigid_ransac_iters", 200))
        self.ransac_thresh = float(cfg.infer.get("rigid_ransac_thresh", 0.08))

    @staticmethod
    def _ensure_points(x: Any) -> np.ndarray:
        """将输入标准化为 (N,3) float64。"""
        arr = np.asarray(x, dtype=np.float64)
        if arr.ndim == 3:
            # 常见形状: (1, J, 3)
            arr = arr.reshape(-1, arr.shape[-1])
        if arr.ndim != 2 or arr.shape[1] != 3:
            raise ValueError(f"Expect points shape (N,3), got {arr.shape}")
        return arr

    @staticmethod
    def _kabsch(src: np.ndarray, dst: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        估计 src->dst 的刚体变换: dst ~= R @ src + t

        Args:
            src: (N,3)
            dst: (N,3)

        Returns:
            R: (3,3)
            t: (3,)
        """
        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)
        src_centered = src - src_mean
        dst_centered = dst - dst_mean

        H = src_centered.T @ dst_centered
        U, _, Vt = np.linalg.svd(H)
        R = Vt.T @ U.T

        # 反射修正，确保 det(R)=1
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

    def _ransac_fit(
        self, src: np.ndarray, dst: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """RANSAC + Kabsch，返回最佳 R,t 以及内点掩码。"""
        n = src.shape[0]
        if n < self.min_points:
            raise ValueError(f"Need at least {self.min_points} points, got {n}")

        if n <= 3:
            R, t = self._kabsch(src, dst)
            inliers = np.ones((n,), dtype=bool)
            return R, t, inliers

        rng = np.random.default_rng()
        best_inliers = None
        best_count = -1

        sample_size = min(4, n)
        for _ in range(self.ransac_iters):
            idx = rng.choice(n, size=sample_size, replace=False)
            try:
                R_try, t_try = self._kabsch(src[idx], dst[idx])
            except np.linalg.LinAlgError:
                continue

            err = self._residuals(src, dst, R_try, t_try)
            inliers = err < self.ransac_thresh
            count = int(inliers.sum())

            if count > best_count:
                best_count = count
                best_inliers = inliers

        if best_inliers is None or best_count < 3:
            # 回退到全点拟合
            R, t = self._kabsch(src, dst)
            inliers = np.ones((n,), dtype=bool)
            return R, t, inliers

        R, t = self._kabsch(src[best_inliers], dst[best_inliers])
        return R, t, best_inliers

    @staticmethod
    def _extract_first_person_points(
        frame_result: Optional[Dict[str, Any]]
    ) -> Optional[np.ndarray]:
        """从单帧 SAM3D 输出中提取第一个人的 3D 关键点。"""
        if frame_result is None:
            return None

        if isinstance(frame_result, np.ndarray) and frame_result.ndim == 0:
            frame_result = frame_result.item()
        if not isinstance(frame_result, dict):
            return None

        k3d = frame_result.get("pred_keypoints_3d", None)
        if k3d is None:
            return None

        k3d = np.asarray(k3d, dtype=np.float64)
        if k3d.ndim == 2 and k3d.shape[1] == 3:
            return k3d
        if k3d.ndim == 3 and k3d.shape[0] > 0 and k3d.shape[2] == 3:
            return k3d[0]
        return None

    def infer_from_points(
        self, left_points_3d: Any, right_points_3d: Any
    ) -> Dict[str, Any]:
        """
        根据两视角对应 3D 点估计左->右相机坐标系的相对刚体关系。

        返回:
            {
              "R": (3,3),
              "t": (3,),
              "inlier_mask": (N,),
              "num_points": int,
              "num_inliers": int,
              "rmse": float,
            }
        """
        src = self._ensure_points(left_points_3d)
        dst = self._ensure_points(right_points_3d)
        if src.shape[0] != dst.shape[0]:
            raise ValueError(
                f"Point count mismatch: left={src.shape[0]}, right={dst.shape[0]}"
            )

        valid = np.isfinite(src).all(axis=1) & np.isfinite(dst).all(axis=1)
        src = src[valid]
        dst = dst[valid]

        if src.shape[0] < self.min_points:
            raise ValueError(
                f"Not enough valid correspondences: {src.shape[0]} < {self.min_points}"
            )

        if self.use_ransac:
            R, t, inliers = self._ransac_fit(src, dst)
        else:
            R, t = self._kabsch(src, dst)
            inliers = np.ones((src.shape[0],), dtype=bool)

        err = self._residuals(src, dst, R, t)
        rmse = (
            float(np.sqrt(np.mean(np.square(err[inliers]))))
            if np.any(inliers)
            else float(np.sqrt(np.mean(np.square(err))))
        )

        return {
            "R": R.astype(np.float32),
            "t": t.astype(np.float32),
            "inlier_mask": inliers,
            "num_points": int(src.shape[0]),
            "num_inliers": int(inliers.sum()),
            "rmse": rmse,
        }

    def infer_from_sam3d(
        self,
        left_frame_result: Optional[Dict[str, Any]],
        right_frame_result: Optional[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:
        """从左右单帧 SAM3D 结果中估计相对刚体关系。"""
        left_pts = self._extract_first_person_points(left_frame_result)
        right_pts = self._extract_first_person_points(right_frame_result)
        if left_pts is None or right_pts is None:
            return None

        try:
            return self.infer_from_points(left_pts, right_pts)
        except ValueError as exc:
            if self.verbose:
                logger.warning("Rigid transform skipped: %s", exc)
            return None
