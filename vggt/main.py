#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
Batch process: run VGGT-based multi-view reconstruction for each subject.
(Single-thread version: no multithreading)

Author: Kaixu Chen
Last Modified: 2025-11-25
"""

import logging
import os
from pathlib import Path
from typing import Dict, List, Tuple

import hydra
from omegaconf import DictConfig, OmegaConf

from .single_view_process import process_dual_view_video

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Utility
# --------------------------------------------------------------------------- #
def find_files(
    subject_dir: Path,
    patterns: List[str],
    recursive: bool = False,
) -> List[Path]:
    """在 subject_dir 下按模式查找文件（视频或 pt）。"""
    files: List[Path] = []
    if recursive:
        for pat in patterns:
            files.extend(subject_dir.rglob(pat))
    else:
        for pat in patterns:
            files.extend(subject_dir.glob(pat))
    return sorted({f.resolve() for f in files})


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
@hydra.main(config_path="../configs", config_name="vggt", version_base=None)
def main(cfg: DictConfig) -> None:
    # logging 设置
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.setLevel(logging.INFO)

    logger.info("==== Config ====\n" + OmegaConf.to_yaml(cfg))

    # 读取路径
    video_root = Path(cfg.paths.video_path).resolve()
    log_out_root = Path(cfg.paths.log_path).resolve()
    inference_output_path = Path(cfg.paths.inference_output_path).resolve()

    if not video_root.exists():
        raise FileNotFoundError(f"video_path not found: {video_root}")

    log_out_root.mkdir(parents=True, exist_ok=True)
    inference_output_path.mkdir(parents=True, exist_ok=True)

    recursive = bool(cfg.dataset.get("recursive", False))

    # 搜索 patterns
    vid_patterns = ["*.mp4", "*.mov", "*.avi", "*.mkv", "*.MP4", "*.MOV"]

    # ---------------------------------------------------------------------- #
    # 扫描 video_root
    # ---------------------------------------------------------------------- #
    subjects_video = sorted([p for p in video_root.iterdir() if p.is_dir()])

    if not subjects_video:
        raise FileNotFoundError(f"No subject folders under: {video_root}")

    logger.info(f"Found {len(subjects_video)} subjects in: {video_root}")

    # { subject_name: [video files] }
    videos_map: Dict[str, List[Path]] = {}
    for subject_dir in subjects_video:
        vids = find_files(subject_dir, vid_patterns, recursive)
        if vids:
            videos_map[subject_dir.name] = vids
        else:
            logger.warning(f"[No video] {subject_dir}")

    # ---------------------------------------------------------------------- #
    # 构建 dual-view 任务（每个 subject 取左右两个视频）
    # ---------------------------------------------------------------------- #
    dual_tasks: List[Tuple[str, Path, Path]] = []
    for subject_name, vids in videos_map.items():
        if len(vids) >= 2:
            # 沿用历史约定：vids[1] 为 left，vids[0] 为 right
            dual_tasks.append((subject_name, vids[1], vids[0]))
        else:
            logger.warning(f"[Skip] {subject_name}: need >=2 videos for dual-view")

    logger.info(f"Total subjects with videos: {len(videos_map)}")
    logger.info(f"Total dual-view pairs: {len(dual_tasks)}")

    if not dual_tasks:
        logger.info("No valid dual-view pairs found. EXIT.")
        logger.info("==== ALL DONE ====")
        return

    # ---------------------------------------------------------------------- #
    # 顺序执行（无多线程）
    # ---------------------------------------------------------------------- #
    for subject_name, left_video_path, right_video_path in dual_tasks:
        logger.info(
            f"[Subject: {subject_name}] Dual-view START: {left_video_path.name} & {right_video_path.name}"
        )
        out_dir = process_dual_view_video(
            left_video_path=left_video_path,
            right_video_path=right_video_path,
            out_root=log_out_root,
            inference_output_path=inference_output_path,
            cfg=cfg,
        )

        if out_dir is None:
            logger.error(
                f"[Subject: {subject_name}] Dual-view FAILED: {left_video_path.name} & {right_video_path.name}"
            )

    logger.info("==== ALL DONE ====")


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
