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
from tqdm import tqdm
from pathlib import Path
from typing import Dict, List, Tuple
import re

import hydra
from omegaconf import DictConfig, OmegaConf

from .prepare_paths import PersonInfo, prepare_paths
from .vggt_fuse import (
    resolve_frame_save_path,
    analyze_frame,
)

logger = logging.getLogger(__name__)


def _collect_frame_indices(frame_dir: Path, pattern: str) -> set[int]:
    frame_indices: set[int] = set()
    for path in frame_dir.glob(pattern):
        match = re.compile(r"frame_(\d+)").search(path.stem)
        if match is not None:
            frame_indices.add(int(match.group(1)))
    return frame_indices


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
@hydra.main(config_path="../configs", config_name="fuse", version_base=None)
def main(cfg: DictConfig) -> None:
    # logging 设置
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.setLevel(logging.INFO)

    logger.info("==== Config ====\n" + OmegaConf.to_yaml(cfg))

    # 准备路径
    person_info: Dict[str, PersonInfo] = prepare_paths(cfg)

    logger.info(f"Total subjects to process: {len(person_info)}")

    # ---------------------------------------------------------------------- #
    # 构建任务
    # ---------------------------------------------------------------------- #

    for subject_name, p_info in person_info.items():
        # ---------------------------------------------------------------------- #
        # 顺序执行 dual-view 任务
        # ---------------------------------------------------------------------- #

        logger.info(f"#" * 50)
        logger.info(f"Processing subject: {subject_name}")

        left_vggt_dir = (
            Path("/workspace/data") / "vggt_npy" / "single" / subject_name / "left"
        )
        right_vggt_dir = (
            Path("/workspace/data") / "vggt_npy" / "single" / subject_name / "right"
        )
        left_sam_dir = (
            Path("/workspace/data")
            / "sam3d_body_results"
            / "person"
            / subject_name
            / "left"
        )
        right_sam_dir = (
            Path("/workspace/data")
            / "sam3d_body_results"
            / "person"
            / subject_name
            / "right"
        )

        frame_sets = [
            _collect_frame_indices(left_vggt_dir, "frame_*_predictions.npz"),
            _collect_frame_indices(right_vggt_dir, "frame_*_predictions.npz"),
            _collect_frame_indices(left_sam_dir, "frame_*_sam_3d_body_outputs.npz"),
            _collect_frame_indices(right_sam_dir, "frame_*_sam_3d_body_outputs.npz"),
        ]

        if any(len(frame_set) == 0 for frame_set in frame_sets):
            return []
        frame_indices = sorted(set.intersection(*frame_sets))

        multi_frame = len(frame_indices) > 1
        for current_frame_idx in tqdm(frame_indices):
            current_save_ply = resolve_frame_save_path(
                p_info.inference_output_path, current_frame_idx, multi_frame
            )
            if current_save_ply is not None:
                current_save_ply.parent.mkdir(parents=True, exist_ok=True)

            analyze_frame(
                run_name=subject_name,
                frame_idx=current_frame_idx,
                data_root=Path("/workspace/data"),
                save_ply=current_save_ply,
                show_plotly=False,  # 逐帧显示可能会很慢，默认关闭
            )

        logger.info(f"#" * 50)

    logger.info("==== ALL DONE ====")


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
