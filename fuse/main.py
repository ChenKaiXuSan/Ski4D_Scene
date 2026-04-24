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
from typing import Dict, List, Tuple

import hydra
from omegaconf import DictConfig, OmegaConf

from .infer import process_one_person
from .prepare_paths import PersonInfo, prepare_paths

logger = logging.getLogger(__name__)


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

        # if "pro" in subject_name.lower():
        #     continue; # 跳过 Pro 版本的 subject

        logger.info(f"#" * 50)
        logger.info(f"Processing subject: {subject_name}")

        process_one_person(
            p_info,
            cfg=cfg,
        )
        logger.info(f"#" * 50)

    logger.info("==== ALL DONE ====")


if __name__ == "__main__":
    os.environ["HYDRA_FULL_ERROR"] = "1"
    main()
