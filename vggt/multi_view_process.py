#!/usr/bin/env python3
# -*- coding:utf-8 -*-
"""
vggt_video_infer.py
从单个视频抽帧并执行 VGGT 推理，可作为函数调用。
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import open3d
import torch
from omegaconf import DictConfig
from tqdm import tqdm

from vggt.load import load_info
from vggt.reproject import reproject_and_visualize
from vggt.save import save_camera_info
from vggt.triangulate import triangulate_one_frame
from vggt.vggt.infer import CameraHead
from vggt.vis.pose_visualization import save_stereo_pose_frame, visualize_3d_joints

from .vis.skeleton_visualizer import SkeletonVisualizer

logger = logging.getLogger(__name__)


K = np.array(
    [
        1116.9289548941917,
        0.0,
        955.77175993563799,
        0.0,
        1117.3341496962166,
        538.91061167202145,
        0.0,
        0.0,
        1.0,
    ]
).reshape(3, 3)

K_dist = np.array(
    [
        -1.1940477842823853,
        -15.440461757486913,
        0.00013163161053023783,
        0.00019082529328353381,
        98.843073622415901,
        -1.3588290520381034,
        -14.555841222727574,
        96.219667412855202,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
        0.0,
    ]
)


# --------------------------------------------------------------------------- #
# Processing functions
# --------------------------------------------------------------------------- #
def process_multi_view_video(
    left_video_path: Path,
    left_pt_path: Path,
    right_video_path: Path,
    right_pt_path: Path,
    out_root: Path,
    inference_output_path: Path,
    cfg: DictConfig,
) -> Optional[Path]:
    """
    处理双目视频。返回输出目录；失败返回 None。

    目前示例代码仍然只对 left_video_path 进行 VGGT 推理，
    主要提供一个“成对管理 + 输出目录区分”的框架。
    后续如果需要真正 multi-view 融合，可以在这里扩展。
    """
    subject = left_video_path.parent.name or "default"

    out_dir = out_root / "multi_view" / subject
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"[Run-MV] {left_video_path} & {right_video_path} → {out_dir} | ")

    # * load info from pt and video
    left_kpts, left_kpt_scores, left_bboxes, left_bboxes_scores, left_frames = (
        load_info(
            video_file_path=left_video_path.as_posix(),
            pt_file_path=left_pt_path.as_posix(),
            assume_normalized=False,
        )
    )
    right_kpts, right_kpt_scores, right_bboxes, right_bboxes_scores, right_frames = (
        load_info(
            video_file_path=right_video_path.as_posix(),
            pt_file_path=right_pt_path.as_posix(),
            assume_normalized=False,
        )
    )

    all_frame_raw_x3d = []
    all_frame_camera_extrinsics = []
    all_frame_camera_intrinsics = []
    all_frame_R = []
    all_frame_t = []
    all_frame_C = []

    camera_head = CameraHead(cfg, out_dir / "vggt_infer")
    visualizer = SkeletonVisualizer()

    # rotat the right frame to left frame coordinate
    if cfg.infer.get("hflip", False):
        right_frames = torch.stack([torch.flip(frame, [1]) for frame in right_frames])
        right_kpts = [kp.copy() for kp in right_kpts]
        for kp in right_kpts:
            kp[:, 0] = left_frames[0].shape[1] - kp[:, 0]
        right_bboxes = [box.copy() for box in right_bboxes]
        for box in right_bboxes:
            x1, y1, x2, y2 = box
            box[0] = left_frames[0].shape[1] - x2
            box[2] = left_frames[0].shape[1] - x1
    else:
        right_frames = right_frames
        right_kpts = right_kpts
        right_bboxes = right_bboxes

    for idx in tqdm(
        range(0, min(len(left_frames), len(right_frames))), desc="Processing frames"
    ):
        # if idx > 10:
        #     break

        # save images
        img_dir = out_dir / "raw_frames" / f"frame_{idx:04d}"
        img_dir.mkdir(parents=True, exist_ok=True)
        draw_kpt_left = visualizer.draw_skeleton_2d(
            image=left_frames[idx].numpy(),
            keypoints=left_kpts[idx],
        )
        draw_kpt_right = visualizer.draw_skeleton_2d(
            image=right_frames[idx].numpy(),
            keypoints=right_kpts[idx],
        )

        cv2.imwrite(
            (img_dir / f"left_{idx:04d}_kpt.png").as_posix(),
            draw_kpt_left,
        )
        cv2.imwrite(
            (img_dir / f"right_{idx:04d}_kpt.png").as_posix(),
            draw_kpt_right,
        )

        # infer vggt
        (
            camera_extrinsics,
            camera_intrinsics_resized,
            R,
            t,
            C,
            world_points_from_depth,
        ) = camera_head.reconstruct_from_frames(
            imgs=[
                left_frames[idx],
                right_frames[idx],
            ],
            frame_id=idx,
        )

        # scale bboxes to match the resized image size
        source_size = left_frames.shape[1:3]  # (H,W)
        target_size = world_points_from_depth.shape[0:2]  # (H,W)

        # 得到的 R,t 是 以第一个相机为参考系的 世界坐标系
        # 所以需要对这个进行平移，将坐标系移动回人物身上

        extract_person_points_left = extract_person_points(
            pointmap=world_points_from_depth[0],
            bbox=left_bboxes[idx],
            img_size=source_size,
        )

        extract_person_points_right = extract_person_points(
            pointmap=world_points_from_depth[1],
            bbox=right_bboxes[idx],
            img_size=source_size,
        )

        left_origin = extract_person_points_left.mean(axis=0)  # (3,)
        right_origin = extract_person_points_right.mean(axis=0)  # (3,)

        # 用一个统一的 origin（比如两者平均）
        origin = 0.5 * (left_origin + right_origin)

        # 更新 t 向量，使新的世界坐标系原点对齐到 new_world_origin
        # 注意这里的 R 是 world->cam 的变换矩阵
        for cam_id in range(len(R)):
            t[cam_id] = t[cam_id] + R[cam_id] @ origin

        # 旋转右视角180度，使其与左视角对齐
        R_align = np.array(
            [
                [-1, 0, 0],
                [0, 1, 0],
                [0, 0, -1],
            ]
        )
        R[1] = R_align @ R[1]
        t[1] = R_align @ t[1]

        t[1][0] = -t[1][0]  # 水平翻转
        t[1][2] = -t[1][2]  # 深度翻转

        x3d, reprojet_err = triangulate_one_frame(
            kptL=left_kpts[idx],
            kptR=right_kpts[idx],
            frame_L=left_frames[idx].numpy(),
            frame_R=right_frames[idx].numpy(),
            K=np.stack(
                [camera_intrinsics_resized[0], camera_intrinsics_resized[1]], axis=0
            ),
            R=np.stack([R[0], R[1]], axis=0),
            T=np.stack([t[0], t[1]], axis=0),
            save_dir=out_dir / "triangulation" / "raw" / f"frame_{idx:04d}",
            dist=None,
            visualize_3d=True,
            frame_num=idx,
        )

        # visualize 2d keypoints and save
        visualizer.draw_camera_with_skeleton(
            R=np.stack([R[0], R[1]], axis=0),
            T=np.stack([t[0], t[1]], axis=0),
            keypoints_3d=x3d,
            save_dir=out_dir / "skeleton_camera_viz" / "raw" / f"frame_{idx:04d}.png",
        )

        # write reprojetion error to log
        with open(out_dir / "raw_reprojection_error.txt", "a") as f:
            f.write(f"Frame {idx:04d} Reprojection Error (in pixels):\n")
            for k, v in reprojet_err.items():
                if isinstance(v, np.ndarray):
                    continue
                f.write(f"  {k}: {v}\n")

        left_bboxes[idx] = scale_bbox(
            bbox=left_bboxes[idx],
            source_size=source_size,
            target_size=target_size,
        )
        right_bboxes[idx] = scale_bbox(
            bbox=right_bboxes[idx],
            source_size=source_size,
            target_size=target_size,
        )
        # 根据bbox大小执行点云配准
        source_point_aligned, transformation = ICP_with_bbox(
            source_points=world_points_from_depth[0],
            target_points=world_points_from_depth[1],
            source_bbox=left_bboxes[idx],
            target_bbox=right_bboxes[idx],
        )

        # 更新R和t
        R_update = transformation[:3, :3]
        t_update = transformation[:3, 3]

        R[1] = R_update @ R[1]
        t[1] = R_update @ t[1] + t_update

        x3d, reprojet_err = triangulate_one_frame(
            kptL=left_kpts[idx],
            kptR=right_kpts[idx],
            frame_L=left_frames[idx].numpy(),
            frame_R=right_frames[idx].numpy(),
            K=np.stack(
                [camera_intrinsics_resized[0], camera_intrinsics_resized[1]], axis=0
            ),
            R=np.stack([R[0], R[1]], axis=0),
            T=np.stack([t[0], t[1]], axis=0),
            save_dir=out_dir / "triangulation" / "update" / f"frame_{idx:04d}",
            dist=None,
            visualize_3d=True,
            frame_num=idx,
        )

        # visualize 2d keypoints and save
        visualizer.draw_camera_with_skeleton(
            R=np.stack([R[0], R[1]], axis=0),
            T=np.stack([t[0], t[1]], axis=0),
            keypoints_3d=x3d,
            save_dir=out_dir
            / "skeleton_camera_viz"
            / "update"
            / f"frame_{idx:04d}.png",
        )

        all_frame_raw_x3d.append(x3d)
        all_frame_camera_extrinsics.append(camera_extrinsics)
        all_frame_camera_intrinsics.append(camera_intrinsics_resized)
        all_frame_R.append(R)
        all_frame_t.append(t)
        all_frame_C.append(C)

    # save 3d info into npz
    save_camera_info(
        out_pt_path=inference_output_path / f"{subject}_multi_view_3d_info.npz",
        all_frame_x3d=all_frame_raw_x3d,
        all_frame_camera_intrinsics=all_frame_camera_intrinsics,
        all_frame_R=all_frame_R,
        all_frame_t=all_frame_t,
        all_frame_C=all_frame_C,
    )

    # R_init = np.array(all_frame_R)  # (T,C,3,3)
    # t_init = np.array(all_frame_t)  # (T,C,3)
    # X3d_init = np.array(all_frame_raw_x3d)  # (T,J,3)
    # x2d = np.array(
    #     [
    #         np.stack([left_kpts[i], right_kpts[i]], axis=0)
    #         for i in range(len(all_frame_raw_x3d))
    #     ]
    # )  # (T,C,J,2)
    # conf2d = np.array(
    #     [
    #         np.stack([left_kpt_scores[i], right_kpt_scores[i]], axis=0)
    #         for i in range(len(all_frame_raw_x3d))
    #     ]
    # )  # (T,C,J)

    # mode = cfg.bundle_adjustment.get("mode", "pose_only")
    # for mode in ["pose_only", "pose_cam_t", "full"]:
    #     bundle_adjustment(
    #         cfg=cfg,
    #         out_dir=out_dir,
    #         R_init=R_init,
    #         t_init=t_init,
    #         X3d_init=X3d_init,
    #         x2d=x2d,
    #         conf2d=conf2d,
    #         all_frame_camera_intrinsics=all_frame_camera_intrinsics,
    #         left_frames=left_frames,
    #         right_frames=right_frames,
    #         left_kpts=left_kpts,
    #         right_kpts=right_kpts,
    #         mode=mode,
    #     )


def extract_person_points(pointmap, bbox, img_size):
    """
    pointmap : (H_pm, W_pm, 3)   # VGGT pointmap
    bbox     : (x1, y1, x2, y2)  # 在原图像分辨率下
    img_size : (H_img, W_img)   # 原图大小，用来做坐标对齐
    return   : (N,3) 人体点云 (在VGGT世界系/第1相机系)
    """

    H_img, W_img = img_size
    H_pm, W_pm = pointmap.shape[:2]

    # ---- ① bbox 从原图映射到 pointmap 分辨率 ----
    sx = W_pm / W_img
    sy = H_pm / H_img

    x1, y1, x2, y2 = bbox
    x1 = int(x1 * sx)
    x2 = int(x2 * sx)
    y1 = int(y1 * sy)
    y2 = int(y2 * sy)

    # ---- ② 裁剪并提取 3D 点 ----
    x1 = np.clip(x1, 0, W_pm - 1)
    x2 = np.clip(x2, 0, W_pm)
    y1 = np.clip(y1, 0, H_pm - 1)
    y2 = np.clip(y2, 0, H_pm)

    crop = pointmap[y1:y2, x1:x2, :]  # (h,w,3)
    P = crop.reshape(-1, 3)

    # ---- ③ 去除无效点 / 背景深度 ----
    mask_valid = np.isfinite(P).all(axis=1)
    P = P[mask_valid]

    if len(P) > 0:
        z = P[:, 2]
        z_med = np.median(z)
        P = P[np.abs(z - z_med) < 3.0 * np.std(z)]  # 过滤远处背景

    return P


def scale_bbox(
    bbox: List[float],
    source_size: Tuple[int, int],
    target_size: Tuple[int, int],
) -> List[float]:
    """
    根据源图像和目标图像的尺寸比例，缩放边界框坐标。

    bbox: [x1, y1, x2, y2]
    source_size: (height, width)
    target_size: (height, width)

    返回: 缩放后的边界框 [x1, y1, x2, y2]
    """
    src_h, src_w = source_size
    tgt_h, tgt_w = target_size

    scale_x = tgt_w / src_w
    scale_y = tgt_h / src_h

    x1, y1, x2, y2 = bbox
    x1_scaled = x1 * scale_x
    y1_scaled = y1 * scale_y
    x2_scaled = x2 * scale_x
    y2_scaled = y2 * scale_y

    return [x1_scaled, y1_scaled, x2_scaled, y2_scaled]


def ICP_with_bbox(
    source_points: np.ndarray,
    target_points: np.ndarray,
    source_bbox: List[float],
    target_bbox: List[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """
    使用 ICP 对齐源点云到目标点云，基于给定的边界框进行裁剪以提高配准精度。
    最终返回整个原始源点云的对齐结果。

    source_points: (N, 3) 原始源点云
    target_points: (M, 3) 目标点云
    source_bbox: [x1, y1, x2, y2]
    target_bbox: [x1, y1, x2, y2]

    返回: (对齐后的整个源点云 (N, 3), 4x4 变换矩阵)
    """
    # TODO: 这里的剪裁逻辑可以根据实际需求调整
    # 确保 BBox 是整数 (像素坐标)
    sx1, sy1, sx2, sy2 = map(int, source_bbox)
    tx1, ty1, tx2, ty2 = map(int, target_bbox)

    # --- 1. 裁剪点云用于 ICP (基于图像索引) ---

    # ⚠️ 修正：正确的 NumPy 索引顺序是 [y_min:y_max, x_min:x_max]
    # 裁剪 (H, W, 3) 数组
    # cropped_source_region = source_points[sy1:sy2, sx1:sx2, :]
    # cropped_target_region = target_points[ty1:ty2, tx1:tx2, :]
    cropped_source_region = source_points
    cropped_target_region = target_points

    # 展平为 N x 3 的点集
    cropped_source_points_N3 = cropped_source_region.reshape(-1, 3)
    cropped_target_points_N3 = cropped_target_region.reshape(-1, 3)

    # 过滤掉无效点 (Z=0, NaN, 或原点附近的点)
    def filter_and_validate(points_N3):
        # 过滤掉范数小于 1e-6 的点 (通常表示无效点 [0, 0, 0])
        valid_mask = np.linalg.norm(points_N3, axis=1) > 1e-6
        return points_N3[valid_mask]

    final_source_points = filter_and_validate(cropped_source_points_N3)
    final_target_points = filter_and_validate(cropped_target_points_N3)

    if len(final_source_points) < 50 or len(final_target_points) < 50:
        # 如果点太少，返回零变换并给出警告 (阈值设为 50，可调整)
        print("警告: 裁剪后的有效点太少，跳过 ICP 配准。")
        return source_points.reshape(-1, 3), np.eye(4)

    # 转换为 Open3D 对象
    source_pc_cropped = open3d.geometry.PointCloud()
    source_pc_cropped.points = open3d.utility.Vector3dVector(final_source_points)

    target_pc_cropped = open3d.geometry.PointCloud()
    target_pc_cropped.points = open3d.utility.Vector3dVector(final_target_points)

    # --- 2. 执行 G-ICP (Point-to-Plane) 配准 ---

    threshold = 0.05
    trans_init = np.eye(4)

    # ⚠️ 必须计算法线才能使用 PointToPlane
    # 设置合理的搜索半径 (radius)
    search_radius = 0.05
    source_pc_cropped.estimate_normals(
        search_param=open3d.geometry.KDTreeSearchParamRadius(radius=search_radius)
    )
    target_pc_cropped.estimate_normals(
        search_param=open3d.geometry.KDTreeSearchParamRadius(radius=search_radius)
    )

    reg_result = open3d.pipelines.registration.registration_icp(
        source_pc_cropped,
        target_pc_cropped,
        threshold,
        trans_init,
        open3d.pipelines.registration.TransformationEstimationPointToPlane(),
        open3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=200),
    )

    transformation = reg_result.transformation

    # --- 3. 应用变换到整个原始点云 ---

    R = transformation[:3, :3]
    t = transformation[:3, 3]

    # 将原始 (H, W, 3) 展平为 (N, 3) 进行矩阵运算
    P_original_N3 = source_points.reshape(-1, 3)

    # 计算公式: P_aligned = R @ P_original.T + t
    source_points_aligned = (R @ P_original_N3.T).T + t

    return source_points_aligned, transformation


def bundle_adjustment(
    cfg,
    out_dir,
    R_init,
    t_init,
    all_frame_camera_intrinsics,
    X3d_init,
    x2d,
    conf2d,
    left_frames,
    right_frames,
    left_kpts,
    right_kpts,
    mode="pose_only",
):
    # bundle adjustment
    out_dir_ba = out_dir / "ba_results" / mode
    out_dir_ba.mkdir(parents=True, exist_ok=True)

    # 对摄像机的K取平均
    avg_K = np.mean(np.array(all_frame_camera_intrinsics), axis=0)

    # 转成 torch
    rvec_init_torch = torch.from_numpy(R_init)  # (T,C,3,3)
    tvec_init_torch = torch.from_numpy(t_init)  # (T,C,3)
    X3d_init_torch = torch.from_numpy(X3d_init)  # (T,J,3)
    K_torch = torch.from_numpy(avg_K).float()  # (C,3,3)
    x2d_torch = torch.from_numpy(x2d).float()  # (T,C,J,2)
    conf2d_torch = torch.from_numpy(conf2d).float()  # (T,C,J)

    R_opt, t_opt, X_opt, history = run_local_ba(
        K_torch=K_torch,  # (C,3,3)
        R_init_torch=rvec_init_torch,  # (T,C,3,3)  可优化
        t_init_torch=tvec_init_torch,  # (T,C,3)    可优化
        X3d_init_torch=X3d_init_torch,  # (T,J,3)    可优化
        x2d_torch=x2d_torch,  # (T,C,J,2)
        conf2d_torch=conf2d_torch,  # (T,C,J)
        num_iters=cfg.bundle_adjustment.get("num_iters", 200),
        lr=cfg.bundle_adjustment.get("lr", 1e-3),
        device=cfg.infer.get("device", "cuda"),
        mode=mode,  # 优化模式,
    )

    # use the optimized results to draw and save
    for idx in range(X_opt.shape[0]):
        frame_L = left_frames[idx].numpy()  # (H,W,3)
        frame_R = right_frames[idx].numpy()  # (H,W,3)
        kptL = left_kpts[idx]  # (J,2)
        kptR = right_kpts[idx]  # (J,2)
        R = R_opt[idx].cpu().numpy()
        t = t_opt[idx].cpu().numpy()  # (C,3)
        K = K_torch.cpu().numpy()  # (3,3)
        X3d = X_opt[idx].cpu().numpy()  # (J,3)

        # reproject and visualize
        ba_reproj_err = reproject_and_visualize(
            img1=frame_L,
            img2=frame_R,
            X3=X3d,
            kptL=kptL,  # (J,2)
            kptR=kptR,  # (J,2)
            K1=K[0],
            dist1=K_dist,
            K2=K[1],
            dist2=K_dist,
            R=R,  # （2，3，3），世界》相机
            T=t,  # （2，3），世界》相机
            joint_names=None,  # 或者传 COCO 的关节名列表
            out_path=out_dir_ba / "ba_reproj" / f"{idx:04d}.jpg",
        )

        # write reprojetion error to log
        with open(out_dir_ba / "ba_reprojection_error.txt", "a") as f:
            f.write(f"Frame {idx:04d} Reprojection Error (in pixels):\n")
            for k, v in ba_reproj_err.items():
                if isinstance(v, np.ndarray):
                    continue
                f.write(f"  {k}: {v}\n")

        # ---- 可视化 3D ---- #
        visualize_3d_joints(
            R=R,
            T=t,
            K=K,
            joints_3d=X3d,
            save_path=out_dir_ba / "3d_joints" / f"{idx:04d}.png",
            title=f"3D Triangulated Result",
            image_size=(frame_L.shape[1], frame_L.shape[0]),
        )

        save_stereo_pose_frame(
            R=R,
            T=t,
            K=K,
            img_left=frame_L,
            img_right=frame_R,
            kpt_left=kptL,
            kpt_right=kptR,
            pose_3d=X3d,
            output_path=out_dir_ba / "stereo_pose_frame" / f"{idx:04d}.jpg",
            repoj_error=ba_reproj_err,
            frame_num=idx,
        )

    return out_dir
