#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from pathlib import Path

import numpy as np
import open3d as o3d
import trimesh


def extrinsic_to_RT(extrinsic):
    """
    extrinsic: (T,3,4) / (T,4,4) / (3,4) / (4,4)
    return: R (T,3,3), t (T,3), C (T,3)
    """
    E = np.asarray(extrinsic)

    if E.ndim == 2:
        E = E[None, ...]
    if E.shape[-2:] == (4, 4):
        E = E[:, :3, :]

    r_mat = E[:, :3, :3]
    t_vec = E[:, :3, 3]
    cam_centers = -np.einsum("tij,tj->ti", r_mat.transpose(0, 2, 1), t_vec)

    return r_mat, t_vec, cam_centers


def extrinsic_to_camera_pose(extrinsic):
    """
    将 world->camera 外参转换为 camera->world 位姿。

    extrinsic: (T,3,4) / (T,4,4) / (3,4) / (4,4)
    return:
        cam_R: (T,3,3) camera->world 旋转
        cam_t: (T,3)   camera 在 world 中的位置
        cam_T: (T,4,4) camera->world 齐次变换
    """
    r_mat, _, cam_centers = extrinsic_to_RT(extrinsic)
    cam_R = r_mat.transpose(0, 2, 1)

    batch = cam_R.shape[0]
    cam_T = np.tile(np.eye(4, dtype=np.float32), (batch, 1, 1))
    cam_T[:, :3, :3] = cam_R.astype(np.float32)
    cam_T[:, :3, 3] = cam_centers.astype(np.float32)

    return cam_R, cam_centers, cam_T


def _save_scene_with_axes_and_cameras(
    pts_flat: np.ndarray,
    colors_rgb: np.ndarray,
    extrinsics: np.ndarray,
    frame_idx: int,
    title: str,
    inference_output_dir: Path,
    scene_axis_length: float = 0.5,
    scene_camera_size: float = 0.05,
    camera_axis_length: float = 0.12,
) -> None:
    """将点云、坐标轴和相机中心一起导出为 glb。"""
    scene = trimesh.Scene()

    scene.add_geometry(
        trimesh.points.PointCloud(
            vertices=np.asarray(pts_flat, dtype=np.float32),
            colors=np.asarray(colors_rgb, dtype=np.uint8),
        ),
        geom_name="point_cloud",
    )

    world_axis = trimesh.creation.axis(axis_length=scene_axis_length)
    scene.add_geometry(world_axis, geom_name="world_axis")

    _, camera_centers, camera_poses = extrinsic_to_camera_pose(extrinsics)
    for idx, (center, pose) in enumerate(zip(camera_centers, camera_poses)):
        cam_marker = trimesh.creation.icosphere(
            subdivisions=2,
            radius=scene_camera_size,
        )
        cam_marker.apply_translation(center)
        scene.add_geometry(cam_marker, geom_name=f"camera_{idx}")

        cam_axis = trimesh.creation.axis(axis_length=camera_axis_length)
        cam_axis.apply_transform(pose)
        scene.add_geometry(cam_axis, geom_name=f"camera_axis_{idx}")

    scene_path = inference_output_dir / f"{frame_idx}_scene_{title}.glb"
    scene.export(scene_path)


def save_glb_with_camera_pose(
    pts_flat,
    colors_rgb,
    extrinsics,
    frame_idx,
    title,
    inference_output_dir: Path | None = None,
    scene_axis_length: float = 0.5,
    scene_camera_size: float = 0.05,
    camera_axis_length: float = 0.12,
) -> Path:
    """保存包含点云与相机位姿的 GLB 文件。"""
    if inference_output_dir is None:
        raise ValueError("inference_output_dir is required for save_glb_with_camera_pose")

    output_dir = Path(inference_output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    _save_scene_with_axes_and_cameras(
        pts_flat=np.asarray(pts_flat, dtype=np.float32),
        colors_rgb=np.asarray(colors_rgb, dtype=np.uint8),
        extrinsics=np.asarray(extrinsics),
        frame_idx=int(frame_idx),
        title=str(title),
        inference_output_dir=output_dir,
        scene_axis_length=scene_axis_length,
        scene_camera_size=scene_camera_size,
        camera_axis_length=camera_axis_length,
    )

    return output_dir / f"{frame_idx}_scene_{title}.glb"


def save_single(
    pts_flat,
    images,
    frame_idx,
    title,
    extrinsics=None,
    inference_output_dir: Path | None = None,
    save_scene_with_axes: bool = False,
    scene_axis_length: float = 0.5,
    scene_camera_size: float = 0.05,
    camera_axis_length: float = 0.12,
):
    """保存单视角点云。"""

    pcd = o3d.geometry.PointCloud()
    img_hw3 = images[0].transpose(1, 2, 0)
    colors_rgb = (img_hw3.reshape(-1, 3) * 255.0).clip(0, 255).astype(np.uint8)

    pcd.points = o3d.utility.Vector3dVector(pts_flat)
    pcd.colors = o3d.utility.Vector3dVector(colors_rgb / 255.0)

    if inference_output_dir is None:
        raise ValueError("inference_output_dir is required for save_single")

    o3d.io.write_point_cloud(
        inference_output_dir / f"{frame_idx}_point_cloud_{title}.ply",
        pcd,
    )

    if save_scene_with_axes and extrinsics is not None:
        _save_scene_with_axes_and_cameras(
            pts_flat=pts_flat,
            colors_rgb=colors_rgb,
            extrinsics=extrinsics,
            frame_idx=frame_idx,
            title=title,
            inference_output_dir=inference_output_dir,
            scene_axis_length=scene_axis_length,
            scene_camera_size=scene_camera_size,
            camera_axis_length=camera_axis_length,
        )


def save_combined_point_cloud(
    left_pts_flat,
    right_pts_flat,
    left_images,
    right_images,
    frame_idx,
    title,
    inference_output_dir: Path | None = None,
) -> None:
    """保存双视角合并点云。"""

    pcd = o3d.geometry.PointCloud()

    left_img_hw3 = left_images[0].transpose(1, 2, 0)
    left_colors_rgb = (
        (left_img_hw3.reshape(-1, 3) * 255.0).clip(0, 255).astype(np.uint8)
    )

    right_img_hw3 = right_images[0].transpose(1, 2, 0)
    right_colors_rgb = (
        (right_img_hw3.reshape(-1, 3) * 255.0).clip(0, 255).astype(np.uint8)
    )

    vertices = np.vstack([left_pts_flat, right_pts_flat])
    colors = np.vstack([left_colors_rgb, right_colors_rgb])

    pcd.points = o3d.utility.Vector3dVector(vertices)
    pcd.colors = o3d.utility.Vector3dVector(colors / 255.0)

    if inference_output_dir is None:
        raise ValueError("inference_output_dir is required for save_combined_point_cloud")

    o3d.io.write_point_cloud(
        inference_output_dir / f"{frame_idx}_merged_point_cloud_{title}.ply",
        pcd,
    )
