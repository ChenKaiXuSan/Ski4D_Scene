#!/usr/bin/env python3
# -*- coding:utf-8 -*-

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Optional

import matplotlib.pyplot as plt
import numpy as np


DATA_ROOT = Path("/workspace/data")


FRAME_PATTERN = re.compile(r"frame_(\d+)")


@dataclass
class SideViewBundle:
    name: str
    world_points: np.ndarray
    images: np.ndarray
    extrinsics: np.ndarray
    frame: np.ndarray
    kpts_2d: np.ndarray
    kpts_3d: np.ndarray
    centered_world_points: np.ndarray
    centered_extrinsics: np.ndarray
    centered_kpts_3d: np.ndarray
    feet_center: np.ndarray


@dataclass
class PersonBundle:
    run_name: str
    frame_idx: int
    left: SideViewBundle
    right: SideViewBundle

    @property
    def left_kpts_2d(self) -> np.ndarray:
        return self.left.kpts_2d

    @property
    def right_kpts_2d(self) -> np.ndarray:
        return self.right.kpts_2d

    @property
    def left_kpts_3d(self) -> np.ndarray:
        return self.left.kpts_3d

    @property
    def right_kpts_3d(self) -> np.ndarray:
        return self.right.kpts_3d

    @property
    def left_centered_kpts_3d(self) -> np.ndarray:
        return self.left.centered_kpts_3d

    @property
    def right_centered_kpts_3d(self) -> np.ndarray:
        return self.right.centered_kpts_3d


@dataclass
class PreparedFrameData:
    person: PersonBundle
    combined_world_points: np.ndarray
    combined_images: np.ndarray
    combined_extrinsics: np.ndarray
    combined_kpts_3d: np.ndarray
    rotation: np.ndarray
    translation: np.ndarray
    scale: float
    transform: np.ndarray
    aligned_right_world_points: np.ndarray
    aligned_right_kpts: np.ndarray
    aligned_right_extrinsics: np.ndarray
    alignment_error: np.ndarray
    aligned_combined_world_points: np.ndarray
    aligned_combined_extrinsics: np.ndarray
    aligned_combined_kpts_3d: np.ndarray


@dataclass(frozen=True)
class FramePaths:
    vggt_path: Path
    sam_path: Path


def load_vggt_info(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    return {key: data[key] for key in data.files}


def load_sam_outputs(path: Path) -> dict[str, Any]:
    data = np.load(path, allow_pickle=True)
    if "outputs" in data:
        return data["outputs"].item()
    return {key: data[key] for key in data.files}


def compute_kpts_in_point_cloud(
    kpts: np.ndarray,
    frame_size: tuple[int, int],
    point_cloud: np.ndarray,
    conf_map: Optional[np.ndarray] = None,
    conf_thresh: Optional[float] = None,
    remove_invalid: bool = True,
) -> np.ndarray:
    """根据 2D 关键点，在 VGGT point cloud 中取对应的 3D 点。"""
    kpts_arr = np.asarray(kpts)

    if kpts_arr.ndim < 2 or kpts_arr.shape[-1] < 2:
        raise ValueError(
            f"kpts should have shape (..., 2) or (..., >=2), but got {kpts_arr.shape}"
        )

    width, height = frame_size
    if width <= 0 or height <= 0:
        raise ValueError(f"frame_size should be positive, but got {frame_size}")

    if point_cloud.ndim != 3 or point_cloud.shape[-1] != 3:
        raise ValueError(
            f"point_cloud should have shape (H, W, 3), but got {point_cloud.shape}"
        )

    pc_height, pc_width, _ = point_cloud.shape

    if conf_map is not None and conf_map.shape != (pc_height, pc_width):
        raise ValueError(
            f"conf_map should have shape {(pc_height, pc_width)}, but got {conf_map.shape}"
        )

    kpts_xy = kpts_arr[..., :2].reshape(-1, 2).astype(np.float32)
    x = np.clip(kpts_xy[:, 0], 0, width - 1)
    y = np.clip(kpts_xy[:, 1], 0, height - 1)

    cols = np.floor(x * pc_width / width).astype(np.int32)
    rows = np.floor(y * pc_height / height).astype(np.int32)
    cols = np.clip(cols, 0, pc_width - 1)
    rows = np.clip(rows, 0, pc_height - 1)

    kpts_3d_flat = point_cloud[rows, cols].astype(np.float32, copy=True)

    if remove_invalid:
        valid_mask = np.isfinite(kpts_3d_flat).all(axis=1)
        valid_mask &= ~(np.abs(kpts_3d_flat).sum(axis=1) == 0)

        if conf_map is not None and conf_thresh is not None:
            valid_mask &= conf_map[rows, cols] >= conf_thresh

        kpts_3d_flat[~valid_mask] = np.nan

    return kpts_3d_flat.reshape(*kpts_arr.shape[:-1], 3)


def normalize(vec: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    norm = np.linalg.norm(vec)
    if norm < eps:
        return vec
    return vec / norm


def compute_body_midline_and_axes(kpts: np.ndarray) -> dict[str, Any]:
    """基于 MHR70 关键点计算人体中心和局部坐标轴。"""
    kpts = np.asarray(kpts, dtype=np.float64)

    left_hip, right_hip = kpts[9], kpts[10]
    left_shoulder, right_shoulder = kpts[5], kpts[6]
    left_heel, right_heel = kpts[17], kpts[20]
    left_ankle, right_ankle = kpts[13], kpts[14]
    left_acromion, right_acromion = kpts[67], kpts[68]
    neck = kpts[69]

    feet_center = (left_heel + right_heel) / 2.0
    hip_center = (left_hip + right_hip) / 2.0
    shoulder_center = (left_shoulder + right_shoulder) / 2.0
    acromion_center = (left_acromion + right_acromion) / 2.0

    midline_vec = acromion_center - hip_center
    if np.linalg.norm(midline_vec) < 1e-5:
        midline_vec = shoulder_center - hip_center
    midline_vec = normalize(midline_vec)

    y_axis = normalize(acromion_center - hip_center)
    x_axis = normalize(right_hip - left_hip)
    z_axis = normalize(np.cross(x_axis, y_axis))
    x_axis = normalize(np.cross(y_axis, z_axis))

    return {
        "feet_center": feet_center,
        "body_center": hip_center,
        "midline_vec": midline_vec,
        "midline_start": hip_center,
        "midline_end": acromion_center,
        "midline_points": {
            "feet_center": feet_center,
            "ankle_center": (left_ankle + right_ankle) / 2.0,
            "hip_center": hip_center,
            "shoulder_center": shoulder_center,
            "acromion_center": acromion_center,
            "neck": neck,
        },
        "x_axis": x_axis,
        "y_axis": y_axis,
        "z_axis": z_axis,
    }


def make_y_rotation(theta: float) -> np.ndarray:
    return np.array(
        [
            [np.cos(theta), 0.0, np.sin(theta)],
            [0.0, 1.0, 0.0],
            [-np.sin(theta), 0.0, np.cos(theta)],
        ],
        dtype=np.float64,
    )


def rotate_view(
    world_points: np.ndarray, extrinsics: np.ndarray, theta: float
) -> tuple[np.ndarray, np.ndarray]:
    rotation = make_y_rotation(theta)
    rotated_world_points = world_points @ rotation.T
    rotated_extrinsics = extrinsics.copy()
    rotated_extrinsics[:, :3, :3] = extrinsics[:, :3, :3] @ rotation.T
    return rotated_world_points, rotated_extrinsics


def camera_centers_from_extrinsics(extrinsics: np.ndarray) -> np.ndarray:
    rotation = extrinsics[:, :3, :3]
    translation = extrinsics[:, :3, 3]
    return -np.einsum("vij,vj->vi", rotation.transpose(0, 2, 1), translation)


def center_world_on_feet(
    world_points: np.ndarray, kpts_3d: np.ndarray, extrinsics: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    body_info = compute_body_midline_and_axes(kpts_3d)
    feet_center = body_info["feet_center"]

    centered_world_points = world_points - feet_center
    centered_kpts_3d = kpts_3d - feet_center

    centered_extrinsics = extrinsics.copy()
    centered_extrinsics[:, :3, 3] = centered_extrinsics[:, :3, 3] + np.einsum(
        "vij,j->vi", centered_extrinsics[:, :3, :3], feet_center
    )

    return centered_world_points, centered_kpts_3d, centered_extrinsics, feet_center


def estimate_similarity_transform(
    source_points: np.ndarray, target_points: np.ndarray
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Estimate similarity transform: target ~= scale * source @ R.T + t."""
    source_points = np.asarray(source_points, dtype=np.float64)
    target_points = np.asarray(target_points, dtype=np.float64)

    if source_points.shape != target_points.shape:
        raise ValueError(
            f"Shape mismatch: source {source_points.shape}, target {target_points.shape}"
        )
    if source_points.ndim != 2 or source_points.shape[1] != 3:
        raise ValueError("Input points must have shape (N, 3)")
    if source_points.shape[0] < 3:
        raise ValueError("At least 3 points are required")

    valid_mask = np.isfinite(source_points).all(axis=1) & np.isfinite(target_points).all(
        axis=1
    )
    if valid_mask.sum() < 3:
        raise ValueError(
            f"Not enough valid correspondences: {valid_mask.sum()} < 3"
        )

    source_valid = source_points[valid_mask]
    target_valid = target_points[valid_mask]

    source_center = source_valid.mean(axis=0)
    target_center = target_valid.mean(axis=0)
    source_centered = source_valid - source_center
    target_centered = target_valid - target_center

    covariance = source_centered.T @ target_centered
    left_svd, singular_values, right_svd_t = np.linalg.svd(covariance)
    correction = np.eye(3, dtype=np.float64)
    if np.linalg.det(right_svd_t.T @ left_svd.T) < 0:
        correction[-1, -1] = -1.0
    rotation = right_svd_t.T @ correction @ left_svd.T

    source_var = np.sum(source_centered**2)
    if source_var <= 1e-12:
        raise ValueError("Degenerate source points: zero variance")
    scale = float(np.sum(np.diag(correction) * singular_values) / source_var)

    translation = target_center - scale * (rotation @ source_center)

    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = scale * rotation
    transform[:3, 3] = translation
    return rotation.astype(np.float32), translation.astype(np.float32), scale, transform


def apply_similarity_transform_to_world_points(
    world_points: np.ndarray, rotation: np.ndarray, translation: np.ndarray, scale: float
) -> np.ndarray:
    transformed = world_points.copy()
    for view_index in range(transformed.shape[0]):
        flat_points = transformed[view_index].reshape(-1, 3)
        flat_points = scale * (flat_points @ rotation.T) + translation
        transformed[view_index] = flat_points.reshape(transformed[view_index].shape)
    return transformed


def apply_similarity_transform_to_extrinsics(
    extrinsics: np.ndarray, rotation: np.ndarray, translation: np.ndarray, scale: float
) -> np.ndarray:
    if abs(scale) < 1e-12:
        raise ValueError("Scale is too close to zero")

    aligned_extrinsics = extrinsics.copy()
    linear = (1.0 / scale) * rotation.T
    for view_index in range(aligned_extrinsics.shape[0]):
        cam_rotation = aligned_extrinsics[view_index, :3, :3]
        cam_translation = aligned_extrinsics[view_index, :3, 3]
        aligned_extrinsics[view_index, :3, :3] = cam_rotation @ linear
        aligned_extrinsics[view_index, :3, 3] = (
            cam_translation - cam_rotation @ linear @ translation
        )
    return aligned_extrinsics


def visualize_point_cloud_with_kpts(
    world_points: np.ndarray,
    images: np.ndarray,
    kpts_3d: np.ndarray,
    extrinsics: np.ndarray,
    max_points_per_view: int = 30000,
    point_size: float = 2.0,
    kpt_color: str = "magenta",
    view_colors: Optional[list[str]] = None,
    title: str = "VGGT Point Cloud",
) -> Any:
    import plotly.graph_objects as go

    if view_colors is None:
        view_colors = ["dodgerblue", "darkorange", "limegreen", "tomato"]

    rng = np.random.default_rng(42)
    fig = go.Figure()

    for view_index in range(world_points.shape[0]):
        points = world_points[view_index].reshape(-1, 3)
        image_hw3 = images[view_index].transpose(1, 2, 0)
        colors = (image_hw3.reshape(-1, 3) * 255.0).clip(0, 255).astype(np.uint8)
        color_strings = [f"rgb({r},{g},{b})" for r, g, b in colors]

        if len(points) > max_points_per_view:
            keep_indices = rng.choice(len(points), max_points_per_view, replace=False)
            points = points[keep_indices]
            color_strings = [color_strings[i] for i in keep_indices]

        fig.add_trace(
            go.Scatter3d(
                x=points[:, 0],
                y=points[:, 1],
                z=points[:, 2],
                mode="markers",
                marker=dict(size=point_size, color=color_strings, opacity=0.8),
                name=f"view {view_index} point cloud",
            )
        )

    fig.add_trace(
        go.Scatter3d(
            x=kpts_3d[:, 0],
            y=kpts_3d[:, 1],
            z=kpts_3d[:, 2],
            mode="markers+text",
            marker=dict(size=8, color=kpt_color, symbol="x"),
            text=[f"Keypoint {i}" for i in range(kpts_3d.shape[0])],
            textposition="bottom center",
            name="3D Keypoints",
        )
    )

    camera_centers = camera_centers_from_extrinsics(extrinsics)
    for view_index, center in enumerate(camera_centers):
        color = view_colors[view_index % len(view_colors)]
        fig.add_trace(
            go.Scatter3d(
                x=[center[0]],
                y=[center[1]],
                z=[center[2]],
                mode="markers+text",
                marker=dict(size=8, color=color, symbol="diamond"),
                text=[f"cam {view_index}"],
                textposition="top center",
                name=f"cam {view_index}",
            )
        )

    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            aspectmode="data",
            bgcolor="rgb(20,20,20)",
            xaxis=dict(backgroundcolor="rgb(20,20,20)", gridcolor="gray"),
            yaxis=dict(backgroundcolor="rgb(20,20,20)", gridcolor="gray"),
            zaxis=dict(backgroundcolor="rgb(20,20,20)", gridcolor="gray"),
        ),
        paper_bgcolor="rgb(30,30,30)",
        font=dict(color="white"),
        legend=dict(itemsizing="constant"),
        margin=dict(l=0, r=0, b=0, t=40),
        width=980,
        height=780,
    )
    return fig


def save_combined_point_cloud(
    output_path: Path,
    left_world_points: np.ndarray,
    right_world_points: np.ndarray,
    left_images: np.ndarray,
    right_images: np.ndarray,
) -> None:
    import open3d as o3d

    point_cloud = o3d.geometry.PointCloud()

    left_image_hw3 = left_images[0].transpose(1, 2, 0)
    left_colors = (left_image_hw3.reshape(-1, 3) * 255.0).clip(0, 255).astype(np.uint8)

    right_image_hw3 = right_images[0].transpose(1, 2, 0)
    right_colors = (
        (right_image_hw3.reshape(-1, 3) * 255.0).clip(0, 255).astype(np.uint8)
    )

    vertices = np.vstack(
        [left_world_points[0].reshape(-1, 3), right_world_points[0].reshape(-1, 3)]
    )
    colors = np.vstack([left_colors, right_colors])

    point_cloud.points = o3d.utility.Vector3dVector(vertices)
    point_cloud.colors = o3d.utility.Vector3dVector(colors / 255.0)
    o3d.io.write_point_cloud(str(output_path), point_cloud, write_ascii=True)


def visualize_open3d_point_cloud(path: Path) -> None:
    import open3d as o3d

    point_cloud = o3d.io.read_point_cloud(str(path))
    print(point_cloud)
    o3d.visualization.draw([{"name": "point_cloud", "geometry": point_cloud}])


def show_frames(left_frame: np.ndarray, right_frame: np.ndarray) -> None:
    plt.figure()
    plt.imshow(left_frame)
    plt.title("Left frame")
    plt.axis("off")

    plt.figure()
    plt.imshow(right_frame)
    plt.title("Right frame")
    plt.axis("off")
    plt.show()


def _collect_frame_indices(frame_dir: Path, pattern: str) -> set[int]:
    frame_indices: set[int] = set()
    for path in frame_dir.glob(pattern):
        match = FRAME_PATTERN.search(path.stem)
        if match is not None:
            frame_indices.add(int(match.group(1)))
    return frame_indices


def list_available_frame_indices(
    run_name: str, data_root: Path = DATA_ROOT
) -> list[int]:
    left_vggt_dir = data_root / "vggt_npy" / "single" / run_name / "left"
    right_vggt_dir = data_root / "vggt_npy" / "single" / run_name / "right"
    left_sam_dir = data_root / "sam3d_body_results" / "person" / run_name / "left"
    right_sam_dir = data_root / "sam3d_body_results" / "person" / run_name / "right"

    frame_sets = [
        _collect_frame_indices(left_vggt_dir, "frame_*_predictions.npz"),
        _collect_frame_indices(right_vggt_dir, "frame_*_predictions.npz"),
        _collect_frame_indices(left_sam_dir, "frame_*_sam_3d_body_outputs.npz"),
        _collect_frame_indices(right_sam_dir, "frame_*_sam_3d_body_outputs.npz"),
    ]

    if any(len(frame_set) == 0 for frame_set in frame_sets):
        return []
    return sorted(set.intersection(*frame_sets))


def build_frame_paths(
    run_name: str,
    side: str,
    frame_idx: int,
    data_root: Path = DATA_ROOT,
) -> FramePaths:
    frame_name = f"frame_{frame_idx:04d}"
    return FramePaths(
        vggt_path=(
            data_root
            / "vggt_npy"
            / "single"
            / run_name
            / side
            / f"{frame_name}_predictions.npz"
        ),
        sam_path=(
            data_root
            / "sam3d_body_results"
            / "person"
            / run_name
            / side
            / f"{frame_name}_sam_3d_body_outputs.npz"
        ),
    )


def resolve_frame_save_path(
    output_path: Optional[Path], frame_idx: int, multi_frame: bool
) -> Optional[Path]:
    if output_path is None:
        return None
    if not multi_frame:
        return output_path
    if output_path.suffix:
        return output_path.with_name(
            f"{output_path.stem}_frame_{frame_idx:04d}{output_path.suffix}"
        )
    return output_path / f"frame_{frame_idx:04d}.ply"


def prepare_side_view_bundle(
    run_name: str,
    side: str,
    frame_idx: int,
    theta: float,
    data_root: Path = DATA_ROOT,
) -> SideViewBundle:
    frame_paths = build_frame_paths(
        run_name=run_name,
        side=side,
        frame_idx=frame_idx,
        data_root=data_root,
    )

    vggt_info = load_vggt_info(frame_paths.vggt_path)
    sam_outputs = load_sam_outputs(frame_paths.sam_path)

    world_points, extrinsics = rotate_view(
        vggt_info["world_points"], vggt_info["extrinsic"], theta=theta
    )

    frame = sam_outputs["frame"]
    kpts_2d = np.asarray(sam_outputs["pred_keypoints_2d"], dtype=np.float32)
    kpts_3d = compute_kpts_in_point_cloud(
        kpts=kpts_2d,
        frame_size=(frame.shape[1], frame.shape[0]),
        point_cloud=world_points[0],
    )
    centered_world_points, centered_kpts_3d, centered_extrinsics, feet_center = (
        center_world_on_feet(world_points, kpts_3d, extrinsics)
    )

    return SideViewBundle(
        name=side,
        world_points=world_points,
        images=vggt_info["images"],
        extrinsics=extrinsics,
        frame=frame,
        kpts_2d=kpts_2d,
        kpts_3d=kpts_3d,
        centered_world_points=centered_world_points,
        centered_extrinsics=centered_extrinsics,
        centered_kpts_3d=centered_kpts_3d,
        feet_center=feet_center,
    )


def prepare_person_bundle(
    run_name: str, frame_idx: int, data_root: Path = DATA_ROOT
) -> PersonBundle:
    left = prepare_side_view_bundle(
        run_name=run_name,
        side="left",
        frame_idx=frame_idx,
        theta=np.pi / 2,
        data_root=data_root,
    )
    right = prepare_side_view_bundle(
        run_name=run_name,
        side="right",
        frame_idx=frame_idx,
        theta=-np.pi / 2,
        data_root=data_root,
    )
    return PersonBundle(run_name=run_name, frame_idx=frame_idx, left=left, right=right)


def prepare_frame_data(
    run_name: str,
    frame_idx: int,
    data_root: Path = DATA_ROOT,
) -> PreparedFrameData:
    person = prepare_person_bundle(
        run_name=run_name,
        frame_idx=frame_idx,
        data_root=data_root,
    )
    left_bundle = person.left
    right_bundle = person.right

    combined_world_points = np.concatenate(
        [left_bundle.centered_world_points, right_bundle.centered_world_points], axis=0
    )
    combined_images = np.concatenate([left_bundle.images, right_bundle.images], axis=0)
    combined_extrinsics = np.concatenate(
        [left_bundle.centered_extrinsics, right_bundle.centered_extrinsics], axis=0
    )
    combined_kpts_3d = np.concatenate(
        [left_bundle.centered_kpts_3d, right_bundle.centered_kpts_3d], axis=0
    )

    rotation, translation, scale, transform = estimate_similarity_transform(
        source_points=right_bundle.centered_kpts_3d,
        target_points=left_bundle.centered_kpts_3d,
    )
    aligned_right_world_points = apply_similarity_transform_to_world_points(
        right_bundle.centered_world_points, rotation, translation, scale
    )
    aligned_right_kpts = (
        scale * (right_bundle.centered_kpts_3d @ rotation.T) + translation
    )
    aligned_right_extrinsics = apply_similarity_transform_to_extrinsics(
        right_bundle.centered_extrinsics, rotation, translation, scale
    )
    alignment_error = np.linalg.norm(
        aligned_right_kpts - left_bundle.centered_kpts_3d, axis=1
    )

    aligned_combined_world_points = np.concatenate(
        [left_bundle.centered_world_points, aligned_right_world_points], axis=0
    )
    aligned_combined_extrinsics = np.concatenate(
        [left_bundle.centered_extrinsics, aligned_right_extrinsics], axis=0
    )
    aligned_combined_kpts_3d = np.concatenate(
        [left_bundle.centered_kpts_3d, aligned_right_kpts], axis=0
    )

    return PreparedFrameData(
        person=person,
        combined_world_points=combined_world_points,
        combined_images=combined_images,
        combined_extrinsics=combined_extrinsics,
        combined_kpts_3d=combined_kpts_3d,
        rotation=rotation,
        translation=translation,
        scale=scale,
        transform=transform,
        aligned_right_world_points=aligned_right_world_points,
        aligned_right_kpts=aligned_right_kpts,
        aligned_right_extrinsics=aligned_right_extrinsics,
        alignment_error=alignment_error,
        aligned_combined_world_points=aligned_combined_world_points,
        aligned_combined_extrinsics=aligned_combined_extrinsics,
        aligned_combined_kpts_3d=aligned_combined_kpts_3d,
    )


def print_view_summary(bundle: SideViewBundle) -> None:
    print(f"[{bundle.name}] images shape: {bundle.images.shape}")
    print(f"[{bundle.name}] feet center: {bundle.feet_center}")
    print(
        f"[{bundle.name}] camera centers:\n{camera_centers_from_extrinsics(bundle.centered_extrinsics)}"
    )


def print_person_summary(person: PersonBundle) -> None:
    print(f"person run={person.run_name}, frame={person.frame_idx}")
    print(
        f"left kpts: 2d={person.left_kpts_2d.shape}, 3d={person.left_kpts_3d.shape}, centered_3d={person.left_centered_kpts_3d.shape}"
    )
    print(
        f"right kpts: 2d={person.right_kpts_2d.shape}, 3d={person.right_kpts_3d.shape}, centered_3d={person.right_centered_kpts_3d.shape}"
    )


def analyze_frame(
    run_name: str,
    frame_idx: int,
    data_root: Path = DATA_ROOT,
    save_ply: Optional[Path] = None,
    show_plotly: bool = True,
    show_frames_flag: bool = False,
    show_open3d_path: Optional[Path] = None,
) -> None:
    frame_data = prepare_frame_data(
        run_name=run_name,
        frame_idx=frame_idx,
        data_root=data_root,
    )
    person = frame_data.person
    left_bundle = person.left
    right_bundle = person.right

    print_person_summary(person)
    print_view_summary(left_bundle)
    print_view_summary(right_bundle)

    if show_frames_flag:
        show_frames(left_bundle.frame, right_bundle.frame)

    if show_plotly:
        left_fig = visualize_point_cloud_with_kpts(
            world_points=left_bundle.centered_world_points,
            images=left_bundle.images,
            kpts_3d=left_bundle.centered_kpts_3d,
            extrinsics=left_bundle.centered_extrinsics,
            title="Left VGGT Point Cloud",
        )
        left_fig.show()

        right_fig = visualize_point_cloud_with_kpts(
            world_points=right_bundle.centered_world_points,
            images=right_bundle.images,
            kpts_3d=right_bundle.centered_kpts_3d,
            extrinsics=right_bundle.centered_extrinsics,
            title="Right VGGT Point Cloud",
        )
        right_fig.show()

    if show_plotly:
        combined_fig = visualize_point_cloud_with_kpts(
            world_points=frame_data.combined_world_points,
            images=frame_data.combined_images,
            kpts_3d=frame_data.combined_kpts_3d,
            extrinsics=frame_data.combined_extrinsics,
            title="Combined VGGT Point Cloud Before Alignment",
        )
        combined_fig.show()

    print("Rigid transform:\n", frame_data.transform)
    print("Scale:", frame_data.scale)
    print("Mean alignment error:", float(np.mean(frame_data.alignment_error)))

    if save_ply is not None:
        save_combined_point_cloud(
            output_path=save_ply,
            left_world_points=left_bundle.centered_world_points,
            right_world_points=frame_data.aligned_right_world_points,
            left_images=left_bundle.images,
            right_images=right_bundle.images,
        )
        print(f"Saved combined point cloud to: {save_ply}")

    if show_open3d_path is not None:
        visualize_open3d_point_cloud(show_open3d_path)


def analyze_run(
    run_name: str,
    frame_idx: Optional[int],
    data_root: Path = DATA_ROOT,
    save_ply: Optional[Path] = None,
    show_plotly: bool = True,
    show_frames_flag: bool = False,
    show_open3d_path: Optional[Path] = None,
) -> None:
    frame_indices = (
        [frame_idx]
        if frame_idx is not None
        else list_available_frame_indices(run_name, data_root)
    )
    if not frame_indices:
        raise ValueError(f"No frames found for run {run_name}")

    multi_frame = len(frame_indices) > 1
    for current_frame_idx in frame_indices:
        current_save_ply = resolve_frame_save_path(
            save_ply, current_frame_idx, multi_frame
        )
        if current_save_ply is not None:
            current_save_ply.parent.mkdir(parents=True, exist_ok=True)
        current_show_open3d_path = resolve_frame_save_path(
            show_open3d_path, current_frame_idx, multi_frame
        )
        analyze_frame(
            run_name=run_name,
            frame_idx=current_frame_idx,
            data_root=data_root,
            save_ply=current_save_ply,
            show_plotly=show_plotly,
            show_frames_flag=show_frames_flag,
            show_open3d_path=current_show_open3d_path,
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VGGT dual-view fusion analysis")
    parser.add_argument("--run-name", default="run_3")
    parser.add_argument("--frame-idx", type=int, default=None)
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument(
        "--save-ply", type=Path, default=Path("/workspace/Ski4D_Scene/fuse/vggt_fuse")
    )
    parser.add_argument("--show-frames", action="store_true")
    parser.add_argument("--hide-plotly", action="store_true")
    parser.add_argument("--show-open3d", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    analyze_run(
        run_name=args.run_name,
        frame_idx=args.frame_idx,
        data_root=args.data_root,
        save_ply=args.save_ply,
        show_plotly=not args.hide_plotly,
        show_frames_flag=args.show_frames,
        show_open3d_path=args.show_open3d,
    )


if __name__ == "__main__":
    main()
