import csv
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

from .config import CaptureConfig
from .errors import SampleSaveError
from .metadata import build_metadata
from .models import CapturedFrames, CaptureResult, SampleFileNames


INDEX_HEADER = [
    "sample_id",
    "timestamp",
    "color_path",
    "d2rgb_path",
    "d2rgb_filtered_path",
    "d2rgb_vis_path",
    "meta_path",
    "pointcloud_path",
]


def format_timestamp(dt: datetime) -> tuple[str, str]:
    sample_id = dt.strftime("%Y%m%d_%H%M%S_") + f"{dt.microsecond // 1000:03d}"
    human_timestamp = dt.strftime("%Y-%m-%d %H:%M:%S.") + f"{dt.microsecond // 1000:03d}"
    return sample_id, human_timestamp


def file_names_for_sample(sample_id: str) -> SampleFileNames:
    return SampleFileNames(
        color=f"{sample_id}_color.png",
        d2rgb=f"{sample_id}_d2rgb.npy",
        d2rgb_filtered=f"{sample_id}_d2rgb_filtered.npy",
        d2rgb_vis=f"{sample_id}_d2rgb_vis.jpg",
        meta=f"{sample_id}_meta.json",
        pointcloud=f"{sample_id}_pointcloud.ply",
    )


def ensure_unique_sample_id(output_dir: Path) -> tuple[str, str]:
    while True:
        sample_id, timestamp = format_timestamp(datetime.now())
        files = file_names_for_sample(sample_id)
        paths = [
            output_dir / files.color,
            output_dir / files.d2rgb,
            output_dir / files.d2rgb_filtered,
            output_dir / files.d2rgb_vis,
            output_dir / files.meta,
            output_dir / files.pointcloud,
        ]
        if not any(path.exists() for path in paths):
            return sample_id, timestamp
        time.sleep(0.001)


def prepare_index_csv(index_path: Path) -> None:
    if index_path.exists():
        return

    with index_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(INDEX_HEADER)


def append_index_row(index_path: Path, result: CaptureResult) -> None:
    with index_path.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                result.sample_id,
                result.timestamp,
                result.files.color,
                result.files.d2rgb,
                result.files.d2rgb_filtered if result.metadata["files"]["d2rgb_filtered_saved"] else "",
                result.files.d2rgb_vis,
                result.files.meta,
                result.files.pointcloud if result.metadata["pointcloud"]["enabled"] else "",
            ]
        )


def save_pointcloud_ply(path: Path, captured: CapturedFrames) -> None:
    if captured.pointcloud_points is None or captured.pointcloud_texture_frame is None:
        raise SampleSaveError("Point cloud export was requested, but no RealSense point cloud data is available.")
    captured.pointcloud_points.export_to_ply(str(path), captured.pointcloud_texture_frame)


def save_capture(
    config: CaptureConfig,
    captured: CapturedFrames,
    capture_context: dict | None = None,
) -> CaptureResult:
    output_dir = config.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    index_path = output_dir / "index.csv"
    prepare_index_csv(index_path)

    sample_id, timestamp = ensure_unique_sample_id(output_dir)
    files = file_names_for_sample(sample_id)
    metadata = build_metadata(config, sample_id, timestamp, captured, files, capture_context)

    result = CaptureResult(
        sample_id=sample_id,
        timestamp=timestamp,
        output_dir=output_dir,
        files=files,
        metadata=metadata,
    )

    try:
        Image.fromarray(captured.color_image).save(result.color_path)
        np.save(result.d2rgb_path, captured.depth_image)
        if captured.filtered_depth_image is not None:
            np.save(result.d2rgb_filtered_path, captured.filtered_depth_image)
        Image.fromarray(captured.depth_visualization_image).save(result.d2rgb_vis_path, quality=95)
        if config.save_pointcloud:
            save_pointcloud_ply(result.pointcloud_path, captured)
        with result.meta_path.open("w", encoding="utf-8") as meta_file:
            json.dump(metadata, meta_file, indent=2, ensure_ascii=False)
        append_index_row(index_path, result)
    except (OSError, RuntimeError) as error:
        raise SampleSaveError(f"Failed to save captured sample {sample_id}: {error}") from error

    return result
