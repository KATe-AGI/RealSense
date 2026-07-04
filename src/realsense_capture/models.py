from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CapturedFrames:
    color_image: np.ndarray
    depth_image: np.ndarray
    depth_visualization_image: np.ndarray
    device_info: dict[str, Any]
    depth_scale: float
    settings: dict[str, Any]
    streams: dict[str, Any]
    frames: dict[str, Any]
    filtered_depth_image: np.ndarray | None = None
    pointcloud_points: Any | None = None
    pointcloud_texture_frame: Any | None = None


@dataclass(frozen=True)
class SampleFileNames:
    color: str
    d2rgb: str
    d2rgb_filtered: str
    d2rgb_vis: str
    meta: str
    pointcloud: str

    def as_dict(self) -> dict[str, str]:
        return {
            "color": self.color,
            "d2rgb": self.d2rgb,
            "d2rgb_filtered": self.d2rgb_filtered,
            "d2rgb_vis": self.d2rgb_vis,
            "meta": self.meta,
            "pointcloud": self.pointcloud,
        }


@dataclass(frozen=True)
class CaptureResult:
    sample_id: str
    timestamp: str
    output_dir: Path
    files: SampleFileNames
    metadata: dict[str, Any]

    @property
    def color_path(self) -> Path:
        return self.output_dir / self.files.color

    @property
    def d2rgb_path(self) -> Path:
        return self.output_dir / self.files.d2rgb

    @property
    def d2rgb_filtered_path(self) -> Path:
        return self.output_dir / self.files.d2rgb_filtered

    @property
    def d2rgb_vis_path(self) -> Path:
        return self.output_dir / self.files.d2rgb_vis

    @property
    def meta_path(self) -> Path:
        return self.output_dir / self.files.meta

    @property
    def pointcloud_path(self) -> Path:
        return self.output_dir / self.files.pointcloud
