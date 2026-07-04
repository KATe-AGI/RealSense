from typing import Any

from .config import CaptureConfig
from .models import CapturedFrames, SampleFileNames


SCHEMA_VERSION = "1.0"


def build_metadata(
    config: CaptureConfig,
    sample_id: str,
    timestamp: str,
    captured: CapturedFrames,
    files: SampleFileNames,
    capture_context: dict[str, Any] | None = None,
) -> dict[str, Any]:
    if capture_context is None:
        capture_context = {
            "capture_mode": "single",
            "duration_s": None,
            "interval_s": None,
            "frame_index": 0,
        }

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "sample_id": sample_id,
        "timestamp": timestamp,
        **capture_context,
        **captured.device_info,
        "depth_scale": captured.depth_scale,
        "capture_config": config.to_metadata(),
        "settings": captured.settings,
        "streams": captured.streams,
        "frames": captured.frames,
        "files": {
            **files.as_dict(),
            "d2rgb_filtered_saved": captured.filtered_depth_image is not None,
        },
        "pointcloud": {
            "enabled": config.save_pointcloud,
            "path": files.pointcloud if config.save_pointcloud else None,
            "method": "pyrealsense2.rs.pointcloud.export_to_ply" if config.save_pointcloud else None,
        },
    }
    return metadata
