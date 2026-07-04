from dataclasses import asdict, dataclass
from pathlib import Path


AUTO_EXPOSURE_MODES = ("on", "off", "default")


@dataclass(frozen=True)
class CaptureConfig:
    output_dir: Path
    color_width: int
    color_height: int
    color_fps: int
    depth_width: int
    depth_height: int
    depth_fps: int
    warmup_frames: int
    auto_exposure: str
    enable_post_processing: bool = False
    stereo_depth_visual_preset: str = "Dynamic"
    stereo_depth_color_scheme: str = "Jet"
    stereo_histogram_equalization_enabled: bool = True
    enable_decimation_filter: bool = True
    enable_rotation_filter: bool = False
    enable_hdr_merge: bool = False
    enable_sequence_id_filter: bool = False
    enable_threshold_filter: bool = False
    enable_depth_to_disparity: bool = True
    enable_spatial_filter: bool = True
    enable_temporal_filter: bool = True
    enable_hole_filling_filter: bool = False
    enable_disparity_to_depth: bool = True
    save_pointcloud: bool = False

    def validate(self) -> None:
        if self.color_width <= 0:
            raise ValueError("color_width must be greater than 0")
        if self.color_height <= 0:
            raise ValueError("color_height must be greater than 0")
        if self.color_fps <= 0:
            raise ValueError("color_fps must be greater than 0")
        if self.depth_width <= 0:
            raise ValueError("depth_width must be greater than 0")
        if self.depth_height <= 0:
            raise ValueError("depth_height must be greater than 0")
        if self.depth_fps <= 0:
            raise ValueError("depth_fps must be greater than 0")
        if self.warmup_frames < 0:
            raise ValueError("warmup_frames must be greater than or equal to 0")
        if self.auto_exposure not in AUTO_EXPOSURE_MODES:
            modes = ", ".join(AUTO_EXPOSURE_MODES)
            raise ValueError(f"auto_exposure must be one of: {modes}")

    def to_metadata(self) -> dict:
        data = asdict(self)
        data["output_dir"] = str(self.output_dir)
        return data
