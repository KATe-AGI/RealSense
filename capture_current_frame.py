import argparse
from pathlib import Path
import sys
import time

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from realsense_capture.camera import RealSenseCamera
from realsense_capture.config import CaptureConfig
from realsense_capture.errors import RealSenseCaptureError
from realsense_capture.storage import save_capture
from realsense_capture.visualization import close_preview_windows, show_preview

COLOR_SIZE_CHOICES = ("1920x1080", "1280x720", "960x540", "848x480", "640x480", "640x360", "424x240", "320x240")
DEPTH_SIZE_CHOICES = ("1280x720", "848x480", "640x480", "640x360", "480x270", "424x240", "320x240", "320x180")
COLOR_FPS_CHOICES = (6, 15, 30, 60)
DEPTH_FPS_CHOICES = (6, 15, 30, 60, 90)


def parse_size(size: str) -> tuple[int, int]:
    width, height = size.split("x", maxsplit=1)
    return int(width), int(height)


"""
Windows CLI examples:

# 单帧采集--(供SDK二开用)
python capture_current_frame.py `
  --mode single `
  --output-dir camera_data

# 连续采集：总时长 10 秒，每 1 秒保存一帧 (采集数据集用)
python capture_current_frame.py `
  --mode continuous `
  --duration-s 20 `
  --interval-s 1 `
  --output-dir camera_data

# 按键采集: 预览窗口持续显示, 按空格保存当前帧, q/Esc 退出 (手眼标定用)
python capture_current_frame.py `
  --mode manual `
  --output-dir camera_data
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture one RealSense color frame and aligned depth frame."
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--mode", choices=("single", "continuous", "manual"), default="single", help="Capture mode: single=one shot, continuous=timed interval, manual=press Space to save. Default: single")
    parser.add_argument("--duration-s", type=float, default=None, help="Total continuous capture duration in seconds. Required when --mode continuous")
    parser.add_argument("--interval-s", type=float, default=1.0, help="Continuous capture save interval in seconds. Default: 1.0")
    parser.add_argument("--color-size", choices=COLOR_SIZE_CHOICES, default="1920x1080", help="Color stream size. Default: 1920x1080")
    parser.add_argument("--color-fps", type=int, choices=COLOR_FPS_CHOICES, default=30, help="Color stream FPS. Default: 30")
    parser.add_argument("--depth-size", choices=DEPTH_SIZE_CHOICES, default="1280x720", help="Depth stream size. Default: 1280x720")
    parser.add_argument("--depth-fps", type=int, choices=DEPTH_FPS_CHOICES, default=30, help="Depth stream FPS. Default: 30")
    parser.add_argument("--warmup-frames", type=int, default=20, help="Frames to discard before saving. Default: 20")
    parser.add_argument("--auto-exposure", choices=("on", "off", "default"), default="on", help="Auto exposure mode: on=force enable, off=force disable, default=keep camera current setting. Default: on")
    parser.add_argument("--enable-post-processing", action="store_true", help="Save Viewer-like Stereo Module post-processing output as d2rgb_filtered.npy. Default: disabled")
    parser.add_argument("--save-pointcloud", action="store_true", help="Save a colored point cloud PLY for each captured sample. Default: disabled")
    parser.add_argument("--preview-window-name", default="RealSense Capture", help="Preview window name for continuous/manual mode. Default: RealSense Capture")
    parser.add_argument("--preview-max-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=(1280, 900), help="Continuous preview max display size as (width, height). Default: (1280, 900)")
    return parser.parse_args(argv)


def validate_runtime_args(args: argparse.Namespace) -> None:
    if args.mode == "continuous" and args.duration_s is None:
        raise ValueError("--duration-s is required when --mode continuous")
    if args.duration_s is not None and args.duration_s <= 0:
        raise ValueError("--duration-s must be greater than 0")
    if args.interval_s <= 0:
        raise ValueError("--interval-s must be greater than 0")
    if args.preview_max_size[0] <= 0 or args.preview_max_size[1] <= 0:
        raise ValueError("--preview-max-size width and height must be greater than 0")


def config_from_args(args: argparse.Namespace) -> CaptureConfig:
    color_width, color_height = parse_size(args.color_size)
    depth_width, depth_height = parse_size(args.depth_size)
    config = CaptureConfig(
        output_dir=args.output_dir,
        color_width=color_width,
        color_height=color_height,
        color_fps=args.color_fps,
        depth_width=depth_width,
        depth_height=depth_height,
        depth_fps=args.depth_fps,
        warmup_frames=args.warmup_frames,
        auto_exposure=args.auto_exposure,
        enable_post_processing=args.enable_post_processing,
        save_pointcloud=args.save_pointcloud,
    )
    config.validate()
    return config


def capture_context(
    mode: str,
    duration_s: float | None,
    interval_s: float | None,
    frame_index: int,
) -> dict:
    return {
        "capture_mode": mode,
        "duration_s": duration_s,
        "interval_s": interval_s,
        "frame_index": frame_index,
    }


def capture_current_frame(config: CaptureConfig):
    with RealSenseCamera(config) as camera:
        captured = camera.capture_once()
        return save_capture(
            config,
            captured,
            capture_context("single", None, None, 0),
        )


def capture_continuous(
    config: CaptureConfig,
    duration_s: float,
    interval_s: float,
    preview_window_name: str,
    preview_max_width: int,
    preview_max_height: int,
) -> list:
    results = []
    with RealSenseCamera(config) as camera:
        camera.warmup()
        start_time = time.monotonic()
        end_time = start_time + duration_s
        next_save_time = start_time
        frame_index = 0

        try:
            while time.monotonic() <= end_time:
                captured = camera.capture_frame()
                now = time.monotonic()

                if now >= next_save_time:
                    result = save_capture(
                        config,
                        captured,
                        capture_context("continuous", duration_s, interval_s, frame_index),
                    )
                    results.append(result)
                    print(f"Saved sample[{frame_index}]: {result.sample_id}")
                    frame_index += 1
                    next_save_time += interval_s

                key = show_preview(
                    captured,
                    preview_window_name,
                    preview_max_width,
                    preview_max_height,
                )
                if key in (ord("q"), 27):
                    break
        finally:
            close_preview_windows()

    return results


def capture_manual(
    config: CaptureConfig,
    preview_window_name: str,
    preview_max_width: int,
    preview_max_height: int,
) -> list:
    results = []
    with RealSenseCamera(config) as camera:
        camera.warmup()
        frame_index = 0
        print("Manual mode: press Space to save, q/Esc to quit.")

        try:
            while True:
                captured = camera.capture_frame()
                key = show_preview(
                    captured,
                    preview_window_name,
                    preview_max_width,
                    preview_max_height,
                )
                if key in (ord("q"), 27):
                    break
                if key == ord(" "):
                    result = save_capture(
                        config,
                        captured,
                        capture_context("manual", None, None, frame_index),
                    )
                    results.append(result)
                    print(f"Saved sample[{frame_index}]: {result.sample_id}")
                    frame_index += 1
        finally:
            close_preview_windows()

    return results


def print_result(result) -> None:
    print(f"Saved sample: {result.sample_id}")
    print(f"Output directory: {result.output_dir}")
    print(f"Color: {result.files.color}")
    print(f"Aligned depth: {result.files.d2rgb}")
    if result.metadata["files"]["d2rgb_filtered_saved"]:
        print(f"Filtered depth: {result.files.d2rgb_filtered}")
    print(f"Depth visualization: {result.files.d2rgb_vis}")
    if result.metadata["pointcloud"]["enabled"]:
        print(f"Point cloud: {result.files.pointcloud}")
    print(f"Metadata: {result.files.meta}")
    print("Index: index.csv")


def main(argv: list[str] | None = None) -> int:
    try:
        args = parse_args(argv)
        validate_runtime_args(args)
        config = config_from_args(args)
        if args.mode == "continuous":
            results = capture_continuous(
                config,
                args.duration_s,
                args.interval_s,
                args.preview_window_name,
                args.preview_max_size[0],
                args.preview_max_size[1],
            )
            print(f"Saved {len(results)} samples")
            return 0
        if args.mode == "manual":
            results = capture_manual(
                config,
                args.preview_window_name,
                args.preview_max_size[0],
                args.preview_max_size[1],
            )
            print(f"Saved {len(results)} samples")
            return 0
        result = capture_current_frame(config)
    except (RealSenseCaptureError, ValueError) as error:
        print(f"Capture failed: {error}", file=sys.stderr)
        return 1

    print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
