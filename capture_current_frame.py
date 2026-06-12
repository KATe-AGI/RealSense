import argparse
from pathlib import Path
import sys
import time

import cv2
import numpy as np


ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from realsense_capture.camera import RealSenseCamera
from realsense_capture.config import CaptureConfig
from realsense_capture.errors import RealSenseCaptureError
from realsense_capture.storage import save_capture

r'''
windows:

# 单帧采集--(供SDK二开用)
python capture_current_frame.py ` 
  --mode single `
  --output-dir E:\camera\test_20260612

# 连续采集：总时长 10 秒，每 1 秒保存一帧 (采集数据集用)
python capture_current_frame.py `
  --mode continuous `
  --duration-s 10 `
  --interval-s 1 `
  --output-dir E:\camera\test_20260612

# 按键采集：预览窗口持续显示，按空格保存当前帧，q/Esc 退出 (手眼标定用)
python capture_current_frame.py `
  --mode manual `
  --output-dir E:\camera\test_20260612

'''


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture one RealSense color frame and aligned depth frame."
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Output directory.")
    parser.add_argument("--mode", choices=("single", "continuous", "manual"), default="single", help="Capture mode: single=one shot, continuous=timed interval, manual=press Space to save. Default: single")
    parser.add_argument("--duration-s", type=float, default=None, help="Total continuous capture duration in seconds. Required when --mode continuous")
    parser.add_argument("--interval-s", type=float, default=1.0, help="Continuous capture save interval in seconds. Default: 1.0")
    parser.add_argument("--color-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=(1920, 1080), help="Color stream size as (width, height). Default: (1920, 1080)")
    parser.add_argument("--color-fps", type=int, default=30, help="Color stream FPS. Default: 30")
    parser.add_argument("--depth-size", type=int, nargs=2, metavar=("WIDTH", "HEIGHT"), default=(1280, 720), help="Depth stream size as (width, height). Default: (1280, 720)")
    parser.add_argument("--depth-fps", type=int, default=30, help="Depth stream FPS. Default: 30")
    parser.add_argument("--warmup-frames", type=int, default=10, help="Frames to discard before saving. Default: 20")
    parser.add_argument("--auto-exposure", choices=("on", "off", "default"), default="on", help="Auto exposure mode: on=force enable, off=force disable, default=keep camera current setting. Default: on")
    parser.add_argument("--enable-post-processing", action="store_true", help="Apply Viewer-like Stereo Module post-processing to d2rgb.npy. Default: disabled")
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
    if args.color_size[0] <= 0 or args.color_size[1] <= 0:
        raise ValueError("--color-size width and height must be greater than 0")
    if args.depth_size[0] <= 0 or args.depth_size[1] <= 0:
        raise ValueError("--depth-size width and height must be greater than 0")
    if args.preview_max_size[0] <= 0 or args.preview_max_size[1] <= 0:
        raise ValueError("--preview-max-size width and height must be greater than 0")


def config_from_args(args: argparse.Namespace) -> CaptureConfig:
    config = CaptureConfig(
        output_dir=args.output_dir,
        color_width=args.color_size[0],
        color_height=args.color_size[1],
        color_fps=args.color_fps,
        depth_width=args.depth_size[0],
        depth_height=args.depth_size[1],
        depth_fps=args.depth_fps,
        warmup_frames=args.warmup_frames,
        auto_exposure=args.auto_exposure,
        enable_post_processing=args.enable_post_processing,
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


def _ensure_rgb(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    return image


def _resize_to_width(image: np.ndarray, target_width: int) -> np.ndarray:
    height, width = image.shape[:2]
    if width == target_width:
        return image
    target_height = max(1, round(height * target_width / width))
    return cv2.resize(image, (target_width, target_height), interpolation=cv2.INTER_AREA)


def compose_preview(
    color_image: np.ndarray,
    depth_visualization_image: np.ndarray,
    max_width: int,
    max_height: int,
) -> np.ndarray:
    color_image = _ensure_rgb(color_image)
    depth_visualization_image = _ensure_rgb(depth_visualization_image)

    color_height, color_width = color_image.shape[:2]
    depth_height, depth_width = depth_visualization_image.shape[:2]
    color_aspect = color_width / color_height
    depth_aspect = depth_width / depth_height
    width_limited_by_height = max_height / (1 / color_aspect + 1 / depth_aspect)
    target_width = max(
        1,
        round(min(max_width, color_width, depth_width, width_limited_by_height)),
    )

    color_preview = _resize_to_width(color_image, target_width)
    depth_preview = _resize_to_width(depth_visualization_image, target_width)
    return np.vstack([color_preview, depth_preview])


def show_preview(captured, window_name: str, max_width: int, max_height: int) -> int:
    preview_rgb = compose_preview(
        captured.color_image,
        captured.depth_visualization_image,
        max_width,
        max_height,
    )
    preview_bgr = cv2.cvtColor(preview_rgb, cv2.COLOR_RGB2BGR)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, preview_bgr.shape[1], preview_bgr.shape[0])
    cv2.imshow(window_name, preview_bgr)
    return cv2.waitKey(1) & 0xFF


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
            cv2.destroyAllWindows()

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
            cv2.destroyAllWindows()

    return results


def print_result(result) -> None:
    print(f"Saved sample: {result.sample_id}")
    print(f"Output directory: {result.output_dir}")
    print(f"Color: {result.files.color}")
    print(f"Aligned depth: {result.files.d2rgb}")
    print(f"Depth visualization: {result.files.d2rgb_vis}")
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
