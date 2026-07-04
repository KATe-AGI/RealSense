import sys
import unittest
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from capture_current_frame import config_from_args, parse_args, validate_runtime_args
from realsense_capture.visualization import compose_preview, _pointcloud_preview_size


OUTPUT_DIR_ARGS = ["--output-dir", r"E:\camera\test_20260612"]
PREVIEW_MAX_WIDTH = 1280
PREVIEW_MAX_HEIGHT = 900


class CliTests(unittest.TestCase):
    def test_continuous_mode_requires_duration(self):
        args = parse_args([*OUTPUT_DIR_ARGS, "--mode", "continuous"])

        with self.assertRaises(ValueError):
            validate_runtime_args(args)

    def test_continuous_duration_and_interval_must_be_positive(self):
        args = parse_args([*OUTPUT_DIR_ARGS, "--mode", "continuous", "--duration-s", "0"])

        with self.assertRaises(ValueError):
            validate_runtime_args(args)

    def test_size_choices_are_parsed_into_config(self):
        args = parse_args([*OUTPUT_DIR_ARGS, "--color-size", "1920x1080", "--depth-size", "1280x720"])

        config = config_from_args(args)

        self.assertEqual((config.color_width, config.color_height), (1920, 1080))
        self.assertEqual((config.depth_width, config.depth_height), (1280, 720))

    def test_save_pointcloud_flag_is_parsed_into_config(self):
        args = parse_args([*OUTPUT_DIR_ARGS, "--save-pointcloud"])

        config = config_from_args(args)

        self.assertTrue(config.save_pointcloud)

        args = parse_args(
            [*OUTPUT_DIR_ARGS, "--mode", "continuous", "--duration-s", "3", "--interval-s", "0"]
        )

        with self.assertRaises(ValueError):
            validate_runtime_args(args)

    def test_compose_preview_stacks_images_vertically(self):
        color = np.zeros((1080, 1920, 3), dtype=np.uint8)
        depth = np.zeros((540, 960, 3), dtype=np.uint8)

        preview = compose_preview(color, depth, PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT)

        self.assertEqual(preview.dtype, np.uint8)
        self.assertEqual(preview.shape[2], 3)
        self.assertLessEqual(preview.shape[0], PREVIEW_MAX_HEIGHT)
        self.assertLessEqual(preview.shape[1], PREVIEW_MAX_WIDTH)
        self.assertGreater(preview.shape[0], preview.shape[1])

    def test_compose_preview_accepts_different_aspect_ratios(self):
        color = np.zeros((1080, 1920, 3), dtype=np.uint8)
        depth = np.zeros((720, 1280, 3), dtype=np.uint8)

        preview = compose_preview(color, depth, PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT)

        self.assertEqual(preview.dtype, np.uint8)
        self.assertEqual(preview.shape[2], 3)
        self.assertLessEqual(preview.shape[0], PREVIEW_MAX_HEIGHT)
        self.assertLessEqual(preview.shape[1], PREVIEW_MAX_WIDTH)

    def test_compose_preview_accepts_pointcloud_image(self):
        color = np.zeros((1080, 1920, 3), dtype=np.uint8)
        depth = np.zeros((720, 1280, 3), dtype=np.uint8)
        pointcloud = np.zeros((540, 960, 3), dtype=np.uint8)

        preview = compose_preview(color, depth, PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT, pointcloud)

        self.assertEqual(preview.dtype, np.uint8)
        self.assertEqual(preview.shape[2], 3)
        self.assertLessEqual(preview.shape[0], PREVIEW_MAX_HEIGHT)
        self.assertLessEqual(preview.shape[1], PREVIEW_MAX_WIDTH)

    def test_pointcloud_preview_size_preserves_depth_aspect_ratio(self):
        captured = type(
            "Captured",
            (),
            {
                "depth_image": np.zeros((1080, 1920), dtype=np.uint16),
                "streams": {
                    "depth_aligned_to_color": {
                        "intrinsics": {
                            "width": 1920,
                            "height": 1080,
                        }
                    }
                },
            },
        )()

        width, height = _pointcloud_preview_size(captured, 640, 900)

        self.assertEqual((width, height), (640, 360))


if __name__ == "__main__":
    unittest.main()
