import sys
import unittest
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from capture_current_frame import (
    compose_preview,
    parse_args,
    validate_runtime_args,
)


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


if __name__ == "__main__":
    unittest.main()
