import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from realsense_capture.config import CaptureConfig
from realsense_capture.models import CapturedFrames
from realsense_capture.storage import INDEX_HEADER, save_capture


def _dummy_capture() -> CapturedFrames:
    return CapturedFrames(
        color_image=np.zeros((4, 6, 3), dtype=np.uint8),
        depth_image=np.array(
            [
                [0, 100, 200, 300, 400, 500],
                [0, 150, 250, 350, 450, 550],
                [0, 200, 300, 400, 500, 600],
                [0, 250, 350, 450, 550, 650],
            ],
            dtype=np.uint16,
        ),
        depth_visualization_image=np.zeros((4, 6, 3), dtype=np.uint8),
        device_info={
            "device_name": "Test Camera",
            "serial_number": "123",
            "firmware_version": "1.0",
            "product_id": "TEST",
            "usb_type_descriptor": "3.2",
        },
        depth_scale=0.001,
        settings={"auto_exposure": [], "warmup_frames": 20},
        streams={
            "color": {"width": 6, "height": 4, "format": "format.rgb8", "fps": 30},
            "depth_aligned_to_color": {
                "width": 6,
                "height": 4,
                "format": "format.z16",
                "fps": 30,
                "intrinsics": {
                    "fx": 6.0,
                    "fy": 4.0,
                    "ppx": 2.5,
                    "ppy": 1.5,
                },
            },
        },
        frames={
            "color": {"frame_number": 1, "timestamp_ms": 1.0},
            "depth": {"frame_number": 1, "timestamp_ms": 1.0},
        },
    )


def _test_config(output_dir: Path) -> CaptureConfig:
    return CaptureConfig(
        output_dir=output_dir,
        color_width=1920,
        color_height=1080,
        color_fps=30,
        depth_width=1280,
        depth_height=720,
        depth_fps=30,
        warmup_frames=20,
        auto_exposure="on",
    )


class StorageTests(unittest.TestCase):
    def test_save_capture_writes_files_metadata_and_index(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _test_config(Path(temp_dir))
            result = save_capture(config, _dummy_capture())

            self.assertTrue(result.color_path.exists())
            self.assertTrue(result.d2rgb_path.exists())
            self.assertTrue(result.d2rgb_vis_path.exists())
            self.assertTrue(result.meta_path.exists())

            metadata = json.loads(result.meta_path.read_text(encoding="utf-8"))
            self.assertEqual(metadata["schema_version"], "1.0")
            self.assertEqual(metadata["sample_id"], result.sample_id)
            self.assertEqual(metadata["capture_mode"], "single")
            self.assertIsNone(metadata["duration_s"])
            self.assertIsNone(metadata["interval_s"])
            self.assertEqual(metadata["frame_index"], 0)
            self.assertEqual(metadata["files"]["color"], result.files.color)
            self.assertEqual(metadata["files"]["d2rgb"], result.files.d2rgb)
            self.assertFalse(metadata["files"]["d2rgb_filtered_saved"])
            self.assertFalse(metadata["pointcloud"]["enabled"])

    def test_save_capture_writes_pointcloud_when_enabled(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = CaptureConfig(
                output_dir=Path(temp_dir),
                color_width=1920,
                color_height=1080,
                color_fps=30,
                depth_width=1280,
                depth_height=720,
                depth_fps=30,
                warmup_frames=20,
                auto_exposure="on",
                save_pointcloud=True,
            )
            captured = _dummy_capture()
            result = save_capture(config, captured)

            self.assertTrue(result.pointcloud_path.exists())
            metadata = json.loads(result.meta_path.read_text(encoding="utf-8"))
            self.assertTrue(metadata["pointcloud"]["enabled"])
            self.assertEqual(metadata["pointcloud"]["path"], result.files.pointcloud)
            self.assertEqual(metadata["pointcloud"]["method"], "aligned_depth_deprojection_binary_ply")

            ply_bytes = result.pointcloud_path.read_bytes()
            header_end = ply_bytes.index(b"end_header\n") + len(b"end_header\n")
            header = ply_bytes[:header_end].decode("ascii")
            self.assertIn("format binary_little_endian 1.0", header)
            self.assertIn("element vertex 20", header)

            vertex_dtype = np.dtype(
                [
                    ("x", "<f4"),
                    ("y", "<f4"),
                    ("z", "<f4"),
                    ("red", "u1"),
                    ("green", "u1"),
                    ("blue", "u1"),
                ]
            )
            vertices = np.frombuffer(ply_bytes, dtype=vertex_dtype, count=20, offset=header_end)
            self.assertTrue(np.all(vertices["z"] > 0))
            self.assertAlmostEqual(float(vertices["z"][0]), 0.1, places=6)

    def test_save_capture_writes_filtered_depth_when_available(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _test_config(Path(temp_dir))
            captured = _dummy_capture()
            captured = CapturedFrames(
                **{
                    **captured.__dict__,
                    "filtered_depth_image": captured.depth_image // 2,
                }
            )
            result = save_capture(config, captured)

            self.assertTrue(result.d2rgb_filtered_path.exists())
            metadata = json.loads(result.meta_path.read_text(encoding="utf-8"))
            self.assertTrue(metadata["files"]["d2rgb_filtered_saved"])

    def test_save_capture_accepts_continuous_context(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            config = _test_config(Path(temp_dir))
            result = save_capture(
                config,
                _dummy_capture(),
                {
                    "capture_mode": "continuous",
                    "duration_s": 3.0,
                    "interval_s": 1.0,
                    "frame_index": 2,
                },
            )

            metadata = json.loads(result.meta_path.read_text(encoding="utf-8"))

            self.assertEqual(metadata["capture_mode"], "continuous")
            self.assertEqual(metadata["duration_s"], 3.0)
            self.assertEqual(metadata["interval_s"], 1.0)
            self.assertEqual(metadata["frame_index"], 2)

            with (Path(temp_dir) / "index.csv").open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.reader(csv_file))

            self.assertEqual(rows[0], INDEX_HEADER)
            self.assertEqual(rows[1][0], result.sample_id)
            self.assertEqual(rows[1][2], result.files.color)


if __name__ == "__main__":
    unittest.main()
