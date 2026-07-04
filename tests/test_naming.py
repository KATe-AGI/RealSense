import sys
import tempfile
import unittest
from datetime import datetime
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from realsense_capture.storage import ensure_unique_sample_id, file_names_for_sample, format_timestamp


class NamingTests(unittest.TestCase):
    def test_format_timestamp_uses_expected_sample_id(self):
        sample_id, timestamp = format_timestamp(datetime(2026, 6, 10, 15, 30, 35, 316000))

        self.assertEqual(sample_id, "20260610_153035_316")
        self.assertEqual(timestamp, "2026-06-10 15:30:35.316")

    def test_file_names_share_sample_id(self):
        files = file_names_for_sample("20260610_153035_316")

        self.assertEqual(files.color, "20260610_153035_316_color.png")
        self.assertEqual(files.d2rgb, "20260610_153035_316_d2rgb.npy")
        self.assertEqual(files.d2rgb_filtered, "20260610_153035_316_d2rgb_filtered.npy")
        self.assertEqual(files.d2rgb_vis, "20260610_153035_316_d2rgb_vis.jpg")
        self.assertEqual(files.meta, "20260610_153035_316_meta.json")
        self.assertEqual(files.pointcloud, "20260610_153035_316_pointcloud.ply")

    def test_unique_sample_id_does_not_return_existing_file_prefix(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            output_dir = Path(temp_dir)
            sample_id, _ = ensure_unique_sample_id(output_dir)
            (output_dir / f"{sample_id}_color.png").touch()

            next_sample_id, _ = ensure_unique_sample_id(output_dir)

        self.assertNotEqual(next_sample_id, sample_id)


if __name__ == "__main__":
    unittest.main()
