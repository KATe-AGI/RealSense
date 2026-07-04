import sys
import unittest
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from realsense_capture.visualization import (
    PointCloudViewState,
    create_aligned_pointcloud_visualization,
    create_depth_visualization,
    create_pointcloud_visualization,
)


class VisualizationTests(unittest.TestCase):
    def test_depth_visualization_shape_dtype_and_zero_mask(self):
        depth = np.array([[0, 100, 200], [300, 400, 500]], dtype=np.uint16)

        vis = create_depth_visualization(depth)

        self.assertEqual(vis.shape, (2, 3, 3))
        self.assertEqual(vis.dtype, np.uint8)
        self.assertTrue(np.all(vis[0, 0] == 0))
        self.assertGreater(int(vis[1, 2].sum()), 0)

    def test_pointcloud_visualization_shape_dtype_and_content(self):
        depth = np.array([[0, 1000, 1000], [1000, 1000, 1000]], dtype=np.uint16)
        color = np.full((2, 3, 3), 128, dtype=np.uint8)
        streams = {
            "depth_aligned_to_color": {
                "intrinsics": {
                    "fx": 3.0,
                    "fy": 2.0,
                    "ppx": 1.0,
                    "ppy": 0.5,
                }
            }
        }

        vis = create_pointcloud_visualization(depth, color, 0.001, streams, output_size=(120, 80))

        self.assertEqual(vis.shape, (80, 120, 3))
        self.assertEqual(vis.dtype, np.uint8)
        self.assertGreater(int(vis.sum()), 0)

    def test_pointcloud_visualization_uses_view_state(self):
        depth = np.full((4, 6), 1000, dtype=np.uint16)
        color = np.full((4, 6, 3), 128, dtype=np.uint8)
        streams = {
            "depth_aligned_to_color": {
                "intrinsics": {
                    "width": 6,
                    "height": 4,
                    "fx": 6.0,
                    "fy": 4.0,
                    "ppx": 2.5,
                    "ppy": 1.5,
                }
            }
        }
        view_state = PointCloudViewState()
        view_state.yaw += 0.5

        vis = create_pointcloud_visualization(
            depth,
            color,
            0.001,
            streams,
            output_size=(120, 80),
            target_points=1000000,
            view_state=view_state,
        )

        self.assertEqual(vis.shape, (80, 120, 3))
        self.assertEqual(vis.dtype, np.uint8)

    def test_pointcloud_visualization_matches_image_orientation(self):
        depth = np.array(
            [
                [1000, 0, 1000],
                [1000, 0, 0],
            ],
            dtype=np.uint16,
        )
        color = np.zeros((2, 3, 3), dtype=np.uint8)
        color[0, 0] = [255, 0, 0]
        color[0, 2] = [0, 255, 0]
        color[1, 0] = [0, 0, 255]
        streams = {"depth_aligned_to_color": {"intrinsics": {"width": 3, "height": 2}}}

        vis = create_aligned_pointcloud_visualization(
            depth,
            color,
            0.001,
            streams,
            output_size=(300, 200),
            target_points=1000000,
        )

        self.assertGreater(int(vis[:80, :80, 0].sum()), 0)
        self.assertGreater(int(vis[:80, 220:, 1].sum()), 0)
        self.assertGreater(int(vis[120:, :80, 2].sum()), 0)


if __name__ == "__main__":
    unittest.main()
