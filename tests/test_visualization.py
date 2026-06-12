import sys
import unittest
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from realsense_capture.visualization import create_depth_visualization


class VisualizationTests(unittest.TestCase):
    def test_depth_visualization_shape_dtype_and_zero_mask(self):
        depth = np.array([[0, 100, 200], [300, 400, 500]], dtype=np.uint16)

        vis = create_depth_visualization(depth)

        self.assertEqual(vis.shape, (2, 3, 3))
        self.assertEqual(vis.dtype, np.uint8)
        self.assertTrue(np.all(vis[0, 0] == 0))
        self.assertGreater(int(vis[1, 2].sum()), 0)


if __name__ == "__main__":
    unittest.main()
