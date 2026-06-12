import cv2
import numpy as np


def create_depth_visualization(depth_image: np.ndarray) -> np.ndarray:
    valid_depth = depth_image[depth_image > 0]
    scaled = np.zeros(depth_image.shape, dtype=np.uint8)

    if valid_depth.size:
        lower = float(np.percentile(valid_depth, 1))
        upper = float(np.percentile(valid_depth, 99))
        if upper <= lower:
            upper = lower + 1.0

        clipped = np.clip(depth_image.astype(np.float32), lower, upper)
        scaled = ((clipped - lower) / (upper - lower) * 255.0).astype(np.uint8)
        scaled[depth_image == 0] = 0

    colored = cv2.applyColorMap(scaled, cv2.COLORMAP_JET)
    colored[depth_image == 0] = 0
    return colored
