import numpy as np

class BaseStrategy:
    def need_subdivide(self, region, threshold) -> bool:
        return False

    def region_value(self, region):
        return np.zeros(3, dtype=np.uint8)


class AverageStrategy(BaseStrategy):
    def need_subdivide(self, region, threshold):
        threshold = int(threshold)

        avg = np.mean(region, axis=(0, 1))
        var = np.mean((region.astype(np.float32) - avg) ** 2)

        return var > threshold
    def region_value(self, region):
        avg = np.mean(region, axis=(0, 1))
        return np.clip(avg, 0, 255).astype(np.uint8)

import numpy as np
import cv2

class RangeStrategy(BaseStrategy):
    def need_subdivide(self, region, threshold):
        # Ensure float for safe subtraction
        region = region.astype(np.int16)

        # Compute channel-wise range (R, G, B)
        min_val = region.min(axis=(0, 1))
        max_val = region.max(axis=(0, 1))

        range_val = max_val - min_val  # per-channel range

        # Use strongest channel difference
        max_range = np.max(range_val)

        return max_range > threshold

    def region_value(self, region):
        # Simple and stable: average color
        return np.mean(region, axis=(0, 1)).astype(np.uint8)