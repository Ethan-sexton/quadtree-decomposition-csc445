import numpy as np

class BaseStrategy:    
    # Allows for determining if the tree needs to recurse based on different criteria
    def need_subdivide(self, region, threshold) -> bool:
        return False
    
    def region_value(self) -> np.float32 | None:
        return None
    
class AverageStrategy(BaseStrategy):
    def __init__(self):
        self.average_rgb = None
        
    def need_subdivide(self, region, threshold):
        self.average_rgb = np.mean(region, axis=(0, 1), dtype=np.float32)
        variance = np.mean((region.astype(np.float32) - self.average_rgb) ** 2)
        return variance > threshold
    
    def region_value(self):
        return np.round(self.average_rgb).astype(np.uint8)