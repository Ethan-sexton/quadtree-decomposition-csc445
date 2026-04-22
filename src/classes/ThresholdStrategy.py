import numpy as np

class BaseStrategy:    
    def need_subdivide(self, region, threshold) -> bool:
        return False
    
    def region_value(self) -> tuple:
        return (0,0,0)
    
class AverageStrategy(BaseStrategy):
    def __init__(self):
        self.average_color = None
        
    def need_subdivide(self, region, threshold) -> bool:
        if region.size == 0:
            return False

        self.average_rgb = tuple(np.mean(region.reshape(-1, region.shape[-1]), axis=0))

        if region.shape[0] <= 1 or region.shape[1] <= 1:
            return False

        diff = np.abs(region.astype(np.float32) - self.average_rgb)
        max_diff = np.max(diff)

        if max_diff > threshold:
            return True
        else:
            return False
    
    def region_value(self):
        return self.average_rgb