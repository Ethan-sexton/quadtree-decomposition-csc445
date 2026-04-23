import cv2
import numpy as np
import imageio
from classes.ThresholdStrategy import BaseStrategy, AverageStrategy
class QuadTree:    
    def __init__(self, data:cv2.typing.MatLike, strat:BaseStrategy, xmin=0, xmax=-1, ymin=0, ymax=-1, threshold:int=0):
        self.threshold = threshold
        self.memo = {}
        if xmin == 0 and ymin == 0 and xmax == -1 and ymax == -1:
            self.memo[0] = data.copy()     
        self.threshold_strategy = strat
        self.data = data
        
        self.xmin = xmin
        self.ymin = ymin
        
        if xmax == -1:
            self.xmax = self.data.shape[0]
        else:
            self.xmax = xmax
        if ymax == -1:
            self.ymax = self.data.shape[1]
        else:
            self.ymax = ymax        
                    
        # References to subtrees 
        self.northeast = None
        self.northwest = None
        self.southeast = None
        self.southwest = None
        
        if self.threshold != 0:
            self.subdivide()
        self.memoize()
            
    def memoize(self):
        # Save the current image in a hash table based on threshold
        region = self.data[self.xmin:self.xmax, self.ymin:self.ymax].copy()
        key = self.threshold
        self.memo[key] = region
    
    def _calculate_file_size(self, thresh):
        # Calculate the file size in bytes of the compressed image at the given threshold
        image = self.memo[thresh]
        success, encoded = cv2.imencode('.jpg', image)
        if success:
            return len(encoded)
        return 0
    
    def subdivide(self):
        region = self.data[self.xmin:self.xmax, self.ymin:self.ymax]

        if self.threshold_strategy.need_subdivide(region, self.threshold):
            xmid = (self.xmin + self.xmax) // 2
            ymid = (self.ymin + self.ymax) // 2

            self.northeast = self.copy(self.data, self.threshold_strategy, xmid, self.xmax, ymid, self.ymax, self.threshold)
            self.northwest = self.copy(self.data, self.threshold_strategy, self.xmin, xmid, ymid, self.ymax, self.threshold)
            self.southwest = self.copy(self.data, self.threshold_strategy, self.xmin, xmid, self.ymin, ymid, self.threshold)
            self.southeast = self.copy(self.data, self.threshold_strategy, xmid, self.xmax, self.ymin, ymid, self.threshold)
        else:
            self.data[self.xmin:self.xmax, self.ymin:self.ymax] = np.full(region.shape, np.array(self.threshold_strategy.region_value(), dtype=np.uint8))


    def copy(self, data:cv2.typing.MatLike, strat:BaseStrategy, xmin=0, xmax=-1, ymin=0, ymax=-1, threshold:int=0):
        return QuadTree(data, strat, xmin, xmax, ymin, ymax, threshold)
    
    def update(self, thresh):
        self.threshold = thresh
        if self.memo.get(self.threshold) is not None:
            self.data = self.memo[self.threshold]
            return
        self.subdivide()
        self.memoize()

    def get_leaf_rectangles(self, thresh):   
        # If children exist, traverse them (use the actual tree structure)
        if self.northeast is not None:
            rects = []
            rects.extend(self.northeast.get_leaf_rectangles(thresh))
            rects.extend(self.northwest.get_leaf_rectangles(thresh))
            rects.extend(self.southwest.get_leaf_rectangles(thresh))
            rects.extend(self.southeast.get_leaf_rectangles(thresh))
            return rects
        else:
            # Leaf node - return this rectangle
            return [(self.xmin, self.xmax, self.ymin, self.ymax)]


class ImageCompression(QuadTree):

    def __init__(self, data:cv2.typing.MatLike, strat:BaseStrategy, xmin=0, xmax=-1, ymin=0, ymax=-1, threshold:int=0):
        self.file_sizes = {}  # Store file sizes by threshold
        self.rectangles = {}  # Store rectangles by threshold
        super().__init__(data, strat, xmin, xmax, ymin, ymax, threshold)

    def display(self, thresh):
        # Returns a copy of an image at the given threshold.
        # If it hasn't been calculated, calculate it then return it
        if self.memo.get(thresh) is None:
            # Create a fresh copy of the original data
            temp_data = self.memo[0].copy()
            self._apply_threshold(temp_data, thresh)
            self.memo[thresh] = temp_data
        return self.memo[thresh].copy()
    
    def memoize(self):
        # Save the current image and rectangles in a hash table based on threshold
        super().memoize()
        key = self.threshold
        self.rectangles[key] = self.get_leaf_rectangles(self.threshold)
        self.file_sizes[key] = self._calculate_file_size(key)
    
    def _calculate_file_size(self, thresh):
        # Calculate the file size in bytes of the compressed image at the given threshold
        image = self.memo[thresh]
        success, encoded = cv2.imencode('.jpg', image)
        if success:
            return len(encoded)
        return 0
    
    def get_file_size(self, thresh):
        # Returns the file size of an image at the given threshold
        if self.file_sizes.get(thresh) is None:
            self.update(thresh)
        return self.file_sizes.get(thresh, 0)
    
    def _apply_threshold(self, data, thresh):
        # Recursively apply threshold to the data without modifying self.data
        region = data[self.xmin:self.xmax, self.ymin:self.ymax]
        if self.threshold_strategy.need_subdivide(region, thresh):
            # Need to subdivide at this threshold
            xmid = (self.xmin + self.xmax) // 2
            ymid = (self.ymin + self.ymax) // 2
            
            # Create temporary children if they don't exist
            if self.northeast is None:
                self.northeast = self.copy(data, self.threshold_strategy, xmid, self.xmax, ymid, self.ymax, thresh)
                self.northwest = self.copy(data, self.threshold_strategy, self.xmin, xmid, ymid, self.ymax, thresh)
                self.southwest = self.copy(data, self.threshold_strategy, self.xmin, xmid, self.ymin, ymid, thresh)
                self.southeast = self.copy(data, self.threshold_strategy, xmid, self.xmax, self.ymin, ymid, thresh)
            
            self.northeast._apply_threshold(data, thresh)
            self.northwest._apply_threshold(data, thresh)
            self.southwest._apply_threshold(data, thresh)
            self.southeast._apply_threshold(data, thresh)
        else:
            # Leaf node at this threshold
            data[self.xmin:self.xmax, self.ymin:self.ymax] = np.full(region.shape, np.array(self.threshold_strategy.region_value(), dtype=np.uint8))
            
    def psnr(self, thresh) -> float:
        # Given a certain threshold, return the peak signal-to-noise ratio
        # between it and the original image. Returns in decibels
        # Note: An input of 0 would normally give infinity/NaN due to division by 0, this 
        # will return a number around 361 to prevent division by 0
        if self.memo.get(thresh) is None:
            self.update(thresh)
        original = self.memo[0].copy()
        processed = self.memo[thresh].copy()
        return cv2.PSNR(original, processed)
        
    def save(self, path="resources\\scream.jpg"):
        cv2.imwrite(path, self.memo[self.threshold])
        
    def animate(self, path="resources\\test.gif", show_tree=False, num_frames=50):
        output = []
        counter = 1
        # Generate frames across a range of thresholds        
        for thresh in self.memo:
            print(f'Animating frame {counter}/{len(self.memo)}')
            image = self.display(thresh)
            if show_tree:
                rects = self.rectangles.get(thresh, [])
                for xmin, xmax, ymin, ymax in rects:
                    cv2.rectangle(image, (ymin, xmin), (ymax, xmax), (0, 255, 0), 2)
            processed = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            output.append(processed)
            counter += 1
        imageio.mimsave(path, output)
        print(f'Saved to {path}')