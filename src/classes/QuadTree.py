import cv2
import numpy as np

class QuadTree:    
    def __init__(self, im, xmin=0, xmax=-1, ymin=0, ymax=-1, threshold=0):
        self.threshold = threshold
        self.image = im
        self.memo = {}
        
        
        self.xmin = xmin
        self.ymin = ymin
        if xmax == -1:
            self.xmax = self.image.shape[0]
        else:
            self.xmax = xmax
        if ymax == -1:
            self.ymax = self.image.shape[1]
        else:
            self.ymax = ymax        
            
        self.average_rgb = ()
        
        # Lists to store subtrees 
        self.northeast = []
        self.northwest = []
        self.southeast = []
        self.southwest = []
 
        self.subdivide()
        self.memoize()
            
    def memoize(self):
        # Save the current region image for this node
        region = self.image[self.xmin:self.xmax, self.ymin:self.ymax].copy()
        key = (self.threshold)
        self.memo[key] = region
        
    def subdivide(self):
        region = self.image[self.xmin:self.xmax, self.ymin:self.ymax]
        if region.size == 0:
            return

        self.average_rgb = tuple(np.mean(region.reshape(-1, region.shape[-1]), axis=0))

        if region.shape[0] <= 1 or region.shape[1] <= 1:
            return

        diff = np.abs(region.astype(np.float32) - self.average_rgb)
        max_diff = np.max(diff)

        if max_diff > self.threshold:
            xmid = (self.xmin + self.xmax) // 2
            ymid = (self.ymin + self.ymax) // 2

            self.northeast = QuadTree(self.image, xmid, self.xmax, ymid, self.ymax, self.threshold)
            self.northwest = QuadTree(self.image, self.xmin, xmid, ymid, self.ymax, self.threshold)
            self.southwest = QuadTree(self.image, self.xmin, xmid, self.ymin, ymid, self.threshold)
            self.southeast = QuadTree(self.image, xmid, self.xmax, self.ymin, ymid, self.threshold)
        else:
            self.image[self.xmin:self.xmax, self.ymin:self.ymax] = np.full(region.shape, np.array(self.average_rgb, dtype=np.uint8))

    def update(self, thresh):
        self.threshold = thresh
        self.subdivide()
        self.memoize()

    def display(self):
        cv2.imshow("Original", self.memo[0])
        cv2.imshow("Processed", self.memo[self.threshold])
        
        cv2.waitKey()
        cv2.destroyAllWindows()
    
    def save(self):
        cv2.imwrite("resources\\processed.jpg", self.memo[self.threshold])