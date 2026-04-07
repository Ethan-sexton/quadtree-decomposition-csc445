import cv2
class QuadTree:    
    def __init__(self, im, xmin=0, xmax=0, ymin=0, ymax=0, threshold=50):
        self.threshold = threshold
        self.image = im
        self.partitioned = False
        self.memo = {"threshold":"path"}
        
        self.xmin = xmin
        self.xmax = self.image.shape[0]
        self.ymin = ymin
        self.ymax = self.image.shape[1]
        
        self.northeast = []
        self.northwest = []
        self.southeast = []
        self.southwest = []
        
        self.children = [self.northeast, self.northwest, 
                         self.southeast, self.southwest]
    
    def memoize(self):
        # Saves the current image in memory so if it is called
        # in the future it doesn't have to recalculate
        pass
    
    def subdivide(self, thresh):
        pass
    
