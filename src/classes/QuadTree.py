import numpy as np
import cv2
import imageio

class QuadTree:
    def __init__(self, data, strat, xmin=0, xmax=-1, ymin=0, ymax=-1):
        self.data = data
        self.strategy = strat

        h, w = data.shape[:2]

        self.xmin = xmin
        self.ymin = ymin
        self.xmax = h if xmax == -1 else xmax
        self.ymax = w if ymax == -1 else ymax

        self.children = []
        self.threshold = 0

        self.leaf_value = None
        self._built = False

    def region(self):
        return self.data[self.xmin:self.xmax, self.ymin:self.ymax]

    def build(self, threshold):
        self.threshold = threshold
        self.children.clear()

        region = self.region()

        if region.shape[0] < 4 or region.shape[1] < 4:
            self.leaf_value = self.strategy.region_value(region)
            return

        if self.strategy.need_subdivide(region, threshold):
            xmid = (self.xmin + self.xmax) // 2
            ymid = (self.ymin + self.ymax) // 2

            self.children = [
                QuadTree(self.data, self.strategy, self.xmin, xmid, self.ymin, ymid),
                QuadTree(self.data, self.strategy, self.xmin, xmid, ymid, self.ymax),
                QuadTree(self.data, self.strategy, xmid, self.xmax, self.ymin, ymid),
                QuadTree(self.data, self.strategy, xmid, self.xmax, ymid, self.ymax),
            ]

            for c in self.children:
                c.build(threshold)
        else:
            self.leaf_value = self.strategy.region_value(region)

    def render(self, out):
        if not self.children:
            out[self.xmin:self.xmax, self.ymin:self.ymax] = self.leaf_value
            return

        for c in self.children:
            c.render(out)

class ImageCompression(QuadTree):
    def __init__(self, data, strat):
        super().__init__(data, strat)

        self.memo = {}
        self.sizes = {}
        self.rectangles = {}

        self.original = data.copy()

    def update(self, threshold):
        if threshold in self.memo:
            return

        self.build(threshold)

        img = np.zeros_like(self.data)
        self.render(img)

        self.memo[threshold] = img

        ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 50])
        self.sizes[threshold] = len(enc) / 1024 if ok else 0

    def display(self, threshold):
        self.update(threshold)
        return self.memo[threshold]

    def get_file_size(self, threshold):
        self.update(threshold)
        return self.sizes[threshold]

    def get_ratio(self, threshold):
        return self.get_file_size(0) / max(self.get_file_size(threshold), 1e-6)

    def psnr(self, threshold):
        self.update(threshold)

        original = self.original.astype(np.float32)
        processed = self.memo[threshold].astype(np.float32)

        mse = np.mean((original - processed) ** 2)
        if mse == 0:
            return 100.0

        return 20 * np.log10(255.0 / np.sqrt(mse))

    def compute_all_thresholds(self, max_thresh):
        for t in range(max_thresh + 1):
            self.update(t)
            
    
    def get_leaf_rectangles(self, thresh):   
        # If children exist, traverse them
        if self.children is not None:
            rects = []
            for child in self.children:
                rects.extend(child.get_leaf_rectangles(thresh))
            return rects
        else:
            # Leaf node - return this rectangle
            return [(self.xmin, self.xmax, self.ymin, self.ymax)]
            
    def animate(self, path, show_tree=False):
        writer = imageio.get_writer(path)
        
        counter = 1
        total = len(self.memo)
        
        for thresh in sorted(self.memo.keys()):
            print(f'Animating frame {counter}/{total}')
            image = self.display(thresh)
            
            if show_tree:
                rects = self.rectangles.get(thresh, [])
                for xmin, xmax, ymin, ymax in rects:
                    cv2.rectangle(image, (ymin, xmin), (ymax, xmax), (0, 255, 0), 2)
            
            writer.append_data(image)
            counter += 1
        
        writer.close()
        print(f'Saved to {path}')
        