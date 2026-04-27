from classes.QuadTree import ImageCompression
import numpy as np
import cv2 
from classes.ThresholdStrategy import AverageStrategy
import time

if __name__ == "__main__":
    image = cv2.imread("resources\\scream.jpg")
    if image is None:
        raise FileNotFoundError()
    t_strat = AverageStrategy()
    image_tree = ImageCompression(image, t_strat)
    
    start = time.time()
    image_tree.compute_all_thresholds(255)
    end = time.time()
    print(f'Total: {end - start:0.2f} s')
    
    image_tree.save()
    image_tree.animate(show_tree=True)
    print(image_tree.psnr(0))
