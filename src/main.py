from classes.QuadTree import ImageCompression
import numpy as np
import cv2 
from classes.GUI import GUI
from classes.ThresholdStrategy import AverageStrategy

if __name__ == "__main__":
    #gui = GUI()
    #gui.start()
    image = cv2.imread("resources\\exterior.jpg")
    if image is None:
        raise FileNotFoundError()
    t_strat = AverageStrategy()
    image_tree = ImageCompression(image, t_strat)
    for i in range(1, 255, 5):
        image_tree.update(i)
    image_tree.save()
    image_tree.animate(show_tree=True)
