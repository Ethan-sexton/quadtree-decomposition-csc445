from classes.QuadTree import QuadTree
import numpy as np
import cv2 
from classes.GUI import GUI


if __name__ == "__main__":
    #gui = GUI()
    #gui.start()
    image = cv2.imread("resources\\image.jpg")
    tree = QuadTree(image)
    tree.update(15)
    tree.display()
    tree.save()