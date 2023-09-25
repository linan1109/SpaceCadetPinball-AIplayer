from typing import Any
import cv2 as cv
import numpy as np
from skimage import metrics

class Number(object):
    def __init__(self):
        self.imgs = [
            cv.imread('./numbers/0.png', cv.IMREAD_GRAYSCALE),
            cv.imread('./numbers/1.png', cv.IMREAD_GRAYSCALE),
            cv.imread('./numbers/2.png', cv.IMREAD_GRAYSCALE),
            cv.imread('./numbers/3.png', cv.IMREAD_GRAYSCALE),
            cv.imread('./numbers/4.png', cv.IMREAD_GRAYSCALE),
            cv.imread('./numbers/5.png', cv.IMREAD_GRAYSCALE),
            cv.imread('./numbers/6.png', cv.IMREAD_GRAYSCALE),
            cv.imread('./numbers/7.png', cv.IMREAD_GRAYSCALE),
            cv.imread('./numbers/8.png', cv.IMREAD_GRAYSCALE),
            cv.imread('./numbers/9.png', cv.IMREAD_GRAYSCALE)
        ]
        

    def compare(self, number_img):
        number_img = cv.cvtColor(number_img, cv.COLOR_BGR2GRAY)
        similarity = []
        for i in range(len(self.imgs)):
            img = self.imgs[i]
            sim = metrics.structural_similarity(img, number_img, full=True)[0]
            if sim > 0.1:
                similarity.append(sim)
            else:
                similarity.append(0)
        return similarity.index(max(similarity))
            
            