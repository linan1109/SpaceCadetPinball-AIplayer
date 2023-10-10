from typing import Any
import cv2 as cv
import numpy as np
from skimage import metrics

class Number(object):
    def __init__(self):
        self.imgs = [
            cv.imread('./patterns/0.png', cv.IMREAD_GRAYSCALE),
            cv.imread('./patterns/1.png', cv.IMREAD_GRAYSCALE),
            cv.imread('./patterns/2.png', cv.IMREAD_GRAYSCALE),
            cv.imread('./patterns/3.png', cv.IMREAD_GRAYSCALE),
            cv.imread('./patterns/4.png', cv.IMREAD_GRAYSCALE),
            cv.imread('./patterns/5.png', cv.IMREAD_GRAYSCALE),
            cv.imread('./patterns/6.png', cv.IMREAD_GRAYSCALE),
            cv.imread('./patterns/7.png', cv.IMREAD_GRAYSCALE),
            cv.imread('./patterns/8.png', cv.IMREAD_GRAYSCALE),
            cv.imread('./patterns/9.png', cv.IMREAD_GRAYSCALE)
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
            

class Words(object):
    def __init__(self) -> None:
        self.pattern_start_new_game = cv.imread('./patterns/F2StartsNewGame.png', cv.IMREAD_GRAYSCALE)
        self.location_start_new_game = (430, 270, 195, 55)
        
        self.pattern_ball_at_start = cv.imread('./patterns/BallAtStart.png', cv.IMREAD_GRAYSCALE)
        self.location_ball_at_start = (341, 423, 21, 21)
        
        self.pattern_game_over = cv.imread('./patterns/GameOver.png', cv.IMREAD_GRAYSCALE)
        self.location_game_over = (430, 270, 160, 35)
        
        self.pattern_cancel = cv.imread('./patterns/Cancel.png', cv.IMREAD_GRAYSCALE)

    def compare(self, img):
        grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        part = grey[self.location_start_new_game[1]:self.location_start_new_game[1] + self.location_start_new_game[3],
                    self.location_start_new_game[0]:self.location_start_new_game[0] + self.location_start_new_game[2]]
        sim = metrics.structural_similarity(self.pattern_start_new_game, part, full=True)[0]
        # print("pattern_start_new_game: ", sim)
        if sim > 0.5:
            return 1
        
        part = grey[self.location_ball_at_start[1]:self.location_ball_at_start[1] + self.location_ball_at_start[3],
                    self.location_ball_at_start[0]:self.location_ball_at_start[0] + self.location_ball_at_start[2]]
        sim = metrics.structural_similarity(self.pattern_ball_at_start, part, full=True)[0]
        # print("pattern_ball_at_start: ", sim)
        if sim > 0.5:
            return 2
        
        
        part = grey[self.location_game_over[1]:self.location_game_over[1] + self.location_game_over[3],
                    self.location_game_over[0]:self.location_game_over[0] + self.location_game_over[2]]
        sim = metrics.structural_similarity(self.pattern_game_over, part, full=True)[0]
        # print("pattern_game_over: ", sim)
        if sim > 0.5:
            return 3
        
        return 0
    
    def findCancel(self, img):
        grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        # find cancel button location
        res = cv.matchTemplate(grey, self.pattern_cancel, cv.TM_CCOEFF_NORMED)
        threshold = 0.01
        loc = np.where(res >= threshold)
        # return all x, y
        return loc[::-1]
        
