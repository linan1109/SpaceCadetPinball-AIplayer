from mss import mss
import cv2 as cv
import numpy as np
import time
import pyautogui
import pywinauto
from pywinauto import mouse
from Pattern import Number, Words

class Frames(object):
    def __init__(self):
        self.sct = mss()
        self.lastFrame = None
        self.x = 0
        self.y = 0
        self.v_x = 0
        self.v_y = 0
        self.last_x = 0
        self.last_y = 0
        self.score = 0
        
        self.frame = None
        self.monitor = None
        self.WindowSize = (640, 480)
        
        self.Number_fun = Number()
        self.Words_fun = Words()
        
    def oneFrame(self):
        img, monitor = self.getFrame()
        img = self.getBallContous()
        self.score = self.recScore()
        # if score > self.score: 
        #     self.score = score
        stage = self.stageChange()
        return img, self.x/monitor['width'], self.y/monitor['height'], self.v_x/monitor['width'], self.v_y/monitor['height'], self.score, stage
        
    def getWindowCoord(self):
        # find the game window
        hwnd = pywinauto.findwindows.find_windows(title='3D Pinball for Windows - Space Cadet')[0]
        # get the game window
        self.gameWindow = pywinauto.application.Application().connect(handle=hwnd)
        # get the game window coordinates
        coordinates = self.gameWindow.window(handle=hwnd).rectangle()
        # change it into a dictionary
        coordinates = {'top': coordinates.top, 'left': coordinates.left, 'width': coordinates.width(), 'height': coordinates.height()}
        # print(coordinates)
        self.monitor = coordinates
        return coordinates
    
    def getFrame(self):
        monitor = self.getWindowCoord()
        # grab the screen
        sct_img = self.sct.grab(monitor)
        # convert to numpy array
        img = np.array(sct_img)
        img = cv.cvtColor(img, cv.COLOR_BGRA2BGR)
        
        # get the inside of the game window
        grey = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        contours, hierarchy = cv.findContours(grey, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE)
        # select contours area > 50
        contours = [c for c in contours if cv.contourArea(c) > 50]
        # if width > height
        if len(contours) > 4:
            # sort the contours by y
            contours = sorted(contours, key=lambda x: cv.boundingRect(x)[1])

            if monitor['width'] / monitor['height'] > 640/480:
                x, y, w, h = cv.boundingRect(contours[1])
                top = y
                bottom = y + h
                
                x, y, w, h = cv.boundingRect(contours[3])
                right = x + w
                
                x, y, w, h = cv.boundingRect(contours[2])
                left = x + w - int((bottom - top) * 0.17)
                
                if right - left > 0 and bottom - top > 0:
                    img = img[top:bottom, left:right]
            else:
                x, y, w, h = cv.boundingRect(contours[3])
                right = x + w
                
                x, y, w, h = cv.boundingRect(contours[2])
                left = x
                top = y
                bottom = y + h
                
                if right - left > 0 and bottom - top > 0:
                    img = img[top:bottom, left:right]
            
        # resize the image
        img = cv.resize(img, self.WindowSize)
        self.frame = img
        return img, monitor
    
    def getBallContous(self):
        img = self.frame
        # add mask to find the ball
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        lower = (0, 0, 80)
        upper = (5, 5, 100)
        mask = cv.inRange(hsv, lower, upper)
        result = cv.bitwise_and(img, img, mask=mask)
        # convert to grayscale
        result = cv.cvtColor(result, cv.COLOR_BGR2GRAY)

        # find the changes
        if self.lastFrame is None:
            self.lastFrame = result
            self.x = 0
            self.y = 0
            self.v_x = 0
            self.v_y = 0
            self.last_x = 0
            self.last_y = 0
        else:
            changes = cv.absdiff(self.lastFrame, result)
            self.lastFrame = result
            # find the contours
            contours, hierarchy = cv.findContours(changes, cv.RETR_TREE, cv.CHAIN_APPROX_SIMPLE) 
            # draw the contours
            cv.drawContours(img, contours, -1, (0, 255, 0), 3)
            
            if len(contours) == 0:
                self.x = 0
                self.y = 0
                self.v_x = 0
                self.v_y = 0
            else:
                self.y = cv.boundingRect(contours[0])[1]
                for c in contours:
                    x, y, w, h = cv.boundingRect(c)
                    if y <= self.y:
                        self.x = x + (w // 2)
                        self.y = y - (h // 2)
                
                self.v_x = self.x - self.last_x
                self.v_y = self.y - self.last_y
                self.last_x = self.x
                self.last_y = self.y
                # draw the rectangle
                cv.rectangle(img, (self.x - 10, self.y), (self.x + 10, self.y + 20), (0, 0, 255), 2)
                # draw the velocity vector
                cv.arrowedLine(img, (self.x, self.y), (self.x + self.v_x, self.y + self.v_y), (255, 0, 0), 2)
        return img


    def recScore(self):
        # find all the numbers
        img = self.frame
        (x, y, w, h) = (482, 229, 136, 23)
        # cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), 2)
        length_of_number = 15
        space = [4,4,6,8,10,10,12,14]
        score = 0
        for i in range(8):
            number_img = img[y:y + h, x + (length_of_number) * i  + space[i] :x + (length_of_number ) * i + length_of_number + space[i]]
            digit = self.Number_fun.compare(number_img)
            cv.rectangle(img, (x + (length_of_number) * i  + space[i], y), (x + (length_of_number ) * i + length_of_number + space[i], y + h), (0, 255, 255), 1)
            score = score * 10 + digit

        # draw the score    
        cv.putText(img, str(score), (x, y - 5), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        return score
        
    def stageChange(self):
        img = self.frame
        stage = self.Words_fun.compare(img)
        if stage > 0:
            print("stage: ", stage)
        return stage

    def findCancel(self):
        locs = self.Words_fun.findCancel(self.frame)
        
        # change locs in img to locs in monitor 
        left = self.monitor['left']
        top = self.monitor['top']
        width = self.monitor['width']
        height = self.monitor['height']
        
        width_ratio = width / self.WindowSize[0]
        height_ratio = height / self.WindowSize[1]
        
        # for loc in locs:
        #     x = left + int(loc[0] * width_ratio)
        #     y = top + int(loc[1] * height_ratio)
        #     mouse.click(button='left', coords=(x, y))
        #     print("click",loc[0], loc[1], x, y)
        #     time.sleep(0.1)
            
        
        for i in range(left, left + 50, 10):
            for j in range(top + height - 50, top + height, 10):
                print(i, j)
                mouse.click(button='left', coords=(i, j))
                # time.sleep(0.01)
        
    