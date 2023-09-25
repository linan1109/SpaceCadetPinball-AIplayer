from mss import mss
import cv2 as cv
import numpy as np
import time
import pyautogui
import pywinauto

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
        
        self.frame = None
        self.FPS = 30
    
    def running(self):
        fps = self.FPS
        while True:
            self.getFrame()
            img = self.getBallContous()
            cv.imshow('test', img)
            if cv.waitKey(1) & 0xFF == ord('q'):
                cv.destroyAllWindows()
                break
            time.sleep(1/fps)
    
    def getWindowCoord(self):
        # find the game window
        hwnd = pywinauto.findwindows.find_windows(title='3D Pinball for Windows - Space Cadet')[0]
        # get the game window
        gameWindow = pywinauto.application.Application().connect(handle=hwnd)
        # get the game window coordinates
        coordinates = gameWindow.window(handle=hwnd).rectangle()
        # change it into a dictionary
        coordinates = {'top': coordinates.top, 'left': coordinates.left, 'width': coordinates.width(), 'height': coordinates.height()}
        # print(coordinates)
        return coordinates
    
    def getFrame(self):
        monitor = self.getWindowCoord()
        # grab the screen
        sct_img = self.sct.grab(monitor)
        # convert to numpy array
        img = np.array(sct_img)
        # convert from BGR to RGB
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        self.frame = img
        return img
    
    def getBallContous(self):
        img = self.frame
        # add mask to find the ball
        hsv = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        lower = (0, 0, 80)
        upper = (5, 5, 100)
        mask = cv.inRange(hsv, lower, upper)
        result = cv.bitwise_and(img, img, mask=mask)
        # convert to grayscale
        result = cv.cvtColor(result, cv.COLOR_RGB2GRAY)

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



if __name__ == "__main__":
    frames = Frames()
    frames.running()
        
    