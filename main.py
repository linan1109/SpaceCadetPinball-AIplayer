from Frame import Frames
import cv2 as cv
import time

class Main:
    def __init__(self):
        self.frame = Frames()
        self.FPS = 30

    def running(self):
        fps = self.FPS
        while True:
            img, _, _, _, _, _ = self.frame.oneFrame()
            cv.imshow('test', img)
            if cv.waitKey(1) & 0xFF == ord('q'):
                cv.destroyAllWindows()
                break
            time.sleep(1/fps)
    
if __name__ == '__main__':
    main = Main()
    main.running()