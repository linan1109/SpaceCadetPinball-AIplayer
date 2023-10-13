from pynput.keyboard import Key, Controller
import time
import cv2 as cv
from Frame import Frames

FPS = 30

def playWithOutModel():
    frame = Frames()
    try:
        fps = FPS
        score = 0
        while True:
            img, x, y, vx, vy, score, stage = frame.oneFrame()
            cv.imshow('playing', img)
            if cv.waitKey(1) & 0xFF == ord('q'):
                cv.destroyAllWindows()
                break
            time.sleep(1/fps)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        cv.destroyAllWindows()
        pass
    finally:
        cv.destroyAllWindows()
        pass
    return score
    

def play(model=None):
    if model is None:
        return playWithOutModel()
    keyboard = Controller()
    frame = Frames()
    fps = FPS
    bestscore = 0
    try:
        while True:
            img, x, y, vx, vy, score, stage = frame.oneFrame()
            if score > bestscore:
                bestscore = score
            # print("score", bestscore)
            cv.imshow('playing', img)
            if stage == 1:
                # game to restart
                restart()
                bestscore = 0
            elif stage == 2:
                # ball at start, press space to start
                start()
            elif stage == 3:
                # game over 
                time.sleep(5)
                img, x, y, vx, vy, score, stage = frame.oneFrame()
                frame.findCancel()
                return bestscore
            else:
                # real play
                left, right = model.predict(x, y, vx, vy)
                print(left, right)
                if left > 0.5 and right > 0.5:
                    # press letter Z and /
                    keyboard.press('z')
                    keyboard.press('/')
                    time.sleep(0.1)
                    keyboard.release('z')
                    keyboard.release('/')
                elif left > 0.5:
                    # press letter Z
                    keyboard.press('z')
                    time.sleep(0.1)
                    keyboard.release('z')
                elif right > 0.5:
                    # press letter /
                    keyboard.press('/')
                    time.sleep(0.1)
                    keyboard.release('/')
            if cv.waitKey(1) & 0xFF == ord('q'):
                cv.destroyAllWindows()
                break
            time.sleep(1/fps)
    except KeyboardInterrupt:
        print("KeyboardInterrupt")
        cv.destroyAllWindows()
        pass
    finally:
        cv.destroyAllWindows()
        pass
    return bestscore
    
def start():
    keyboard = Controller()
    keyboard.press(Key.space)
    time.sleep(3)
    keyboard.release(Key.space)

def restart():
    keyboard = Controller()
    # press F2
    keyboard.press(Key.f2)
    time.sleep(0.01)
    keyboard.release(Key.f2)
    time.sleep(1)