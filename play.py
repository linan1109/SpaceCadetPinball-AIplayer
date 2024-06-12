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
    

def play(left_model=None, right_model=None):
    if left_model is None or right_model is None:
        return playWithOutModel()
    keyboard = Controller()
    frame = Frames()
    fps = FPS
    bestscore = 0
    num_l_move = 0
    num_r_move = 0
    state_prime = [0, 0, 0, 0]
    left = 0
    right = 0
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
                # train model based on last action and this state
                state = state_prime
                state_prime = [x, y, vx, vy]
                reward = get_reward(score, num_l_move, num_r_move)
                left_model.memory.put((state, left, reward, state_prime))
                right_model.memory.put((state, right, reward, state_prime))

                # get action from model
                left = left_model.get_action(state_prime, False)
                right = right_model.get_action(state_prime, False)
                print(left, right)
                if left > 0.5 and right > 0.5:
                    # press letter Z and /
                    keyboard.press('z')
                    keyboard.press('/')
                    time.sleep(0.1)
                    keyboard.release('z')
                    keyboard.release('/')
                    num_l_move += 1
                    num_r_move += 1
                elif left > 0.5:
                    # press letter Z
                    keyboard.press('z')
                    time.sleep(0.1)
                    keyboard.release('z')
                    num_l_move += 1
                elif right > 0.5:
                    # press letter /
                    keyboard.press('/')
                    time.sleep(0.1)
                    keyboard.release('/')
                    num_r_move += 1
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

def get_reward(score, num_l_move, num_r_move):
    return score - num_l_move - num_r_move
