
import cv2 as cv
import time
from play import play
from model import *

class Main:
    def __init__(self):
        pass
    
    def runningWithModel(self, model=None):
        play(model)
    
    def runGeneAlo(self):
        score, gen, id = ModelController().genetic_algorithm()
        print(f"Best score: {score}, generation: {gen}, id: {id}")
        
    
if __name__ == '__main__':
    main = Main()
    # model = loadModel()
    # main.runningWithModel(None)
    main.runGeneAlo()