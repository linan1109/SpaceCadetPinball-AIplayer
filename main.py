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
        score, gen, id = ModelController().genetic_algorithm(
            num_generations=20, 
            population_size=8, 
            num_best_players=5, 
            num_parents=3
        )
        print(f"Best score: {score}, generation: {gen}, id: {id}")
        # log the result
        with open("./log/log.txt", "a") as f:
            f.write(f"Best score: {score}, generation: {gen}, id: {id}\n")
    
if __name__ == '__main__':
    main = Main()
    model = loadModel('./models/model_16_0_.pt')
    main.runningWithModel(model)
    
    
    # main.runningWithModel(None)
    # main.runGeneAlo()