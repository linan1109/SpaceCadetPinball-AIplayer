import torch.nn as nn
import torch
from torchrl.collectors import SyncDataCollector
import random
from play import play
from SAC import Agent

def loadModel(path="./models/model.pt"):
    model = Agent()
    model.load_state_dict(torch.load(path))
    return model
    

class ModelController(object):
    def __init__(self) -> None:
        pass
        
    def genetic_algorithm(self, num_generations=50, population_size=8, num_best_players=5, num_parents=3):
        population = self.initialize_population(population_size)
        bset_score = 0
        best_generation = 0
        best_id = 0
        for generation in range(num_generations):
            score_list = []
            for (chromosome_left, chromosome_right) in population:
                score = self.evaluate_fitness(chromosome_left, chromosome_right)
                score_list.append(score)
            # sort the index by their scores descendingly
            score_list, indexs = zip(*sorted(zip(score_list, range(len(score_list))), reverse=True))
            population = [population[i] for i in indexs][:num_best_players]
            
            # save models and score
            for i in range(num_best_players):
                torch.save(population[i][0].state_dict(), "./models/model_left_" + str(generation) + "_" + str(i) + "_" + ".pt")
                torch.save(population[i][1].state_dict(), "./models/model_right_" + str(generation) + "_" + str(i) + "_" + ".pt")
                with open("./log/scores.txt", "a") as f:
                    f.write("./models/model_" + str(generation) + "_" + str(i) + ":" + str(score_list[i]) + "\n")
                if score_list[i] > bset_score:
                    bset_score = score_list[i]
                    best_generation = generation
                    best_id = i
            
            # make new chromosomes
            for _ in range(population_size):
                new_population = []
                # for both left and right
                # create new chromosomes
                for _ in range(population_size):
                    # randomly select parents
                    parents = self.select_parents(population, num_parents)
                    child_left = self.create_chromosome()
                    child_right = self.create_chromosome()
                    for param in child_left.parameters():
                        param.requires_grad = False
                        param.data = torch.zeros_like(param)
                    for param in child_right.parameters():
                        param.requires_grad = False
                        param.data = torch.zeros_like(param)

                    random_list = [random.random() for _ in range(num_parents)]
                    # softmax random_list to make it a probability distribution and sum to 1
                    random_list = torch.softmax(torch.tensor(random_list), dim=-1)
                    # and the bigger the score, the bigger the probability to be selected///////////
                    random_list, _ = torch.sort(random_list, descending=True)
                    # for parent in parents:
                        # for child_param, parent_param in zip(child.parameters(), parent.parameters()):
                            # child_param.data = child_param.data + parent_param.data * random_list.pop()
                    for i in range(num_parents):
                        for child_param, parent_param in zip(child_left.parameters(), parents[i].parameters()):
                            child_param.data = child_param.data + parent_param.data * random_list[i]
                        for child_param, parent_param in zip(child_right.parameters(), parents[i].parameters()):
                            child_param.data = child_param.data + parent_param.data * random_list[i]
                    # add some noise
                    for param in child_left.parameters():
                        param.requires_grad = False
                        param.data = param.data * (1 + torch.randn_like(param) * 0.003)
                    for param in child_right.parameters():
                        param.requires_grad = False
                        param.data = param.data * (1 + torch.randn_like(param) * 0.003)
                    new_population.append((child_left, child_right))
            population = new_population 
        return bset_score, best_generation, best_id


    def evaluate_fitness(self, chromosome_left, chromosome_right):
        # play the game 
        score = play(chromosome_left, chromosome_right)
        print("final score: ", score)
        return score
    
    def initialize_population(self, population_size):
        return [(self.create_chromosome(), self.create_chromosome()) for _ in range(population_size)]
    
    def create_chromosome(self):
        # create a chromosome with random weights
        model = Agent()
        for param in model.parameters():
            param.requires_grad = False
            param.data = torch.randn_like(param) * 0.1
        return model
    
    def select_parents(self, population, num_parents):
        parents = []
        for _ in range(num_parents):
            parent = random.choice(population)
            parents.append(parent)
        return parents
