import torch.nn as nn
import torch
from torchrl.collectors import SyncDataCollector
import random
from play import play

INPUT_SIZE = 4
OUTPUT_SIZE = 2
HIDDEN_SIZE = 256


def loadModel(path="./models/model.pt"):
    model = PlayModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
    model.load_state_dict(torch.load(path))
    return model
    

class ModelController(object):
    def __init__(self) -> None:
        self.model = PlayModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
        
    def genetic_algorithm(self, num_generations=10, population_size=8, num_best_players=5, num_parents=5):
        population = self.initialize_population(population_size)
        bset_score = 0
        best_generation = 0
        best_id = 0
        for generation in range(num_generations):
            score_list = []
            for chromosome in population:
                score = self.evaluate_fitness(chromosome)
                score_list.append(score)
            # sort the index by their scores descendingly
            score_list, indexs = zip(*sorted(zip(score_list, range(len(score_list))), reverse=True))
            population = [population[i] for i in indexs][:num_best_players]
            
            # save models and score
            for i in range(num_best_players):
                torch.save(population[i].state_dict(), "./models/model_" + str(generation) + "_" + str(i) + "_" + ".pt")
                with open("./log/scores.txt", "a") as f:
                    f.write("./models/model_" + str(generation) + "_" + str(i) + ":" + str(score_list[i]) + "\n")
                if score_list[i] > bset_score:
                    bset_score = score_list[i]
                    best_generation = generation
                    best_id = i
            
            # make new chromosomes
            for _ in range(population_size):
                new_population = []
                # create new chromosomes
                for _ in range(population_size):
                    # randomly select parents
                    parents = self.select_parents(population, num_parents)
                    # average the weights of the parents and add some noise
                    child = self.create_chromosome()
                    # set the weights of the child to small random noise
                    for param in child.parameters():
                        param.data = param.data + torch.randn_like(param) * 0.0005
                    # average the weights of the parents
                    for parent in parents:
                        for child_param, parent_param in zip(child.parameters(), parent.parameters()):
                            child_param.data = child_param.data + parent_param.data / num_parents
                    
                    new_population.append(child)
            population = new_population
            
        return bset_score, best_generation, best_id


    def evaluate_fitness(self, chromosome):
        # play the game with the chromosome
        score = play(chromosome)
        print("final score: ", score)
        return score
    
    def initialize_population(self, population_size):
        return [self.create_chromosome() for _ in range(population_size)]
    
    def create_chromosome(self):
        # create a chromosome with random weights
        model = PlayModel(INPUT_SIZE, HIDDEN_SIZE, OUTPUT_SIZE)
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

class PlayModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(PlayModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        out = torch.relu(self.fc1(x))
        out = torch.relu(self.fc2(out))
        out = torch.relu(self.fc3(out))
        out = self.fc4(out)
        # output as two probabilities for the two actions
        out = torch.softmax(out, dim=-1)
        return out
    
    def predict(self, x, y, vx, vy):
        # predict the action
        out = self.forward(torch.tensor([x, y, vx, vy], dtype=torch.float32))
        return out[0].item(), out[1].item()