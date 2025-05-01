import numpy as np
import random
import copy
import gymnasium as gym
from evogym.envs import *
from evogym import EvoWorld, EvoSim, EvoViewer, sample_robot, get_full_connectivity, is_connected
import utils
from fixed_controllers import *
import csv
import matplotlib.pyplot as plt
import os
from data_utils import write_to_csv_file
from GA_utils import evaluate_fitness, create_random_robot, standard_mutate, one_point_crossover, standard_tournament_selection, simulate_and_save
import multiprocessing
import csv

robot_structure = np.array([
            [1, 3, 1, 0, 0],
            [4, 1, 3, 2, 2],
            [3, 4, 4, 4, 4],
            [2, 0, 0, 3, 2],
            [2, 0, 0, 0, 2]
        ])
       
controllers = [alternating_gait, hopping_motion, sinusoidal_wave]
controller_names = ["walking", "hopping", "slithering"]

scenarios = ["Walker-v0", "BridgeWalker-v0", "DownStepper-v0", "ObstacleTraverser-v0"]

data = []

for i in range (0, len(controllers)):
    for scenario in scenarios:
        new_line = [controller_names[i], scenario]
        fitness = evaluate_fitness(scenario=scenario, controller=controllers[i], robot_structure=robot_structure)
        new_line.append(fitness)
        data.append(new_line)
        simulate_and_save(robot_structure, f"gif_{controller_names[i]}_{scenario}.gif", scenario,500, controllers[i])
    
with open("threshold_baselines.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerows(data)