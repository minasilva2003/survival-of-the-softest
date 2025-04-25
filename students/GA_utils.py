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

#function to evaluate fitness based on robot simulation
def evaluate_fitness(scenario, controller, robot_structure, view=False):
        """Evaluate the fitness of a robot structure."""
        if not is_connected(robot_structure):
            return 0.0
        try:
            connectivity = get_full_connectivity(robot_structure)
            env = gym.make(scenario, body=robot_structure, connections=connectivity)
            env.reset()
            sim = env.sim
            viewer = EvoViewer(sim)
            viewer.track_objects('robot')
            t_steps = 0
            t_reward = 0
            action_size = sim.get_dim_action_space('robot')  # Get correct action size

            while True:
                actuation = controller(action_size, t_steps)
                if view:
                    viewer.render('screen')
                ob, reward, terminated, truncated, info = env.step(actuation)
                t_reward += reward
                t_steps += 1

                if terminated or truncated:
                    env.reset()
                    break

            viewer.close()
            env.close()
            return t_reward
        except (ValueError, IndexError):
            return 0.0


#function to create a random robot structure
def create_random_robot(grid_size):
    random_robot, _ = sample_robot(grid_size)
    return random_robot

#function to mutate one random voxel in the robot structure
def standard_mutate(robot, voxel_types):
    offspring = copy.deepcopy(robot)
    x, y = random.randint(0, offspring.shape[0] - 1), random.randint(0, offspring.shape[1] - 1)
    offspring[x, y] = random.choice(voxel_types)
    return offspring

#function to one point crossover between two parent robots
def one_point_crossover(parent1, parent2):
    parent1_flat = parent1.flatten()
    parent2_flat = parent2.flatten()
    crossover_point = random.randint(1, len(parent1_flat) - 1)
    child1_flat = np.concatenate((parent1_flat[:crossover_point], parent2_flat[crossover_point:]))
    child2_flat = np.concatenate((parent2_flat[:crossover_point], parent1_flat[crossover_point:]))
    child1 = child1_flat.reshape(parent1.shape)
    child2 = child2_flat.reshape(parent2.shape)
    return child1, child2

#function to perform tournament selection of parents
def standard_tournament_selection(population, fitnesses, tournament_size):
    tournament = random.sample(list(zip(population, fitnesses)), tournament_size)
    return max(tournament, key=lambda x: x[1])[0]


def simulate_and_save(best_robot, filename, scenario, steps, controller):
    """Simulate the best robot and save the result as a GIF."""
    print("Best robot structure found:")
    print(best_robot)
    print("Best fitness score:")
    utils.simulate_best_robot(best_robot, scenario=scenario, steps=steps)
    utils.create_gif(best_robot, filename=filename, scenario=scenario, steps=steps, controller=controller)

