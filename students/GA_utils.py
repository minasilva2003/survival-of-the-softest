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

#function to mutate random voxels up to a maximum number, with decaying probability
def decaying_mutate(robot, voxel_types, base_mutation_rate, max_mutations=3):
   
    offspring = copy.deepcopy(robot)
    num_mutations = 0

    for attempt in range(max_mutations):
        # Decay mutation probability exponentially with each attempt
        prob = base_mutation_rate * np.exp(-attempt)

        if random.random() < prob:
            # Pick random position
            x = random.randint(0, offspring.shape[0] - 1)
            y = random.randint(0, offspring.shape[1] - 1)
            
            # Mutate voxel
            offspring[x, y] = random.choice(voxel_types)
            num_mutations += 1

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


#function to choose probabilistic_tournament_selection
def probabilistic_tournament_selection(population, fitnesses, tournament_size, temperature):

    indices = random.sample(range(len(population)), tournament_size)
    competitors = [population[i] for i in indices]
    competitor_fitnesses = [fitnesses[i] for i in indices]
    baseline = 0.3

    # Step 2: Rank competitors (highest fitness first)
    sorted_group = sorted(zip(competitors, competitor_fitnesses), key=lambda x: x[1], reverse=True)
    
    sorted_competitors = [competitor[0] for competitor in sorted_group]

    for competitor in sorted_competitors[:-1]:

        #if temperature is high, probability is LOW
        #when temperature is low, probability is HIGH
        prob = baseline + (1-baseline) * (1-temperature)

        if random.random() < prob:
            return competitor
    
    return sorted_competitors[-1]
    


    # Step 3: Assign probabilities based on rank
    # Example: Probability of selection = exp(rank) / sum(exp(rank))
    ranks = np.arange(len(ranked))  # 0 = best, 1 = second best, etc.
    probs = np.exp(-ranks)  # better ranks get higher probability
    probs /= np.sum(probs)  # normalize

    # Step 4: Select one based on computed probabilities
    chosen_index = np.random.choice(len(ranked), p=probs)
    selected_individual = ranked[chosen_index][0]

    return selected_individual


def simulate_and_save(best_robot, filename, scenario, steps, controller):
    """Simulate the best robot and save the result as a GIF."""
    print("Best robot structure found:")
    print(best_robot)
    print("Best fitness score:")
    utils.simulate_best_robot(best_robot, scenario=scenario, steps=steps)
    utils.create_gif(best_robot, filename=filename, scenario=scenario, steps=steps, controller=controller)



#function to remove worst individuals and add new individuals
def immigrant_function(population, fitnesses, immigrant_pool_size, scenario, controller, grid_size):
  
    # Rank the population by fitness (lowest to highest)
    ranked_population = sorted(zip(population, fitnesses), key=lambda x: x[1])

    # Remove the lowest-ranking individuals
    survivors = [individual[0] for individual in ranked_population[immigrant_pool_size:]]
    survivor_fitnesses = [individual[1] for individual in ranked_population[immigrant_pool_size:]]

    # Create new robots to replace the removed individuals
    immigrants = [create_random_robot(grid_size) for _ in range(immigrant_pool_size)]
    immigrant_fitnesses = [evaluate_fitness(scenario, controller, robot) for robot in immigrants]

    # Combine the survivors and the new immigrants
    updated_population = survivors + immigrants
    updated_fitnesses = survivor_fitnesses + immigrant_fitnesses

    return updated_population, updated_fitnesses


#function to do uniform_crossover
def uniform_crossover(parent1, parent2):
 
    # Ensure both parents have the same shape
    assert parent1.shape == parent2.shape, "Parents must have the same shape for uniform crossover."

    # Create empty children with the same shape as the parents
    child1 = np.empty_like(parent1)
    child2 = np.empty_like(parent2)

    # Perform uniform crossover
    for i in range(parent1.shape[0]):
        for j in range(parent1.shape[1]):
            if random.random() < 0.5:  # 50% chance to inherit from parent1
                child1[i, j] = parent1[i, j]
                child2[i, j] = parent2[i, j]
            else:  # 50% chance to inherit from parent2
                child1[i, j] = parent2[i, j]
                child2[i, j] = parent1[i, j]

    return child1, child2