import numpy as np
import random
import gymnasium as gym
from evogym.envs import *
from evogym import EvoViewer, get_full_connectivity
from neural_controller import *
import os
import csv
import matplotlib.pyplot as plt
import torch
import multiprocessing
from data_utils import save_best_controller


class DE_Controller:
    def __init__(self, population_size=50, num_generations=10, steps=500, mutation_factor=0.8, crossover_rate=0.7, scenario='DownStepper-v0', seed=42, directory="results/de/"):
        self.num_generations = num_generations
        self.population_size = population_size
        self.steps = steps
        self.scenario = scenario
        self.directory = directory
        self.mutation_factor = mutation_factor
        self.crossover_rate = crossover_rate
        self.seed = seed
        self.brain = None

        np.random.seed(self.seed)
        random.seed(self.seed)

        # Robot structure
        self.robot_structure = np.array([
            [1, 3, 1, 0, 0],
            [4, 1, 3, 2, 2],
            [3, 4, 4, 4, 4],
            [3, 0, 0, 3, 2],
            [0, 0, 0, 0, 2]
        ])
        self.connectivity = get_full_connectivity(self.robot_structure)

        # Create directory for saving results
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
            print(f"Created directory: {self.directory}")
        else:
            print(f"Directory already exists: {self.directory}")


    def reshape_individual(self, individual):
        # Reshape flat weights into model format
        shapes = [p.shape for p in self.brain.parameters()]
        new_weights = []
        idx = 0
        for shape in shapes:
            size = np.prod(shape)
            new_weights.append(individual[idx:idx+size].reshape(shape))
            idx += size

        return new_weights

    def evaluate_fitness(self, weights, view=False):
        """Evaluate the fitness of a neural controller with given weights."""
        brain = NeuralController(self.input_size, self.output_size)
        set_weights(brain, weights)  # Load weights into the network

        env = gym.make(self.scenario, max_episode_steps=self.steps, body=self.robot_structure, connections=self.connectivity)
        sim = env.sim
        viewer = EvoViewer(sim)
        viewer.track_objects('robot')

        state = env.reset()[0]  # Get initial state
        t_reward = 0
        for t in range(self.steps):
            state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Convert to tensor
            action = brain(state_tensor).detach().numpy().flatten()  # Get action
            if view:
                viewer.render('screen')
            state, reward, terminated, truncated, info = env.step(action)
            t_reward += reward
            if terminated or truncated:
                env.reset()
                break

        viewer.close()
        env.close()
        return t_reward

    def de_search(self, run_number):
        """Perform Differential Evolution search."""
        best_weights = None
        best_fitness = -float('inf')
        fitness_history = []  # Store fitness for each iteration

        # Prepare CSV file
        iteration_csv = self.directory + f'fitness_run_{run_number}.csv'
        with open(iteration_csv, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Iteration', 'Fitness'])

            ###### Step 1: Initialize population
            # each individual is a randomized flattened vector of weights, 
            #from a uniform distribution between -1 and 1
            brain = NeuralController(self.input_size, self.output_size)
            self.brain = brain
            mean_weights = np.concatenate([p.detach().numpy().flatten() for p in brain.parameters()])
            dim = mean_weights.shape[0]
            population = np.random.uniform(-1, 1, (self.population_size, dim))  # Random initialization

            with multiprocessing.Pool() as pool:
                    reshaped_inds = pool.map(self.reshape_individual, population)
                    old_fitnesses = pool.map(self.evaluate_fitness, reshaped_inds)
            
            # Generational Loop
            for it in range(self.num_generations):
                new_population = []
                fitnesses = []
                trials = []

                for i in range(self.population_size):

                    ###### Step 2: Geometric mutation
                    #Each mutant is a linear combination of three randomly chosen individuals
                    indices = list(range(self.population_size))
                    indices.remove(i)
                    a, b, c = population[np.random.choice(indices, 3, replace=False)]
                    mutant = a + self.mutation_factor * (b - c)
                    mutant = np.clip(mutant, -1, 1)  # Ensure mutant is within bounds

                    ###### Step 3: Crossover
                    # A trial vector is created as a copy of th current individual in the population loop
                    # This individual is crossed over with the mutant vector
                    trial = np.copy(population[i])
                    for j in range(dim):
                        if random.random() < self.crossover_rate:
                            trial[j] = mutant[j]

                    trials.append(trial)

                ###### Step 4: Evaluate fitness
                #reshape all trials and evaluate their fitnesses
                with multiprocessing.Pool() as pool:
                    reshaped_inds = pool.map(self.reshape_individual, trials)
                    trial_fitnesses = pool.map(self.evaluate_fitness, reshaped_inds)
            
                
                ###### Step 5: Replace individual in population with the trial, if the trial is better
                for i in range (self.population_size):
                    if trial_fitnesses[i] > old_fitnesses[i]:
                        new_population.append(trials[i])
                        fitnesses.append(trial_fitnesses[i])
                    else:
                        new_population.append(population[i])
                        fitnesses.append(old_fitnesses[i])

                population = np.array(new_population)

                ###### Step 6: Get best individual
                best_idx = np.argmax(fitnesses)
                if fitnesses[best_idx] > best_fitness:
                    best_fitness = fitnesses[best_idx]
                    shapes = [p.shape for p in brain.parameters()]
                    new_weights = []
                    idx_flat = 0
                    for shape in shapes:
                        size = np.prod(shape)
                        new_weights.append(population[best_idx][idx_flat:idx_flat+size].reshape(shape))
                        idx_flat += size
                    best_weights = new_weights

                ###### Step 7: Update old fitnesses
                old_fitnesses = fitnesses

                # Logging
                writer.writerow([it + 1, fitnesses[best_idx]])
                fitness_history.append(fitnesses[best_idx])

                print(f"Iteration {it + 1}: Best Fitness = {fitnesses[best_idx]}")

        # Plotting
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.num_generations + 1), fitness_history, marker='o', label='Fitness')
        plt.title(f'Differential Evolution Fitness Over Iterations (Run {run_number})')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        plt.grid(True)
        plt.legend()
        graph_filename = self.directory + f'fitness_plot_run_{run_number}.png'
        plt.savefig(graph_filename)
        plt.close()
        print(f"Fitness plot for run {run_number} saved to {graph_filename}")

        save_best_controller(best_weights, self.directory + f'controller_run_{run_number}.json')
        return best_weights, best_fitness

    def execute_runs(self, n_runs):
        best_fitnesses = []  # Store the best fitness of each run

        # Get input and output sizes
        env = gym.make(self.scenario, max_episode_steps=self.steps, body=self.robot_structure, connections=self.connectivity)
        self.input_size = env.observation_space.shape[0]
        self.output_size = env.action_space.shape[0]
        env.close()

        for run in range(1, n_runs + 1):
            print(f"Executing run {run}/{n_runs}...")
            best_weights, best_fitness = self.de_search(run_number=run)
            best_fitnesses.append(best_fitness)

            # Save the best weights for this run
            weights_filename = self.directory + f'best_weights_run_{run}.npy'
            np.save(weights_filename, best_weights)
            print(f"Best weights for run {run} saved to {weights_filename}")

            self.seed += 1
            np.random.seed(self.seed)
            random.seed(self.seed)

        # Save the best fitnesses to a CSV file
        output_csv = self.directory + 'best_fitnesses.csv'
        with open(output_csv, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Run', 'Best Fitness'])
            for run, fitness in enumerate(best_fitnesses, start=1):
                writer.writerow([run, fitness])

        print(f"All runs completed. Best fitnesses saved to {output_csv}.")


if __name__ == "__main__":
    de_algorithm = DE_Controller(population_size=5,
                                  num_generations=2,
                                  steps=500,
                                  mutation_factor=0.1,
                                  crossover_rate=0.8,
                                  scenario="DownStepper-v0",
                                  directory="results/de/DownStepper-v0/")
    
    de_algorithm.execute_runs(2)


if __name__ == "__main__":
    de_algorithm = DE_Controller(population_size=50,
                                  num_generations=100,
                                  steps=500,
                                  mutation_factor=0.1,
                                  crossover_rate=0.8,
                                  scenario="ObstacleTraverser-v0",
                                  directory="results/de/ObstacleTraverser-v0/")
    
    de_algorithm.execute_runs(5)