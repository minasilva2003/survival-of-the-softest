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
from data_utils import create_directory, write_to_csv_file, plot_graph, average_csv_files
from GA_utils import evaluate_fitness, create_random_robot, standard_mutate, one_point_crossover, standard_tournament_selection, simulate_and_save

class GeneticAlgorithm:
    def __init__(self, population_size=10, num_generations=10, tournament_size=5, mutation_rate=0.1, crossover_rate=0.7,
                 elitism_count=0, steps=500, grid_size=(5, 5), voxel_types=[0, 1, 2, 3, 4], controller=hopping_motion,
                 scenario='Walker-v0', directory="results/genetic_algorithm/"):
        self.population_size = population_size
        self.num_generations = num_generations
        self.tournament_size = tournament_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_count = elitism_count
        self.steps = steps
        self.grid_size = grid_size
        self.voxel_types = voxel_types  # Empty, Rigid, Soft, Active (+/-)
        self.controller = controller
        self.scenario = scenario  # Scenario is now an attribute of the class
        self.directory = directory

        create_directory(self.directory)  # Create directory for saving results

    # perform genetic algorithm to evolve the best robot structure
    def genetic_algorithm(self, run_number):
        best_robot = None
        best_fitness = -float('inf')
        best_fitness_history = []  # Store fitness for each generation
        #average_fitness_history = []  # Store average fitness for each generation
        
        # Initialize population
        population = [create_random_robot(self.grid_size) for _ in range(self.population_size)]

        for generation in range(self.num_generations):

            # Step 1: Evaluate fitness
            fitnesses = [evaluate_fitness(scenario = self.scenario, robot_structure=robot, controller=self.controller) for robot in population]
            best_fitness = max(fitnesses)
            #avg_fitness = sum(fitnesses) / len(fitnesses)
            best_fitness_history.append(best_fitness)
            #average_fitness_history.append(avg_fitness)
            print(f"Generation {generation + 1}: Best Fitness = {best_fitness}")

            # Step 2: Keep the elite
            elites = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)[:self.elitism_count]
            offspring = [elite[0] for elite in elites]

            # Step 3: Create offspring
            while len(offspring) < self.population_size:

                # 3.1. tournament selection
                parent1 = standard_tournament_selection(population, fitnesses, tournament_size=self.tournament_size)
                parent2 = standard_tournament_selection(population, fitnesses, tournament_size=self.tournament_size)

                # 3.2. crossover
                if random.random() < self.crossover_rate:
                    child1, child2 = one_point_crossover(parent1, parent2)
                else:
                    child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

                # 3.3. mutation
                if random.random() < self.mutation_rate:
                    child1 = standard_mutate(child1, voxel_types=self.voxel_types)
                if random.random() < self.mutation_rate:
                    child2 = standard_mutate(child2, voxel_types=self.voxel_types)

                # 3.4. add children to offspring
                offspring.append(child1)
                if len(offspring) < self.population_size:
                    offspring.append(child2)


            # Step 4: Create new population
            population = offspring[:self.population_size]  # Ensure population size is maintained
           
        #get fitnesses of the final population
        fitnesses = [evaluate_fitness(scenario=self.scenario, controller=self.controller, robot_structure=robot) for robot in population]
        best_fitness = max(fitnesses)
        best_fitness_history.append(best_fitness)

        # Get the best robot
        best_robot = population[fitnesses.index(max(fitnesses))]

        # Save information to CSV files
        write_to_csv_file(self.directory + f'best_fit_run_{run_number}.csv', best_fitness_history)
        #write_to_csv_file(self.directory + f'avg_fit_run_{run_number}.csv', average_fitness_history)

        # Plot the fitness history
        plot_graph(best_fitness_history, self.num_generations+1, 'Generation', 'Best_Fitness', 'Fitness', f"Best Fitness for Each Generation, Run {run_number}", self.directory + f'best_fit_plot_run_{run_number}.png')
        #plot_graph(average_fitness_history, self.num_generations+1, 'Generation', 'Average_Fitness', 'Fitness', f"Average Fitness for Each Generation, Run {run_number}", self.directory + f'avg_fitness_plot_run_{run_number}.png') 
                          
        return best_robot, best_fitness



    # Execute multiple runs of the genetic algorithm
    def execute_runs(self, n_runs):
        best_fitnesses = []
       
        for run in range(1, n_runs + 1):
            print(f"Executing run {run}/{n_runs}...")
            
            best_robot, best_fitness = self.genetic_algorithm(run)
            best_fitnesses.append(best_fitness)

            # Save the simulation and GIF for this run
            gif_filename = self.directory + f'best_robot_run_{run}.gif'
            simulate_and_save(scenario=self.scenario, steps=self.steps, controller=self.controller, best_robot=best_robot, filename=gif_filename)

           
        # Save the best fitnesses to a CSV file
        write_to_csv_file(self.directory + f'all_runs_best_fit.csv', best_fitnesses)
        
        # Average the best fitness CSV files
        average_csv_files(self.directory, 'best_fit', 'all_runs_avg_fit.csv')
        
     

       

# Example usage
if __name__ == "__main__":

    ga = GeneticAlgorithm(num_generations=100, 
                          population_size=50,
                          tournament_size=5, 
                          mutation_rate=0.5, 
                          crossover_rate=0.8,
                          elitism_count=2,
                          scenario='Walker-v0',
                          controller=alternating_gait,
                          directory="results/genetic_algorithm/official_experiments/Walker-v0/walking/")
                         
    ga.execute_runs(n_runs=5)

    ga = GeneticAlgorithm(num_generations=100, 
                          population_size=50,
                          tournament_size=5, 
                          mutation_rate=0.5, 
                          crossover_rate=0.8,
                          elitism_count=2,
                          scenario='BridgeWalker-v0',
                          controller=alternating_gait,
                          directory="results/genetic_algorithm/official_experiments/BridgeWalker-v0/walking/")
                         
    ga.execute_runs(n_runs=5)

    