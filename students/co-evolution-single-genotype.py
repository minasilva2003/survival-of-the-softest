import numpy as np
import random
import copy
import os
import gymnasium as gym
import multiprocessing
from always_connected_GA import GeneticAlgorithm
from neural_controller import *
from cma_es_library import CMA_ES_Controller
from data_utils import write_to_csv_file, create_directory, plot_graph, save_best_robot, save_best_controller, average_csv_files
from evogym import EvoViewer, get_full_connectivity, is_connected
import cma
import time
from GA_utils import create_random_robot, standard_mutate, standard_tournament_selection, one_point_crossover

class CoEvolution:
    def __init__(self, population_size, num_generations, crossover_rate, mutation_rate, elitism_count, tournament_size,  steps, scenario, grid_size=(5, 5), voxel_types=[0, 1, 2, 3, 4], directory="results/co-evolution/"):
     

        self.steps = steps
        self.scenario = scenario
        self.population_size = population_size
        self.num_generations = num_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_count = elitism_count
        self.tournament_size = tournament_size
        self.grid_size = grid_size
        self.voxel_types = voxel_types
        self.directory = directory

        create_directory(self.directory)

    #get input and output size for Neural Controller
    def get_input_and_output_size(self, robot):
       
        #check observation space to withdraw input and output size
        env = gym.make(self.scenario, max_episode_steps=self.steps, body=robot, connections=get_full_connectivity(robot))
        input_size = env.observation_space.shape[0]
        output_size = env.action_space.shape[0]
        env.close()
        return input_size, output_size

    def reshape_controller(self, individual):
        # Reshape flat weights into model format
        controller = individual["controller"]
        brain = individual["brain"]
        shapes = [p.shape for p in brain.parameters()]
        new_controller = []
        idx = 0
        for shape in shapes:
            size = np.prod(shape)
            new_controller.append(controller[idx:idx+size].reshape(shape))
            idx += size

        return new_controller

    def initialize_population(self):
        # Initialize controller population

        population = []

        for _ in range (self.population_size):
            individual={}
            individual["robot"] = create_random_robot(self.grid_size)

            input_size, output_size = self.get_input_and_output_size(individual["robot"])
            brain = NeuralController(input_size, output_size)
            mean_weights = np.concatenate([p.detach().numpy().flatten() for p in brain.parameters()])
            dim = mean_weights.shape[0]
            individual["controller"] = mean_weights
            individual["brain"] = brain
            population.append(individual)
        
        return population
    

    ###### Evaluate fitness of a robot-controller pairing 
    def evaluate_fitness(self, individual, view=False):
      
        robot, weights = individual["robot"], self.reshape_controller(individual)

        brain = individual["brain"]
        try:
            set_weights(brain, weights)
        except Exception as e:
            print("Brain mismatch")
            return 0 

        #perform simulation
        env = gym.make(self.scenario, max_episode_steps=self.steps, body=robot, connections=get_full_connectivity(robot))
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

    
    def controller_mutate(self, weights, mutation_strength=0.1):
   
        mutated_weights = weights.copy()
        for i in range(len(mutated_weights)):
            if np.random.rand() < self.mutation_rate:
                mutated_weights[i] += np.random.normal(0, mutation_strength)
        return mutated_weights
    

    def controller_crossover(self, parent1, parent2):
      
        if parent1.shape[0] < parent2.shape[0]:
            size = parent1.shape[0]
        else:
            size = parent2.shape[0]

        cxpoint1 = np.random.randint(1, size - 1)
        cxpoint2 = np.random.randint(cxpoint1 + 1, size)

        child1 = parent1.copy()
        child2 = parent2.copy()

        # Swap the genes between the two crossover points
        child1[cxpoint1:cxpoint2] = parent2[cxpoint1:cxpoint2]
        child2[cxpoint1:cxpoint2] = parent1[cxpoint1:cxpoint2]

        return child1, child2

    
    def co_evolution(self, run_number):

        #Track best fitness and individuals
        best_robot = None
        best_controller = None
        best_fitness = -float('inf')
        best_fitness_history = []

        ###### Step 1: Initialize population
        population = self.initialize_population()
        print("Population initialized")

        ###### Generational Loop
        for gen in range(self.num_generations):
            
            ###### Step 2: evaluate fitness
            with multiprocessing.Pool() as pool:
                fitnesses = pool.map(self.evaluate_fitness, population)

            best_fitness = max(fitnesses)
            best_fitness_history.append(best_fitness)
            print(f"Generation {gen}: {best_fitness}")
                
            ###### Step 3: get elite
            elites = sorted(zip(population, fitnesses), key=lambda x: x[1], reverse=True)[:self.elitism_count]
            new_population = [elite[0] for elite in elites]
        

            ###### Step 4: get other offspring
            while len(new_population) < self.population_size:

                # 3.1. tournament selection
                parent1 = standard_tournament_selection(population, fitnesses, tournament_size=self.tournament_size)
                parent2 = standard_tournament_selection(population, fitnesses, tournament_size=self.tournament_size)

                child1 ={}
                child2 = {}
                child1["brain"] = parent1["brain"]
                child2["brain"] = parent2["brain"]


                # 3.2.1 crossover for robots
                if random.random() < self.crossover_rate:
                    connected=False
                    while not connected:
                        child1["robot"], child2["robot"] = one_point_crossover(parent1["robot"], parent2["robot"])
                        connected = is_connected(child1["robot"]) and is_connected(child2["robot"])
                else:   
                    child1["robot"], child2["robot"] = copy.deepcopy(parent1["robot"]), copy.deepcopy(parent2["robot"])


                # 3.2.2 crossover for controllers
                if random.random() < self.crossover_rate:
                    child1["controller"], child2["controller"] = self.controller_crossover(parent1["controller"], parent2["controller"])
                else:
                    child1["controller"], child2["controller"] = copy.deepcopy(parent1["controller"]), copy.deepcopy(parent2["controller"])
                

                # 3.3.1 mutation for robots
                if random.random() < self.mutation_rate:
                    connected=False
                    while not connected:
                        child1["robot"] = standard_mutate(child1["robot"], voxel_types=self.voxel_types)
                        connected = is_connected(child1["robot"])
                if random.random() < self.mutation_rate:
                    connected=False
                    while not connected:
                        child2["robot"] = standard_mutate(child2["robot"], voxel_types=self.voxel_types)
                        connected = is_connected(child2["robot"])

                
                # 3.3.2 mutation for controllers
                child1["controller"] = self.controller_mutate(child1["controller"])
                child2["controller"] = self.controller_mutate(child2["controller"])

                # 3.4. add children to offspring
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            population = new_population[:self.population_size]

        #get fitnesses of the final population
        with multiprocessing.Pool() as pool:
            fitnesses = pool.map(self.evaluate_fitness, population)

        # Get the best individual
        best_fitness = max(fitnesses)
        best_fitness_history.append(best_fitness)
        best_individual = population[fitnesses.index(best_fitness)]
        best_robot, best_controller = best_individual["robot"], self.reshape_controller(best_individual)

        ###### Step 9: Logging
        write_to_csv_file(self.directory + f"best_fit_run_{run_number}.csv")
        plot_graph(best_fitness_history, self.num_generations+1, 'Generation', 'Best_Fitness', 'Fitness', f"Best Fitness for Each Generation, Run {run_number}", self.directory + f'best_fit_plot_run_{run_number}.png')
        save_best_robot(self.directory + f"best_robot_run_{run_number}.json", best_robot)
        save_best_controller(self.directory + f"best_controller_run_{run_number}.json", best_controller)
        
        return best_robot, best_controller, best_fitness


    # Execute multiple runs of the genetic algorithm
    def execute_runs(self, n_runs):
        best_fitnesses = []
       
        for run in range(1, n_runs + 1):
            print(f"Executing run {run}/{n_runs}...")
            _, _, best_fitness = self.co_evolution(run)
            best_fitnesses.append(best_fitness)

        # Save the best fitnesses to a CSV file
        write_to_csv_file(self.directory + f'all_runs_best_fit.csv', best_fitnesses)
        
        # Average the best fitness CSV files
        average_csv_files(self.directory, 'best_fit', 'all_runs_avg_fit.csv')
        
     

if __name__ == "__main__":

    # Initialize and run co-evolution
    co_op = CoEvolution(population_size = 10,
                                num_generations = 1,
                                crossover_rate = 0.8,
                                mutation_rate = 0.5,
                                tournament_size = 5,
                                elitism_count = 2,
                                steps = 500, 
                                scenario = "Walker-v0",
                                directory = "results/co_evolution"
                                )

    co_op.execute_runs(2)