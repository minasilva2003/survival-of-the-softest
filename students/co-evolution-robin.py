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

class CoEvolutionGA_CMAES:
    def __init__(self, ga_params, cma_es_params, steps, scenario, population_size=50, num_generations=100, pairing_strategy="random", directory="results/co-evolution/"):
        """
        Initialize the co-evolution framework.

        Args:
            ga_params (dict): Parameters for the Genetic Algorithm (GA) for evolving robot structures.
            cma_es_params (dict): Parameters for CMA-ES for evolving controllers.
            num_generations (int): Number of generations for co-evolution.
            pairing_strategy (str): Strategy for pairing robots and controllers ("random", "best", etc.).
            directory (str): Directory to save results.
        """
        self.ga_params = ga_params
        self.cma_es_params = cma_es_params
        self.steps = steps
        self.scenario = scenario
        self.population_size = population_size
        self.num_generations = num_generations
        self.pairing_strategy = pairing_strategy
        self.directory = directory
        self.input_size = None
        self.output_size = None
        self.seed = random.randint(1, 0xFFFFFFFF)
        self.robot_population = None
        self.controller_population = None

        create_directory(self.directory)

        np.random.seed(self.seed)
        random.seed(self.seed)

        self.input_size, self.output_size = self.get_input_and_output_size(np.array([
                                                                                    [1, 3, 1, 0, 0],
                                                                                    [4, 1, 3, 2, 2],
                                                                                    [3, 4, 4, 4, 4],
                                                                                    [3, 0, 0, 3, 2],
                                                                                    [0, 0, 0, 0, 2]
                                                                                ]))

        self.brain = None


    def check_if_valid_pair(self, robot):
        input_size, output_size = self.get_input_and_output_size(robot)

        if input_size == self.input_size and output_size == self.output_size:
            return 0
        
        else:
            return -(abs(input_size - self.input_size) + abs(output_size - self.output_size))
        

    #get input and output size for Neural Controller
    def get_input_and_output_size(self, robot):
       
        #check observation space to withdraw input and output size
        env = gym.make(self.scenario, max_episode_steps=self.steps, body=robot, connections=get_full_connectivity(robot))
        input_size = env.observation_space.shape[0]
        output_size = env.action_space.shape[0]
        env.close()
        return input_size, output_size

    def reshape_controller(self, controller):
        # Reshape flat weights into model format
        shapes = [p.shape for p in self.brain.parameters()]
        new_controller = []
        idx = 0
        for shape in shapes:
            size = np.prod(shape)
            new_controller.append(controller[idx:idx+size].reshape(shape))
            idx += size

        return new_controller

    def initialize_populations(self):

        # Initialize controller population
        brain = NeuralController(self.input_size, self.output_size)
        self.brain = brain
        mean_weights = np.concatenate([p.detach().numpy().flatten() for p in brain.parameters()])
        dim = mean_weights.shape[0]

        self.es = cma.CMAEvolutionStrategy(mean_weights, self.cma_es_params["sigma"], {'popsize': self.population_size, 'seed': self.seed})

        self.controller_population = self.es.ask()

        # Initialize robot population 
        self.robot_population = [create_random_robot(self.ga_params["grid_size"]) for _ in range(self.population_size)]

    
    def alter_robot_population(self, fitnesses):

        new_population = []

        while len(new_population) < self.population_size:
            
            ### Step 1: parent selection
            parent1 = standard_tournament_selection(self.robot_population, fitnesses, tournament_size=self.ga_params["tournament_size"])
            parent2 = standard_tournament_selection(self.robot_population, fitnesses, tournament_size=self.ga_params["tournament_size"])

            ### Step 2: crossover
            if random.random() < self.ga_params["crossover_rate"]:
                connected=False
                while not connected:
                    child1, child2 = one_point_crossover(parent1, parent2)
                    connected = is_connected(child1) and is_connected(child2)
            else:   
                child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)

            ### Step 3: mutation
            if random.random() < self.ga_params["mutation_rate"]:
                connected=False
                while not connected:
                    child1 = standard_mutate(child1, voxel_types=self.ga_params["voxel_types"])
                    connected = is_connected(child1)
            
            if random.random() < self.ga_params["mutation_rate"]:
                connected=False
                while not connected:
                    child2 = standard_mutate(child2, voxel_types=self.ga_params["voxel_types"])
                    connected = is_connected(child2)

            # 3.4. add children to offspring
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)

        return new_population


    def alter_controller_population(self, fitnesses):
        
        #reverse fitness signs because cma-es minimizes fitness
        reversed_fitnesses = [-x for x in fitnesses]

        #update cma-es with new fitness values
        self.es.tell(self.controller_population, reversed_fitnesses)

        #get new population
        return self.es.ask()
    

    ###### Evaluate fitness of a robot-controller pairing 
    def evaluate_fitness(self, pairing, view=False):
      
        robot, controller = pairing

        weights = self.reshape_controller(controller)

        diff = self.check_if_valid_pair(robot)

        if diff != -0:
            #print("Size mistmatch")
            return diff

        input_size, output_size = self.get_input_and_output_size(robot)

        #load weights into NN
        brain = NeuralController(input_size, output_size)

        set_weights(brain, weights)

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
        #print(t_reward)
        return t_reward
        


    def evaluate_fitness_in_pairs_random(self):
      
        

        ###### Step 1: Evaluate each robot with random controller pairing
        pairings = [(robot, random.choice(self.controller_population)) for robot in self.robot_population]

        
        with multiprocessing.Pool() as pool:
            robot_fitnesses = pool.map(self.evaluate_fitness, pairings)
       
        #robot_fitnesses = [self.evaluate_fitness(pairing) for pairing in pairings]
        
        #get best pairing
        best_idx = np.argmax(robot_fitnesses)
        best_pair_1 = pairings[best_idx]
        best_fit_1 = robot_fitnesses[best_idx]

        ###### Step 2: Evaluate each controller with random robot pairing
        pairings = [(random.choice(self.robot_population), controller) for controller in self.controller_population]

        
        with multiprocessing.Pool() as pool:
            controller_fitnesses = pool.map(self.evaluate_fitness, pairings)
        
        #controller_fitnesses = [self.evaluate_fitness(pairing) for pairing in pairings]
        
        #get best pairing
        best_idx = np.argmax(controller_fitnesses)
        best_pair_2 = pairings[best_idx]
        best_fit_2 = controller_fitnesses[best_idx]

        ###### Step 3: return fitnesses and best pairing
        if best_fit_1 > best_fit_2:
            return robot_fitnesses, controller_fitnesses, best_pair_1, best_fit_1
        else:
            return robot_fitnesses, controller_fitnesses, best_pair_2, best_fit_2
        
        
    def evaluate_fitness_in_pairs_round_robin(self):
            
            pairings = []
            
            ###### Step 1: Robots

            # assign 3 controllers to each robot
            for robot in self.robot_population:
                for _ in range (3):
                    pairings.append((robot, random.choice(self.controller_population)))

            # evaluate fitnesses
            with multiprocessing.Pool() as pool:
                aux_fitnesses = pool.map(self.evaluate_fitness, pairings)

            robot_fitnesses = []

            # average fitness for each three pairings
            for idx in range(0, len(aux_fitnesses), 3):
                robot_fitnesses.append((aux_fitnesses[idx] + aux_fitnesses[idx+1] + aux_fitnesses[idx+2]) / 3)

            print(robot_fitnesses)

            #get best pairing
            best_idx = np.argmax(robot_fitnesses)
            best_pair_1 = pairings[best_idx]
            best_fit_1 = robot_fitnesses[best_idx]

            ####################################################################

            ###### Step 2: Controllers

            pairings = []

            # assign 3 robots to each pairing
            for controller in self.controller_population:
                for _ in range (3):
                    pairings.append((random.choice(self.robot_population), controller))

            # evaluate fitnesses
            with multiprocessing.Pool() as pool:
                aux_fitnesses = pool.map(self.evaluate_fitness, pairings)

            controller_fitnesses = []

            # average fitness for each three pairings
            for idx in range(0, len(aux_fitnesses), 3):
                controller_fitnesses.append((aux_fitnesses[idx] + aux_fitnesses[idx+1] + aux_fitnesses[idx+2]) / 3)

            #get best pairing
            best_idx = np.argmax(controller_fitnesses)
            best_pair_2 = pairings[best_idx]
            best_fit_2 = controller_fitnesses[best_idx]

            ###### Step 3: return fitnesses and best pairing
            if best_fit_1 > best_fit_2:
                return robot_fitnesses, controller_fitnesses, best_pair_1, best_fit_1
            else:
                return robot_fitnesses, controller_fitnesses, best_pair_2, best_fit_2
            

    def evaluate_fitness_in_pairs_tournament(self, robot_fitnesses, controller_fitnesses, size=5):

        pairings = []
        
        ###### Step 1: Robots

        # do a controller tournament for each robot
        for robot in self.robot_population:
            
            tournament = random.sample(list(zip(self.controller_population, controller_fitnesses)), size)
            winner = max(tournament, key=lambda x: x[1])[0]

            pairings.append((robot, winner))
    
        
        # evaluate fitnesses
        with multiprocessing.Pool() as pool:
            robot_fitnesses = pool.map(self.evaluate_fitness, pairings)

        #get best pairing
        best_idx = np.argmax(robot_fitnesses)
        best_pair_1 = pairings[best_idx]
        best_fit_1 = robot_fitnesses[best_idx]

        ####################################################################

        ###### Step 2: Controllers

        pairings = []

        # do a robot tournament for each controller
        for controller in self.controller_population:
            
            tournament = random.sample(list(zip(self.robot_population, robot_fitnesses)), size)
            winner = max(tournament, key=lambda x: x[1])[0]

            pairings.append((winner, controller))
    
        # evaluate fitnesses
        with multiprocessing.Pool() as pool:
            controller_fitnesses = pool.map(self.evaluate_fitness, pairings)

        #get best pairing
        best_idx = np.argmax(controller_fitnesses)
        best_pair_2 = pairings[best_idx]
        best_fit_2 = controller_fitnesses[best_idx]

        ###### Step 3: return fitnesses and best pairing
        if best_fit_1 > best_fit_2:
            return robot_fitnesses, controller_fitnesses, best_pair_1, best_fit_1
        else:
            return robot_fitnesses, controller_fitnesses, best_pair_2, best_fit_2
        

  
    
    def co_evolution(self, run_number):

        #Track best fitness and individuals
        best_robot = None
        best_controller = None
        best_fitness = -float('inf')
        best_fitness_history = []

        ###### Step 1: Initialize population
        self.initialize_populations()
        print("Population initialized")

        old_robot_fitnesses = []
        old_controller_fitnesses = []

        ###### Generational Loop
        for gen in range(self.num_generations):
            
            ###### Step 2: evaluate fitness in pairs
            if self.pairing_strategy == "random":
                robot_fitnesses, controller_fitnesses, best_pair, best_fit = self.evaluate_fitness_in_pairs_random()

            elif self.pairing_strategy == "round_robin" or (self.pairing_strategy == "tournament" and gen == 0):
                robot_fitnesses, controller_fitnesses, best_pair, best_fit = self.evaluate_fitness_in_pairs_round_robin()

            elif self.pairing_strategy == "tournament":
                robot_fitnesses, controller_fitnesses, best_pair, best_fit = self.evaluate_fitness_in_pairs_tournament(old_robot_fitnesses, old_controller_fitnesses)

            old_robot_fitnesses = robot_fitnesses
            old_controller_fitnesses = controller_fitnesses

            #In certain intervals, check all robots against all controllers
            if (gen+1) % 10 == 0:
                all_pairings = [(robot, controller) for robot in self.robot_population for controller in self.controller_population]
                with multiprocessing.Pool() as pool:
                    all_fitnesses = pool.map(self.evaluate_fitness, all_pairings)
                best_overall_idx = np.argmax(all_fitnesses)
                best_overall_pair = all_pairings(best_overall_idx)
                best_overall_fit = all_fitnesses(best_overall_idx)

                #If the fitness found in all vs all comparisons is better than paired comparison, replace best pair and best fit
                if best_overall_fit > best_fit:
                    best_fit, best_pair = best_overall_fit, best_overall_pair

            #Log best fitness found in this generation
            best_fitness_history.append(best_fit)
            print(f"Generation {gen}: {best_fit}")
                
            ###### Step 3: get elite
            robot_elite, controller_elite = best_pair

            ###### Step 4: alter robot population
            new_robot_population = self.alter_robot_population(robot_fitnesses)

            ###### Step 5: alter controller population
            new_controller_population = self.alter_controller_population(controller_fitnesses)

            ###### Step 6: carry over elites
            new_robot_population[0] = robot_elite
            new_controller_population[0] = controller_elite

            ###### Step 7: replace populations
            self.robot_population = new_robot_population
            self.controller_population = new_controller_population

            ###### Step 8: track best individual
            if best_fit > best_fitness:
                best_fitness = best_fit
                best_robot, best_controller = best_pair[0], self.reshape_controller(best_pair[1])


        ###### Step 9: Logging
        write_to_csv_file(self.directory + f"best_fit_run_{run_number}.csv", best_fitness_history)
        plot_graph(best_fitness_history, self.num_generations, 'Generation', 'Best_Fitness', 'Fitness', f"Best Fitness for Each Generation, Run {run_number}", self.directory + f'best_fit_plot_run_{run_number}.png')
        save_best_robot(self.directory + f"best_robot_run_{run_number}.json", best_robot)
        save_best_controller(best_controller, self.directory + f"best_controller_run_{run_number}.json")
        
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

    ga_params = {
        "tournament_size": 5,
        "mutation_rate": 0.5,
        "crossover_rate": 0.8,
        "grid_size": (5, 5),
        "voxel_types": [0, 1, 2, 3, 4],
    }

    
    cma_es_params = {
        "sigma": 0.5,
    }

    # Initialize and run co-evolution
    co_op = CoEvolutionGA_CMAES(ga_params = ga_params,
                                cma_es_params = cma_es_params,
                                steps = 500, 
                                scenario = "GapJumper-v0", 
                                population_size = 50,
                                num_generations = 100,
                                pairing_strategy = "round_robin",
                                directory = "results/co_evolution/robin/GapJumper-v0/"
                                )

    co_op.execute_runs(5)

    # Initialize and run co-evolution
    co_op = CoEvolutionGA_CMAES(ga_params = ga_params,
                                cma_es_params = cma_es_params,
                                steps = 500, 
                                scenario = "CaveCrawler-v0", 
                                population_size = 50,
                                num_generations = 100,
                                pairing_strategy = "round_robin",
                                directory = "results/co_evolution/robin/CaveCrawler-v0/"
                                )

    co_op.execute_runs(5)