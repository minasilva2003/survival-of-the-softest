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

class RandomStructureSearch:
    def __init__(self, num_generations=10, min_grid_size=(5, 5), max_grid_size=(5, 5), steps=500, scenario='Walker-v0', controller=hopping_motion, directory="results/random_search/"):
        self.num_generations = num_generations
        self.min_grid_size = min_grid_size
        self.max_grid_size = max_grid_size
        self.steps = steps
        self.scenario = scenario
        self.controller = controller
        self.voxel_types = [0, 1, 2, 3, 4]  # Empty, Rigid, Soft, Active (+/-)

        #create directory for saving results
        self.directory = directory
        if not os.path.exists(self.directory):
            os.makedirs(self.directory)
            print(f"Created directory: {self.directory}")
        else:
            print(f"Directory already exists: {self.directory}")
    
   

    def evaluate_fitness(self, robot_structure, view=False):
        """Evaluate the fitness of a robot structure."""
        try:
            connectivity = get_full_connectivity(robot_structure)
            env = gym.make(self.scenario, max_episode_steps=self.steps, body=robot_structure, connections=connectivity)
            env.reset()
            sim = env.sim
            viewer = EvoViewer(sim)
            viewer.track_objects('robot')
            t_reward = 0
            action_size = sim.get_dim_action_space('robot')  # Get correct action size
            for t in range(self.steps):
                # Update actuation before stepping
                actuation = self.controller(action_size, t)
                if view:
                    viewer.render('screen')
                ob, reward, terminated, truncated, info = env.step(actuation)
                t_reward += reward

                if terminated or truncated:
                    env.reset()
                    break

            viewer.close()
            env.close()
            return t_reward
        except (ValueError, IndexError) as e:
            return 0.0



    def create_random_robot(self):
        """Generate a valid random robot structure."""
        grid_size = (
            random.randint(self.min_grid_size[0], self.max_grid_size[0]),
            random.randint(self.min_grid_size[1], self.max_grid_size[1]),
        )
        random_robot, _ = sample_robot(grid_size)
        return random_robot



    def random_search(self, run_number):
        """Perform a random search to find the best robot structure."""
        best_robot = None
        best_fitness = -float('inf')
        fitness_history = []  # Store fitness for each iteration
        
        # Prepare the CSV file to store fitness for each iteration
        iteration_csv = self.directory + f'fitness_run_{run_number}.csv'
        
        with open(iteration_csv, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Iteration', 'Fitness'])  # Header row

            for it in range(self.num_generations):
                robot = self.create_random_robot()
                fitness_score = self.evaluate_fitness(robot)

                # Log the fitness of the current iteration
                writer.writerow([it + 1, fitness_score])
                fitness_history.append(fitness_score)  # Add fitness to history

                if fitness_score > best_fitness:
                    best_fitness = fitness_score
                    best_robot = robot

                print(f"Iteration {it + 1}: Fitness = {fitness_score}")
        
        # Plot the fitness history
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, self.num_generations + 1), fitness_history, marker='o', label='Fitness')
        plt.title(f'Fitness Over Iterations (Run {run_number})')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        plt.grid(True)
        plt.legend()
        graph_filename = self.directory + f'fitness_plot_run_{run_number}.png'
        plt.savefig(graph_filename)
        plt.close()
        print(f"Fitness plot for run {run_number} saved to {graph_filename}")
        
        return best_robot, best_fitness



    def simulate_and_save(self, best_robot, filename):
        """Simulate the best robot and save the result as a GIF."""
        print("Best robot structure found:")
        print(best_robot)
        print("Best fitness score:")
        print(self.evaluate_fitness(best_robot))
        utils.simulate_best_robot(best_robot, scenario=self.scenario, steps=self.steps)
        utils.create_gif(best_robot, filename=filename, scenario=self.scenario, steps=self.steps, controller=self.controller)



    def execute_runs(self, n_runs):
        """Execute multiple runs of the random search, save results, and generate a CSV file."""
        best_fitnesses = []
        all_fitness_histories = []  # Store fitness histories for all runs

        for run in range(1, n_runs + 1):
            print(f"Executing run {run}/{n_runs}...")
            best_robot, best_fitness = self.random_search(run)
            best_fitnesses.append(best_fitness)

            # Save the simulation and GIF for this run
            gif_filename = self.directory + f'best_robot_run_{run}.gif'
            self.simulate_and_save(best_robot, filename=gif_filename)

             # Read the fitness history from the CSV file for plotting later
            iteration_csv = self.directory + f'fitness_run_{run}.csv'
            fitness_history = []
            with open(iteration_csv, mode='r') as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    fitness_history.append(float(row['Fitness']))
            all_fitness_histories.append(fitness_history)

        # Save the best fitnesses to a CSV file
        output_csv = self.directory + 'best_fitnesses.csv'

        with open(output_csv, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Run', 'Best Fitness'])
            for run, fitness in enumerate(best_fitnesses, start=1):
                writer.writerow([run, fitness])

        print(f"All runs completed. Best fitnesses saved to {output_csv}.")

         # Plot all fitness histories in one plot
        plt.figure(figsize=(12, 8))
        for run, fitness_history in enumerate(all_fitness_histories, start=1):
            plt.plot(range(1, len(fitness_history) + 1), fitness_history, marker='o', label=f'Run {run}')
        plt.title('Fitness Over Iterations for All Runs')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness')
        plt.grid(True)
        plt.legend()
        combined_plot_filename = self.directory + 'combined_fitness_plot.png'
        plt.savefig(combined_plot_filename)
        plt.close()
        print(f"Combined fitness plot saved to {combined_plot_filename}.")

# Example usage
if __name__ == "__main__":
    search = RandomStructureSearch(num_generations=100, scenario='Walker-v0', controller=hopping_motion, directory="results/random_search/Walker-v0/")
    search.execute_runs(n_runs=5)

    #search = RandomStructureSearch(num_generations=100, scenario='BridgeWalker-v0', controller=hopping_motion, directory="results/random_search/BridgeWalker-v0/")
    #search.execute_runs(n_runs=5)