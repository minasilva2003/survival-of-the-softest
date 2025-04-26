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


class RandomControllerSearch:
    def __init__(self, num_generations=10, steps=500, scenario='DownStepper-v0', seed=42, directory="results/random_controller/"):
        self.num_generations = num_generations
        self.steps = steps
        self.scenario = scenario
        self.seed = seed
        self.directory = directory

        # Set random seeds
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

    def random_search(self, run_number):
        """Perform a random search to find the best neural controller weights."""
        best_weights = None
        best_fitness = -float('inf')
        fitness_history = []  # Store fitness for each iteration

        # Prepare the CSV file to store fitness for each iteration
        iteration_csv = self.directory + f'fitness_run_{run_number}.csv'
        with open(iteration_csv, mode='w', newline='') as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(['Iteration', 'Fitness'])  # Header row

            for it in range(self.num_generations):
                # Generate random weights for the neural network
                brain = NeuralController(self.input_size, self.output_size)
                random_weights = [np.random.randn(*param.shape) for param in brain.parameters()]

                # Evaluate the fitness of the current weights
                fitness_score = self.evaluate_fitness(random_weights)

                # Log the fitness of the current iteration
                writer.writerow([it + 1, fitness_score])
                fitness_history.append(fitness_score)

                if fitness_score > best_fitness:
                    best_fitness = fitness_score
                    best_weights = random_weights

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

        return best_weights, best_fitness

    def execute_runs(self, n_runs):
        """Execute multiple runs of the random search, save results, and generate a combined plot."""
        best_fitnesses = []
        all_fitness_histories = []  # Store fitness histories for all runs

        # Get input and output sizes
        env = gym.make(self.scenario, max_episode_steps=self.steps, body=self.robot_structure, connections=self.connectivity)
        self.input_size = env.observation_space.shape[0]
        self.output_size = env.action_space.shape[0]
        env.close()

        for run in range(1, n_runs + 1):
            print(f"Executing run {run}/{n_runs}...")
            best_weights, best_fitness = self.random_search(run)
            best_fitnesses.append(best_fitness)

            # Save the best weights to a file
            weights_filename = self.directory + f'best_weights_run_{run}.npy'
            np.save(weights_filename, best_weights)
            print(f"Best weights for run {run} saved to {weights_filename}")

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
        print(f"Combined fitness plot saved to {combined_plot_filename}")


# Example usage
if __name__ == "__main__":
    search = RandomControllerSearch(num_generations=100, scenario='DownStepper-v0', directory="results/random_controller/DownStepper-v0/")
    search.execute_runs(n_runs=5)