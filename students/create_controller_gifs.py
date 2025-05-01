import os
import json
import torch


import gymnasium as gym
from evogym.envs import *
from evogym import EvoViewer, get_full_connectivity
from neural_controller import *
import os
import matplotlib.pyplot as plt
import torch
import imageio


robot_structure = np.array([
            [1, 3, 1, 0, 0],
            [4, 1, 3, 2, 2],
            [3, 4, 4, 4, 4],
            [3, 0, 0, 3, 2],
            [0, 0, 0, 0, 2]
        ])

connectivity = get_full_connectivity(robot_structure)

def process_weights_and_simulate(directory, robot_structure, connectivity, scenario, steps, gif_directory, duration=0.066):
    """
    Loops through JSON files in a directory, reads weights, simulates the robot, and saves the result as a GIF.

    Args:
        directory (str): The directory containing JSON files with weights.
        robot_structure (np.ndarray): The structure of the robot.
        connectivity (np.ndarray): The connectivity of the robot.
        scenario (str): The simulation scenario (e.g., 'DownStepper-v0').
        steps (int): The number of simulation steps.
        gif_directory (str): The directory to save the resulting GIFs.
    """
    # Ensure the directory exists
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    # Ensure the GIF directory exists
    if not os.path.exists(gif_directory):
        os.makedirs(gif_directory)

    # Loop through all JSON files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)

            # Read the weights from the JSON file
            with open(filepath, 'r') as file:
                try:
                    weights = json.load(file)
                except json.JSONDecodeError:
                    print(f"Error reading JSON file: {filepath}")
                    continue

            # Convert weights from lists to tensors
            weights = [torch.tensor(w, dtype=torch.float32) for w in weights]

            # Simulate the robot with the given weights
            print(f"Simulating robot with weights from file: {filename}")
            brain = NeuralController(robot_structure.shape[0] * robot_structure.shape[1], connectivity.shape[0])
            set_weights(brain, weights)  # Load weights into the neural controller

            env = gym.make(scenario, max_episode_steps=steps, body=robot_structure, connections=connectivity)
            sim = env.sim
            viewer = EvoViewer(sim)
            viewer.track_objects('robot')

            state = env.reset()[0]  # Get initial state

            action_size = sim.get_dim_action_space('robot')  # Get correct action size
            frames = []

            for t in range(steps):
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # Convert to tensor
                action = brain(state_tensor).detach().numpy().flatten()  # Get action
                

                state, reward, terminated, truncated, info = env.step(action)
                
                frame = viewer.render('rgb_array')
                frames.append(frame)
                
                if terminated or truncated:
                    env.reset()
                    break

            # Save the simulation as a GIF
            gif_filename = os.path.splitext(filename)[0] + ".gif"
            gif_filepath = os.path.join(gif_directory, gif_filename)
            imageio.mimsave(gif_filepath, frames, duration=duration, optimize=True)
            print(f"Saved simulation as GIF: {gif_filepath}")

            viewer.close()
            env.close()


process_weights_and_simulate("students/results/de/DownStepper-v0/", robot_structure, connectivity, "DownStepper-v0",500,"students/results/de/DownStepper-v0/")