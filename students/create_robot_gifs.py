import os
import json
from GA_utils import simulate_and_save
from fixed_controllers import *
import gymnasium as gym
from evogym.envs import *
from evogym import EvoViewer, get_full_connectivity
from neural_controller import *
import os
import matplotlib.pyplot as plt
import torch
import imageio



def process_robot_structures(directory, scenario, steps, duration=0.066):
    """
    Loops through JSON files in a directory, reads robot structures, and simulates and saves them as GIFs.

    Args:
        directory (str): The directory containing JSON files with robot structures.
        scenario (str): The simulation scenario (e.g., 'DownStepper-v0').
        steps (int): The number of simulation steps.
        controller (function): The controller function for the robots.
    """
    # Ensure the directory exists
    if not os.path.exists(directory):
        print(f"Directory '{directory}' does not exist.")
        return

    for i in range(1,6):
        robot_filename = os.path.join(directory, f"best_robot_run_{i}.json")
        controller_filename = os.path.join(directory, f"best_controller_run_{i}.json")

        # Read the robot structure from the JSON file
        with open(robot_filename, 'r') as file:
            try:
                robot_structure = np.array(json.load(file))
            except json.JSONDecodeError:
                print(f"Error reading JSON file: {robot_filename}")
                continue

        connectivity = get_full_connectivity(robot_structure)

        # Read the weights from the JSON file
        with open(controller_filename, 'r') as file:
            try:
                weights = json.load(file)
            except json.JSONDecodeError:
                print(f"Error reading JSON file: {controller_filename}")
                continue

        # Convert weights from lists to tensors
        weights = [torch.tensor(w, dtype=torch.float32) for w in weights]

        # Simulate the robot with the given weights
        print(f"Simulating robot with weights from file: {controller_filename}")
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
        gif_filename = f"gif_{i}.gif"
        gif_filepath = os.path.join(directory, gif_filename)
        imageio.mimsave(gif_filepath, frames, duration=duration, optimize=True)
        print(f"Saved simulation as GIF: {gif_filepath}")

        viewer.close()
        env.close()



process_robot_structures("results/co_evolution/tournament/CaveCrawler-v0/", "CaveCrawler-v0", 500)