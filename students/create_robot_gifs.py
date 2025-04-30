import os
import json
from GA_utils import simulate_and_save
from fixed_controllers import *

def process_robot_structures(directory, scenario, steps, controller):
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

    # Loop through all JSON files in the directory
    for filename in os.listdir(directory):
        if filename.endswith(".json"):
            filepath = os.path.join(directory, filename)

            # Read the robot structure from the JSON file
            with open(filepath, 'r') as file:
                try:
                    robot_structure = np.array(json.load(file))
                except json.JSONDecodeError:
                    print(f"Error reading JSON file: {filepath}")
                    continue

            # Generate a GIF filename based on the JSON filename
            gif_filename = os.path.splitext(filename)[0] + ".gif"
            gif_filepath = os.path.join(directory, gif_filename)

            # Simulate and save the robot
            print(f"Processing file: {filename}")
            simulate_and_save(robot_structure, gif_filepath, scenario, steps, controller)




process_robot_structures("students/results/hyper_genetic_algorithm/Walker-v0/walking/", "Walker-v0", 500, alternating_gait)