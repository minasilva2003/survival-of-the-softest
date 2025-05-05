import csv
import numpy as np

def compute_fitness_stats(filepath):
    """
    Reads a CSV file containing fitness values and computes the average and standard deviation.

    Args:
        filepath (str): The path to the CSV file.

    Returns:
        None
    """
    fitness_values = []

    # Read the CSV file
    with open(filepath, 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            fitness_values.append(float(row[1]))  # Read the "Best Fitness" column

    # Compute average and standard deviation
    avg_fitness = np.mean(fitness_values)
    std_fitness = np.std(fitness_values)

    # Print the results
    print(f"Average Fitness: {avg_fitness:.4f}")
    print(f"Standard Deviation: {std_fitness:.4f}")

compute_fitness_stats("students/results_more/genetic_algorithm/always_connected/BridgeWalker-v0/walking/all_runs_best_fit.csv")