import os
import numpy as np
import matplotlib.pyplot as plt

def plot_average_fitness_evolution(directory, n_runs, threshold, output_filename="average_fitness_plot.png"):
    """
    Processes fitness evolution files, computes the average and standard deviation across runs,
    and plots the average fitness evolution with ±std and a fixed threshold.

    Args:
        directory (str): The directory containing fitness evolution files (fitness_run_{n}.csv).
        n_runs (int): The number of runs to process.
        threshold (float): The fixed threshold to plot as a red line.
        output_filename (str): The filename to save the resulting plot (default: "average_fitness_plot.png").
    """
    all_fitnesses = []

    # Loop through all runs and collect fitness evolution data
    for run in range(1, n_runs + 1):
        filepath = os.path.join(directory, f"fitness_run_{run}.csv")
        if not os.path.exists(filepath):
            print(f"File not found: {filepath}")
            continue

        # Read fitness values from the CSV file
        fitness_values = []
        with open(filepath, 'r') as file:
            next(file)  # Skip the header
            for line in file:
                _, fitness = line.strip().split(',')
                fitness_values.append(float(fitness))

        all_fitnesses.append(fitness_values)

    # Ensure we have data to process
    if len(all_fitnesses) == 0:
        print("No fitness data found. Exiting.")
        return

    # Convert to a NumPy array for easier computation
    all_fitnesses = np.array(all_fitnesses)

    # Compute average and standard deviation across runs
    avg_fitness = np.mean(all_fitnesses, axis=0)
    std_fitness = np.std(all_fitnesses, axis=0)

    # Plot the results
    generations = np.arange(1, len(avg_fitness) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(generations, avg_fitness, label="Average Fitness", color="blue")
    plt.fill_between(generations, avg_fitness - std_fitness, avg_fitness + std_fitness, color="blue", alpha=0.2, label="±1 Std Dev")
    plt.axhline(y=threshold, color="red", linestyle="--", label=f"Threshold = {threshold}")

    # Add labels, legend, and grid
    plt.title("Average Fitness Evolution Across Runs")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)

    # Save the plot
    output_path = os.path.join(directory, output_filename)
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")

plot_average_fitness_evolution("results/de/ObstacleTraverser-v0/", 5, 2.2164728042066884)