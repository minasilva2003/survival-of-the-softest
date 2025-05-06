import os
import numpy as np
import matplotlib.pyplot as plt

def plot_average_fitness_evolution(directory1, directory2, final_directory, algorithm1, algorithm2, n_runs, threshold, output_filename="average_fitness_plot.png"):
    """
    Processes fitness evolution files, computes the average and standard deviation across runs,
    and plots the average fitness evolution with ±std and a fixed threshold.

    Args:
        directory (str): The directory containing fitness evolution files (fitness_run_{n}.csv).
        n_runs (int): The number of runs to process.
        threshold (float): The fixed threshold to plot as a red line.
        output_filename (str): The filename to save the resulting plot (default: "average_fitness_plot.png").
    """
    all_fitnesses1 = []
    all_fitnesses2 = []

    # Loop through all runs and collect fitness evolution data
    for run in range(1, n_runs + 1):
        filepath1 = os.path.join(directory1, f"best_fit_run_{run}.csv")
        if not os.path.exists(filepath1):
            print(f"File not found: {filepath1}")
            continue

        filepath2 = os.path.join(directory2, f"best_fit_run_{run}.csv")
        if not os.path.exists(filepath2):
            print(f"File not found: {filepath2}")
            continue


        # Read fitness values from the CSV file
        fitness_values1 = []
        fitness_values2 = []
        with open(filepath1, 'r') as file:
            next(file)  # Skip the header
            for line in file:
                _, fitness = line.strip().split(',')
                fitness_values1.append(float(fitness))

        with open(filepath2, 'r') as file:
            next(file)  # Skip the header
            for line in file:
                _, fitness = line.strip().split(',')
                fitness_values2.append(float(fitness))

        all_fitnesses1.append(fitness_values1)
        all_fitnesses2.append(fitness_values2)

    # Ensure we have data to process
    if len(all_fitnesses1) == 0 or len(all_fitnesses2) == 0:
        print("No fitness data found. Exiting.")
        return

    # Convert to a NumPy array for easier computation
    all_fitnesses1 = np.array(all_fitnesses1)
    all_fitnesses2 = np.array(all_fitnesses2)


    # Compute average and standard deviation across runs
    avg_fitness1 = np.mean(all_fitnesses1, axis=0)
    std_fitness1 = np.std(all_fitnesses1, axis=0)
    avg_fitness2 = np.mean(all_fitnesses2, axis=0)
    std_fitness2 = np.std(all_fitnesses2, axis=0)

    # Plot the results
    generations = np.arange(1, len(avg_fitness1) + 1)
    plt.figure(figsize=(10, 6))
    plt.plot(generations, avg_fitness1, label=f"Average Best Fitness {algorithm1}", color="blue")
    plt.fill_between(generations, avg_fitness1 - std_fitness1, avg_fitness1 + std_fitness1, color="blue", alpha=0.2, label=f"± Std {algorithm1}")
    plt.plot(generations, avg_fitness2, label=f"Average Best Fitness {algorithm2}", color="green")
    plt.fill_between(generations, avg_fitness2 - std_fitness2, avg_fitness2 + std_fitness2, color="green", alpha=0.2, label=f"± Std {algorithm2}")
    plt.axhline(y=threshold, color="red", linestyle="--", label=f"Threshold = {threshold}")

    # Add labels, legend, and grid
    plt.title("Average Best Fitness Evolution Across Runs")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.legend()
    plt.grid(True)

    # Save the plot
    output_path = os.path.join(final_directory, output_filename)
    plt.savefig(output_path)
    plt.close()
    print(f"Plot saved to {output_path}")

plot_average_fitness_evolution("results/co_evolution/robin/CaveCrawler-v0/",
                               "results/co_evolution/tournament/CaveCrawler-v0/",
                               "results/co_evolution/",
                               "robin",
                               "tournament",
                               5,
                               2.86138843004549,
                               "coevolution_cave_crawler")

