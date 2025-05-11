import os
import csv
import matplotlib.pyplot as plt
import numpy as np
import json

# Function to create a directory if it doesn't exist
def create_directory(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

# Function to write fitness data to a CSV file
def write_to_csv_file(filename, data):

    with open(filename, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['N', 'Fitness'])

        for i in range(1,len(data)+1):
            writer.writerow([i, data[i-1]])


# Function to plot fitness data           
def plot_graph(data, n, x_label, y_label, label, title, filename):
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, n + 1), data, marker='o', label=label)
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.grid(True)
    plt.legend()
    plt.savefig(filename)
    plt.close()
    

def average_csv_files(directory, prefix, output_file):
    all_data = []

    # Read all CSV files in the directory
    for filename in os.listdir(directory):
        if filename.startswith(prefix) and filename.endswith('.csv'):
            file_path = os.path.join(directory, filename)
            with open(file_path, mode='r', encoding='utf-8') as csv_file:
                reader = csv.reader(csv_file)
                next(reader)  # Skip the header row
                file_data = [float(row[1]) for row in reader]  # Assuming the second column contains the values
                all_data.append(file_data)

    # Ensure all files have the same number of rows
    min_length = min(len(data) for data in all_data)
    all_data = [data[:min_length] for data in all_data]

    # Calculate the average for each row
    averaged_data = np.mean(all_data, axis=0)

    # Save the averaged results to a new CSV file
    output_path = os.path.join(directory, output_file)
    with open(output_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(['Row', 'Average'])  # Header row
        for i, value in enumerate(averaged_data, start=1):
            writer.writerow([i, value])

    plot_graph(averaged_data, len(averaged_data), 'Generation', 'Average Best Fitness', 'Average Best Fitness', f'Average Best Fitness Per Generation', os.path.join(directory, 'average_plot.png'))



def save_best_robot(filename, robot_structure):

    with open(filename, mode='w', encoding='utf-8') as f:
        if isinstance(robot_structure, np.ndarray):
            robot_structure = robot_structure.tolist()  # Convert to list if it's a NumPy array
        json.dump(robot_structure, f, indent=4)
    print(f"Best robot structure saved to {filename}")



def save_best_controller(weights, filename):

    # Convert weights to a JSON-serializable format (lists)
    weights_serializable = [w.tolist() for w in weights]

    # Save to a JSON file
    with open(filename, mode='w', encoding='utf-8') as f:
        json.dump(weights_serializable, f, indent=4)
    print(f"Best controller saved to {filename}")