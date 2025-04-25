import os
import subprocess

def install_requirements():
    """Installs required Python packages and ensures correct environment setup."""
    packages = [
        "torch==2.5.1",  # PyTorch for neural networks
        "numpy==1.23.5",  # Numerical computing
        "gymnasium==0.28",  # OpenAI Gym for RL environments
        "evogym==2.0.0",  # EvoGym for evolutionary robotics
        "imageio==2.36.1"
    ]
    
    print("Installing required packages...")
    for package in packages:
        subprocess.run(["pip", "install", package], check=True)
    
    print("All packages installed successfully.")

if __name__ == "__main__":
    install_requirements()
