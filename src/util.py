"""
This module contains utility functions for the project.
It includes functions for loading and saving data, as well as for processing and transforming data.
"""

import os
import pandas as pd
import numpy as np
import re
import matplotlib.pyplot as plt
from qiskit.visualization import plot_histogram


def extract_gate_times(series):
    times = []
    for entry in series.dropna():
        for pair in entry.split(';'):
            match = re.match(r"\d+_\d+:(\d+)", pair)
            if match:
                times.append(int(match[1]))
    return times

def calculate_fidelity(counts_ideal, counts_noisy):
    """
    Calculate a simple fidelity measure between two count dictionaries
    (classical fidelity between two probability distributions)
    """
    # Get all possible bit strings
    all_bitstrings = set(counts_ideal.keys()) | set(counts_noisy.keys())
    
    # Calculate total shots
    total_ideal = sum(counts_ideal.values())
    total_noisy = sum(counts_noisy.values())
    
    # Calculate classical fidelity
    fidelity = 0
    for bitstring in all_bitstrings:
        p_ideal = counts_ideal.get(bitstring, 0) / total_ideal
        p_noisy = counts_noisy.get(bitstring, 0) / total_noisy
        fidelity += np.sqrt(p_ideal * p_noisy)
    
    return fidelity

def get_qubit_counts(simulator, transpiled_circuit, num_qubits):
    result_noise = simulator.run(transpiled_circuit).result()
    counts_noise = result_noise.get_counts(0)  # Get counts for the first circuit
    plot_histogram(
        counts_noise,
        title=f"Counts for {num_qubits}-qubit GHZ state with device noise model",
    )
    plt.show()


def visualize_noise_impact(simulator, transpiled_circuit, noise_model, num_qubits=2, shots=1024):
    """
    Visualize the impact of noise on quantum circuit execution
    
    Args:
        calibration_file: Path to IBM calibration data CSV file
        circuit_type: Type of test circuit ('bell', 'ghz', or 'qft')
        num_qubits: Number of qubits to use in the circuit
        shots: Number of shots for the simulation
    """
    
    # Run non-customized noisy simulation
    job_ideal = simulator.run(transpiled_circuit, shots=shots)
    result_ideal = job_ideal.result()
    counts_ideal = result_ideal.get_counts()
    
    # Run customized noisy simulation
    job_noisy = simulator.run(transpiled_circuit, noise_model=noise_model, shots=shots)
    result_noisy = job_noisy.result()
    counts_noisy = result_noisy.get_counts()
    
    # Calculate fidelity (simple version - just compare probability distributions)
    fidelity = calculate_fidelity(counts_ideal, counts_noisy)
    
    # Plot results
    fig, axs = plt.subplots(1, 2, figsize=(15, 6))
    plot_histogram(counts_ideal, ax=axs[0], title="Non-Customized Results")
    plot_histogram(counts_noisy, ax=axs[1], title=f"Customized Noisy Results (Fidelity: {fidelity:.4f})")
    plt.tight_layout()
    
    # Show selected qubits properties
    print(f"Simulation results for {num_qubits}-qubit circuit:")
    print(f"Selected qubits: {num_qubits}")
    print(f"Simulation fidelity: {fidelity:.4f}")
    
    plt.show()



def save_simulation_results(results, output_path):
    """
    Save simulation results to a file.
    This function saves the provided simulation results to the specified output path.
    """
    # Check if the output directory exists
    output_dir = os.path.dirname(output_path)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Save the results to a CSV file
    results.to_csv(output_path, index=False)