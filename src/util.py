"""
This module contains utility functions for the project.
It includes functions for loading and saving data, as well as for processing and transforming data.
"""

import os
import pandas as pd
import numpy as np
import re

def extract_gate_times(series):
    times = []
    for entry in series.dropna():
        for pair in entry.split(';'):
            match = re.match(r"\d+_\d+:(\d+)", pair)
            if match:
                times.append(int(match[1]))
    return times

def check_difference(a, b):
    """
    Check if two values are different.
    This function compares two values and returns True if they are different, False otherwise.
    """
    return a != b

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