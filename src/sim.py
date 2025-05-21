"""
This module contains the function for simulation of the noise model.
"""

import os
import re
import pandas as pd
import numpy as np

from qiskit.circuit.random import random_circuit
from qiskit_ibm_runtime.fake_provider import (
    FakeBrisbane,
    FakeSherbrooke,
)
from qiskit_aer.noise import (
    NoiseModel,
    QuantumError,
    ReadoutError,
    pauli_error,
    depolarizing_error,
    thermal_relaxation_error,
)
from qiskit_aer import AerSimulator
from util import *


def load_calibration_data(calibration_file: str) -> pd.DataFrame:
    """
    Load calibration data from a file.
    This function reads the calibration data from the specified file path and returns it as a DataFrame.
    """
    file_path = os.path.join(os.path.dirname(__file__), calibration_file)

    # Check if the file exists
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Calibration file not found: {file_path}")

    # Load the calibration data
    calibration_data = pd.read_csv(file_path)

    return calibration_data


def create_noise_model_from_calibration(data: pd.DataFrame) -> tuple[NoiseModel, list]:
    """
    This function generates a noise model based on the provided parameters.
    Such as T1, T2, and gate errors.
    @param data: DataFrame containing the calibration data.
    @return: NoiseModel object, operational qubits
    """

    # Make the empty noise model
    noise_model = NoiseModel()

    # Tracking an operational qubits
    operational_qubits = []

    # NOTE: the gate time is tunable
    AVG_GATE_TIME = np.mean(extract_gate_times(data["Gate time (ns)"]))

    operational_qubits = []
    for _, row in data.iterrows():
        qubit_idx = int(row["Qubit"])

        # Check if the qubit is operational
        if isinstance(row["Operational"], bool):
            operational = row["Operational"]
        else:
            operational = str(row["Operational"]).lower() == "true"

        if operational:
            operational_qubits.append(qubit_idx)

    qubit_data = {}
    ecr_connections = {}

    for _, row in data.iterrows():
        qubit_idx = int(row["Qubit"])

        # Skip if the qubit is not operational
        if qubit_idx not in operational_qubits:
            continue

        # T1/T2 relaxation errors (in microseconds, convert to nanoseconds)
        t1 = row["T1"] * 1e-3 if not pd.isna(row["T1"]) else None
        t2 = row["T2"] * 1e-3 if not pd.isna(row["T2"]) else None

        qubit_data[qubit_idx] = {"T1": t1, "T2": t2}

        ######################
        # Add readout errors #
        ######################
        if not pd.isna(row["prob_meas0_prep1"]) and not pd.isna(
            row["prob_meas1_prep0"]
        ):
            # Probabilities of measureing 0 when prepared in 0, and 1 when preapred in 1
            p_meas0_prep0 = 1.0 - row["prob_meas1_prep0"]
            p_meas1_prep1 = 1.0 - row["prob_meas0_prep1"]

            # Create readout error
            readout_error = ReadoutError(
                [[p_meas0_prep0, 1 - p_meas0_prep0], [p_meas1_prep1, 1 - p_meas1_prep1]]
            )
            noise_model.add_readout_error(readout_error, [qubit_idx])

        ###################
        # Add gate errors #
        ###################

        # 1. ID error (thermal relaxation errors)
        if not pd.isna(row["ID error"]):
            if t1 is not None and t2 is not None:
                try:
                    noise_model.add_quantum_error(
                        thermal_relaxation_error(t1, t2, AVG_GATE_TIME),
                        "id",
                        [qubit_idx],
                    )
                except Exception as e:
                    # print(f"Warning: Could not add thermal relaxation error for qubit {qubit_idx}: {e}")
                    continue

        # 2. SX gate error (depolarizing error) without T1/T2
        if not pd.isna(row["sx error"]):
            # sx_error = depolarizing_error(row['√x (sx) error'], 1)
            noise_model.add_quantum_error(
                depolarizing_error(row["sx error"], 1), "sx", [qubit_idx]
            )

        # 3. X gate error
        if not pd.isna(row["Pauli-X error"]):
            # x_error = depolarizing_error(row['x (x) error'], 1)
            noise_model.add_quantum_error(
                depolarizing_error(row["Pauli-X error"], 1), "x", [qubit_idx]
            )

        # 4. RZ gate error
        if (
            not pd.isna(row["Z-axis rotation (rz) error"])
            and row["Z-axis rotation (rz) error"] > 0
        ):
            # rz_error = depolarizing_error(row['rz (rz) error'], 1)
            noise_model.add_quantum_error(
                depolarizing_error(row["Z-axis rotation (rz) error"], 1),
                "rz",
                [qubit_idx],
            )

        # 5. ECR gate error
        if not pd.isna(row["ECR error"]) and row["ECR error"]:
            # Parse gate times if available
            gate_times = {}
            if not pd.isna(row["Gate time (ns)"]) and row["Gate time (ns)"]:
                for gt_entry in row["Gate time (ns)"].split(";"):
                    if not gt_entry or ":" not in gt_entry:
                        continue
                    gt_connection, gt_value = gt_entry.split(":")
                    gate_times[gt_connection] = (
                        float(gt_value) * 1e-9
                    )  # convert ns to seconds

            # Parse ECR errors
            ecr_entries = row["ECR error"].split(";")
            for entry in ecr_entries:
                if not entry or ":" not in entry:
                    continue

                connection, error_rate = entry.split(":")
                target_qubit = int(connection.split("_")[1])

                # Skip if target qubit not operational
                if target_qubit not in operational_qubits:
                    continue

                # Store this connection for later processing
                ecr_connections[(qubit_idx, target_qubit)] = {
                    "error_rate": float(error_rate),
                    "gate_time": gate_times.get(connection, 660e-9),  # default: 660ns
                }

    # Process two-qubit gate errors using stored data
    for (control, target), conn_data in ecr_connections.items():
        error_rate = conn_data["error_rate"]
        gate_time = conn_data["gate_time"]

        # TODO: Need clarification on how to handle this
        # Add thermal relaxation error if T1/T2 available for both qubits
        # if (control in qubit_data and target in qubit_data and
        #     qubit_data[control]['T1'] is not None and qubit_data[control]['T2'] is not None and
        #     qubit_data[target]['T1'] is not None and qubit_data[target]['T2'] is not None):

        # For multi-qubit thermal_relaxation_error, we need to adapt our custom function
        # Here we would handle the tensor product of single-qubit errors
        # For now, we'll use an approximation by adding separate errors

        # Control qubit thermal relaxation
        # t1t2_control = thermal_relaxation_error(
        #     qubit_data[control]['T1'],
        #     qubit_data[control]['T2'],
        #     gate_time
        # )
        # noise_model.add_quantum_error(t1t2_control, 'cx', [control, target])
        # noise_model.add_quantum_error(t1t2_control, 'ecr', [control, target], [0])

        # # Target qubit thermal relaxation
        # t1t2_target = thermal_relaxation_error(
        #     qubit_data[target]['T1'],
        #     qubit_data[target]['T2'],
        #     gate_time
        # )
        # noise_model.add_quantum_error(t1t2_target, 'cx', [control, target], [1])
        # noise_model.add_quantum_error(t1t2_target, 'ecr', [control, target], [1])

        # Add depolarizing error component
        depol_error = depolarizing_error(error_rate, 2)  # 2-qubit gate
        noise_model.add_quantum_error(depol_error, "cx", [control, target])
        noise_model.add_quantum_error(depol_error, "ecr", [control, target])

    return noise_model, operational_qubits


def simulate_noise_model(calibration_file: str, num_qubits: int):
    """
    Simulate the noise model.
    This function simulates the noise model based on the provided parameters.
    It generates a set of simulated data points and returns them as a list.
    """
    ################
    # Main Process #
    ################

    # Example of using the noise model in a Qiskit simulation

    # 1. Create a random circuit
    qc = random_circuit(num_qubits=num_qubits, depth=3, measure=True)

    # 2. Get the Fake Provider backend properties
    fake_backend = FakeBrisbane()
    print(f"Fake Backend Name: {fake_backend.backend_name}")

    # 3. Extract the noise model
    basis_noise_model = NoiseModel.from_backend(fake_backend)

    # 4. Create a noise model with calibration data
    # Load the calibration data
    calibration_data = load_calibration_data(calibration_file)

    # Create a calibrated noise model
    calibrated_noise_model, operational_qubits = create_noise_model_from_calibration(
        calibration_data
    )

    # 5. Transpile the circuit based on the Fake backend
    from qiskit import transpile

    transpiled_circuit = transpile(qc, backend=fake_backend)

    # 6. Construct the Aer simulator
    simulator = AerSimulator(
        noise_model=calibrated_noise_model,
        coupling_map=fake_backend.configuration().coupling_map,
        basis_gates=fake_backend.configuration().basis_gates,
    )

    # 7. Get the counts and plot the histogram
    # get_qubit_counts(simulator, transpiled_circuit, num_qubits)

    # 8. Visualize the noise impact
    visualize_noise_impact(simulator, transpiled_circuit, calibrated_noise_model, num_qubits)
