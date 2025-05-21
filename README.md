# README

This project is for the quantum computing simulation by the user customized noise model.


## Qiskit Noise Model

The Qiskit Aer device noise model automatically generates a simplified noise model for a real device. This model is generated using the calibration information reported in the properties of a device and takes into account.

However, we propose the simulation method to directly tune the noise model with the calibration data.

The table shows the errors defined in `qiskit_aer.noise.errors` for noise models.

|        Error Name        |                                                                    Definition                                                                    | Error Component |
|:------------------------:|:------------------------------------------------------------------------------------------------------------------------------------------------:|:---------------:|
| readout_error            | A error occurs when measurement                                                                                                                  |                 |
| depolarizing_error       | An n-qubit depolarizing error channel parameterized by a depolarization probability p                                                            | Gate Error      |
| thermal_relaxation_error | A single qubit thermal relaxation channel parameterized by relaxation time constants T1, T2, gate time t, and excited state thermal population p | T1, T2          |
| phase_dampling_error     | A single-qubit phase damping error channel given by a phase damping parameter r                                                                  |                 |
| pauli_error              | An n-qubit Pauli error channel (mixed unitary) gives as a list of Pauli's and probabilities                                                      | Gate Error      |


In addition to the errors mentioned in the table above, there are several additional errors.

## Process

The code execution process is simple.

1. Input the calibration data of the QPU.
2. Create a random empty noise model, and then add errors based on the calibration data to the noise model.
3. Run the simulation using the generated noise model (the backend can be freely set)

## Issue
- [ ] Modify the error_model according to the calibration data
- [ ] Check the fidelity of simulation result

## Reference
The Qiskit API reference for the Noise Model

* <b> https://qiskit.github.io/qiskit-aer/apidocs/aer_noise.html#module-qiskit_aer.noise </b>

* [Quantum Errors] https://docs.quantum.ibm.com/guides/build-noise-models#2-specific-qubit-quantum-error

* [FakeProvider] https://docs.quantum.ibm.com/api/qiskit-ibm-runtime/fake-provider 
