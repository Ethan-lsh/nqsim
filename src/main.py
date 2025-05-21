import argparse

import sim
from util import save_simulation_results

parser = argparse.ArgumentParser(description="A simple argument parser.")
parser.add_argument(
    "-c",
    "--calibration",
    type=str,
    required=True,
    dest="calibration_file",
    help="calibration file path.",
)
parser.add_argument(
    "-o",
    "--output",
    type=str,
    required=False,
    default="output.txt",
    help="Output file path.",
)
parser.add_argument(
   "-n",
   "--num_qubits",
   type=int,
   required=True,
   default=3,
   dest="num_qubits",
   help="Number of qubits to simulate.",
)

args = parser.parse_args()



if __name__ == "__main__":
   print("Executing main.py")
   print(f"Number of qubits: {args.num_qubits}")
   print(f"Calibration file: {args.calibration_file}")
   print(f"Output file: {args.output}")
   print("=========================")

   # Execute the simulation
   print("[ Simulation options ]")
   result = sim.simulate_noise_model(args.calibration_file, args.num_qubits)

#    save_simulation_results(result, args.output)


# %%
