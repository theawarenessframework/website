# sim_core_dynamics.py (L1)

import pandas as pd
import numpy as np
import argparse
from itertools import product
from tqdm import tqdm

# Import the core solver from the utility file
try:
    from framework_utils import solve_core_dynamics, DEFAULT_INIT_STATE, DEFAULT_LAMBDA
except ImportError:
    print("Error: framework_utils.py not found. Please ensure it's in the same directory.")
    exit(1)

OUTPUT_FILE_SINGLE = "core_timeseries_single.csv"
OUTPUT_FILE_SWEEP = "core_timeseries_sweep.csv"

def run_l1_simulation(lambda_A0, init_state, output_file):
    """Runs a single simulation and saves the full time-series."""
    t, O, M, P, R = solve_core_dynamics(init_state, lambda_A0)

    df_out = pd.DataFrame({
        "time": t, "O": O, "M": M, "P": P, "R": R,
        "lambda_A0": lambda_A0, "init_O": init_state[0], "init_M": init_state[1], "init_P": init_state[2]
    })

    df_out.to_csv(output_file, index=False)
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="L1: Core Dynamics Simulation (R=O*M*P).")
    parser.add_argument(
        "--lam", type=float, default=DEFAULT_LAMBDA,
        help="Specific lambda_A0 value for independent test run."
    )
    parser.add_argument(
        "--sweep", action="store_true",
        help="Run the full parameter sweep instead of a single test case."
    )
    args = parser.parse_args()

    if args.sweep:
        print("L1: Running Full Parameter Sweep...")
        # *** UPDATED to your 8 initial states ***
        all_initial_states = [
            (0.0, 0.0, 0.0),
            (25.0, 25.0, 25.0),
            (0.0, 25.0, 50.0),
            (50.0, 50.0, 50.0),
            (25.0, 50.0, 75.0),
            (75.0, 75.0, 75.0),
            (50.0, 75.0, 100.0),
            (100.0, 100.0, 100.0)
        ]
        all_lambda_values = np.arange(0.1, 0.9, 0.2) # 0.1, 0.3, 0.5, 0.7
        
        all_results = []
        total_cases = len(all_initial_states) * len(all_lambda_values)
        
        for init, lamb in tqdm(list(product(all_initial_states, all_lambda_values)), total=total_cases):
            t, O, M, P, R = solve_core_dynamics(init, lamb)
            df_case = pd.DataFrame({
                "time": t, "O": O, "M": M, "P": P, "R": R, 
                "lambda_A0": lamb, "init_O": init[0], "init_M": init[1], "init_P": init[2]
            })
            all_results.append(df_case)

        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(OUTPUT_FILE_SWEEP, index=False)
        print(f"✅ L1 Sweep Complete. Data saved to {OUTPUT_FILE_SWEEP}")
    else:
        print(f"L1: Running Independent Test Case (λ={args.lam})") 
        run_l1_simulation(args.lam, DEFAULT_INIT_STATE, OUTPUT_FILE_SINGLE)
        print(f"✅ L1 Independent Test Complete. Data saved to {OUTPUT_FILE_SINGLE}")