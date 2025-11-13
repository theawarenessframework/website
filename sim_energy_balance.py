# sim_energy_balance.py (L2)

import pandas as pd
import numpy as np
import argparse
import sys
import os

INPUT_FILE_DEFAULT = "core_timeseries_single.csv" 
OUTPUT_FILE = "energy_metrics.csv"

def compute_thermodynamics(df_core):
    """
    Computes Energy (E) and Entropy (S) from core time-series.
    Groups by each individual run to get accurate final-state values.
    """
    # Group by each individual run (lambda_A0 and init_state)
    grouped = df_core.groupby(["lambda_A0", "init_O", "init_M", "init_P"])
    
    all_metrics = []
    
    for name, group in grouped:
        O, M, P = group["O"].values, group["M"].values, group["P"].values
        
        Energy = O**2 + M**2 + P**2
        
        # Placeholder for Entropy
        S = np.zeros_like(O); mask = (O > 0) & (M > 0) & (P > 0)
        S[mask] = - (O[mask] * np.log(O[mask]) + M[mask] * np.log(M[mask]) + P[mask] * np.log(P[mask]))
        
        metrics = {
            "lambda_A0": name[0],
            "init_O": name[1],
            "init_M": name[2],
            "init_P": name[3],
            "E_mean": np.mean(Energy),
            "S_mean": np.mean(S[mask]), # Only average valid entropy values
            "O_final": O[-1],
            "M_final": M[-1],
            "P_final": P[-1]
        }
        all_metrics.append(metrics)
        
    df_metrics = pd.DataFrame(all_metrics)
    return df_metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="L2: Resource/Energy Balance (E=O^2+M^2+P^2).")
    parser.add_argument(
        "--input_file", type=str, default=None,
        help=f"Input file from L1 (e.g., core_timeseries_sweep.csv). If None, runs L1 default case internally."
    )
    args = parser.parse_args()

    if args.input_file and os.path.exists(args.input_file):
        df_core = pd.read_csv(args.input_file)
        print(f"L2 (Chained Mode): Loaded L1 data from {args.input_file}")
    else:
        print(f"L2 (Independent Mode): Running L1 default test case to generate input.")
        try:
            # We import the L1 *function* not the script
            from sim_core_dynamics import run_l1_simulation
            from framework_utils import DEFAULT_INIT_STATE, DEFAULT_LAMBDA
        except ImportError:
            print("Error: Could not import dependencies. Ensure all files are in the same directory.")
            sys.exit(1)
            
        run_l1_simulation(DEFAULT_LAMBDA, DEFAULT_INIT_STATE, INPUT_FILE_DEFAULT) 
        df_core = pd.read_csv(INPUT_FILE_DEFAULT)

    # --- Run L2 Computation ---
    df_energy = compute_thermodynamics(df_core)
    
    df_energy.to_csv(OUTPUT_FILE, index=False)
    print(f"✅ L2 Computation Complete. Energy metrics saved to {OUTPUT_FILE}")