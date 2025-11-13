# sim_spatial_field.py (L3)

import numpy as np
import argparse
import sys
import os
import pandas as pd

INPUT_FILE_DEFAULT = "energy_metrics.csv"
OUTPUT_FILE = "spatial_state.npy"

def laplacian_2d(field):
    """
    Calculates the discrete 2D Laplacian of a field using the 5-point stencil.
    """
    lap = np.roll(field, 1, axis=0) + np.roll(field, -1, axis=0) + \
          np.roll(field, 1, axis=1) + np.roll(field, -1, axis=1) - 4 * field
    return lap

def run_spatial_simulation(df_energy):
    """
    Simulates spatial coupling (L3) using a 2D Diffusion system.
    """
    print("\nL3: Simulating Spatial Field (2D Diffusion Update)...")
    
    GRID_SIZE = 50 
    TIME_STEPS = 100
    DIFFUSION_RATE = 0.05
    DT = 0.1
    
    # Use the first case's final state for L3 initialization
    base_state = df_energy.iloc[0][['O_final', 'M_final', 'P_final']].values 
    
    spatial_state = np.zeros((3, GRID_SIZE, GRID_SIZE))
    noise = np.random.randn(GRID_SIZE, GRID_SIZE) * 0.01
    
    for i in range(3): spatial_state[i, :, :] = base_state[i] + noise
    spatial_state[spatial_state < 0] = 0.0

    spatial_history = [spatial_state.copy()]
    
    for t in range(TIME_STEPS):
        O, M, P = spatial_state[0], spatial_state[1], spatial_state[2]
        
        lap_O, lap_M, lap_P = laplacian_2d(O), laplacian_2d(M), laplacian_2d(P)
        
        dO_dt = DIFFUSION_RATE * lap_O
        dM_dt = DIFFUSION_RATE * lap_M
        dP_dt = DIFFUSION_RATE * lap_P
        
        new_O = O + DT * dO_dt
        new_M = M + DT * dM_dt
        new_P = P + DT * dP_dt
        
        spatial_state[0] = np.maximum(new_O, 0.0)
        spatial_state[1] = np.maximum(new_M, 0.0)
        spatial_state[2] = np.maximum(new_P, 0.0)

        spatial_history.append(spatial_state.copy())

    spatial_history_array = np.array(spatial_history)
    
    np.save(OUTPUT_FILE, spatial_history_array)
    print(f"L3 Simulation finished. Output array shape: {spatial_history_array.shape}")
    return spatial_history_array # Return the array for plotting

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="L3: Spatial Coupling (Network of ROMP triads).")
    parser.add_argument(
        "--input_file", type=str, default=None,
        help=f"Input file from L2 ({INPUT_FILE_DEFAULT}). If None, runs L2 default case internally."
    )
    args = parser.parse_args()

    if args.input_file and os.path.exists(args.input_file):
        df_energy = pd.read_csv(args.input_file)
        print(f"L3 (Chained Mode): Loaded L2 data from {args.input_file}")
    else:
        print(f"L3 (Independent Mode): Running L1 and L2 default test cases to generate input.")
        try:
            from sim_energy_balance import compute_thermodynamics
            from sim_core_dynamics import run_l1_simulation
            from framework_utils import DEFAULT_INIT_STATE, DEFAULT_LAMBDA
        except ImportError:
            print("Error: Could not import dependency scripts.")
            sys.exit(1)
            
        run_l1_simulation(DEFAULT_LAMBDA, DEFAULT_INIT_STATE, "temp_l1.csv") 
        df_core_temp = pd.read_csv("temp_l1.csv")
        df_energy = compute_thermodynamics(df_core_temp)
        os.remove("temp_l1.csv") # Clean up

    run_spatial_simulation(df_energy)
    print(f"✅ L3 Simulation Complete. Spatial field saved to {OUTPUT_FILE}")