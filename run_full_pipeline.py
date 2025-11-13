# run_full_pipeline.py (Final, Robust Corrected Version)

import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from itertools import product
from scipy.integrate import solve_ivp
import seaborn as sns 

# --- Import required functions from modular files ---
try:
    from framework_utils import solve_core_dynamics
    from sim_energy_balance import compute_thermodynamics
    from sim_spatial_field import run_spatial_simulation
    from sim_temporal_hierarchy import analyze_temporal_hierarchy
except ImportError as e:
    print(f"FATAL ERROR: Could not import modular functions. Details: {e}")
    sys.exit(1)


# --- Global Configuration & Constants ---
OUTPUT_DIR = "romp_final_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)
L1_SWEEP_OUTPUT = "core_timeseries_sweep.csv"; L3_SPATIAL_OUTPUT = "spatial_state.npy"; 
L4_MOD_OUTPUT = "modulation_metrics.csv"; L2_METRICS_OUTPUT = "energy_metrics.csv"
INITIAL_STATES = [
    (0.0, 0.0, 0.0), (25.0, 25.0, 25.0), (0.0, 25.0, 50.0), (50.0, 50.0, 50.0), 
    (25.0, 50.0, 75.0), (75.0, 75.0, 75.0), (50.0, 75.0, 100.0), (100.0, 100.0, 100.0)
]
LAMBDA_VALUES = np.arange(0.1, 0.9, 0.2) # 0.1, 0.3, 0.5, 0.7


# --- Plotting Functions (Local Definitions with Final Fix) ---

def plot_l1_dynamics(df_l1):
    """Generates a plot of the L1 dynamics for the (25,25,25) test case."""
    plt.figure(figsize=(10, 6))
    
    # *** FINAL FIX: Use np.isclose for ALL float comparisons (lambda_A0, init_O, init_M, init_P) ***
    df_plot = df_l1[
        np.isclose(df_l1['lambda_A0'], 0.5) & 
        np.isclose(df_l1['init_O'], 25.0) & 
        np.isclose(df_l1['init_M'], 25.0) & 
        np.isclose(df_l1['init_P'], 25.0)
    ]
    
    if df_plot.empty: 
        print("Warning: Could not find L1 plot case (lambda=0.5, init=(25,25,25)). Skipping L1 plot.")
        return

    plt.plot(df_plot["time"], df_plot["O"], label="Observation (O)", linewidth=2)
    plt.plot(df_plot["time"], df_plot["M"], label="Memory (M)", linewidth=2)
    plt.plot(df_plot["time"], df_plot["P"], label="Pattern (P)", linewidth=2)
    plt.plot(df_plot["time"], df_plot["R"], label="Awareness (R = OMP)", linestyle='--', color='black', alpha=0.7)
    
    plt.title(r"L1: Core Dynamics Time-Series ($\lambda_{A0}=0.5$ / Init=(25,25,25))")
    plt.xlabel("Time (t)"); plt.ylabel("Component Magnitude"); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "L1_Core_Dynamics_Time_Series.png"), dpi=150); plt.close()
    print("✅ New Plot 1 (L1 Time-Series) saved.")

def plot_l3_spatial_snapshot(spatial_history_array):
    """Generates a plot of the final L3 spatial field."""
    final_state = spatial_history_array[-1]
    R_final_field = final_state[0] * final_state[1] * final_state[2]

    plt.figure(figsize=(8, 7))
    plt.imshow(R_final_field, cmap='viridis', origin='lower')
    plt.colorbar(label='Awareness Magnitude (R)')
    plt.title(r"L3: Spatial Snapshot of Final Awareness Field ($R=O \cdot M \cdot P$)")
    plt.xlabel("Spatial X-coordinate"); plt.ylabel("Spatial Y-coordinate"); plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "L3_Spatial_Snapshot.png"), dpi=150); plt.close()
    print("✅ New Plot 2 (L3 Spatial Snapshot) saved.")

def plot_l5_integrated_metrics(df_l1, df_l2):
    """Generates the integrated Triptych plot (L5) focusing on metrics vs lambda."""
    sns.set(style="whitegrid", context="talk")
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    
    # 1. Calculate Efficiency (eta) from L1/L2
    df_plot = df_l1.groupby(["lambda_A0", "init_O", "init_M", "init_P"]).agg(
        R_mean=('R', 'mean'), R_final=('R', 'last')
    ).reset_index()
    df_plot['R_final'] = df_plot['R_final']
    df_plot['Efficiency_eta'] = df_plot['R_mean'] / df_plot['R_final']
    
    # Group L2 for mean Energy/Entropy (as L2 already aggregates over time per lambda)
    df_l2_avg = df_l2.groupby("lambda_A0").agg(
        E_mean=('E_mean', 'mean'), S_mean=('S_mean', 'mean')
    ).reset_index()
    
    # Plot 1: Efficiency (L1/L2)
    ax1.plot(df_plot['lambda_A0'], df_plot['Efficiency_eta'], marker='o', label=r'Efficiency ($\eta = \bar{R}/R_{final}$)', color='tab:blue')
    ax1.set_ylabel(r'Efficiency ($\eta$)')
    ax1.set_title(r"L5: Integrated Metrics Triptych vs $\lambda_{A0}$")

    # Plot 2: Energy and Entropy (L2)
    ax2.plot(df_l2_avg['lambda_A0'], df_l2_avg['E_mean'], marker='s', label=r'Energy ($\bar{E}$)', color='tab:red')
    ax2.plot(df_l2_avg['lambda_A0'], df_l2_avg['S_mean'], marker='^', label=r'Entropy ($\bar{S}$)', color='tab:orange')
    ax2.set_ylabel('Energy/Entropy Magnitude')

    # Plot 3: Final Awareness (L1)
    ax3.plot(df_plot['lambda_A0'], df_plot['R_final'], marker='d', label=r'Final Awareness ($R_{final}$)', color='tab:green')
    ax3.set_ylabel(r'Final Awareness ($R_{final}$)')
    ax3.set_xlabel(r'Global Prior Awareness Bias ($\lambda_{A0}$)')
    
    # Correctly collect handles/labels from all axes for the single legend
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    
    ax2.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=3, fancybox=True, shadow=True)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(os.path.join(OUTPUT_DIR, "L5_Integrated_Metrics_Triptych.png"), dpi=150); plt.close()
    print("✅ New Plot 3 (L5 Integrated Triptych) saved.")


# --- Full Pipeline Orchestration ---
def run_l1_sweep_orchestration():
    """Runs the L1 sweep and saves the output."""
    print("L1: Running Full Parameter Sweep (using framework_utils.solve_core_dynamics)...")
    all_results = []
    
    total_runs = len(INITIAL_STATES) * len(LAMBDA_VALUES)
    
    for init, lamb in product(INITIAL_STATES, LAMBDA_VALUES):
        t, O, M, P, R = solve_core_dynamics(init, lamb) 
        df_case = pd.DataFrame({
            "time": t, "O": O, "M": M, "P": P, "R": R, 
            "lambda_A0": lamb, "init_O": init[0], "init_M": init[1], "init_P": init[2] 
        })
        all_results.append(df_case)
    df_sweep = pd.concat(all_results, ignore_index=True)
    df_sweep.to_csv(L1_SWEEP_OUTPUT, index=False)
    print(f"✅ L1 Sweep Complete. Data saved to {L1_SWEEP_OUTPUT}")
    return df_sweep

def run_full_pipeline_orchestration():
    """Runs the complete L1-L5 pipeline."""
    # 1. L1: Run Core Dynamics Sweep
    df_l1_sweep = run_l1_sweep_orchestration()
    plot_l1_dynamics(df_l1_sweep) 

    # 2. L2: Compute Thermodynamics/Metrics
    print("\nL2: Computing Energy Metrics...")
    df_l2_metrics = compute_thermodynamics(df_l1_sweep)
    df_l2_metrics.to_csv(L2_METRICS_OUTPUT, index=False)
    print(f"✅ L2 Complete. Data saved to {L2_METRICS_OUTPUT}")

    # 3. L3: Run Spatial Simulation
    print("\nL3: Running Spatial Simulation...")
    spatial_data = run_spatial_simulation(df_l2_metrics)
    print(f"✅ L3 Complete. Spatial field saved to {L3_SPATIAL_OUTPUT}")
    plot_l3_spatial_snapshot(spatial_data) 
    
    # 4. L4: Analyze Temporal Hierarchy
    print("\nL4: Analyzing Temporal Hierarchy...")
    spatial_data_load = np.load(L3_SPATIAL_OUTPUT) 
    analyze_temporal_hierarchy(spatial_data_load)
    df_l4_metrics = pd.read_csv(L4_MOD_OUTPUT) 
    print(f"✅ L4 Complete. Metrics saved to {L4_MOD_OUTPUT}")

    # 5. L5: Integration and Plotting
    print("\nL5: Running Final Integration and Plotting...")
    plot_l5_integrated_metrics(df_l1_sweep, df_l2_metrics) 
    
    print("\n✅ All framework steps and visualizations complete. The L5 Integrated Triptych is ready for display.")

if __name__ == "__main__":
    from run_full_pipeline import run_full_pipeline_orchestration
    run_full_pipeline_orchestration()