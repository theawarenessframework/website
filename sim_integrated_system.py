# sim_integrated_system.py (L5)

import pandas as pd
import numpy as np
import argparse
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Set up defaults for inputs
L1_DEFAULT = "core_timeseries_sweep.csv" # Use sweep by default
L2_DEFAULT = "energy_metrics.csv"
L4_DEFAULT = "modulation_metrics.csv"
OUTPUT_DIR = "romp_final_plots"
os.makedirs(OUTPUT_DIR, exist_ok=True)
sns.set(style="whitegrid", context="talk")

def plot_l5_integrated_metrics(df_l1, df_l2):
    """
    Generates the final 3-axis plot for Efficiency, Energy, and Entropy.
    """
    # L1 Data: Calculate Efficiency (eta) for each run
    # Group by all run parameters to get R_mean and R_final for *each* run
    df_plot_eta = df_l1.groupby(["lambda_A0", "init_O", "init_M", "init_P"]).agg(
        R_mean=('R', 'mean'),
        R_final=('R', 'last') 
    ).reset_index()
    
    # Calculate Efficiency for each individual run
    df_plot_eta['Efficiency_eta'] = df_plot_eta['R_mean'] / df_plot_eta['R_final']
    
    # Now, average the efficiency for each lambda_A0
    df_plot_final = df_plot_eta.groupby("lambda_A0").mean(numeric_only=True).reset_index()
    
    # L2 Data: Average the E_mean and S_mean (already averaged per run in compute_thermodynamics)
    df_plot_E_S = df_l2.groupby("lambda_A0").mean(numeric_only=True).reset_index()

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Plot 1: Efficiency (Primary)
    ax1.plot(df_plot_final["lambda_A0"], df_plot_final["Efficiency_eta"], marker="o", color='darkblue', label=r"Efficiency $\eta$")
    ax1.set_xlabel(r"Global Prior Awareness ($\lambda_{A0}$)")
    ax1.set_ylabel(r"Efficiency $\eta$ (Mean/Final R)", color='darkblue')
    ax1.tick_params(axis='y', labelcolor='darkblue')

    # Plot 2: Energy (Secondary)
    ax2 = ax1.twinx()
    ax2.plot(df_plot_E_S["lambda_A0"], df_plot_E_S["E_mean"], marker="s", linestyle='--', color='red', label=r"Mean Energy $E$")
    ax2.set_ylabel(r"Mean Energy $E$", color='red')
    ax2.tick_params(axis='y', labelcolor='red')

    # Plot 3: Entropy (Tertiary)
    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('outward', 60))
    ax3.plot(df_plot_E_S["lambda_A0"], df_plot_E_S["S_mean"], marker="^", linestyle=':', color='green', label=r"Mean Entropy $S$")
    ax3.set_ylabel(r"Mean Entropy $S$", color='green')
    ax3.tick_params(axis='y', labelcolor='green')

    # *** FIX for ValueError: Combine legends correctly ***
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='best')

    plt.title(r"L5: Integrated System Metrics (Efficiency, Energy, and Entropy)")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "L5_Integrated_Metrics_Triptych.png"), dpi=300)
    print(f"✅ L5 Plot saved to {OUTPUT_DIR}/L5_Integrated_Metrics_Triptych.png")
    plt.close()


def compute_global_coherence(df_l2, df_l4):
    """
    Placeholder: Integrates Energy and Temporal metrics to yield a global index.
    """
    print("L5: Computing Global Coherence Index...")
    avg_energy = df_l2['E_mean'].mean()
    mod_index_row = df_l4[df_l4['metric'] == 'modulation_index']
    
    if not mod_index_row.empty:
        mod_index = mod_index_row['value'].iloc[0]
        global_coherence_index = mod_index / (1 + avg_energy)
        print(f"Global Coherence Index: {global_coherence_index:.4f}")
    else:
        print("Warning: Could not calculate Global Coherence Index.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="L5: Integrated System and Final Plotting.")
    parser.add_argument("--l1_file", type=str, default=L1_DEFAULT, help="Input file from L1.")
    parser.add_argument("--l2_file", type=str, default=L2_DEFAULT, help="Input file from L2.")
    parser.add_argument("--l4_file", type=str, default=L4_DEFAULT, help="Input file from L4.")
    args = parser.parse_args()

    # 1. Load Data
    try:
        df_l1 = pd.read_csv(args.l1_file)
        df_l2 = pd.read_csv(args.l2_file)
        df_l4 = pd.read_csv(args.l4_file)
    except FileNotFoundError as e:
        print(f"Error loading input file: {e}. Please ensure you have run the preceding scripts.")
        sys.exit(1)

    # 2. R = O * M * P Verification Plot (using L1/L2 data)
    plot_l5_integrated_metrics(df_l1, df_l2)
    
    # 3. Compute Integrated Metrics
    compute_global_coherence(df_l2, df_l4)
    
    print("✅ L5 Integration Complete. Final plots and metrics generated.")