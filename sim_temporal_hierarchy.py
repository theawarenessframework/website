# sim_temporal_hierarchy.py (L4)

import numpy as np
import pandas as pd
import argparse
import os
import sys

INPUT_FILE_DEFAULT = "spatial_state.npy"
OUTPUT_FILE = "modulation_metrics.csv"

def analyze_temporal_hierarchy(spatial_history_array):
    """
    Analyzes the spatial history array (T, OMP_Index, X, Y)
    for temporal metrics by first calculating global R(t).
    """
    print("\nL4: Analyzing Temporal Hierarchy...")
    T, OMP_index, X, Y = spatial_history_array.shape
    
    # 1. Calculate R(t, x, y) = O * M * P
    R_field_all_time = (
        spatial_history_array[:, 0, :, :] * spatial_history_array[:, 1, :, :] * spatial_history_array[:, 2, :, :]
    ) # New shape: (T, X, Y)

    # 2. Calculate global R(t) by averaging over the spatial dimensions
    global_R_t = np.mean(R_field_all_time, axis=(1, 2)) # Final shape: (T,)
    
    # --- Metric Calculations ---
    dt = 2.0 # Placeholder dt
    
    # Power Spectral Density (PSD)
    psd = np.fft.fft(global_R_t)[:T//2]
    freqs = np.fft.fftfreq(T, d=dt)[:T//2]
    
    # Filter out near-zero frequencies before finding peak
    valid_freq_mask = freqs > 1e-6 
    peak_frequency = freqs[valid_freq_mask][np.argmax(np.abs(psd)[valid_freq_mask])] if np.any(valid_freq_mask) else 0.0
    
    # Simple metrics
    modulation_index = np.std(global_R_t) / np.mean(global_R_t)
    
    df_out = pd.DataFrame({
        "metric": ["modulation_index", "peak_frequency", "mean_R"],
        "value": [modulation_index, peak_frequency, np.mean(global_R_t)]
    })
    
    df_out.to_csv(OUTPUT_FILE, index=False)
    print(f"L4 Analysis finished. Metrics saved to {OUTPUT_FILE}")
    return df_out # Return dataframe for pipeline

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="L4: Temporal Hierarchy (Modulation and Memory).")
    parser.add_argument(
        "--input_file", type=str, default=None,
        help=f"Input file from L3 ({INPUT_FILE_DEFAULT}). If None, runs L3 default case internally."
    )
    args = parser.parse_args()

    if args.input_file and os.path.exists(args.input_file):
        try:
            spatial_data = np.load(args.input_file)
            print(f"L4 (Chained Mode): Loaded L3 data from {args.input_file}")
            analyze_temporal_hierarchy(spatial_data)
        except Exception as e:
            print(f"Error processing {args.input_file}: {e}")
            sys.exit(1)
    else:
        print(f"L4 (Independent Mode): Using dummy data for testing.")
        spatial_data = np.random.rand(101, 3, 50, 50) # Dummy array matching L3 shape
        analyze_temporal_hierarchy(spatial_data)

    print(f"✅ L4 Analysis Complete. Modulation metrics saved to {OUTPUT_FILE}")