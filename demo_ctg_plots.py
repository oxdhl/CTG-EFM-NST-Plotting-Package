#!/usr/bin/env python3
"""
demo_ctg_plots.py
=================

This demo script uses the ctg_plotter module to generate and save five example CTG plots:
  A. A 20‑minute trace at 1 cm scale.
  B. A 20‑minute trace at 4 cm scale.
  C. A 45‑minute unsplit trace at 1 cm scale.
  D. A 45‑minute split trace at 1 cm scale.
  E. A 10‑minute trace at 1 cm scale with trimming enabled.

For unsplit cases, if the signal duration is less than 30 minutes the x‑axis is padded to 30 minutes,
unless the parameter trim_to_length is True. The figure sizes are computed so that the physical
distance between adjacent major x‑ticks (1 minute) matches that between adjacent major y‑ticks (20 BPM).
Output filenames reflect the mode, signal duration, scale, and whether trimming is enabled.
 
Usage:
    python demo_ctg_plots.py
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from ctg_plotter import plot_ctg

def generate_synthetic_ctg(duration_minutes=40, sampling_freq=4):
    total_points = int(duration_minutes * 60 * sampling_freq)
    t = np.linspace(0, duration_minutes, total_points)
    FHR = 140 + 10 * np.sin(2 * np.pi * t / 10)
    MHR = 80 + 5 * np.sin(2 * np.pi * t / 15)
    TOCO = 20 + 10 * np.abs(np.sin(2 * np.pi * t / 5))
    Movements = (np.random.rand(total_points) > 0.98).astype(int)
    return FHR, MHR, TOCO, Movements

def cm_to_inches(cm):
    return cm / 2.54

def demo_case(duration, split, scale_cm, trim, demo_number, save_dir=None):
    """
    Generate a demo CTG plot.
    
    Parameters
    ----------
    duration : int
        Signal duration in minutes.
    split : bool
        True for split mode (each segment is 30 minutes), False for unsplit.
    scale_cm : float
        Scale in centimeters per minute (i.e. width in cm per minute).
    trim : bool
        If True, do not pad the x-axis for signals shorter than 30 minutes.
    demo_number : int
        Used in the output filename.
    save_dir : str, optional
        Directory to save the plots.
    """
    FHR, MHR, TOCO, Movements = generate_synthetic_ctg(duration_minutes=duration)
    if not split and (duration < 30) and (not trim):
        x_range = 30
    else:
        # For unsplit signals with duration >= 30 minutes, use the full signal length.
        # In split mode, each segment is 30 minutes.
        x_range = duration if (not split and (duration >= 30)) else 30
    width_cm = x_range * scale_cm
    width_inch = cm_to_inches(width_cm)
    height_inch = 7 * width_inch / x_range
    figsize = (width_inch, height_inch)
    
    # Build the filename.
    mode = "split" if split else "unsplit"
    fname = f"{mode}_{duration}min_{scale_cm}cm_trim{trim}.png"
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
    figs = plot_ctg(FHR, sampling_freq=4, MHR=MHR, TOCO=TOCO, Movements=Movements,
                    Plot_missing=False, Split=split, interactive=False,
                    Save=True, filename=fname, figsize=figsize,
                    font_size=None, scale_cm=scale_cm, trim_to_length=trim,
                    show=False, save_dir=save_dir)
    for fig in figs:
        plt.show()
    print(f"Demo {demo_number} saved as {os.path.join(save_dir, fname) if save_dir else fname}")

if __name__ == '__main__':
    print("=== CTG Plot Demo Script ===\n")
    # Define a directory to save the demo plots.
    save_directory = "demo_plots"
    # A. 20-minute plot at 1cm scale.
    demo_case(duration=20, split=False, scale_cm=1, trim=False, demo_number="A", save_dir=save_directory)
    # B. 20-minute plot at 4cm scale.
    demo_case(duration=20, split=False, scale_cm=4, trim=False, demo_number="B", save_dir=save_directory)
    # C. 45-minute unsplit plot at 1cm scale.
    demo_case(duration=45, split=False, scale_cm=1, trim=False, demo_number="C", save_dir=save_directory)
    # D. 45-minute split plot at 1cm scale.
    demo_case(duration=45, split=True, scale_cm=1, trim=False, demo_number="D", save_dir=save_directory)
    # E. 10-minute plot at 1cm scale with trimming enabled.
    demo_case(duration=10, split=False, scale_cm=1, trim=True, demo_number="E", save_dir=save_directory)
    
    print("\nDemo completed. Check the generated files in the specified directory.")
