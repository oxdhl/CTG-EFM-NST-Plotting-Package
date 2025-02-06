#!/usr/bin/env python3
"""
demo_ctg_plots.py
=================

This demo script uses the ctg_plotter module to generate and save six example CTG plots:
  1. A sub‑30‑minute trace (20‑minute signal) at 1 cm scale.
  2. A sub‑30‑minute trace (20‑minute signal) at 4 cm scale.
  3. A split 45‑minute trace at 1 cm scale.
  4. A split 45‑minute trace at 4 cm scale.
  5. A 45‑minute unsplit trace at 1 cm scale.
  6. A 45‑minute unsplit trace at 4 cm scale.

For unsplit cases, if the signal duration is less than 30 minutes the x‑axis is padded to 30 minutes,
unless the parameter trim_to_length is True. The figure sizes are computed so that the physical
distance between adjacent major x‑ticks (1 minute) matches that between adjacent major y‑ticks (20 BPM).
Output filenames reflect the mode, signal duration, scale, and whether trimming is enabled.
 
Usage:
    python demo_ctg_plots.py
"""

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

def demo_case(duration, split, scale_cm, trim, demo_number):
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
    
    if split:
        fname = f"split_{duration}min_{scale_cm}cm_trim{trim}.png"
    else:
        fname = f"unsplit_{duration}min_{scale_cm}cm_trim{trim}.png"
    
    figs = plot_ctg(FHR, sampling_freq=4, MHR=MHR, TOCO=TOCO, Movements=Movements,
                    Plot_missing=False, Split=split, interactive=False,
                    Save=True, filename=fname, figsize=figsize,
                    font_size=None, scale_cm=scale_cm, trim_to_length=trim, show=True)
    for fig in figs:
        plt.show()
    print(f"Demo {demo_number} saved as {fname}")

if __name__ == '__main__':
    print("=== CTG Plot Demo Script ===\n")
    # 1. Sub-30-minute trace (20-minute signal) at 1 cm scale, no trimming.
    demo_case(duration=20, split=False, scale_cm=1, trim=False, demo_number=1)
    # 2. Sub-30-minute trace (20-minute signal) at 4 cm scale, no trimming.
    demo_case(duration=20, split=False, scale_cm=4, trim=False, demo_number=2)
    # 3. Split 45-minute trace at 1 cm scale.
    demo_case(duration=45, split=True, scale_cm=1, trim=False, demo_number=3)
    # 4. Split 45-minute trace at 4 cm scale.
    demo_case(duration=45, split=True, scale_cm=4, trim=False, demo_number=4)
    # 5. 45-minute unsplit trace at 1 cm scale, with trimming.
    demo_case(duration=45, split=False, scale_cm=1, trim=True, demo_number=5)
    # 6. 45-minute unsplit trace at 4 cm scale, with trimming.
    demo_case(duration=45, split=False, scale_cm=4, trim=True, demo_number=6)
    
    print("\nDemo completed. Check the generated files with descriptive filenames.")
