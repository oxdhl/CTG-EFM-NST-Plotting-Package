#!/usr/bin/env python3
"""
ctg_plotter.py
==============
This module provides the function `plot_ctg` for generating professional cardiotocography (CTG)
plots from time‐series data. It supports configurable sampling frequencies, missing data handling,
and segmentation (or padding) using a 30‐minute criterion. Both static (Matplotlib) and interactive
(Plotly) modes are supported.

Features (static mode):
  • The FHR subplot background is divided into horizontal color bands:
       – Red from 50–80 and from 180–210 (color "#f8e9e9")
       – Yellow from 80–110 and from 150–180 (color "#fcfce9")
       – White from 110–150 (color "#ffffff")
  • The FHR subplot y‐limits are fixed at 50–210.
  • Major y‐ticks (and vertical annotations) are set at 60, 80, 100, 120, 140, 160, 180, and 200;
    minor ticks appear every 10 BPM in between.
  • The FHR subplot’s y‐label is “BPM” and the TOCO subplot’s y‐label is “UC.”
  • The x‐axis has major ticks every 1 minute (labels are shown only for multiples of 10)
    and minor ticks every 0.5 minute.
  • In unsplit mode, if the signal is shorter than 30 minutes the x‐axis is padded to 30 minutes
    unless the parameter trim_to_length is True.
  • In unsplit mode for signals longer than 30 minutes the full signal is plotted.
  • In split mode, the signal is divided into 30‐minute segments and the x‑axis numbering remains continuous.
  • Vertical text annotations (showing the y‐values) are added every 10 minutes,
    but are omitted at time 0 and at the left/right borders.
  • The global font size applies to all text. Its default value is set according to the scale:
    if scale_cm == 1 then font size is 8; if scale_cm == 4 then it is 20.
  • The horizontal position of the y‑axis labels for both subplots is fixed at (–0.03, 0.5).
  • An optional parameter “show” allows immediate display of the generated plot.
  
Dependencies:
  - numpy, matplotlib  
Optional (for interactive mode):
  - plotly (and kaleido for saving images)
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    _has_plotly = True
except ImportError:
    _has_plotly = False

def plot_ctg(FHR, sampling_freq=4, MHR=None, TOCO=None, Movements=None,
             Plot_missing=False, Split=False, figsize=(11.69, 8.27), config=None,
             interactive=False, Save=False, filename=None, dpi=300, 
             font_size=None, scale_cm=None, trim_to_length=False, show=False):
    """
    Plots CTG traces with the specified styling and options.
    
    Parameters
    ----------
    FHR : array-like
        Mandatory time-series array for Fetal Heart Rate.
    sampling_freq : float, optional
        Sampling frequency in Hz (default 4 Hz).
    MHR : array-like, optional
        Maternal Heart Rate (must be the same length as FHR).
    TOCO : array-like, optional
        Uterine activity (must be the same length as FHR).
    Movements : array-like, optional
        Binary array (0 or 1) for fetal movements (must be the same length as FHR).
    Plot_missing : bool, optional
        If True, convert -1 values to 0; otherwise, values ≤ 0 become NaN.
    Split : bool, optional
        If True, split the signal into 30‐minute segments.
    figsize : tuple, optional
        Figure size in inches.
    config : dict, optional
        Custom styling parameters.
    interactive : bool, optional
        If True, generate an interactive Plotly plot.
    Save : bool, optional
        If True, save the figure(s) to file (filename required).
    filename : str, optional
        Filename (or base filename if Split=True) for saving.
    dpi : int, optional
        Resolution for saving (default 300 DPI).
    font_size : int, optional
        Global font size to use for all text. If not provided and if scale_cm is given,
        then if scale_cm == 1 then font size defaults to 8; if scale_cm == 4 then to 20.
        Otherwise, if unsplit and signal length >=30 then 20, else 7.
    scale_cm : float, optional
        Scale in centimeters per minute (e.g. 1 or 4). This is used to determine the default font size.
    trim_to_length : bool, optional
        If True, do not pad the x-axis for signals shorter than 30 minutes.
    show : bool, optional
        If True, display the plot(s) after generation.
    
    Returns
    -------
    figs : list
        A list of figure objects.
    """
    if FHR is None:
        raise ValueError("FHR signal must be provided.")
    FHR = np.array(FHR)
    n_points = len(FHR)
    if MHR is not None:
        MHR = np.array(MHR)
        if len(MHR) != n_points:
            raise ValueError("MHR must be the same length as FHR.")
    if TOCO is not None:
        TOCO = np.array(TOCO)
        if len(TOCO) != n_points:
            raise ValueError("TOCO must be the same length as FHR.")
    if Movements is not None:
        Movements = np.array(Movements)
        if len(Movements) != n_points:
            raise ValueError("Movements must be the same length as FHR.")

    default_config = {
        "FHR_color": "black",
        "MHR_color": "navy",
        "TOCO_color": "black",  # updated to black
        "Movement_marker": "triangle",  # '^' for Matplotlib; 'triangle-up' for Plotly.
        "Movement_color": "darkgreen",
        "linewidth_FHR": 0.75,
        "linewidth_MHR": 0.50,
        "linewidth_TOCO": 0.75,
        "movement_size": 3,
    }
    if config is not None:
        default_config.update(config)
    def process_signal(arr):
        arr = np.array(arr, dtype=float)
        if Plot_missing:
            arr[arr == -1] = 0
        else:
            arr[arr <= 0] = np.nan
        return arr
    FHR = process_signal(FHR)
    if MHR is not None:
        MHR = process_signal(MHR)
    if TOCO is not None:
        TOCO = process_signal(TOCO)
    time = np.linspace(0, n_points / sampling_freq / 60, num=n_points)
    # Segmentation always uses 30 minutes per segment.
    if Split:
        segment_duration = 30
        segments = []
        start = 0
        while start < n_points:
            t_start = time[start]
            t_end = t_start + segment_duration
            end = np.searchsorted(time, t_end, side="right")
            if end == start:
                end = start + 1
            segments.append((start, end, t_start))
            start = end
    else:
        if (time[-1] < 30) and (not trim_to_length):
            segments = [(0, n_points, time[0])]  # pad later to 30 minutes
        else:
            segments = [(0, n_points, time[0])]
    # Determine default global font size if not provided.
    if font_size is None:
        if scale_cm is not None:
            if scale_cm == 1:
                font_size = 8
            elif scale_cm == 4:
                font_size = 20
            else:
                font_size = 7
        else:
            if (not Split) and (time[-1] >= 30):
                font_size = 20
            else:
                font_size = 7
    figs = []
    for seg_idx, (start, end, seg_start_time) in enumerate(segments):
        if not Split:
            if (time[-1] < 30) and (not trim_to_length):
                x_min = 0
                x_lim = 30
            else:
                x_min = time[0]
                x_lim = time[-1]
        else:
            x_min = time[start]
            if trim_to_length and (time[-1] < x_min + 30):
                x_lim = time[-1]
            else:
                x_lim = x_min + 30
        time_seg = time[start:end]
        y_min, y_max = 50, 210
        major_y = np.arange(60, 201, 20)
        minor_y = np.arange(60, 201, 10)
        if interactive:
            if not _has_plotly:
                raise ImportError("Plotly must be installed for interactive plotting.")
            if TOCO is not None:
                fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                    vertical_spacing=0.05, row_heights=[0.7, 0.3])
            else:
                fig = go.Figure()
            fig.update_layout(font=dict(size=font_size))
            fig.add_trace(go.Scatter(
                x=time_seg, y=FHR[start:end], mode='lines', name='FHR',
                line=dict(color=default_config["FHR_color"],
                          width=default_config["linewidth_FHR"])
            ), row=1, col=1 if TOCO is not None else None)
            if MHR is not None:
                fig.add_trace(go.Scatter(
                    x=time_seg, y=MHR[start:end], mode='lines', name='MHR',
                    line=dict(color=default_config["MHR_color"],
                              width=default_config["linewidth_MHR"])
                ), row=1, col=1 if TOCO is not None else None)
            if Movements is not None:
                idx = np.where(Movements[start:end] == 1)[0]
                if idx.size:
                    fig.add_trace(go.Scatter(
                        x=time_seg[idx], y=[60]*len(idx),
                        mode='markers', name='Movements',
                        marker=dict(symbol='triangle-up',
                                    size=default_config["movement_size"] * 3,
                                    color=default_config["Movement_color"])
                    ), row=1, col=1 if TOCO is not None else None)
            shapes = [
                dict(type="rect", xref="x", yref="y", x0=x_min, x1=x_lim, y0=50, y1=80,
                     fillcolor="#f8e9e9", line_width=0, layer="below"),
                dict(type="rect", xref="x", yref="y", x0=x_min, x1=x_lim, y0=80, y1=110,
                     fillcolor="#fcfce9", line_width=0, layer="below"),
                dict(type="rect", xref="x", yref="y", x0=x_min, x1=x_lim, y0=110, y1=150,
                     fillcolor="#ffffff", line_width=0, layer="below"),
                dict(type="rect", xref="x", yref="y", x0=x_min, x1=x_lim, y0=150, y1=180,
                     fillcolor="#fcfce9", line_width=0, layer="below"),
                dict(type="rect", xref="x", yref="y", x0=x_min, x1=x_lim, y0=180, y1=210,
                     fillcolor="#f8e9e9", line_width=0, layer="below")
            ]
            if TOCO is not None:
                shapes.append(dict(type="rect", xref="x2", yref="y2",
                                   x0=x_min, x1=x_lim, y0=0, y1=100,
                                   fillcolor="#f2fefe", line_width=0, layer="below"))
            layout_update = {
                "xaxis": dict(range=[x_min, x_lim],
                              tick0=0, dtick=1, minor=dict(dtick=0.5),
                              gridcolor="#b4b4b3"),
                "yaxis": dict(range=[y_min, y_max],
                              tickvals=major_y, minor=dict(dtick=10),
                              gridcolor="#b4b4b3", title="BPM"),
            }
            if TOCO is not None:
                layout_update.update({
                    "xaxis2": dict(range=[x_min, x_lim],
                                   tick0=0, dtick=1, minor=dict(dtick=0.5),
                                   gridcolor="#b4b4b3"),
                    "yaxis2": dict(range=[0, 100],
                                   tickvals=[25,50,75],
                                   gridcolor="#b4b4b3", title="UC")
                })
            ann = []
            for xt in np.arange(np.ceil(x_min/10)*10, x_lim+0.001, 10):
                if np.isclose(xt, 0) or np.isclose(xt, x_min) or np.isclose(xt, x_lim):
                    continue
                for y_val in major_y:
                    ann.append(dict(x=xt, y=y_val, text=str(y_val),
                                    showarrow=False,
                                    font=dict(size=font_size, color="#484847"),
                                    xanchor="center", yanchor="middle", xref="x", yref="y"))
                if TOCO is not None:
                    for y_val in [25,50,75]:
                        ann.append(dict(x=xt, y=y_val, text=str(y_val),
                                        showarrow=False,
                                        font=dict(size=font_size, color="#484847"),
                                        xanchor="center", yanchor="middle", xref="x", yref="y2"))
            layout_update["annotations"] = ann
            fig.update_layout(layout_update, shapes=shapes,
                              width=figsize[0]*dpi, height=figsize[1]*dpi)
            figs.append(fig)
            if Save:
                if filename is None:
                    raise ValueError("Filename must be specified when Save is True.")
                if Split and len(segments) > 1:
                    base, ext = filename.rsplit('.', 1)
                    save_filename = f"{base}_segment{seg_idx+1}.{ext}"
                else:
                    save_filename = filename
                try:
                    fig.write_image(save_filename, scale=1,
                                    width=fig.layout.width, height=fig.layout.height)
                except Exception as e:
                    raise RuntimeError("Error saving interactive figure. Ensure kaleido is installed.") from e
            if show:
                fig.show()
        else:
            if TOCO is not None:
                fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True,
                                               figsize=figsize, dpi=dpi,
                                               gridspec_kw={'height_ratios': [0.7, 0.3]})
            else:
                fig, ax1 = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
            ax1.axhspan(50, 80, facecolor='#f8e9e9', zorder=0)
            ax1.axhspan(80, 110, facecolor='#fcfce9', zorder=0)
            ax1.axhspan(110, 150, facecolor='#ffffff', zorder=0)
            ax1.axhspan(150, 180, facecolor='#fcfce9', zorder=0)
            ax1.axhspan(180, 210, facecolor='#f8e9e9', zorder=0)
            ax1.plot(time_seg, FHR[start:end],
                     color=default_config["FHR_color"],
                     linewidth=default_config["linewidth_FHR"],
                     label="FHR")
            if MHR is not None:
                ax1.plot(time_seg, MHR[start:end],
                         color=default_config["MHR_color"],
                         linewidth=default_config["linewidth_MHR"],
                         label="MHR")
            if Movements is not None:
                idx = np.where(Movements[start:end] == 1)[0]
                if idx.size:
                    ax1.scatter(time_seg[idx],
                                np.full_like(time_seg[idx], 60),
                                marker='^',
                                s=default_config["movement_size"] * 10,
                                color=default_config["Movement_color"],
                                label="Movements")
            ax1.set_xlim(x_min, x_lim)
            ax1.set_ylim(y_min, y_max)
            ax1.set_ylabel("BPM", fontsize=font_size)
            plt.setp(ax1.get_xticklabels(), fontsize=font_size)
            plt.setp(ax1.get_yticklabels(), fontsize=font_size)
            major_xticks = np.arange(np.floor(x_min), x_lim+1, 1)
            xticklabels = [str(int(tick)) if np.isclose(tick % 10, 0, atol=1e-6) else ""
                           for tick in major_xticks]
            ax1.set_xticks(major_xticks)
            ax1.set_xticklabels(xticklabels, fontsize=font_size)
            ax1.set_xticks(np.arange(np.floor(x_min), x_lim+0.5, 0.5), minor=True)
            ax1.set_yticks(major_y)
            ax1.set_yticks(minor_y, minor=True)
            ax1.grid(which='major', linestyle='-', linewidth=0.5,
                     color='#b4b4b3', zorder=1)
            ax1.grid(which='minor', linestyle='--', linewidth=0.5,
                     color='#b4b4b3', zorder=1)
            for xt in np.arange(np.ceil(x_min/10)*10, x_lim+0.001, 10):
                if np.isclose(xt, x_min) or np.isclose(xt, x_lim) or np.isclose(xt, 0):
                    continue
                for y_val in major_y:
                    ax1.text(xt, y_val, str(y_val),
                             ha='center', va='center', color="#484847",
                             fontsize=font_size,
                             zorder=10, clip_on=False)
                if TOCO is not None:
                    for y_val in [25,50,75]:
                        ax2.text(xt, y_val, str(y_val),
                                 ha='center', va='center', color="#484847",
                                 fontsize=font_size,
                                 zorder=10, clip_on=False)
            if TOCO is not None:
                ax2.plot(time_seg, TOCO[start:end],
                         color=default_config["TOCO_color"],
                         linewidth=default_config["linewidth_TOCO"],
                         label="TOCO")
                ax2.axhline(y=10, color='grey', linestyle='--', linewidth=1)
                ax2.set_xlim(x_min, x_lim)
                ax2.set_ylim(0, 100)
                ax2.set_ylabel("UC", fontsize=font_size)
                ax2.set_xlabel("Time (minutes)", fontsize=font_size)
                plt.setp(ax2.get_xticklabels(), fontsize=font_size)
                plt.setp(ax2.get_yticklabels(), fontsize=font_size)
                ax2.set_xticks(major_xticks)
                ax2.set_xticklabels(xticklabels, fontsize=font_size)
                ax2.set_xticks(np.arange(np.floor(x_min), x_lim+0.5, 0.5), minor=True)
                ax2.set_yticks([25, 50, 75])
                ax2.grid(which='major', linestyle='-', linewidth=0.5,
                         color='#b4b4b3', zorder=1)
                ax2.grid(which='minor', linestyle='--', linewidth=0.5,
                         color='#b4b4b3', zorder=1)
                ax2.set_facecolor('#f2fefe')
                for xt in np.arange(np.ceil(x_min/10)*10, x_lim+0.001, 10):
                    if np.isclose(xt, x_min) or np.isclose(xt, x_lim) or np.isclose(xt, 0):
                        continue
                    for y_val in [25,50,75]:
                        ax2.text(xt, y_val, str(y_val),
                                 ha='center', va='center', color="#484847",
                                 fontsize=font_size,
                                 zorder=10, clip_on=False)
            # Align y-axis labels to the same horizontal position.
            ax1.yaxis.set_label_coords(-0.03, 0.5)
            if TOCO is not None:
                ax2.yaxis.set_label_coords(-0.03, 0.5)
            plt.tight_layout()
            figs.append(fig)
            if Save:
                if filename is None:
                    raise ValueError("Filename must be specified when Save is True.")
                if Split and len(segments) > 1:
                    base, ext = filename.rsplit('.', 1)
                    save_filename = f"{base}_segment{seg_idx+1}.{ext}"
                else:
                    save_filename = filename
                fig.savefig(save_filename, dpi=dpi)
            if show:
                plt.show()
    return figs

if __name__ == '__main__':
    import matplotlib
    matplotlib.use('Agg')
    duration = 40
    sampling_freq = 4
    n_pts = int(duration * 60 * sampling_freq)
    t = np.linspace(0, duration, n_pts)
    FHR = 140 + 10 * np.sin(2 * np.pi * t / 10)
    MHR = 80 + 5 * np.sin(2 * np.pi * t / 15)
    TOCO = 20 + 10 * np.abs(np.sin(2 * np.pi * t / 5))
    Movements = (np.random.rand(n_pts) > 0.98).astype(int)
    figs = plot_ctg(FHR, sampling_freq=sampling_freq, MHR=MHR, TOCO=TOCO,
                    Movements=Movements, Plot_missing=False, Split=True,
                    Save=False, interactive=False)
    print("CTG plots generated successfully.")
