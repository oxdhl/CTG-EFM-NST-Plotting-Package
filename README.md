# **CTG/EFM/NST Plotting Package**  

Developed by **Oxford Digital Health Labs**  
**Nuffield Department of Women's and Reproductive Health**  
**University of Oxford**  
**Women's Centre (Level 3), John Radcliffe Hospital, OX3 9DU, United Kingdom**  

Contact: **digitalhealthlabs@wrh.ox.ac.uk**  
Website: [https://oxdhl.com/](https://oxdhl.com/)  

---

## **1. Overview**  

This package provides a **professional-grade plotting tool** for **Cardiotocography (CTG)**, also known as **Electronic Fetal Monitoring (EFM)** or the **Non-Stress Test (NST)** in the United States. The tool is designed for clinical and research applications, supporting:  

- **Standardised plotting conventions** for fetal heart rate (FHR) and uterine contractions  
- **Customisable scale settings** to match regional conventions:  
  - **1 cm per minute** (Used in the UK, Europe, and most non-US countries)  
  - **4 cm per minute** (Used in the United States)  
- **Automatic segmentation for long traces** (≥30 minutes)  
- **Customisable font sizes and visual settings**  
- **Missing data handling**  
- **Support for interactive plots (Plotly) and publication-quality static plots (Matplotlib)**  
- **Multiple export formats** (PNG, PDF, SVG, EPS)  

This package is designed for **clinical research, AI model validation, and real-time fetal monitoring analysis**.

---

## **2. Installation Guide**  

### **2.1 System Requirements**  

Ensure you have the following software installed:  

- Python **≥3.7**  
- Matplotlib **≥3.4**  
- NumPy **≥1.21**  
- Plotly **(optional, for interactive mode)**  

### **2.2 Install Required Python Packages**  

To install all dependencies, run:  

```bash
pip install numpy matplotlib plotly
```

If using Conda:  

```bash
conda install numpy matplotlib
conda install -c plotly plotly
```

To enable **Plotly export functionality**, install **Kaleido**:  

```bash
pip install kaleido
```

---

## **3. Usage Guide**  

### **3.1 Running the Demo Script**  

To generate example plots, run:  

```bash
python demo_ctg_plots.py
```

This script will generate multiple sample plots demonstrating different configurations, including **split vs. unsplit traces, scaling differences, and handling of short traces**.  

---

### **3.2 Understanding Example Plots**  

| Filename | Mode | Duration | Scale | Trimmed |
|----------|------|----------|------|---------|
| `unsplit_20min_1cm_trimFalse.png` | Unsplit | 20 min | 1 cm/min (UK/Non-USA) | No |
| `unsplit_20min_4cm_trimFalse.png` | Unsplit | 20 min | 4 cm/min (USA) | No |
| `split_45min_1cm_trimFalse.png` | Split | 45 min | 1 cm/min | No |
| `split_45min_4cm_trimFalse.png` | Split | 45 min | 4 cm/min | No |
| `unsplit_45min_1cm_trimTrue.png` | Unsplit | 45 min | 1 cm/min | Yes |
| `unsplit_45min_4cm_trimTrue.png` | Unsplit | 45 min | 4 cm/min | Yes |

- **Unsplit Mode** → The entire CTG/EFM trace is plotted in one figure  
- **Split Mode** → If the trace is **longer than 30 minutes**, it is broken into multiple 30-minute plots  
- **Trimmed Mode** → If `trim=True`, no padding is applied to short traces  

---

## **4. Using the Plotting Module (`ctg_plotter.py`)**  

For integration into research pipelines, import the function:

```python
from ctg_plotter import plot_ctg
```

### **4.1 Example Usage**  

```python
import numpy as np
from ctg_plotter import plot_ctg

# Generate synthetic CTG/EFM data
duration = 40  # 40 minutes
sampling_freq = 4  # 4 Hz
total_points = int(duration * 60 * sampling_freq)

time = np.linspace(0, duration, total_points)
FHR = 140 + 10 * np.sin(2 * np.pi * time / 10)  # Simulated FHR data
MHR = 80 + 5 * np.sin(2 * np.pi * time / 15)  # Simulated MHR data
TOCO = 20 + 10 * np.abs(np.sin(2 * np.pi * time / 5))  # Simulated uterine contractions
Movements = (np.random.rand(total_points) > 0.98).astype(int)  # Random fetal movements

# Generate and save plot
plot_ctg(FHR, sampling_freq=4, MHR=MHR, TOCO=TOCO, Movements=Movements,
         Plot_missing=False, Split=True, interactive=False,
         Save=True, filename="ctg_output.png", scale_cm=4, trim_to_length=False, show=True)
```

---

## **5. Plot Features and Configurations**  

### **5.1 Background Colour Bands**  

The **FHR subplot** includes standardised background colour bands for clinical interpretation:  

- **Red (50-80 BPM & 180-210 BPM)** → `#f8e9e9`  
- **Yellow (80-110 BPM & 150-180 BPM)** → `#fcfce9`  
- **White (110-150 BPM)** → `#ffffff`  

The **TOCO subplot** has a **light blue background** (`#f2fefe`).

### **5.2 Axis and Grid Details**  

#### **FHR Subplot (Upper Plot)**  

- **X-Axis (Time in Minutes)**  
  - **Major grid** every **1 minute**  
  - **Minor grid** every **30 seconds**  
  - **Labels** appear every **10 minutes**  

- **Y-Axis (BPM)**  
  - **Range:** `50–210 BPM`  
  - **Major ticks:** `60, 80, 100, 120, 140, 160, 180, 200`  
  - **Minor ticks:** Every **10 BPM**  

#### **TOCO Subplot (Lower Plot)**  

- **X-Axis** → Matches FHR subplot  
- **Y-Axis (Uterine Contractions)**  
  - **Range:** `0–100`  
  - **Major ticks:** `25, 50, 75`  

---

## **6. How to Cite**  

If you use this package in your research, please cite:  

> Oxford Digital Health Labs. **CTG/EFM/NST Plotting Package: A High-Quality Fetal Monitoring Visualisation Tool**. Nuffield Department of Women's and Reproductive Health, University of Oxford. Version X.X (2025).  

### **BibTeX Citation Format**  

```bibtex
@software{oxdhl_ctgplotter_2025,
  author = {Oxford Digital Health Labs},
  title = {CTG/EFM/NST Plotting Package: A High-Quality Fetal Monitoring Visualisation Tool},
  institution = {Nuffield Department of Women's and Reproductive Health, University of Oxford},
  year = {2025},
  version = {X.X},
  url = {https://oxdhl.com/}
}
```

For further information or updates, visit:  
[https://oxdhl.com/](https://oxdhl.com/)  

---

## **7. License & Contributions**  

This package is released under an **open-source license**. Contributions, bug reports, and feature requests are welcome.

### **Developers & Contributors**  

- **Oxford Digital Health Labs**  
- **Nuffield Department of Women's and Reproductive Health**  
- **University of Oxford**  

📧 Contact: **digitalhealthlabs@wrh.ox.ac.uk**  
🔗 Website: [https://oxdhl.com/](https://oxdhl.com/)