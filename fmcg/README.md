# py-fMCG

**A Python package for fetal magnetocardiography (fMCG) data analysis and processing**

![Work in Progress](https://img.shields.io/badge/status-work--in--progress-yellow)


## Overview

py-fMCG is a comprehensive Python package designed for the analysis and processing of fetal magnetocardiography (fMCG) data accquired at the German Heart Center Munich. The package provides a complete pipeline, comprising of pre-processing, reconstruction and post-processing for extracting fetal and maternal cardiac signals from magnetometer recordings, enabling non-invasive monitoring of fetal heart activity.


## Installation

### Prerequisites

- Python 3.8 or higher
- PyTorch (for GPU acceleration, optional)
- CUDA toolkit (for GPU support, optional)

### Installation

```bash
git clone https://github.com/JonasEmrich/py-fMCG.git
cd py-fMCG
pip install -r requirements.txt
```

Moreover, change the paths in `workspace:config.py` to the correct directories.

**Preferred usage:** Place your code in the parent directory when working with `py-fMCG`.

**Alternative:** If you place code inside the `py-fMCG` folder, you can still import the framework by modifying the Python path:

```python
import os, sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
```


## ICA Example

In addition to the forward model–based reconstruction approach, this repository provides a spline-based independent component selection method built on independent component analysis (ICA).

See the Jupyter notebook `basic_ICA_example.ipynb` for a complete walkthrough, including:

* Data loading and preprocessing
* Manual signal processing steps
* Heartbeat detection and analysis
* Visualization of results


## Package Structure

```
fmcg/
├── analysis/           # Signal analysis and heartbeat detection
│   ├── ecg_segment.py     # ECG segmentation utilities
│   └── heartbeats.py      # Heartbeat detection and analysis
├── configs/            # Measurement and system / sensor configurations
├── fitting/            # Inverse problem solving
│   ├── field_model.py     # Magnetic field forward model
│   ├── inverse_solver.py  # Gradient-based optimization
│   └── basis.py           # Fitting basis functions
├── ics/               # Independent Component Subtraction
├── pipeline/          # InverseSolving processing pipeline
│   ├── pipeline.py        # Complete analysis workflow
│   └── pipeline_utils.py  # Pipeline utilities
├── signal/            # Signal processing utilities
│   ├── filtering.py       # Frequency domain filtering
│   ├── artifact_removal.py # Artifact detection and removal
│   └── whitening.py       # Noise whitening methods
└── utils/             # General utilities and plotting
    ├── data.py            # Data loading utilities
    ├── utils.py           # General helper functions
    └── plotting/          # Visualization utilities
```

## Core Components

### 1. Pipeline (`fmcg.Pipeline`)
The main class orchestrating the complete fMCG analysis workflow:
- Data loading and validation
- Preprocessing (filtering, artifact removal, whitening)
- Inverse problem solving (dipole fitting)
- Post-processing (ICA separation, heartbeat detection)
- Results saving and visualization

### 2. Signal Processing (`fmcg.signal`)
- **Filtering**: Bandpass and notch filters for noise removal
- **Artifact Removal**: Wavelet-based artifact detection
- **Whitening**: PCA/ZCA whitening using noise covariance

### 3. Forward Model (`fmcg.fitting.field_model.ForwardModel`)
Implements the magnetic field forward model for dipole sources:
- Biot-Savart law for magnetic field computation
- GPU-accelerated tensor operations
- Support for multiple dipole configurations

### 4. Inverse Solver (`fmcg.fitting.InverseSolver`)
Gradient-based optimization for dipole parameter estimation:
- Adam/SGD optimizers with learning rate scheduling
- Regularization and constraints
- Real-time loss monitoring

### 5. Heartbeat Analysis (`fmcg.analysis.heartbeats`)
- ICA-based fetal/maternal signal separation
- Peak detection and heart rate calculation
- Beat averaging and morphology analysis

