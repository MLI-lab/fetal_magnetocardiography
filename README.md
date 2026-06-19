# Fetal Magnetocardiography Signal Reconstruction & Analysis

A Python package for processing and analyzing fetal magnetocardiography (fMCG) signals, separating fetal and maternal cardiac signals from multichannel magnetometer recordings.

## Installation

```bash
git clone <repo-url>
cd fmcg
pip install -e .
```

## Features

- **Signal Processing**: Bandpass filtering, PCA/ZCA whitening, artifact removal
- **Source Separation**: ICA-based fetal/maternal cardiac signal separation
- **Spatial Filtering**: LMMSE-based spatial filtering as an alternative to ICA
- **Dipole Fitting**: Gradient-based inverse solver using PyTorch (GPU-accelerated)
- **Analysis**: Heartbeat detection, heart rate computation, beat averaging

## Examples

See the `examples/` directory for Jupyter notebooks demonstrating different workflows:

| Notebook | Description |
|---|---|
| `synthetic_data_example.ipynb` | Full pipeline benchmark on synthetic fMCG data |
| `basic_ICA_example.ipynb` | ICA-based source separation |
| `basic_forward_model_example.ipynb` | Dipole fitting / forward model reconstruction |
| `basic_spatial_filtering_example.ipynb` | LMMSE spatial filtering approach |
| `comparison.ipynb` | Side-by-side comparison of all methods |
| `ICA_annotate.ipynb` | Interactive GUI for labeling ICA components |

> **Data requirements**: The synthetic data experiments can be reproduced without real patient data using real empyt device measurements for realistic noise conditions. All other notebooks require real fMCG measurements, which cannot be publicly shared due to patient privacy constraints.


## Module Structure

- `fmcg.config` - Configuration dataclasses (`PipelineConfig`, `DataConfig`, `PreprocessingConfig`, ...)
- `fmcg.pipeline` - Main processing pipeline and multi-approach orchestration
- `fmcg.fitting` - Dipole inverse solver and forward field model
- `fmcg.ica` - ICA decomposition and component labeling
- `fmcg.spatial_filtering` - LMMSE spatial filtering
- `fmcg.signal` - Filtering, whitening, artifact removal utilities
- `fmcg.analysis` - Heartbeat detection and heart rate analysis
- `fmcg.utils` - Data loading, plotting, and general utilities
- `fmcg.configs` - Hardware-specific sensor geometry configurations


## License
This software is licensed under GNU GPL v3 for open-source use in academic and research contexts. Commercial licenses are available upon request.

## Contact
For questions, please contact me via GitHub or institutional email.