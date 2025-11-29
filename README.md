# fMCG: Fetal Magnetocardiography Signal Processing

A Python package for processing and analyzing fetal magnetocardiography (fMCG) signals.

## Installation

```bash
pip install -e .
```

For development with editable install:
```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from fmcg import Pipeline

# Main workflow
pipeline = Pipeline()
# ... configure and run pipeline
```

## Features

- **Signal Processing**: Filtering, whitening, artifact removal
- **Analysis**: Heartbeat detection and segmentation
- **Fitting**: Inverse solvers and field models
- **ICS**: Independent Component Subtraction methods

## Documentation

See `examples/` directory for tutorial notebooks:
- `basic_ICA_example.ipynb` - Introduction to ICA methods
- `ICA_annotate.ipynb` - Interactive component labeling

## Module Structure

- `fmcg.analysis` - Heartbeat detection and analysis
- `fmcg.signal` - Signal processing utilities
- `fmcg.fitting` - Inverse problem solving
- `fmcg.utils` - Data loading and utilities
- `fmcg.ics` - Independent Component Subtraction

## License

TBD
