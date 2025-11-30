# fMCG: Fetal Magnetocardiography Signal Processing

A Python package for processing and analyzing fetal magnetocardiography (fMCG) signals.

## Installation

```bash
pip install -e .
```

For development with testing support:
```bash
pip install -e ".[dev]"
```

## Quick Start

```python
from fmcg import Pipeline, PipelineConfig, DataConfig

# Create configuration
config = PipelineConfig(
    data=DataConfig(
        ds_path="/path/to/data",
        patient="P001",
        series="S01",
        sig_group_names="R001",
        noise_group_names="R002",
    ),
    device="cpu",  # or "cuda:0" for GPU
)

# Run pipeline with automatic resource cleanup
with Pipeline() as pipeline:
    results = pipeline.run(config, system_config, measurement_config)
```

## Features

- **Signal Processing**: Filtering, whitening, artifact removal
- **Analysis**: Heartbeat detection and segmentation
- **Fitting**: Inverse solvers and field models
- **ICS**: Independent Component Subtraction methods
- **Type-Safe Configuration**: Dataclass-based config system with IDE support
- **Context Manager**: Automatic resource cleanup (GPU memory, etc.)

## Examples

See the `examples/` directory for detailed usage:
- `basic_usage.py` - Introduction to the pipeline
- `custom_config.py` - Advanced configuration options
- Tutorial notebooks in `fmcg/examples/` (ICA annotation, etc.)

## Testing

Run tests:
```bash
pytest
```

Run tests with coverage:
```bash
pytest --cov=fmcg --cov-report=html
```

## Module Structure

- `fmcg.config` - Configuration dataclasses
- `fmcg.pipeline` - Main processing pipeline
- `fmcg.analysis` - Heartbeat detection and analysis
- `fmcg.signal` - Signal processing utilities
- `fmcg.fitting` - Inverse problem solving
- `fmcg.utils` - Data loading and utilities
- `fmcg.ics` - Independent Component Subtraction

## Configuration

The package uses dataclass-based configuration for type safety:

```python
from fmcg.config import PipelineConfig, PreprocessingConfig, SolverConfig

config = PipelineConfig(
    data=...,
    preprocessing=PreprocessingConfig(
        whitening="PCA",
        bandpass_low=1.0,
        bandpass_high=40.0,
    ),
    solver=SolverConfig(
        num_dipoles=2,
        optimizer="lamb",
    ),
)
```

Backward compatible with dict configs:
```python
config_dict = {...}  # Your old dict config
config = PipelineConfig.from_dict(config_dict)
```

## License

TBD
