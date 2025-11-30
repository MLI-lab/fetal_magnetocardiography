"""
Basic Usage Example for fMCG Package

This example demonstrates the simplest way to use the fMCG pipeline with
the new dataclass-based configuration system and context manager.

Run this example:
    python examples/basic_usage.py

Note: This example uses synthetic data for demonstration. For real data,
replace the synthetic data generation with your actual data loading.
"""

import numpy as np
from fmcg import Pipeline, PipelineConfig, DataConfig, SolverConfig, PreprocessingConfig


def generate_simple_synthetic_data():
    """Generate minimal synthetic fMCG data for demonstration."""
    print("Generating synthetic fMCG data...")

    fs = 500  # Sampling frequency (Hz)
    duration = 2  # seconds
    n_samples = int(fs * duration)
    n_sensors = 16
    n_axes = 3

    # Create simple sinusoidal signals
    time = np.linspace(0, duration, n_samples)
    fetal_hr = 140  # bpm
    maternal_hr = 80  # bpm

    # Simple synthetic sensor data
    data = np.random.randn(n_samples, n_sensors * n_axes) * 1e-9

    print(f"  - Duration: {duration}s")
    print(f"  - Sampling rate: {fs} Hz")
    print(f"  - Sensors: {n_sensors} x {n_axes} axes")
    print(f"  - Shape: {data.shape}")

    return data, time, fs


def main():
    """Demonstrate basic fMCG pipeline usage."""

    print("\n" + "=" * 60)
    print("fMCG Package - Basic Usage Example")
    print("=" * 60 + "\n")

    # Step 1: Generate or load your data
    # --------------------------------------------
    data, time, fs = generate_simple_synthetic_data()

    # Step 2: Create configuration using dataclasses
    # --------------------------------------------
    print("\nCreating pipeline configuration...")

    # Method 1: Using dataclasses (recommended)
    config = PipelineConfig(
        data=DataConfig(
            ds_path="/path/to/your/data",  # Update this
            patient="P001",
            series="S01",
            sig_group_names="R001",
            noise_group_names="R002",
        ),
        preprocessing=PreprocessingConfig(
            bandpass_low=1.0,
            bandpass_high=40.0,
            downsample_factor=2,
        ),
        solver=SolverConfig(
            num_dipoles=2,
            optimizer="lamb",
        ),
        device="cpu",  # Use "cuda:0" for GPU
        save_plots=False,
        save_data=False,
    )

    print("  ✓ Configuration created")
    print(f"  - Device: {config.device}")
    print(f"  - Preprocessing: {config.preprocessing.bandpass_low}-{config.preprocessing.bandpass_high} Hz")
    print(f"  - Solver: {config.solver.num_dipoles} dipoles, {config.solver.optimizer} optimizer")

    # Step 3: Use Pipeline with context manager
    # --------------------------------------------
    print("\nRunning pipeline (context manager handles cleanup)...")

    with Pipeline() as pipeline:
        print("  ✓ Pipeline initialized")

        # Note: Actual pipeline.run() requires measurement configs and real data
        # For this demo, we just show the setup

        print("  ✓ Pipeline ready to run")
        print("\n  To run the full pipeline:")
        print("  pipeline.run(config, system_config, measurement_config)")

    print("  ✓ Pipeline closed (GPU memory cleaned up)")

    # Step 4: Alternative - using dict config (backward compatible)
    # --------------------------------------------
    print("\nAlternative: Using dict configuration (backward compatible)...")

    config_dict = {
        "data": {
            "ds_path": "/path/to/data",
            "patient": "P001",
            "series": "S01",
            "sig_group_names": "R001",
            "noise_group_names": "R002",
        },
        "device": "cpu",
    }

    # Convert dict to dataclass
    config_from_dict = PipelineConfig.from_dict(config_dict)
    print("  ✓ Dict config converted to dataclass")
    print(f"  - Patient: {config_from_dict.data.patient}")

    print("\n" + "=" * 60)
    print("Example complete!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("  1. Use PipelineConfig dataclass for type-safe configuration")
    print("  2. Use Pipeline as context manager for automatic cleanup")
    print("  3. Dict configs still supported for backward compatibility")
    print("\nNext steps:")
    print("  - See examples/custom_config.py for advanced configuration")
    print("  - See examples/ directory for more examples")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
