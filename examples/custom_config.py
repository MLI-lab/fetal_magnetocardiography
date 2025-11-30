"""
Custom Configuration Example for fMCG Package

This example shows how to customize various configuration options using
the dataclass-based system, including solver parameters, learning rates,
and preprocessing settings.

Run this example:
    python examples/custom_config.py
"""

import numpy as np
from fmcg.config import (
    PipelineConfig,
    DataConfig,
    PreprocessingConfig,
    SolverConfig,
    PostProcessingConfig,
    LearningRateConfig,
    InitializerConfig,
)


def example_minimal_config():
    """Example 1: Minimal configuration with defaults."""
    print("\n" + "=" * 60)
    print("Example 1: Minimal Configuration")
    print("=" * 60)

    config = PipelineConfig(
        data=DataConfig(
            ds_path="/path/to/data",
            patient="P001",
            series="S01",
            sig_group_names="R001",
            noise_group_names="R002",
        )
    )

    print("✓ Using all defaults")
    print(f"  - Device: {config.device}")
    print(f"  - Whitening: {config.preprocessing.whitening}")
    print(f"  - Num dipoles: {config.solver.num_dipoles}")
    print(f"  - Optimizer: {config.solver.optimizer}")

    return config


def example_custom_preprocessing():
    """Example 2: Custom preprocessing settings."""
    print("\n" + "=" * 60)
    print("Example 2: Custom Preprocessing")
    print("=" * 60)

    config = PipelineConfig(
        data=DataConfig(
            ds_path="/path/to/data",
            patient="P001",
            series="S01",
            sig_group_names="R001",
            noise_group_names="R002",
        ),
        preprocessing=PreprocessingConfig(
            whitening="ZCA",  # Use ZCA instead of PCA
            bandpass_low=0.5,  # Lower frequency
            bandpass_high=60.0,  # Higher frequency
            bandpass_order=6,  # Higher order filter
            downsample_factor=4,  # More downsampling
            process_segments=False,  # Disable segmentation
        ),
    )

    print("✓ Custom preprocessing configured")
    print(f"  - Whitening: {config.preprocessing.whitening}")
    print(f"  - Bandpass: {config.preprocessing.bandpass_low}-{config.preprocessing.bandpass_high} Hz")
    print(f"  - Filter order: {config.preprocessing.bandpass_order}")
    print(f"  - Downsample: {config.preprocessing.downsample_factor}x")

    return config


def example_custom_solver():
    """Example 3: Custom solver with learning rate schedule."""
    print("\n" + "=" * 60)
    print("Example 3: Custom Solver Configuration")
    print("=" * 60)

    config = PipelineConfig(
        data=DataConfig(
            ds_path="/path/to/data",
            patient="P001",
            series="S01",
            sig_group_names="R001",
            noise_group_names="R002",
        ),
        solver=SolverConfig(
            num_dipoles=2,
            optimizer="adam",  # Use Adam instead of LAMB
            parallel_processing=True,
            max_workers=8,  # More parallel workers
            lr=LearningRateConfig(
                niter=1000,  # More iterations
                warmup=100,  # Longer warmup
                lr_groups=[5e-2, 5e-2, 1e-3],  # Custom learning rates
                early_stopping_patience=100,
            ),
            initializer=InitializerConfig(
                method="pseudo_inv_regularized",
                rcond=0.5,  # Different regularization
                determine_scaling=True,
            ),
        ),
    )

    print("✓ Custom solver configured")
    print(f"  - Optimizer: {config.solver.optimizer}")
    print(f"  - Iterations: {config.solver.lr.niter}")
    print(f"  - Warmup: {config.solver.lr.warmup}")
    print(f"  - LR groups: {config.solver.lr.lr_groups}")
    print(f"  - Workers: {config.solver.max_workers}")

    return config


def example_custom_dipole_positions():
    """Example 4: Custom dipole positions and bounds."""
    print("\n" + "=" * 60)
    print("Example 4: Custom Dipole Positions")
    print("=" * 60)

    # Define custom bounds for dipole positions
    r_bnds = np.array([
        [(-0.2, 0.2), (-0.2, -0.01), (-0.2, 0.2)],  # Fetal: tighter bounds
        [(-0.4, 0.4), (-0.4, -0.01), (0.2, 0.7)]    # Maternal: wider bounds
    ])

    # Define custom initial positions
    r_init = np.array([[[0.01, -0.03, 0.01], [0.0, -0.06, 0.25]]])

    config = PipelineConfig(
        data=DataConfig(
            ds_path="/path/to/data",
            patient="P001",
            series="S01",
            sig_group_names="R001",
            noise_group_names="R002",
        ),
        solver=SolverConfig(
            num_dipoles=2,
            r_bnds=r_bnds,
            r_init=r_init,
        ),
    )

    print("✓ Custom positions configured")
    print(f"  - Bounds shape: {config.solver.r_bnds.shape}")
    print(f"  - Initial positions shape: {config.solver.r_init.shape}")
    print(f"  - Fetal init: {config.solver.r_init[0, 0, :]}")
    print(f"  - Maternal init: {config.solver.r_init[0, 1, :]}")

    return config


def example_gpu_config():
    """Example 5: GPU configuration."""
    print("\n" + "=" * 60)
    print("Example 5: GPU Configuration")
    print("=" * 60)

    config = PipelineConfig(
        data=DataConfig(
            ds_path="/path/to/data",
            patient="P001",
            series="S01",
            sig_group_names="R001",
            noise_group_names="R002",
        ),
        device="cuda:0",  # Use first GPU
        solver=SolverConfig(
            parallel_processing=False,  # Disable CPU parallelism for GPU
            num_dipoles=2,
        ),
    )

    print("✓ GPU configuration")
    print(f"  - Device: {config.device}")
    print(f"  - CPU parallelism disabled (using GPU instead)")

    return config


def example_convert_to_dict():
    """Example 6: Converting between dict and dataclass."""
    print("\n" + "=" * 60)
    print("Example 6: Dict ↔ Dataclass Conversion")
    print("=" * 60)

    # Start with dataclass
    config_dataclass = PipelineConfig(
        data=DataConfig(
            ds_path="/path/to/data",
            patient="P001",
            series="S01",
            sig_group_names="R001",
            noise_group_names="R002",
        ),
        device="cuda:0",
    )

    # Convert to dict (for backward compatibility or serialization)
    config_dict = config_dataclass.to_dict()
    print("✓ Dataclass → Dict")
    print(f"  - Type: {type(config_dict)}")
    print(f"  - Keys: {list(config_dict.keys())[:5]}...")

    # Convert back to dataclass
    config_back = PipelineConfig.from_dict(config_dict)
    print("✓ Dict → Dataclass")
    print(f"  - Type: {type(config_back)}")
    print(f"  - Patient: {config_back.data.patient}")
    print(f"  - Device: {config_back.device}")

    return config_back


def main():
    """Run all configuration examples."""
    print("\n" + "=" * 70)
    print("  fMCG Package - Custom Configuration Examples")
    print("=" * 70)

    # Run all examples
    example_minimal_config()
    example_custom_preprocessing()
    example_custom_solver()
    example_custom_dipole_positions()
    example_gpu_config()
    example_convert_to_dict()

    print("\n" + "=" * 70)
    print("All examples complete!")
    print("=" * 70)
    print("\nKey points:")
    print("  • Dataclasses provide type safety and IDE autocomplete")
    print("  • All parameters have sensible defaults")
    print("  • Easy to customize specific settings")
    print("  • Backward compatible with dict configs")
    print("  • Numpy arrays supported for positions/bounds")
    print("\nSee also:")
    print("  • examples/basic_usage.py - Simple getting started example")
    print("  • fmcg/config.py - Full configuration class definitions")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
